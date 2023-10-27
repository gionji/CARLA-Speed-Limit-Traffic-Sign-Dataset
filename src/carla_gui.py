import math
import tkinter as tk
import carla
import random
import threading
import time
import queue
import numpy as np
import cv2
import time
import torch
from PIL import Image, ImageTk

import yolov7_detector as yolo



import sys
sys.path.insert(0, '/home/gionji/yolov7')

CLASS_MAPPING = {
    'SL_30_KMH': 0,
    'SL_60_KMH': 1,
    'SL_90_KMH': 2,
}

MODEL_PATH = '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt'  # Path to your YOLOv7 model file
CLASS_FILE = '/home/gionji/yolov7/runs/train/carla_gio_072/class_names.txt'  # Path to a file containing class names




class CarlaWeatherApp:
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port
        self.client = carla.Client(host, port)
        self.world = None
        self.weather_parameters = None
        self.available_towns = []
        self.current_town = None
        self.sliders = {}
        self.checkboxes = {}
        self.randomization_thread = None
        self.randomization_running = False
        self.output_image = None

        self.camera = None

        self.spawnable_actors = []
        self.selected_actor = None
        self.spawn_count = None

        # Store previously spawned actors
        self.spawned_actors = []

        # load a yolo model
        self.yolo_detector = yolo.YOLOv7Detector(MODEL_PATH, classes=[0,1,2])

        self.root = tk.Tk()
        self.root.title("CARLA Weather Control")

        # Create four frames to divide the main window
        self.frame1 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame2 = tk.Frame(self.root, bg="white", width=400, height=50)
        self.frame3 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame4 = tk.Frame(self.root, bg="white", width=400, height=50)

        self.frame1.grid(row=0, column=0, rowspan=2, columnspan=2)
        self.frame2.grid(row=2, column=0, rowspan=2, columnspan=2)
        self.frame3.grid(row=0, column=2, rowspan=2, columnspan=2)
        self.frame4.grid(row=2, column=2, rowspan=2, columnspan=2)

        self.initialize_carla()
        self.fetch_available_towns()
        self.fetch_spawnable_actors()

        self.create_town_selector()
        self.create_sliders()
        self.create_randomization_controls()
        self.create_spawn_button()
        self.create_output_frame()
        self.create_prediction_frame()
        self.create_spawn_actor_frame() 
        self.create_detection_frame()
        

    def initialize_carla(self):
        try:
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.weather_parameters = self.world.get_weather()
            self.current_town = self.world.get_map().name
        except Exception as e:
            print(f"Error initializing CARLA simulation: {str(e)}")

    def fetch_available_towns(self):
        self.available_towns = [town for town in self.client.get_available_maps()]


    def fetch_spawnable_actors(self):
        # Retrieve the spawnable actors (you can customize this based on your needs)
        # Here, we are fetching vehicle blueprints for demonstration purposes
        actor_filter = "*vehicle.*"
        self.spawnable_actors = self.world.get_blueprint_library().filter( actor_filter )


    def create_town_selector(self):
        # Create a frame for town selection
        town_frame = tk.Frame(self.frame1, bg="white")
        town_frame.pack()

        # Create a dropdown menu for selecting towns
        town_label = tk.Label(town_frame, text="Select Town:")
        town_label.pack(side=tk.LEFT)
        town_var = tk.StringVar(self.frame1)
        town_var.set(self.current_town)  # Set the default town to the current town

        town_dropdown = tk.OptionMenu(town_frame, town_var, *self.available_towns)
        town_dropdown.pack(side=tk.LEFT)

        # Create an "Apply" button to change the town
        apply_town_button = tk.Button(town_frame, text="Apply Town", command=lambda: self.apply_town(town_var.get()))
        apply_town_button.pack(side=tk.LEFT)


    def create_spawn_button(self):
        # Create a frame for the spawn button
        spawn_frame = tk.Frame(self.frame1, bg="white")
        spawn_frame.pack()

        # Create a "Spawn Cars" button
        spawn_button = tk.Button(spawn_frame, text="Spawn Cars in Front of Speed Limit Signs", command=self.spawn_car_in_front_of_speed_limit_signs)
        spawn_button.pack()

        spawn_camera_random_button = tk.Button(spawn_frame, text="Spawn camera random", command=self.spawn_camera)
        spawn_camera_random_button.pack()

        spawn_camera_satelite_button = tk.Button(spawn_frame, text="Spawn camera satelite", command=self.spawn_camera_satelite)
        spawn_camera_satelite_button.pack()
    

    def create_spawn_actor_frame(self):
        # Create a frame for spawning actors
        actor_frame = tk.Frame(self.frame1, bg="white")
        actor_frame.pack()

        # Create a dropdown menu for selecting spawnable actors
        actor_label = tk.Label(actor_frame, text="Select Actor:")
        actor_label.pack()
        self.actor_var = tk.StringVar(self.frame1)
        self.actor_var.set(self.spawnable_actors[0].id)  # Set the default actor to the first in the list
        actor_dropdown = tk.OptionMenu(actor_frame, self.actor_var, *self.get_actor_names())
        actor_dropdown.pack()

        # Create an entry field for specifying the number of objects to spawn
        count_label = tk.Label(actor_frame, text="Spawn Count:")
        count_label.pack()
        self.spawn_count = tk.Entry(actor_frame)
        self.spawn_count.pack()

        # Create a "Spawn" button to initiate random spawning
        spawn_button = tk.Button(actor_frame, text="Spawn", command=self.spawn_actors)
        spawn_button.pack()



    def create_output_frame(self):
        # Create a frame for the output image
        image_frame = tk.Frame(self.frame3, bg="white")
        image_frame.pack()  # Position it on the right with padding
        # Create a canvas to display the image
        self.output_canvas = tk.Canvas(image_frame, width=640, height=480)
        self.output_canvas.pack()

    def update_output_image(self, img):
        self.output_image = img
        # Display the output image in the canvas
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.output_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.output_canvas.image = img


    def create_prediction_frame(self):
        # Create a frame for the output image
        image_frame = tk.Frame(self.frame4, bg="white")
        image_frame.pack()  # Position it on the right with padding
        # Create a canvas to display the image
        self.prediction_canvas = tk.Canvas(image_frame, width=640, height=480)
        self.prediction_canvas.pack()

    def update_prediction_image(self, img):
        self.prediction_image = img
        # Display the output image in the canvas
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.prediction_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.prediction_canvas.image = img


    def create_sliders(self):
        # Create a frame for weather sliders
        sliders_frame = tk.Frame(self.frame1, bg="white")
        sliders_frame.pack()

        # Create horizontal sliders for weather parameters
        #weather_parameters_names = ['cloudiness', 'dust_storm', 'fog_density', 'fog_distance', 'fog_falloff', 'mie_scattering_scale', 'precipitation', 'precipitation_deposits', 'rayleigh_scattering_scale', 'scattering_intensity', 'sun_altitude_angle', 'sun_azimuth_angle', 'wetness', 'wind_intensity']
        weather_parameters_names = ['cloudiness',  'fog_distance', 'fog_falloff', 'scattering_intensity', 'sun_altitude_angle', 'sun_azimuth_angle', 'wetness']

        for param_name in weather_parameters_names:
            param_value = getattr(self.weather_parameters, param_name, None)
            if isinstance(param_value, (float, int)):
                param_frame = tk.Frame(sliders_frame)
                param_frame.pack(anchor=tk.W)
                
                # Create a checkbox for automatic edits
                edit_var = tk.IntVar()
                edit_checkbox = tk.Checkbutton(param_frame, text="Edit Automatically", variable=edit_var)
                edit_checkbox.grid(row=0, column=0)
                self.checkboxes[param_name] = edit_var
                
                slider = tk.Scale(param_frame, label=param_name, from_=0, to=100, orient=tk.HORIZONTAL)
                slider.set(param_value)
                self.sliders[param_name] = slider
                slider.grid(row=0, column=1)

        # Create an "Apply" button to apply weather settings
        apply_button = tk.Button(sliders_frame, text="Apply Weather", command=self.apply_weather_settings)
        apply_button.pack()

    def create_randomization_controls(self):
        # Create a frame for randomization controls
        randomization_frame = tk.Frame(self.frame1, bg="white")
        randomization_frame.pack()

        # Create a "Start Randomization" button
        start_button = tk.Button(randomization_frame, text="Start Randomization", command=self.start_randomization)
        start_button.grid(row=0, column=0)

        # Create a "Stop Randomization" button
        stop_button = tk.Button(randomization_frame, text="Stop Randomization", command=self.stop_randomization)
        stop_button.grid(row=0, column=1)

    def create_detection_frame(self):
        # Create a frame for the spawn button
        detection_frame = tk.Frame(self.frame1, bg="white")
        detection_frame.pack()

        # Create a "Spawn Cars" button
        run_yolo_button = tk.Button(detection_frame, text="Run yolo on shown frame", command=self.run_yolo)
        run_yolo_button.pack()
        

    def get_actor_names(self):
        # Get the names of spawnable actors
        actor_names = [actor.id for actor in self.spawnable_actors]
        return actor_names
    

    def spawn_actors(self):
        # Remove previously spawned actors
        for actor in self.spawned_actors:
            try:
                actor.destroy()
            except:
                pass

        selected_actor_id = self.actor_var.get()
        spawn_count = self.spawn_count.get()
        
        try:
            spawn_count = int(spawn_count)
        except ValueError:
            print("Invalid spawn count. Please enter an integer.")
            return
        
        selected_actor = None
        for actor in self.spawnable_actors:
            if actor.id == selected_actor_id:
                selected_actor = actor
                break

        if selected_actor is not None:
            # Spawn the selected actor the specified number of times
            for _ in range(spawn_count):
                spawn_location = self.get_random_spawn_location()
                if spawn_location:
                    try:
                        actor = self.world.spawn_actor(selected_actor, spawn_location)
                        self.spawned_actors.append(actor)
                    except Exception as e:
                        print('Error spawning oject: ', e)
        else:
            print("Selected actor not found.")



    def get_random_spawn_location(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            return random.choice(spawn_points)
        return None




    def apply_town(self, selected_town):
        # Change the map to the selected town
        try:
            self.client.load_world(selected_town)
            print(f"Changed map to: {selected_town}")
        except Exception as e:
            print(f"Error changing map: {str(e)}")

    def apply_weather_settings(self):
        # Get slider values and update the weather parameters
        for param_name, slider in self.sliders.items():
            new_value = slider.get()
            setattr(self.weather_parameters, param_name, new_value)

        # Apply the changes to CARLA
        self.world.set_weather(self.weather_parameters)


    def start_randomization(self):
        if self.randomization_thread is None:
            self.randomization_thread = threading.Thread(target=self.randomize_weather)
            self.randomization_thread.start()

    def stop_randomization(self):
        self.randomization_running = False
        if self.randomization_thread:
            self.randomization_thread.join()
            self.randomization_thread = None

    def randomize_weather(self):
        self.randomization_running = True
        while self.randomization_running:
            for param_name, checkbox in self.checkboxes.items():
                if checkbox.get() == 1:
                    new_value = random.randint(0, 100)
                    self.sliders[param_name].set(new_value)
                    setattr(self.weather_parameters, param_name, new_value)
                    self.world.set_weather(self.weather_parameters)
            time.sleep(0.2)

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def get_class_id_and_class_name_from_substring(self, substring):
        for key, value in CLASS_MAPPING.items():
            if substring in key:
                return value, key
        raise ValueError(f"Substring '{substring}' not found in CLASS_MAPPING keys")
    
    def bbox_to_yolo_annotation(self, class_id, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        # Calculate normalized coordinates and dimensions
        bbox_width = (x2 - x1) / image_width
        bbox_height = (y2 - y1) / image_height
        center_x = (x1 + x2) / (2.0 * image_width)
        center_y = (y1 + y2) / (2.0 * image_height)

        # Construct YOLO annotation string
        annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        return annotation


    def can_objects_face_each_other(self, ego, sign):
        digits = 1

        (Xa, Ya, Ra) = ego
        (Xb, Yb, Rb) = sign

        tetha_a = -(Ra + 90)
        tetha_b = -(Rb + 90)

        tetha_a = -(Ra + 90)
        tetha_b = -(Rb + 90)

        # Calculate direction vectors
        ego_direction =  ( round(math.cos(math.radians(tetha_a)), 2), round(math.sin(math.radians(tetha_a)), 2) )
        sign_direction = ( -round(math.cos(math.radians(tetha_b)), 2), round(math.sin(math.radians(tetha_b)), 2) )

        # Calculate vectors from A to B and from B to A
        difference_between_ego_and_sign = (round(Xb - Xa, digits), round(Yb - Ya, digits))
        difference_between_sign_and_ego = (round(Xa - Xb, digits), round(Ya - Yb, digits))
        
        # Calculate the dot products
        dot_product_A_to_B = round(np.dot(ego_direction,  difference_between_ego_and_sign), digits)
        dot_product_B_to_A = round(np.dot(sign_direction, difference_between_sign_and_ego), digits)
        
        # Check if both objects are within the FoV of each other
        see_each_other = dot_product_A_to_B < 0 and dot_product_B_to_A < 0 

        return see_each_other


    def spawn_camera_on_waypoint(self):
        return
    
    ## TODO make a view where specify the position

    def spawn_camera_satelite(self):
        # TODO specify altitude 
        altitude = 300
        lat = random.uniform(-100, 100)
        long = random.uniform(-100, 100)
        camera_tranform = carla.Transform(carla.Location(x=lat,y=long,z=altitude), carla.Rotation(pitch = -90))
        self.spawn_camera(camera_tranform)
        return


    def spawn_camera(self, transform=None):
        # Get the CARLA world
        world = self.client.get_world()

        # spawn and set camera attributes
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()  
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        if transform is None:
            up_val = 5
            x_rand = random.uniform(-2, up_val)
            y_rand = random.uniform(-2, up_val)
            z_rand = random.uniform( 0, up_val)
            yaw_rand = random.uniform( 0, 359)
            camera_init_trans = carla.Transform(carla.Location(x=x_rand,y=y_rand,z=z_rand), carla.Rotation(yaw = yaw_rand))
        else:
            camera_init_trans = transform
        
        self.camera = world.spawn_actor(camera_bp, camera_init_trans)#, attach_to=sign)

        image_queue = queue.Queue()
        self.camera.listen(image_queue.put)
        transform = self.camera.get_transform()

        # Get the world to camera matrix
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

        # Calculate the camera projection matrix to project from 3D -> 2D
        K = self.build_projection_matrix(image_w, image_h, fov)

        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        self.update_output_image(img)

        self.camera.destroy()




    def spawn_car_in_front_of_speed_limit_signs(self, distance_in_front=10.0):
        # Get the CARLA world
        world = self.client.get_world()

        # Step 1: Retrieve the list of traffic signs
        #traffic_signs = world.get_actors().filter('traffic.traffic_sign')
        #ts_bboxes = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        traffic_signs = world.get_actors().filter('traffic.speed_limit.*')

        # spawn and set camera attributes
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()  
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        ## set the camera relative position to the sign
        relative_camera_sign_location = carla.Location(x=-1,y=+4,z=1.9)
        camera_init_trans = carla.Transform(relative_camera_sign_location, carla.Rotation(yaw = -90))
        
        bounding_box_set_traffic_signs=world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

        random_sign = random.choice(traffic_signs)

        traffic_signs_list = list()
        traffic_signs_list.append(random_sign)

        for sign in traffic_signs_list:
            traffic_sign_max_speed = str(sign.type_id).split('.')[2]
            ## get the corresponding class_id and name
            class_id, class_name = self.get_class_id_and_class_name_from_substring(traffic_sign_max_speed)

            self.camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=sign)
            print('Traffic sign max speed: ', traffic_sign_max_speed)  

            image_queue = queue.Queue()
            self.camera.listen(image_queue.put)
            transform = self.camera.get_transform()

            # Get the world to camera matrix
            world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

            # Calculate the camera projection matrix to project from 3D -> 2D
            K = self.build_projection_matrix(image_w, image_h, fov)

            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            annotations = list()

            for bb in bounding_box_set_traffic_signs:
                # Filter for distance from ego traffic_sign_actor
                # TODO: aggiungi controllo face to face
                random_sign
                ego_coord = (transform.location.x, transform.location.y, transform.rotation.yaw )
                sign_coord = (bb.location.x, bb.location.y, bb.rotation.yaw )
                is_facing = self.can_objects_face_each_other(ego_coord, sign_coord )

                if bb.location.distance(transform.location) < 50:
                    print(is_facing, ego_coord, sign_coord)
                    # Calculate the dot product between the forward vector
                    # of the traffic_sign_actor and the vector between the traffic_sign_actor
                    # and the bounding box. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = transform.get_forward_vector()
                    ray = bb.location - transform.location

                    if forward_vec.dot(ray) > 1:
                        # Cycle through the vertices
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]

                        # Join the vertices into edges
                        p1 = self.get_image_point(bb.location, K, world_2_camera)
                        
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = self.get_image_point(vert, K, world_2_camera)
                            # Find the rightmost vertex
                            if p[0] > x_max:
                                x_max = p[0]
                            # Find the leftmost vertex
                            if p[0] < x_min:
                                x_min = p[0]
                            # Find the highest vertex
                            if p[1] > y_max:
                                y_max = p[1]
                            # Find the lowest  vertex
                            if p[1] < y_min:
                                y_min = p[1]              

                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,255,255, 255), 1)

                        bbox_pos = (x_min, x_max, y_min, y_max)
                        text_orig = (int(x_min), int(y_min))

                        # print the max speed on the label
                        # Add the YOLO annotation text to the image
                        cv2.putText(img, str(traffic_sign_max_speed), text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # convert the bboxes to a yolo annotation line
                        yolo_annotation = self.bbox_to_yolo_annotation(class_id, bbox_pos, image.width, image.height)
                        # append to a list
                        annotations.append(yolo_annotation)

                    # print the bboxes array
                    for i in range(0, len(annotations)):
                        cv2.putText(img, str(annotations[i]), (50, 10 + (20*i)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    # make a classification

                    # check the results with a ground truth

            self.update_output_image(img)

            self.camera.destroy()


    def compare_prediction_and_truth():
        return

    def run_yolo(self, image=None):
        if image is None:
            image = self.output_image

        frame=image.copy()       
        pred = self.yolo_detector.detect(frame)
        self.update_prediction_image(pred)

    def run(self):
        self.root.mainloop()



class Experiment:
    def __init__(self, host):
        self.host = host



if __name__ == '__main__':
    app = CarlaWeatherApp()

    app.run()