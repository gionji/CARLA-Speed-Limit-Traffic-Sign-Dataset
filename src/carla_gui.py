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
        self.camera_image = None

        # last image captured
        self.clean_camera_image = None

        self.max_sign_distance_from_camera = 20

        self.camera = None

        self.spawnable_actors = []
        self.selected_actor = None
        self.spawn_count = None

        # Store previously spawned actors
        self.spawned_actors = []
        self.spawned_pedestrians = []

        # load a yolo model
        self.yolo_detector = yolo.YOLOv7Detector(MODEL_PATH, classes=[0,1,2])

        # score function
        self.score = DetectionScorer()

        self.root = tk.Tk()
        self.root.title("CARLA Weather Control")

        # Create four frames to divide the main window
        self.frame1 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame2 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame3 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame4 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame5 = tk.Frame(self.root, bg="white", width=400, height=300)
        self.frame6 = tk.Frame(self.root, bg="white", width=400, height=300)

        self.frame1.grid(row=0, column=0, rowspan=2, columnspan=2)
        self.frame2.grid(row=2, column=0, rowspan=2, columnspan=2)
        self.frame3.grid(row=0, column=2, rowspan=2, columnspan=2)
        self.frame4.grid(row=2, column=2, rowspan=2, columnspan=2)
        self.frame5.grid(row=4, column=0, rowspan=2, columnspan=6)
        #self.frame6.grid(row=2, column=6, rowspan=2, columnspan=2)

        self.initialize_carla()
        self.fetch_available_towns()
        self.fetch_spawnable_actors()

        self.create_town_selector()
        self.create_sliders()
        self.create_spawn_camera_button()
        self.create_output_frame()
        self.create_detection_frame()
        self.create_prediction_frame()
        self.create_spawn_actor_frame() 
        self.create_pedestrians_frame()
        self.create_log_frame()
        self.create_randomization_controls()


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
        actor_filter = "*static.*"
        self.spawnable_actors = self.world.get_blueprint_library().filter( actor_filter )



### GUI
    def create_town_selector(self):
        # Create a frame for town selection
        town_frame = tk.Frame(self.frame1, bg="white")
        town_frame.pack(side=tk.TOP, padx=10, pady=10)

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


    def create_spawn_camera_button(self):
        # Create a frame for the spawn button
        spawn_frame = tk.Frame(self.frame2, bg="white")
        spawn_frame.pack(padx=10, pady=10)
        
        spawn_camera_random_button = tk.Button(spawn_frame, text="Spawn camera random", command=self.spawn_camera)
        spawn_camera_random_button.pack()

        spawn_camera_satelite_button = tk.Button(spawn_frame, text="Spawn camera satelite", command=self.spawn_camera_satelite)
        spawn_camera_satelite_button.pack()

        # Create a "Spawn Cars" button
        spawn_button = tk.Button(spawn_frame, text="Spawn Camera in Front of Speed Limit Signs", command=self.spawn_car_in_front_of_speed_limit_signs)
        spawn_button.pack()

        # Create an entry field for specifying the number of objects to spawn
        count_label = tk.Label(spawn_frame, text="Labelind max distance (m) :")
        count_label.pack()
        self.max_sign_distance_from_camera = tk.Entry(spawn_frame)
        self.max_sign_distance_from_camera.insert(0, '20')
        self.max_sign_distance_from_camera.pack()


    def create_spawn_actor_frame(self):
        # Create a frame for spawning actors
        actor_frame = tk.Frame(self.frame2, bg="white")
        actor_frame.pack(padx=10, pady=10)

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
        spawn_button = tk.Button(actor_frame, text="Spawn Asset", command=self.spawn_actors)
        spawn_button.pack()


    def create_output_frame(self):
        # Create a frame for the output image
        image_frame = tk.Frame(self.frame3, bg="white")
        image_frame.pack(padx=10, pady=10)  # Position it on the right with padding
        # Create a canvas to display the image
        self.camera_canvas = tk.Canvas(image_frame, width=640, height=480)
        self.camera_canvas.pack()


    def create_prediction_frame(self):
        # Create a frame for the output image
        image_frame = tk.Frame(self.frame4, bg="white")
        image_frame.pack(padx=10, pady=10)  # Position it on the right with padding
        # Create a canvas to display the image
        self.prediction_canvas = tk.Canvas(image_frame, width=640, height=480)
        self.prediction_canvas.pack()

    
    def update_camera_canvas(self, img):
        img = cv2.resize(img, (640, 640))
        self.camera_image = img
        # Display the output image in the canvasnum_pedestrians
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.camera_canvas.image = img

    def update_prediction_canvas(self, img):
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
        sliders_frame.pack(padx=10, pady=10)

        # Create horizontal sliders for weather parameters
        #weather_parameters_names = ['cloudiness', 'dust_storm', 'fog_density', 'fog_distance', 'fog_falloff', 'mie_scattering_scale', 'precipitation', 'precipitation_deposits', 'rayleigh_scattering_scale', 'scattering_intensity', 'sun_altitude_angle', 'sun_azimuth_angle', 'wetness', 'wind_intensity']
        weather_parameters_names = ['cloudiness',  'fog_distance', 'fog_falloff', 
                                    'scattering_intensity', 
                                    'sun_altitude_angle', 'sun_azimuth_angle', 
                                    'wetness']

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
        randomization_frame = tk.Frame(self.frame2, bg="white")
        randomization_frame.pack(padx=10, pady=10)
        # Create a "Start Randomization" button
        start_button = tk.Button(randomization_frame, text="Start Randomization", command=self.start_randomization)
        start_button.grid(row=0, column=0)
        # Create a "Stop Randomization" button
        stop_button = tk.Button(randomization_frame, text="Stop Randomization", command=self.stop_randomization)
        stop_button.grid(row=0, column=1)

        run_single_experiment_button = tk.Button(randomization_frame, text="Run single experiment", command=self.run_single_experiment)
        run_single_experiment_button.grid(row=1, column=0)

        run_full_experiment_button = tk.Button(randomization_frame, text="Run full experiment", command=self.run_full_experiment)
        run_full_experiment_button.grid(row=1, column=1)


    def create_detection_frame(self):
        # Create a frame for the spawn button
        detection_frame = tk.Frame(self.frame4, bg="white")
        detection_frame.pack(padx=10, pady=10)
        # Create a "Spawn Cars" button
        run_yolo_button = tk.Button(detection_frame, text="Run yolo on shown frame", command=self.run_yolo)
        run_yolo_button.pack()


    def create_log_frame(self):
        # Create a frame for the output log area
        output_frame = tk.Frame(self.frame5, bg="white")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a Text widget for the log
        self.log = tk.Text(output_frame, height=10, state=tk.DISABLED)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a Scrollbar and associate it with the log Text widget
        scrollbar = tk.Scrollbar(output_frame, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.config(yscrollcommand=scrollbar.set)

    def create_pedestrians_frame(self):
        # Frame to input the number of pedestrians
        entry_frame = tk.Frame(self.frame2)
        entry_frame.pack()

        entry_label = tk.Label(entry_frame, text="Number of Pedestrians:")

        self.num_pedestrians = tk.StringVar()
        self.num_pedestrians.set("1")
        num_pedestrians_entry = tk.Entry(entry_frame, textvariable=self.num_pedestrians)

        entry_label.pack(side=tk.LEFT)
        num_pedestrians_entry.pack(side=tk.LEFT)

        # Button to spawn pedestrians
        self.spawn_button = tk.Button(entry_frame, text="Spawn Pedestrians", command=self.spawn_pedestrians)
        self.spawn_button.pack()

        # Button to delete pedestrians
        self.delete_button = tk.Button(entry_frame, text="Delete Pedestrians", command=self.delete_pedestrians)
        self.delete_button.pack()

        # Button to delete pedestrians
        self.randomize_pedestrian_velocity_button = tk.Button(entry_frame, 
                                                              text="Randomize movement", 
                                                              command=self.randomize_pedestrian_velocity)
        self.randomize_pedestrian_velocity_button.pack()


    def append_to_log(self, text):
        text = '\n' + str(text)
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.yview(tk.END) 
        self.log.config(state=tk.DISABLED)

    def delete_log(self, text):
        self.log.config(state=tk.NORMAL)
        self.log.delete(1.0, tk.END)
        self.log.config(state=tk.DISABLED)
        
    ### #############################################################################################################################3

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
        self.spawn_car_in_front_of_speed_limit_signs()


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
            self.append_to_log('New experiment ...')
            for param_name, checkbox in self.checkboxes.items():
                if checkbox.get() == 1:
                    new_value = random.randint(0, 100)
                    self.sliders[param_name].set(new_value)
                    setattr(self.weather_parameters, param_name, new_value)
                    self.world.set_weather(self.weather_parameters)
                    self.append_to_log( (param_name, new_value) ) 
            
            self.run_full_experiment()

            time.sleep(0.01)

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



    def bbox_to_yolo_annotation(self, class_id, bbox, image_height, image_width):
        x1, x2, y1, y2 = bbox
        # Calculate normalized coordinates and dimensions
        bbox_width = (x2 - x1) / image_width
        bbox_height = (y2 - y1) / image_height
        center_x = (x1 + x2) / (2.0 * image_width)
        center_y = (y1 + y2) / (2.0 * image_height)

        # Construct YOLO annotation string
        annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        return annotation

 
    def is_bbox_in_the_frame(self, bbox, frame):
        xmin, xmax, ymin, ymax = bbox
        height, width, channels = frame.shape
        # Check if all coordinates of the bounding box are within the window boundaries
        if xmin >= 0 and xmax <= width and ymin >= 0 and ymax <= height:
            return True
        else:
            return False


    def is_object_face_another_object(self, ego, sign):
        digits = 1

        (Xa, Ya, Ra) = ego
        (Xb, Yb, Rb) = sign

        tetha_a = Ra
        tetha_b = Rb
    
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

        self.update_camera_canvas(img)

        self.camera.destroy()


    def spawn_pedestrians(self):
            # Get the blueprint library to create pedestrians
            blueprint_library = self.world.get_blueprint_library()

            # Select the pedestrian blueprint
            pedestrian_bp = random.choice( blueprint_library.filter("*walker.pedestrian*") )

            # Define the pedestrian spawn points
            spawn_points = self.world.get_map().get_spawn_points()

            # Spawn specified number of pedestrians
            num_to_spawn = int(self.num_pedestrians.get())

            print('Number of pedestrian to be spawned: ', num_to_spawn)

            for i in range(num_to_spawn):
                print('Spawning pedestrina nr. ', i)
                spawn_point = random.choice(spawn_points)
                pedestrian = self.world.spawn_actor(pedestrian_bp, spawn_point)
                self.spawned_pedestrians.append(pedestrian)

                # Set pedestrian attributes (e.g., speed, behavior, appearance)
                # Example:
                pedestrian.set_target_velocity(carla.Vector3D(x=1.0, y=0.0, z=1.0))
                pedestrian.apply_control(carla.WalkerControl(speed=1.5))

    def delete_pedestrians(self):
        # Destroy all spawned pedestrians
        for pedestrian in self.spawned_pedestrians:
            pedestrian.destroy()
        # Clear the list of spawned pedestrians
        self.spawned_pedestrians = []


    def randomize_pedestrian_velocity(self):
        for pedestrian in self.spawned_pedestrians:
            xrand = random.uniform(-1.0, 1.0)
            yrand = random.uniform(-1.0, 1.0)
            pedestrian.set_target_velocity(carla.Vector3D(x=xrand, y=yrand, z=0))
        return



    def spwan_random_object_in_front_of_the_camera(self):
        return
        


    def spawn_car_in_front_of_speed_limit_signs(self, distance_in_front=10.0, full_experiment=False):
        # Get the CARLA world
        world = self.client.get_world()

        #traffic_signs = world.get_actors().filter('traffic.traffic_sign')
        #ts_bboxes = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

        # spawn and set camera attributes
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()  
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        ## set the camera relative position to the sign
        relative_camera_sign_location = carla.Location(x=-1,y=distance_in_front,z=1.9)
        camera_init_trans = carla.Transform(relative_camera_sign_location, carla.Rotation(yaw = -90))
                
        traffic_signs = world.get_actors().filter('traffic.speed_limit.*')
        bounding_box_set_traffic_signs=world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

        random_sign = random.choice(traffic_signs)

        traffic_signs_list = list()

        if full_experiment:
            traffic_signs_list = traffic_signs
        else:
            traffic_signs_list.append(random_sign)

        experiment_score = 0

        ## In teoria questo cicla su tutti!
        for sign in traffic_signs_list:
            traffic_sign_max_speed = str(sign.type_id).split('.')[2]
            ## get the corresponding class_id and name
            class_id, class_name = self.get_class_id_and_class_name_from_substring(traffic_sign_max_speed)

            self.camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=sign)
            #print('Traffic sign max speed: ', traffic_sign_max_speed, '-----------------')  

            image_queue = queue.Queue()
            self.camera.listen(image_queue.put)
            camera_transform = self.camera.get_transform()

            # Get the world to camera matrix
            world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

            # Calculate the camera projection matrix to project from 3D -> 2D
            K = self.build_projection_matrix(image_w, image_h, fov)

            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            clean_image = img.copy()

            # list containing every bbox, in a yolo aoutput format
            annotations = list()

            max_dist = int(self.max_sign_distance_from_camera.get())

            for bb in bounding_box_set_traffic_signs:                
                # Filter for distance from ego traffic_sign_actor
                if bb.location.distance(camera_transform.location) < max_dist:                    
                    # Calculate the dot product between the forward vector
                    forward_vec = camera_transform.get_forward_vector()
                    ray = bb.location - camera_transform.location

                    # Control if the camara is facing the sign
                    ego_tetha  = - (int(camera_transform.rotation.yaw) - 0)
                    ego_coord  =   (int(camera_transform.location.x), int(camera_transform.location.y),  ego_tetha)

                    sign_tetha = - (int(bb.rotation.yaw) - 0)
                    sign_coord =   (int(bb.location.x), int(bb.location.y), sign_tetha )

                    is_facing  = self.is_object_face_another_object( ego_coord, sign_coord )

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

                        bbox_pos  = (int(x_min), int(x_max), int(y_min), int(y_max))
                        text_orig = (int(x_min), int(y_min))

                        #check if the bounding box is on screen
                        #is_bbox_in_the_frame = self.is_bbox_in_the_frame(bbox_pos, img)

                        # Add the YOLO annotation text to the image
                        cv2.putText(img, str(((class_id * 30) + 30)), text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # convert the bboxes to a yolo annotation line
                        yolo_annotation = self.bbox_to_yolo_annotation(class_id, bbox_pos, img.shape[0], img.shape[1])
                        # append to a list
                        annotations.append(yolo_annotation)

                    # print the bboxes array
                    for i in range(0, len(annotations)):
                        cv2.putText(img, str(annotations[i]), (50, 10 + (20*i)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # for every sign (in this case just one. one random picked)

            self.clean_camera_image = cv2.resize(clean_image, (640, 640))
            self.update_camera_canvas(img)   
                  
            # get predictions
            pred_img, preds = self.run_yolo()
   
            preds_list = [list(map(float, item.split())) for item in preds]
            truth_list = [list(map(float, item.split())) for item in annotations]

            #score, scores = self.score.calculate_confusion_matrix(truth_list, preds_list)
            scores = self.score.calculate_confusion_matrix(truth_list, preds_list)

            print( scores )

            #experiment_score += score
 
            self.camera.destroy()
        
        return experiment_score

    



    def convert_to_yolo_annotation(self, predictions, image_height, image_width):
        yolo_annotations = []
        for prediction in predictions[0]:
            x1, y1, x2, y2, confidence, class_id = prediction.tolist()

            # Calculating center x, center y, width, and height of the bounding box
            bbox_width  = (x2 - x1) / image_width
            bbox_height = (y2 - y1) / image_height
            center_x    = (x1 + x2) / (2.0 * image_width)
            center_y    = (y1 + y2) / (2.0 * image_height)

            # Construct YOLO annotation string
            annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
            yolo_annotations.append(annotation)

        return yolo_annotations



    def run_yolo(self, image=None):
        if image is None:
            image = self.clean_camera_image

        frame=image.copy()      
        pred_img , preds = self.yolo_detector.detect( frame )

        # trasform the prediction format as the input
        yolo_preds_annotations = self.convert_to_yolo_annotation(preds, pred_img.shape[0], pred_img.shape[1])

        self.update_prediction_canvas(pred_img)

        return pred_img, yolo_preds_annotations



    def run_single_experiment(self):
        score = self.spawn_car_in_front_of_speed_limit_signs(full_experiment=False)
        self.append_to_log(('Experiment score: ', score) )
        return
    

    def run_full_experiment(self):
        score = self.spawn_car_in_front_of_speed_limit_signs(full_experiment=True)
        self.append_to_log(('Experiment score: ', score) )
        return


    def run(self):
        self.root.mainloop()


    def initialize_app(self):
        #load yolo
        self.yolo_detector = yolo.YOLOv7Detector(MODEL_PATH, classes=[0,1,2])

        self.spawn_car_in_front_of_speed_limit_signs()

        self.update_camera_canvas(self.camera_image)
        self.run_yolo()
        return



class DetectionScorer:
    def __init__(self):
        # Define scores for different cases
        self.scores = {
            'true_positive_detection':   1.0,
            'false_positive_detection':  1.0,
            'true_negative_detection':   1.0,
            'false_negative_detection':  1.0,

            'true_positive_classification': 0.0,
            'false_positive_classification': 0.0,
            'true_negative_classification': 0.0,
            'false_negative_classification': 0.0,
        }

    def calculate_scores(self, truth, preds, threshold=0.7):
        scores = {
            'detection': {
                'true_positive':  0,
                'false_positive': 0,
                'true_negative':  0,
                'false_negative': 0
            },
            'classification': {
                'true_positive':  0,
                'false_positive': 0,
                'true_negative':  0,
                'false_negative': 0
            }
        }
        total_score = 0  # Initialize total score

        if len(truth) > len(preds):
            truth, preds = preds, truth

        for truth_box in truth:
            best_iou = 0
            best_match = None
            for pred_box in preds:
                if int(truth_box[0]) == int(pred_box[0]):
                    iou = self.calculate_iou(truth_box, pred_box)
                    print('iou', iou)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_box

            if best_iou > threshold:
                if best_match:
                    scores['detection']['true_positive'] += self.scores['true_positive_detection']
                    # Check for correct classification
                    if int(truth_box[0]) == int(best_match[0]):
                        scores['classification']['true_positive'] += self.scores['true_positive_classification']
                    else:
                        scores['classification']['false_positive'] += self.scores['false_positive_classification']
                else:
                    scores['detection']['false_negative'] += self.scores['false_negative_detection']

            else:
                if best_match:
                    scores['detection']['false_positive']     += self.scores['false_positive_detection']
                    scores['classification']['true_negative'] += self.scores['true_negative_classification']
                else:
                    scores['detection']['true_negative'] += self.scores['true_negative_detection']
                    scores['classification']['false_negative'] += self.scores['false_negative_classification']

        # Sum all the scores
        for key, value in scores.items():
            for sub_key, sub_value in value.items():
                total_score += sub_value

        return total_score, scores
    

    def count_correct_detections(self, truth, preds, threshold=0.5):
        correct_detections = 0

        # To avoid out of range errors, loop through the smaller list
        if len(truth) > len(preds):
            truth, preds = preds, truth

        for truth_box in truth:
            best_iou = 0
            for pred_box in preds:
                if int(truth_box[0]) == int(pred_box[0]):
                    iou = self.calculate_iou(truth_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou

            if best_iou > threshold:
                correct_detections += 1

        return correct_detections


    def calculate_iou(self, box1, box2):
        # Extract coordinates and areas
        box1_class, box1_x, box1_y, box1_w, box1_h = box1
        box2_class, box2_x, box2_y, box2_w, box2_h = box2

        # Convert the coordinates from YOLO format to regular bounding box format
        box1_x1, box1_x2 = box1_x - box1_w / 2, box1_x + box1_w / 2
        box1_y1, box1_y2 = box1_y - box1_h / 2, box1_y + box1_h / 2
        box2_x1, box2_x2 = box2_x - box2_w / 2, box2_x + box2_w / 2
        box2_y1, box2_y2 = box2_y - box2_h / 2, box2_y + box2_h / 2

        # Calculate the intersection coordinates
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # Calculate the intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the individual areas of the bounding boxes
        box1_area = box1_w * box1_h
        box2_area = box2_w * box2_h

        # Calculate the union area
        union = box1_area + box2_area - intersection

        # Calculate IOU
        iou = intersection / union if union > 0 else 0

        return iou
    
    def calculate_confusion_matrix(self, ground_truth, predicted, threshold=0.7):
        # Initialize confusion matrix
        confusion_matrix = {}

        for gt_box in ground_truth:
            gt_class = gt_box[0]
            matched = False

            for pred_box in predicted:
                pred_class = pred_box[0]

                # Check if classes match
                if pred_class == gt_class:
                    gt_xmin, gt_xmax, gt_ymin, gt_ymax = gt_box[1:]
                    pred_xmin, pred_xmax, pred_ymin, pred_ymax = pred_box[1:]

                    # Calculate Intersection over Union (IoU)
                    x_left = max(gt_xmin, pred_xmin)
                    x_right = min(gt_xmax, pred_xmax)
                    y_top = max(gt_ymin, pred_ymin)
                    y_bottom = min(gt_ymax, pred_ymax)

                    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
                    gt_box_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
                    pred_box_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
                    union_area = gt_box_area + pred_box_area - intersection_area

                    iou = intersection_area / union_area

                    if iou > threshold:  # Threshold for considering a match
                        matched = True
                        if gt_class not in confusion_matrix:
                            confusion_matrix[gt_class] = {gt_class: 1}
                        else:
                            if gt_class not in confusion_matrix[pred_class]:
                                confusion_matrix[gt_class][pred_class] = 1
                            else:
                                confusion_matrix[gt_class][pred_class] += 1

            # If no match found for ground truth box, it's a false negative
            if not matched:
                if gt_class not in confusion_matrix:
                    confusion_matrix[gt_class] = {gt_class: 0}
                else:
                    confusion_matrix[gt_class][gt_class] = 0

        return confusion_matrix




if __name__ == '__main__':
    app = CarlaWeatherApp()

    try:
        app.initialize_app()
    except:
        pass

    app.run()