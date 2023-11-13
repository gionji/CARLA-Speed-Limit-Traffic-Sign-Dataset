import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os
import string


def initialize_dataset_folders(output_folder_name_with_path):
    # Extract the dataset name from the output folder path
    dataset_name = os.path.basename(os.path.normpath(output_folder_name_with_path))

    # Generate a random HEX_MARK
    hex_mark = ''.join(random.choices(string.hexdigits, k=8))

    # Create the dataset folder with HEX_MARK
    dataset_folder = os.path.join(output_folder_name_with_path, dataset_name + "_" + hex_mark)
    os.makedirs(dataset_folder, exist_ok=True)

    # Create the data folder
    data_folder = os.path.join(dataset_folder, "data")
    os.makedirs(data_folder, exist_ok=True)

    # Create the images and labels folders within the data folder
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    return dataset_folder, images_folder, labels_folder


def generate_yolo_annotations(annotation_filename, bounding_boxes_with_classes, image_width, image_height):
    with open(annotation_filename, "w") as annotation_file:
        for (class_id, x_min, x_max, y_min, y_max) in bounding_boxes_with_classes:
            # Calculate normalized coordinates
            x_center = (x_min + x_max) / (2 * image_width)
            y_center = (y_min + y_max) / (2 * image_height)
            box_width = (x_max - x_min) / image_width
            box_height = (y_max - y_min) / image_height

            # Write annotation line to the file
            annotation_line = f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
            annotation_file.write(annotation_line)


def save_frame_to_path(frame, file_path):
    try:
        cv2.imwrite(file_path, frame)
        print(f"Frame saved to {file_path}")
    except Exception as e:
        print(f"Error saving frame: {str(e)}")



class DatasetExporter:
    def __init__(self, output_folder_name_with_path, image_width, image_height):
        # Extract the dataset name from the output folder path
        dataset_name = os.path.basename(os.path.normpath(output_folder_name_with_path))

        # Generate a random HEX_MARK
        self.hex_mark = ''.join(random.choices(string.hexdigits, k=8))

        # Create the dataset folder with HEX_MARK
        dataset_folder = os.path.join(output_folder_name_with_path, dataset_name + "_" + self.hex_mark)
        os.makedirs(dataset_folder, exist_ok=True)

        # Create the data folder
        data_folder = os.path.join(dataset_folder, "data")
        os.makedirs(data_folder, exist_ok=True)

        # Create the images and labels folders within the data folder
        images_folder = os.path.join(data_folder, "images")
        labels_folder = os.path.join(data_folder, "labels")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        self.dataset_name = dataset_name
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.frame_number = 0
        self.image_width = image_width
        self.image_height = image_height

    def generate_hex_mark(self):
        return ''.join(random.choices(string.hexdigits, k=8))

    def save_frame_and_annotations(self, frame, bounding_boxes_with_classes, image_ext='png'):
        hex_mark = self.hex_mark
        image_filename = f"{self.dataset_name}_{hex_mark}_{self.frame_number:04d}.{image_ext}"
        annotation_filename = f"{self.dataset_name}_{hex_mark}_{self.frame_number:04d}.txt"

        image_path = os.path.join(self.images_folder, image_filename)
        annotation_path = os.path.join(self.labels_folder, annotation_filename)

        # Save the frame as an image
        cv2.imwrite(image_path, frame)

        # Generate and save YOLO annotations
        self.generate_yolo_annotations(annotation_path, bounding_boxes_with_classes)

        self.frame_number += 1

    def generate_yolo_annotations(self, annotation_filename, bounding_boxes_with_classes):
        with open(annotation_filename, "w") as annotation_file:
            for (class_id, x_min, x_max, y_min, y_max) in bounding_boxes_with_classes:
                x_center = (x_min + x_max) / (2 * self.image_width)
                y_center = (y_min + y_max) / (2 * self.image_height)
                box_width = (x_max - x_min) / self.image_width
                box_height = (y_max - y_min) / self.image_height

                annotation_line = f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
                annotation_file.write(annotation_line)




class MapDrawer:
    def __init__(self, map_size, map_position, base_img):
        self.map_size = map_size
        self.map_position = map_position
        self.base_img = base_img

    def draw_object(self, x, y, rotation_degrees):
        # Calculate the object's position on the map
        map_x = self.map_position[0] + int((x / self.base_img.shape[1]) * self.map_size[0])
        map_y = self.map_position[1] + int((y / self.base_img.shape[0]) * self.map_size[1])

        # Draw a triangle for the object at its position and orientation on the map
        triangle_points = np.array([
            [map_x,   map_y],
            [map_x + 5, map_y + 10],
            [map_x - 5, map_y + 10]
        ], np.int32)
        triangle_points = triangle_points.reshape((-1, 1, 2))
        rotated_triangle = self.rotate_triangle(triangle_points, map_x, map_y, rotation_degrees)

        cv2.fillPoly(self.base_img, [rotated_triangle], (0, 255, 0))  # Green triangle

    def rotate_triangle(self, triangle, x, y, angle_degrees):
        # Create a rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle_degrees, 1)

        # Apply the rotation matrix to the triangle vertices
        rotated_triangle = cv2.transform(triangle, rotation_matrix)

        return rotated_triangle

    def get_map_with_objects(self):
        return self.base_img




def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
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


def get_random_weather():
    """
    Randomizes weather params.
    # https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters
    :return: Weather dict
    """

    weather_mod = dict()
    weather_mod['cloudiness'] = random.uniform(0, 100)
    weather_mod['precipitation'] = random.uniform(0, 100) # Possible to rain without any clouds.
    weather_mod['precipitation_deposits'] = random.uniform(0, 100)
    weather_mod['wind_intensity'] = random.uniform(0, 100) # Makes very little difference
    weather_mod['sun_azimuth_angle'] = random.uniform(0, 360)
    weather_mod['sun_altitude_angle'] = np.random.choice([random.uniform(0, 90), random.uniform(-90, 0)],
                                                            p=[.999, .001]) # Mixture distribution
    weather_mod['fog_density'] = np.random.exponential(scale=5)
    weather_mod['fog_distance'] = 5
    weather_mod['wetness'] = np.random.choice([0, random.uniform(0, 100)], p=[.5, .5])
    
    return weather_mod




def get_yaw_from_3d_vector(vector):
    # Calculate the yaw angle using the arctan2 functionq
    yaw = math.atan2(vector.x, vector.y)
    
    # Convert the yaw angle from radians to degrees
    yaw_degrees = math.degrees(yaw)
    
    # Ensure the angle is within the range [0, 360] degrees(
    if yaw_degrees < 0:
        yaw_degrees += 360.0

    #yaw_degrees -= 180
    
    return yaw_degrees


def can_objects_face_each_other(ego, sign):
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

    return see_each_other, ego_direction, sign_direction, difference_between_ego_and_sign, difference_between_sign_and_ego, dot_product_A_to_B, dot_product_B_to_A


def is_bounding_box_within_frame(bbox, window_width, window_height):
    xmin, xmax, ymin, ymax = bbox

    # Check if all coordinates of the bounding box are within the window boundaries
    if xmin >= 0 and xmax <= window_width and ymin >= 0 and ymax <= window_height:
        return True
    else:
        return False

    
def spawn_pedestrian(self):
    # Get the blueprint library to create pedestrians
    blueprint_library = self.world.get_blueprint_library()

    # Select the pedestrian blueprint
    pedestrian_bp = random.choice(blueprint_library.filter("walker.pedestrian"))

    # Define the pedestrian spawn point
    spawn_points = self.world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # Spawn a pedestrian
    pedestrian = self.world.spawn_actor(pedestrian_bp, spawn_point)

    # Set pedestrian attributes (e.g., speed, behavior, appearance)
    # Example:
    pedestrian.set_velocity(carla.Vector3D(x=1.0, y=0.0, z=0.0))
    pedestrian.apply_control(carla.WalkerControl(speed=1.5))

    # When done, remember to destroy the actors when finished
    pedestrian.destroy()
    return


# Function to save frames to a video
def save_frames_as_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
   


# Define a custom distance function for Location objects
def location_distance(loc1, loc2):
    #print('Vertex to be matched! : ',loc1, loc2)
    return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)# + (loc1.z - loc2.z) ** 2)



def main():
    client = carla.Client('localhost', 2000)

    # set the map name or get it from the running carla instance
    world  = client.get_world()
    #world = client.load_world('Town07_more_speedsign')
    #world = client.load_world('Town01')

    debug= world.debug

    bp_lib = world.get_blueprint_library()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    # Create a new carla.Transform in a single line
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 

    # spawn camera
    camera_bp = bp_lib.find('sensor.camera.rgb')

    camera_init_trans = carla.Transform(carla.Location(x=2,z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    vehicle.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Get the world to camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()  
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    list_actor = world.get_actors()

    # set the traffic lights state
    color = random.randint(0,2)
    for actor_ in list_actor:
        if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is 
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red     
            if color == 0:
                actor_.set_state(carla.TrafficLightState.Green)          
            elif color ==1:
                actor_.set_state(carla.TrafficLightState.Yellow)       
            else:
                actor_.set_state(carla.TrafficLightState.Red) 
            actor_.set_yellow_time(0)
            actor_.set_green_time(1.0)
            actor_.set_red_time(0)
    traffic_light_color = ["green","orange","red"]

    # spawn other vehicles around the map
    for i in range(0):
        vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    # Set up the set of bounding boxes from the level
    bounding_box_set_traffic_signs  = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

    # Remember the edge pairs
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    # Retrieve the first image
    world.tick()
    image = image_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

    # Display the image in an OpenCV display window
    cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ImageWindowName',img)
    cv2.waitKey(1)

    '''
    bike_ids = []
    motorbike_ids = []

    if (world.get_actors().filter('*bike*').filter('*vehicle*'))!=[]:
        for bike in world.get_actors().filter('*bike*').filter('*vehicle*'):
            bike_ids.append(bike.id)

    motorbike_list = [world.get_actors().filter('*yamaha*').filter('*vehicle*') , world.get_actors().filter('*vespa*').filter('*vehicle*') , world.get_actors().filter('*kawasaki*').filter('*vehicle*') , world.get_actors().filter('*harley*').filter('*vehicle*')]
    bounding_box_set_traffic_signs
    if (motorbike_list)!=[]:
        for motorbikes in motorbike_list:
            for motorbike in motorbikes:
                motorbike_ids.append(motorbike.id)

    visible_traffic_light = carla.Location(0,0,0)
    color = ""
    '''
    
    ## list containing on vertex of the bounding boxes ------------------------------------------------------------------------------------------------
    # First contains the bboxes positions in the world of the speed limits but not the class
    # Second contains the bboxes of the speed "barrirer" and the type of the speedlimit, so the class
    vtx_bboxes = list()
    vtx_sl_actors = list()

    ts_bboxes = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
    speedlimit_actors = world.get_actors().filter('traffic.speed_limit.*')

    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print(speedlimit_actors)

    # fill the list before to order it
    for bb in ts_bboxes:
        vtx_bboxes.append( bb.get_world_vertices(carla.Transform())[0] )

    for speedlimit_actor in speedlimit_actors:
        vtx_sl_actors.append( speedlimit_actor.bounding_box.get_world_vertices(speedlimit_actor.get_transform())[0] )

    A = ts_bboxes
    B = speedlimit_actors

    # Calculate the pairwise distances between Location objects in A and B
    distances = np.zeros((len(A), len(B)))
    for i, loc1 in enumerate(A):
        for j, loc2 in enumerate(B):
            distances[i, j] = location_distance(
                loc1.get_world_vertices(carla.Transform())[0], 
                loc2.bounding_box.get_world_vertices(loc2.get_transform())[0] 
                )

    # Solve the assignment problem to find the optimal order
    row_indices, col_indices = linear_sum_assignment(distances)

    # Reorder list B based on the optimal assignment
    ordered_actors_vector = [B[ int(col) ] for col in col_indices]

    #for i in range(0, len(ordered_actors_vector)):
    #    print( ordered_actors_vector[i].bounding_box.get_world_vertices(ordered_actors_vector[i].get_transform())[0]  )

    ##--------------------------------------------------------------------------------------------------------------------------------- 
    pause_update = False
    slomo = False
    fake = False
    show_map = False
    is_recording_on = False

    ## initialize the databese exporter
    #database_exporter = DatasetExporter("./database_output", image_w, image_h)

    #array containing the frames to save as a video
    frames = []

    while True:

        key = cv2.waitKey(1)

        if key == ord('q'):
            # Quit the script when 'q' is pressed
            break
        elif key == ord('p'):
            # Toggle pause/resume when 'p' is pressed
            pause_update = not pause_update
        elif key == ord('s'):
            # Toggle slomotion/resume when 'p' is pressed
            slomo = not slomo
        elif key == ord('w'):
            # Trandomize weather when 'w' is pressed
            weather = get_random_weather()
            if weather is not None:
                if isinstance(weather, str):
                    world.set_weather(getattr(carla.WeatherParameters, weather))
                elif isinstance(weather, dict):
                    world.set_weather(carla.WeatherParameters(**weather))
        elif key == ord('f'):
            # Enable adverarial fake miscalssification  when 'f' is pressed
            fake = not fake
        elif key == ord('m'):
            # THide/show map when 'm' is pressed
            show_map = not show_map
        elif key == ord('r'):
            # THide/show map when 'm' is pressed
            is_recording_on = not is_recording_on
            print('Recording: ', is_recording_on)
        elif key == ord('y'):    
            # Save frames as a video
            video_name = 'carla_video.avi'
            save_frames_as_video(frames, video_name, fps=15)
            print("The video was successfully saved")

        if pause_update: 
            continue

        if slomo:
            vehicle.enable_constant_velocity(carla.Vector3D(2, 0, 0))
            vehicle.constant_velocity_enabled = True
        else:
            vehicle.disable_constant_velocity()
            vehicle.constant_velocity_enabled = False

        # Retrieve and reshape the image
        world.tick()

        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        clean_img = img.copy()

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Add the text on the left side of the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1

        ## get ego car position, forward, yaw vectors
        ego_position    = vehicle.get_transform().location
        ego_forward_vec = vehicle.get_transform().get_forward_vector()
        ego_yaw         = get_yaw_from_3d_vector(ego_forward_vec)
        ego_coord       = (int(ego_position.x), int(ego_position.y), int(ego_yaw))

        ## these contains the basic info of ego and signs like class, position and orientation. cold be use to draw a map on screen
        ego_tuple = ('EGO', int(ego_position.x), int(ego_position.y), int(ego_yaw))
        #this will be filled in the main loop
        tgt_tuple_list = list()

        output_bboxes = list()

        ## print the iformations on the LEFT , BOTTOM
        if show_map:
            corner_string = "Ego Position: " + str(int(ego_position.x)) + "  " + str(int(ego_position.y)) + " Heading: " + str(int(ego_yaw))
            cv2.putText(img, corner_string, (20, 20), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        signs_on_frame = 0
                
        for i in range(0, len(bounding_box_set_traffic_signs)):   
            bb = bounding_box_set_traffic_signs[i]

            ## get the sign class/speed limit in kmh
            try:
                speed_limit_str = ordered_actors_vector[i].type_id.split('limit.')[1]
                orientation_taken_by_actors_list = ordered_actors_vector[i].bounding_box.rotation.yaw 
            except:
                print()
                continue
            bbox_by_actor = ordered_actors_vector[i].bounding_box.get_world_vertices(ordered_actors_vector[i].get_transform())[0] 
            bbox_by_actor_pos = (speed_limit_str, int(bbox_by_actor.x), int(bbox_by_actor.y), int(orientation_taken_by_actors_list))
            
            # append the sign class, position and orientation to the list for draw a map
            sign_orientation = int(bb.rotation.yaw)
            bbox_by_bbs_pos = (speed_limit_str, int(bb.location.x), int(bb.location.y), int(sign_orientation))
            distance_between_bbox_and_actor = np.sqrt((bbox_by_actor_pos[1] - bbox_by_bbs_pos[1]) ** 2 + (bbox_by_actor_pos[2] - bbox_by_bbs_pos[2]) ** 2)  

            tgt_tuple_list.append( bbox_by_bbs_pos )

            # Filter for distance from ego vehicle
            if bb.location.distance(vehicle.get_transform().location) < 30:
                #is the sign oin front of the camera
                ray = bb.location - vehicle.get_transform().location
                dot_product = ego_forward_vec.dot(ray)
                                                                
                if dot_product > 1:
                    signs_on_frame += 1
                    # Cycle through the vertices
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    for edge in edges:
                        # Join the vertices into edges
                        #p1 = get_image_point(bb.location, K, world_2_camera)
                        
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
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

                    ## Filter just the traffic sign i can see the front face
                    sign_coord = (int(bb.location.x), int(bb.location.y), sign_orientation)

                    ##  calcultate if the sigh if faceing the car    
                    can_i_see_the_face, direction_ego, direction_sign, diff_AB, diff_BA, dot_product_A_to_B, dot_product_B_to_A, = can_objects_face_each_other( ego_coord, sign_coord ) 
                    is_bbox_complete = is_bounding_box_within_frame(( x_min, x_max, y_min, y_max ), image_w, image_h) 

                    if can_i_see_the_face and is_bbox_complete and distance_between_bbox_and_actor < 3.18:
                        output_bboxes.append( (speed_limit_str, x_min, x_max, y_min, y_max)  )

                    # tect position params
                    variation = 2
                    x_min = x_min + random.randint(-variation, variation)
                    x_max = x_max + random.randint(-variation, variation)
                    y_min = y_min + random.randint(-variation, variation)
                    y_max = y_max + random.randint(-variation, variation)

                    text_x = int(x_min) - 30  # Adjust the position as needed
                    text_y = int(y_min) -30  # Center the text vertically
                    #compose the string to be print
                    sign_position_str = ('x,y,r: '   + str(sign_coord) +
                                    ' | dirs: '      + str(direction_ego) + ' ' + str(direction_sign) + 
                                    ' | diffs: '     + str(diff_AB)       + ' ' + str(diff_BA) + 
                                    ' | dots a_AB: ' + "{:.2f}".format(dot_product_A_to_B) + " {:.2f}".format(dot_product_B_to_A)
                                    )
                    
                    speed_colors = {
                        '30' : (0,255,0, 255),
                        '60' : (255,255,0, 255),
                        '90' : (0,255,255, 255),
                        'STOP' : (0,0,250, 255),
                        'ONE WAY' : (255,0,255, 255)
                    }

                    if fake:
                        random_key = random.choice( list(speed_colors.keys()) )
                        random_value = speed_colors[random_key]
                        speed_limit_str = random_key
                        font_color = random_value
                    else:
                        font_color = speed_colors[speed_limit_str]

                    if is_bbox_complete and can_i_see_the_face:
                        # draw the speed on the bbox
                        cv2.putText(img, speed_limit_str, (text_x, text_y), font, 2, font_color, 3, cv2.LINE_AA)

                        if distance_between_bbox_and_actor > 3.18:
                            cv2.putText(img, 'WRONG', (text_x, text_y+40), font, 2, (0, 0, 255), 1, cv2.LINE_AA)

                        # draw the bbox on the image frame
                        box_color = font_color
                        box_thickness = 2
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), box_color, box_thickness)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), box_color, box_thickness)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), box_color, box_thickness)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), box_color, box_thickness)

                        ## draw a line connecting the sign with le string
                        cv2.line(img, (20, 20 * signs_on_frame + 20), (int(x_max),int(y_max)), font_color, 1)

                        ## pritn info on the ilst
                        _str = speed_limit_str + ' -- ' + sign_position_str
                        cv2.putText(img, _str, (20, 20 * signs_on_frame + 40), font, 0.4, font_color, font_thickness, cv2.LINE_AA)

                        '''
                        print('\nBoxx')
                        print(bbox_by_actor_pos)
                        print(bbox_by_bbs_pos)
                        print(distance_between_bbox_and_actor)
                        '''
                    else:
                        font_color = (0, 0, 255)

        ## draw the map
        if show_map:
            map_size = (400, 400)  # Size of the map
            map_position = (20, image_h - map_size[1] - 20)  # Position of the map in the screen (BOTTOM_RIGHT corner)
            drawer = MapDrawer(map_size, map_position, img)

            tgt_tuple_list.append( ego_tuple )
            objects = tgt_tuple_list

            for obj in objects:
                drawer.draw_object(obj[1], obj[2], obj[3])
            
            img = drawer.get_map_with_objects()

        if signs_on_frame > 0 and is_recording_on:
            frames.append(img.copy()[:, :, :3])

        ## bboxes with class
        #if len(output_bboxes) > 0 :
        #    print( 'Speed limits in scene: ', output_bboxes )
        #    database_exporter.save_frame_and_annotations( clean_img, output_bboxes )

        cv2.imshow('ImageWindowName', img)

    cv2.destroyAllWindows()
    camera.destroy()

    for npc in world.get_actors().filter('*vehicle*'):
        npc.destroy()


if __name__ == "__main__":
    main()
