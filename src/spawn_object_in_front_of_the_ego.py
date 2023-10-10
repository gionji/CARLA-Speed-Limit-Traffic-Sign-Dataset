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



def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K




def move_waypoint_horizontally(W, position, meters):
    """
    Move a waypoint horizontally (left or right) by a specified number of meters.

    Args:
        W (carla.Waypoint): The original waypoint.
        position (str): 'left' to move left, 'right' to move right.
        meters (float): The number of meters to move.
    
    Returns:
        carla.Transform: The updated transform of the waypoint.
    """
    # Get the current transform of the waypoint
    transform = W.transform

    # Extract the location and rotation
    location = transform.location
    rotation = transform.rotation

    # Calculate the new location based on position ('left' or 'right') and meters
    yaw_radians = math.radians(rotation.yaw) # Convert yaw to radians
    if position == 'left':
        new_x = location.x - meters * math.sin(yaw_radians)
        new_y = location.y + meters * math.cos(yaw_radians)
    elif position == 'right':
        new_x = location.x + meters * math.sin(yaw_radians)
        new_y = location.y - meters * math.cos(yaw_radians)
    else:
        raise ValueError("Invalid position. Use 'left' or 'right'.")
    
    new_yaw = rotation.yaw + 90

    # Update the location in the transform
    transform.location.x = new_x
    transform.location.y = new_y
    transform.rotation.yaw = new_yaw

    return transform



def spawn_something(world):
    map= world.get_map()
    blueprint_library = world.get_blueprint_library()

    landmarks = world.get_map().get_all_landmarks()
    roads = map.get_topology()

    ##get the traffic sign asset

    #[print(item) for item in world.get_blueprint_library()]

    speed_limit_value = 30
    object01_bp = blueprint_library.find('static.prop.streetsign01')
    object02_bp = blueprint_library.find('static.prop.streetsign')
    
    waypoints = map.generate_waypoints(distance=10.0)

    for i in range(0, 40):
        wp = waypoints[i]
        new_transform = move_waypoint_horizontally(wp, 'left', 3 )
        world.spawn_actor(object01_bp, new_transform)
        world.spawn_actor(object02_bp, wp.transform)
    return

    for waypoint in waypoints:
        if not waypoint.is_intersection:
            new_transform = move_waypoint_horizontally(waypoint, 'left', 3 )
            #print(f"Waypoint ID: {waypoint.id}")
            #print(f"Location: {waypoint.transform.location}")w
            #print(f"New Location: {new_transform.location}")
            #print(f"Road ID: {waypoint.road_id}")
            #print(f"Lane ID: {waypoint.lane_id}")
            #print(f"Is Intersection: {waypoint.is_intersection}")
            #print(f"-------------------------------------------------")

            # Spawn the traffic sign actor
            world.spawn_actor(object01_bp, new_transform)
            world.spawn_actor(object02_bp, waypoint.transform)
    return


def spawn_and_delete_an_actor_in_front_of_the_ego():
    return


# Define a function to spawn and delete an object in front of the ego car
def spawn_and_delete_object(world, ego_vehicle, actor_blueprint, relative_position, old_object=None, noise_range=0):
    # Destroy the previous object if it exists
    if old_object is not None:
        old_object.destroy()

    # Get the ego car's transform
    ego_transform = ego_vehicle.get_transform()

    # Extract the ego car's position and orientation
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation

    # Calculate the rotation matrix based on the ego car's orientation
    rotation_matrix = carla.Transform(carla.Location(), ego_rotation)

    # Apply uniform noise to the relative position on the X and Y axes
    noise_x = random.uniform(-noise_range, noise_range)
    noise_y = random.uniform(-noise_range, noise_range)
    noisy_relative_position = relative_position + carla.Vector3D(noise_x, noise_y, 0.0)

    # Transform the noisy relative position using the rotation matrix
    transformed_relative_position = rotation_matrix.transform(noisy_relative_position)

    # Calculate the spawn location by adding the transformed relative position to the ego car's location
    spawn_location = ego_location + transformed_relative_position

    # Create a new transform for the spawn location with the same rotation as the ego car
    spawn_transform = carla.Transform(spawn_location, ego_rotation)

    # Spawn the object at the calculated location and rotation
    spawned_object = world.spawn_actor(actor_blueprint, spawn_transform)

    return spawned_object



def main():
    client = carla.Client('localhost', 2000)

    # set the map name or get it from the running carla instance
    #world  = client.get_world()
    #world = client.load_world('Town07_more_speedsign')
    world = client.load_world('Town02')

    debug= world.debug

    bp_lib = world.get_blueprint_library()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # spawn camera
    camera_bp = bp_lib.find('sensor.camera.rgb')

    camera_init_trans = carla.Transform(carla.Location(x=2,z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    vehicle.set_autopilot(True)

    [print(item) for item in world.get_blueprint_library()]
    #spawn_something(world)

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
            actor_.set_yellow_time(1.0)
            actor_.set_green_time(1.0)
            actor_.set_red_time(1.0)
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
    # We filter for traffic lights and traffic signs
    # bounding_box_set_traffic_lights = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
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
    

    ## list containing on vertex of the bounding boxes
    # First contains the bboxes positions in the world of the speed limits but not the class
    # Second contains the bboxes of the speed "barrirer" and the type of the speedlimit, so the class
    vtx_bboxes = list()
    vtx_sl_actors = list()

    ts_bboxes = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
    speedlimit_actors = world.get_actors().filter('traffic.speed_limit.*')

    # fill the list before to order it
    for bb in ts_bboxes:
        vtx_bboxes.append( bb.get_world_vertices(carla.Transform())[0] )

    for speedlimit_actor in speedlimit_actors:
        vtx_sl_actors.append( speedlimit_actor.bounding_box.get_world_vertices(speedlimit_actor.get_transform())[0] )

    A = ts_bboxes
    B = speedlimit_actors

    # Define a custom distance function for Location objects
    def location_distance(loc1, loc2):
        return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)

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
   
    pause_update = False

    object_to_spawn = world.get_blueprint_library().find('static.prop.bird_september_07')
    spawned_object = None

    while True:

        key = cv2.waitKey(1)

        if key == ord('q'):
            # Quit the script when 'q' is pressed
            break
        elif key == ord('p'):
            # Toggle pause/resume when 'p' is pressed
            pause_update = not pause_update

        if pause_update: 
            continue

        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Define a function to spawn and delete an object in front of the ego car
        spawned_object = spawn_and_delete_object(world, vehicle, object_to_spawn, carla.Vector3D(10, 3, 1), spawned_object)
 
        cv2.imshow('ImageWindowName', img)

    cv2.destroyAllWindows()
    camera.destroy()

    for npc in world.get_actors().filter('*vehicle*'):
        npc.destroy()


if __name__ == "__main__":
    main()
