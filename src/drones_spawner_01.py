import carla
import random
from math import cos, sin, radians
import time
import signal
import sys
import numpy as np
import cv2


def spawn_vehicle_with_cameras(client):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Choose a vehicle blueprint
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

    # Choose a random spawn point on the road
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Attach RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust location as needed
    rgb_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Attach instance segmentation camera
    segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    segmentation_camera = world.spawn_actor(segmentation_bp, camera_transform, attach_to=vehicle)

    # Attach depth camera
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)

    return vehicle, rgb_camera, segmentation_camera, depth_camera



def spawn_objects(client, num_objects, altitude_range):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawned_objects = []

    for _ in range(num_objects):
        try:
            blueprint = random.choice(blueprint_library.filter('*plant*'))
            altitude = random.uniform(*altitude_range)
            
            spawn_point = carla.Transform()
            spawn_point.location.x = random.uniform(-200, 200)  # adjust range as needed
            spawn_point.location.y = random.uniform(-200, 200)  # adjust range as needed
            spawn_point.location.z = altitude

            actor = world.spawn_actor(blueprint, spawn_point)
            spawned_objects.append(actor)
        except Exception as e:
            print('Error spawning: ', e)

    return spawned_objects


def randomize_objects(objects):
    for obj in objects:
        new_transform = obj.get_transform()

        # Randomly change rotation
        new_transform.rotation.yaw = random.uniform(0, 360)
        new_transform.rotation.pitch = random.uniform(0, 360)
        new_transform.rotation.roll = random.uniform(0, 360)

        # Randomly change position by an offset in a random direction
        offset_distance = random.uniform(0.5, 1)  # adjust range as needed
        offset_angle = random.uniform(0, 360)

        new_transform.location.x += offset_distance * cos(radians(offset_angle))
        new_transform.location.y += offset_distance * sin(radians(offset_angle))

        obj.set_transform(new_transform)


def capture_images(rgb_camera, segmentation_camera, depth_camera):
    return None

def get_bboxes():
    return None

def save_frames():
    return None

def cleanup_and_exit(objects):
    print("Cleaning up and exiting...")
    for obj in objects:
        obj.destroy()
    sys.exit(0)

def signal_handler(sig, frame):
    cleanup_and_exit(spawned_objects)



if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    vehicle, rgb_camera, segmentation_camera, depth_camera = spawn_vehicle_with_cameras(client)

    spawned_objects = spawn_objects(client, num_objects=400, altitude_range=(20, 30))

    # Set up the signal handler to catch Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Randomize the positions and rotations of the spawned objects
    try:
        while True:
            randomize_objects(spawned_objects)

            #rgb_image, segmentation_image, depth_image = capture_images(rgb_camera, segmentation_camera, depth_camera)


            # Display or save the images as needed
            #cv2.imshow('RGB Image', rgb_image)
            #cv2.imshow('Segmentation Image', segmentation_image)
            #cv2.imshow('Depth Image', depth_image)
            #cv2.waitKey(1000)  # Display images for 1 second
            #cv2.destroyAllWindows()

            time.sleep(1)

    except KeyboardInterrupt:
        cleanup_and_exit(spawned_objects)
        rgb_camera.destroy()
        segmentation_camera.destroy()
        depth_camera.destroy()
        vehicle.destroy()
