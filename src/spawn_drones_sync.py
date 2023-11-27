#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

import random
from math import cos, sin, radians
import time
import signal
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self
    

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data



def spawn_objects(client, num_objects, altitude_range, objects_names):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawned_objects = []

    blueprint_list = list()

    for name in objects_names:
        blueprint_list.append(blueprint_library.find(name))

    for _ in range(num_objects):
        try:
            blueprint = random.choice(blueprint_library.filter('*plant*'))
            altitude = random.uniform(*altitude_range)
            
            spawn_point = carla.Transform()
            spawn_point.location.x = random.uniform(-200, 200)  # adjust range as needed
            spawn_point.location.y = random.uniform(-200, 200)  # adjust range as needed
            spawn_point.location.z = altitude

            #blueprint.set_attribute('size', str(100))

            actor = world.spawn_actor(blueprint, spawn_point)

            spawned_objects.append(actor)
        except Exception as e:
            print('Error spawning: ', e)

    return spawned_objects



def randomize_objects(objects, pitch_roll_range, yaw_range, distance_offset):
    for obj in objects:
        new_transform = obj.get_transform()

        # Randomly change rotation
        new_transform.rotation.yaw   = random.uniform(yaw_range[0], yaw_range[1])
        new_transform.rotation.pitch = random.uniform(pitch_roll_range[0], pitch_roll_range[1])
        new_transform.rotation.roll  = random.uniform(pitch_roll_range[0], pitch_roll_range[1])

        # Randomly change position by an offset in a random direction
        offset_distance = random.uniform(distance_offset[0], distance_offset[1])  # adjust range as needed
        offset_angle = random.uniform(0, 360)

        new_transform.location.x += offset_distance * cos(radians(offset_angle))
        new_transform.location.y += offset_distance * sin(radians(offset_angle))

        obj.set_transform(new_transform)


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (1920, 1080),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    ############################## PARAMETERS ###############################
    # TODO
    objects_names = ['static.prop.plantpot01',
                     'static.prop.plantpot02',
                     'static.prop.plantpot03' ]

    flying_objects_num = 500
    altitude_range=(20, 30)
    cameras_pitch = +35

    pitch_roll_range = (-20, 20)
    yaw_range = (0, 360)

    camera_roll_range = (-20, 20)
    camera_pitch_range = (0, 70)
    camera_yaw_range = (0, 360)

    distance_offset_range = (0.2, 5)

    set_random_camera_location = True

    update_camera_transform_cycles = 20
    update_weather_cycles = 200
    ############################## PARAMETERS ###############################

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        spawned_flying_objects = spawn_objects(client, num_objects=flying_objects_num, altitude_range=altitude_range, objects_names=objects_names)

        # Spwn a random vehicle
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # Spawn the  cameras and attach them to the vehicle
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '110')

        camera_rgb = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=cameras_pitch)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        blueprint = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '110')

        camera_instseg = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=cameras_pitch)),
            attach_to=vehicle)
        actor_list.append(camera_instseg)

        blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '110')

        camera_depth = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=cameras_pitch)),
            attach_to=vehicle)
        actor_list.append(camera_depth)

        counter = 0

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_instseg, camera_depth, fps=1) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                counter += 1

                # rotate and change the positions of existing spwned objects
                randomize_objects(spawned_flying_objects, pitch_roll_range, yaw_range, distance_offset_range)
                # randomize weather
                # change camera position

                timestamp = time.time()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)

                #  TODO calculate bboxes and save the images and annotations
                save_images(timestamp, image_rgb, image_semseg, image_depth)
                bboxes = get_bboxes(image_rgb, image_semseg, image_depth)
                save_annotations(timestamp , bboxes)


                # Choose a random waypoint or the next waypoint and update the car location.
                if counter % update_camera_transform_cycles == 0:
                    if set_random_camera_location:
                        start_pose = random.choice(m.get_spawn_points())
                        #also rotate it
                        start_pose.rotation.yaw   = random.uniform(camera_yaw_range[0], camera_yaw_range[1])
                        start_pose.rotation.pitch = random.uniform(camera_pitch_range[0], camera_pitch_range[1])
                        start_pose.rotation.roll  = random.uniform(camera_roll_range[0], camera_roll_range[1])
                        #waypoint = m.get_waypoint(start_pose.location)
                        #Apply it
                        vehicle.set_transform(start_pose)
                    else:
                        waypoint = random.choice(waypoint.next(1.5))
                        vehicle.set_transform(waypoint.transform)


                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)
                #draw_image(display, image_depth, blend=True)

                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        for obj in spawned_flying_objects:
            obj.destroy()

        pygame.quit()
        print('done.')



def save_images( timestamp, image_rgb, image_semseg, image_depth ):
    return

def get_bboxes( image_rgb, image_semseg, image_depth ):
    return

def save_annotations( timestamp, bboxes):
    return


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')