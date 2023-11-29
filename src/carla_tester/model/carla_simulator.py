import carla

from carla_tester import utils
import queue
import numpy as np

import cv2


BBOX = 1
ACTOR = 0


class CarlaSimulator:
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port

        self.world  = None
        self.current_weather_parameters = None
        self.current_town = None

        # {'actor' : actor_where_to_attach_cameras, 'cameras' : cameras, 'queues' : queues}
        # For each actor i have the correspondant cameras(anothed dict) and queues(dict)
        # Initialized by: 

        self.spawned_cameras = []

        try:
            self.connect_to_carla()
        except Exception as e:
            print(f"TimeoutException during initialization: {str(e)}")


    def connect_to_carla(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print('CarlaSimulator->self.world', self.world)

        if self.world is not None:
            self.current_weather_parameters = self.world.get_weather()
            self.current_town = self.world.get_map().name        
            print("Connected to CARLA.")


    def destroy_all_cameras(self):
        try:
            for camera in self.spawned_cameras:
                camera.destroy()
            self.spawned_cameras = []
        except Exception as e:
            raise e
        
        
    def get_world(self):
        return self.world 

    def fetch_available_towns(self):
        self.available_towns = [town for town in self.client.get_available_maps()]

    def fetch_spawnable_statics(self):
        # Retrieve the spawnable actors (you can customize this based on your needs)
        filter = "*static.*"
        self.spawnable_statics = self.world.get_blueprint_library().filter( filter )

    def fetch_spawnable_vehicles(self):
        # Retrieve the spawnable actors (you can customize this based on your needs)
        filter = "*vehicles.*"
        self.spawnable_vehicles = self.world.get_blueprint_library().filter( filter )


    def set_town(self, selected_town):
        # Change the map to the selected town
        try:
            self.client.load_world(selected_town)
            print(f"Changed map to: {selected_town}")
        except Exception as e:
            print(f"Error changing map: {str(e)}")


    def get_weather_settings(self):
        return 


    def set_weather_settings(self, weather_parameters):
        # Apply the changes to CARLA
        self.world.set_weather(weather_parameters)


    def get_detection_targets(self, actors_filter_string=None, bbox_carla_object=None): #to do general
        # get actors
        traffic_signs_actors = self.world.get_actors().filter('traffic.speed_limit.*') # use actor_filter string
        # get bboxes
        traffic_signs_bboxes = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns) # use bbox carla object

        print('traffic_signs_actors', traffic_signs_actors)

        for actor in traffic_signs_actors:
            ##dovrebbe essere la posizione del cartello
            actor_transform = actor.get_transform()
            actor_location = actor_transform.location
            actor_rotation = actor_transform.rotation
            actor_type = actor.type_id
            speed_limit = actor.type_id.split('limit.')[1]

        for bbox in traffic_signs_bboxes:
            bbox_location = bbox.location
            bbox_rotation = bbox.rotation
            bbox_extent = bbox.extent

        speedlimits_signs = utils.get_speed_limit_signs( traffic_signs_actors, traffic_signs_bboxes )
        
        self.targets = speedlimits_signs

        return speedlimits_signs
    


    def add_cameras_to_a_single_target(self, 
                                       world, 
                                       actor_where_to_attach_cameras, 
                                       camera_relative_transform=None, 
                                       existing_camera_set=None):
        
        # if i didn't pass the relative position i use a default
        if camera_relative_transform == None:
            relative_obj_cam_transform = carla.Transform( carla.Location(x=-1,y=6,z=1.9), 
                                                          carla.Rotation(yaw = -90) ) 
        else:
            relative_obj_cam_transform = camera_relative_transform
    
        camera_bps = {
            "rgb": world.get_blueprint_library().find('sensor.camera.rgb'),
            "instance": world.get_blueprint_library().find('sensor.camera.instance_segmentation'),
            "depth": world.get_blueprint_library().find('sensor.camera.depth'),
            #"segmentation": world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
        }

        camera_transforms = {
            "rgb": relative_obj_cam_transform,
            "instance": relative_obj_cam_transform,
            "depth": relative_obj_cam_transform,
            #"segmentation": carla.Transform(relative_obj_cam_position, relative_obj_cam_rotation),
        }

        cameras = {}
        queues = {}    

        for camera_type, camera_bp in camera_bps.items():
            camera = world.spawn_actor(camera_bp, camera_transforms[camera_type], attach_to=actor_where_to_attach_cameras)
            cameras[camera_type] = camera
            self.spawned_cameras.append( camera )
            # Create a queue for each camera
            queues[camera_type] = queue.Queue()
            # Listen to the camera and put images in the respective queue
            camera.listen( queues[camera_type].put ) 

        camera_set = {'actor' : actor_where_to_attach_cameras, 'cameras' : cameras, 'queues' : queues}

        return camera_set
        

        
    def get_images(self, cameras_set):
        images = {}

        cameras = cameras_set['cameras']
        queues = cameras_set['queues']

        for image_type, camera in cameras.items():
            # Use the corresponding queue for the current camera type
            image_queue = queues[image_type]

            sensor_transform = camera.get_transform()

            width = int(float(camera.attributes['image_size_x']))
            height = int(float(camera.attributes['image_size_y']))
            fov = float(camera.attributes['fov'])

            # Calculate the camera projection matrix to project from 3D -> 2D
            K = utils.build_projection_matrix(width, height, fov)
            # Get the world to camera matrix
            world_2_camera = np.array(sensor_transform.get_inverse_matrix())

            # Get image from the queue
            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            images[image_type] = {'image': img }

        return images, K, world_2_camera


    def get_truth(self, camera_transform, image, targets, K, world_2_camera):
        #print('DBG -----> ', 'get_bboxes(self, target, cameras_set, image)')
        img = image.copy()
        image_h = image.shape[0]
        image_w = image.shape[1]

        selected_bboxes = list()

        for tgt in targets:
            bbox = tgt[BBOX]
            class_id = utils.get_speed_limit( tgt[ACTOR] )
            distance_from_target = bbox.location.distance( camera_transform.location )
            actor_transform = tgt[ACTOR].get_transform()

            if distance_from_target < 40:
                # Conditions to add a bbox to the output or not
                forward_vec = camera_transform.get_forward_vector()
                ray = bbox.location - camera_transform.location 
                        
                is_facing = utils.is_object_face_another_object(
                    camera_transform,
                    tgt[ACTOR].get_transform(),
                    ego_axis=(1, -1), 
                    ego_rot=-1, 
                    #ego_phase=-(90 - camera_relative_transform.rotation.yaw), 
                    ego_phase = 180,
                    sign_axis=(1, -1), 
                    sign_rot=-1, 
                    sign_phase=90
                )

                if forward_vec.dot(ray) > 1 and is_facing:
                    # Cycle through the vertices
                    verts = [v for v in bbox.get_world_vertices(carla.Transform())]

                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = utils.get_image_point(vert, K, world_2_camera)
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

                    bbox_pos  = (int(x_min), int(x_max), int(y_min), int(y_max))
                    text_orig = (int(x_min), int(y_min))
                    
                    # is in the screen
                    is_bbox_complete = utils.is_bounding_box_within_frame(bbox_pos, image)

                    '''
                    # is covered
                    utils.calculate_visibility(self.sensor_data['instance']['image'],
                                               self.sensor_data['depth']['image'],
                                               bbox_pos,
                                               distance_from_target)
                    '''
                    if is_bbox_complete:
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,255,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,255,255, 255), 1)
                        cv2.putText(img, str( class_id ), text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        selected_bbox = utils.bbox_to_yolo_annotation(class_id, bbox_pos, image_h, image_w)
                        selected_bboxes.append(selected_bbox)
        
        return img, selected_bboxes


