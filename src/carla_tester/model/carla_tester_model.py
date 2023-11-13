import carla
import cv2

import queue
import numpy as np
from carla_tester import utils

import random
import time

import itertools
import json
import datetime


BBOX = 1
ACTOR = 0


class CarlaSimulator:
    def __init__(self, host="localhost", port=2000):
        try:
            self.host = host
            self.port = port
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.weather_parameters = self.world.get_weather()
            self.current_town = self.world.get_map().name

        except Exception as e:
            print(f"Error initializing CARLA simulation: {str(e)}")

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

    def set_weather_settings(self, weather_parameters):
        # Apply the changes to CARLA
        self.world.set_weather(weather_parameters)





from . import yolov7_detector as yolo

class YoloDetector:
    def __init__(self):
        self.model_path = None
        self.classes = None
        self.detector = None

    def load_model(self, model_path, classes=[0, 1, 2]):
        self.model_path = model_path
        self.classes = classes
        self.detector = yolo.YOLOv7Detector(model_path, classes=self.classes)

    def run_yolo(self, frame):   
        pred_img, preds = self.detector.detect( frame )

        # trasform the prediction format as the input
        preds_annotations = utils.convert_to_yolo_annotation(preds, 
                                                            pred_img.shape[0], 
                                                            pred_img.shape[1])

        return pred_img, preds_annotations
    


class ExperimentTest:
    def __init__(self, carla_simulator):
        self.carla_simulator = carla_simulator
        self.target_index = 0

    def initialize(self):
        self.carla_simulator.set_town('Town10HD_Opt')
        #self.carla_simulator.set_town('Town01')
        # initialize det/classifier
        self.yolo = YoloDetector()
        self.yolo.load_model( '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt' )

    def add_cameras(self, attach_to=None, camera_relative_transform=None):
        world = self.carla_simulator.world

        # if i didn't pass the relative position i use a default
        if camera_relative_transform == None:
            relative_obj_cam_position = carla.Location(x=-1,y=10,z=1.9)
            relative_obj_cam_rotation = carla.Rotation(yaw = -90)     
            relative_obj_cam_transform = carla.Transform(relative_obj_cam_position, relative_obj_cam_rotation) 
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

        attached_cameras = {}

        for camera_type, camera_bp in camera_bps.items():
            camera = world.spawn_actor(camera_bp, camera_transforms[camera_type], attach_to=attach_to)
            attached_cameras[camera_type] = camera

        self.attached_cameras = attached_cameras

    
    def get_images(self):
        sensor_data = {}
        for sensor_name, sensor in self.attached_cameras.items():

            image_queue = queue.Queue()
            sensor.listen(image_queue.put)
            sensor_transform = sensor.get_transform()

            width = int(float(sensor.attributes['image_size_x']))
            height = int(float(sensor.attributes['image_size_y']))
            fov = float(sensor.attributes['fov'])

            # Calculate the camera projection matrix to project from 3D -> 2D
            self.K = utils.build_projection_matrix(width, height, fov)
            # Get the world to camera matrix
            self.world_2_camera = np.array(sensor_transform.get_inverse_matrix())

            # Get image from the queue
            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            '''
            rgb_image = np.array(image_data)
            image_data = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            '''
            
            sensor_data[sensor_name] = {'image': img }

        return sensor_data
    

    ## TODO eventually move to CarlaSimulator
    def destroy_sensors(self):
        for sensor in self.attached_cameras.values():
            sensor.destroy()
        #self.camera_rgb.destroy()


    def get_detection_targets(self):
        world = self.carla_simulator.world
        # get actors
        traffic_signs_actors = world.get_actors().filter('traffic.speed_limit.*')
        # get bboxes
        traffic_signs_bboxes=world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

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

            #print(obj.get_transform())
        speedlimits_signs = utils.get_speed_limit_signs(traffic_signs_actors,
                                                        traffic_signs_bboxes)

        return speedlimits_signs
    

    def get_bboxes(self, target, image, camera_relative_transform):
        img = image.copy()
        image_h = self.sensor_data['rgb']['image'].shape[0]
        image_w = self.sensor_data['rgb']['image'].shape[1]

        selected_bboxes = list()

        for sign in self.targets:
            bbox = sign[BBOX]
            class_id = utils.get_speed_limit( sign[ACTOR] )
            distance_from_target = bbox.location.distance( self.attached_cameras['rgb'].get_transform().location )
            camera_transform = self.attached_cameras['rgb'].get_transform()
            actor_transform = sign[ACTOR].get_transform()

            if distance_from_target < 40:
                # Conditions to add a bbox to the output or not
                forward_vec = camera_transform.get_forward_vector()
                ray = bbox.location - camera_transform.location 
                        
                is_facing = utils.is_object_face_another_object(
                    camera_transform,
                    sign[ACTOR].get_transform(),
                    ego_axis=(1, -1), 
                    ego_rot=-1, 
                    ego_phase=-(90 - camera_relative_transform.rotation.yaw), 
                    sign_axis=(1, -1), 
                    sign_rot=-1, 
                    sign_phase=0
                )

                if forward_vec.dot(ray) > 1 and is_facing:
                    # Cycle through the vertices
                    verts = [v for v in bbox.get_world_vertices(carla.Transform())]

                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = utils.get_image_point(vert, self.K, self.world_2_camera)
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
                    is_bbox_complete = utils.is_bounding_box_within_frame(bbox_pos, self.sensor_data['rgb']['image'])

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
                    

    def next(self):
        self.target_index = (self.target_index + 1) % len(self.targets)
        return self.get_scene_results()
    

    def previous(self):
        self.target_index = (self.target_index - 1) % len(self.targets)
        return self.get_scene_results()
    

    def get_scene_results(self):
        target = self.targets[ self.target_index ]

        #
        camera_relative_transform = utils.get_camera_spherical_relative_transform(4, -45, 20)
        ## add cameras in the scene attaching to an actor
        self.add_cameras( attach_to=target[ACTOR], camera_relative_transform=camera_relative_transform )

        ## get images from the spawned sensors
        self.sensor_data = self.get_images()
        ## generate the image with the bounding boxes
        bboxes_image, yolo_truth = self.get_bboxes( target, self.sensor_data['rgb']['image'], camera_relative_transform )
        ## add the rgb image with bboxes in the sensorsdata dict
        self.sensor_data['bboxes'] = {}
        self.sensor_data['bboxes']['image'] = bboxes_image 
        ## calculate map
        # map = utils.draw_carla_actors_on_image(self.targets, camera=self.attached_cameras['rgb'])
        ## calculate predictions
        preds, yolo_preds = self.yolo.run_yolo( self.sensor_data['rgb']['image'] )
        self.sensor_data['preds'] = {}
        self.sensor_data['preds']['image'] = preds
        # SCORE: calculate difference between truth and predictions
        # Calculate detection and classification metrics
        in_classes = {'speed_limit_signs' : 8}
        out_classes = {'speed_limit_signs' : 6}
        bboxes_from_istseg, instseg_img = utils.find_bounding_boxes(self.sensor_data['instance']['image'],
                                                                    self.sensor_data['depth']['image'],
                                                                    in_classes,
                                                                    out_classes)
        self.sensor_data['instseg'] = {}
        self.sensor_data['instseg']['image'] = instseg_img


        metrics = utils.calculate_metrics_detection_classification(yolo_truth, yolo_preds, 0.5)

        #for metric, value in metrics.items():
        #    print(f"{metric}: {value}")

        self.destroy_sensors()

        return self.sensor_data



    def note_scene(parameters, truth, prediction):
        return


    def run(self):
        self.initialize()
        self.add_cameras()
        self.targets = self.get_detection_targets()

        if len(self.targets) > 0:
            self.target_index = 0

        return

    def generate_random_weather_parameters(self, parameter_names):
        weather_parameters = carla.WeatherParameters()
        for param_name in parameter_names:
            if hasattr(carla.WeatherParameters, param_name):
                random_value = random.randint(0, 100)  # Assuming weather parameters are in range [0.0, 1.0]
                setattr(weather_parameters, param_name, random_value)
        return weather_parameters
    

    def run_one_iteration(self):
        self.initialize()
        self.targets = self.get_detection_targets()

        self.logger = ExperimentLogger('pipponee')
        max_iterations =1000

        for i in range( max_iterations ):
                       
            target_count = 0

            iou_th = 0.5

            ## Parameters you wanna change
            weather_params = self.generate_random_weather_parameters( ['wetness', 'cloudiness', 'precipitation', 'sun_altitude_angle', 'sun_azimuth_angle'] )
            self.carla_simulator.set_weather_settings(weather_params)

            initial_params = {
                "weather" : str(weather_params),
                "tgt_trans": [str(t[ACTOR].get_transform()) for t in self.targets]
            }


            self.logger.start_iteration(i , initial_params)

            # Variables to track time and precision-recall values
            total_time = 0
            precision_values = []
            recall_values = []

            for target in self.targets:
                target_count += 1
                #print('[', target_count, ']',target[ACTOR].type_id.split('limit.')[1], 'Camera location: ' ,target[ACTOR].get_transform().location) 

                start_time = time.time() 

                ## add cameras in the scene attaching to an actor
                self.add_cameras( attach_to=target[ACTOR] )

                ### Collect the data
                ## get images from the spawned sensors
                self.sensor_data = self.get_images()
                ## generate the image with the bounding boxes
                bboxes_image, yolo_truth = self.get_bboxes( target, self.sensor_data['rgb']['image'] )
                ## add the rgb image with bboxes in the sensorsdata dict
                self.sensor_data['bboxes'] = {}
                self.sensor_data['bboxes']['image'] = bboxes_image 
                ## calculate map
                # map = utils.draw_carla_actors_on_image(self.targets, camera=self.attached_cameras['rgb'])
                ## calculate predictions
                preds, yolo_preds = self.yolo.run_yolo( self.sensor_data['rgb']['image'] )
                self.sensor_data['preds'] = {}
                self.sensor_data['preds']['image'] = preds
                # SCORE: calculate difference between truth and predictions
                metrics = utils.calculate_metrics_detection_classification(yolo_truth, yolo_preds, iou_th)
                # Update precision and recall values
                precision_values.append(metrics["precision"])
                recall_values.append(metrics["recall"])

                #bboxes_from_istseg = utils.find_bounding_boxes(self.sensor_data['instance']['image'], [8])
                #print('joakim', bboxes_from_istseg)

                #After processing the frame
                frame_info = {
                    "tgt": target_count,
                    "bb_truth": yolo_truth,  # Replace with actual ground truth bboxes
                    "bb_pred": yolo_preds,  # Replace with actual prediction bboxes,
                    "prec": metrics["precision"],
                    "recall":metrics["recall"],
                    "cam_tran": str(self.attached_cameras['rgb'].get_transform())
                }

                # Save frame information to a JSON file
                self.logger.save_frame(frame_info)

                self.destroy_sensors()

                end_time = time.time()
                iteration_time = end_time - start_time
                #print("Time taken for iteration:", iteration_time)
                total_time += iteration_time

            iteration_metrics = {
                "average_precision" : np.mean(np.array(precision_values)),
                "average_recall" : np.mean(np.array(recall_values))
            }

            self.logger.end_iteration(iteration_metrics)

            print( "Iteration ", i, '/', max_iterations, 'terminated' )
        self.logger.finish()

        return
    

class ExperimentLogger:
    def __init__(self, experiment_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{experiment_name}_{timestamp}.json"
        self.experiment_name = experiment_name
        self.log_data = {'experiment_name': experiment_name, 'iterations': []}
        self.current_iteration = None

    def start_iteration(self, iteration_number, iteration_info):
        # Customize the iteration information here
        self.current_iteration = {'iteration_number' : iteration_number , 'iteration_info': iteration_info, 'frames': []}

    def save_frame(self, frame_dict):
        if self.current_iteration is not None:
            self.current_iteration['frames'].append(frame_dict)

    def end_iteration(self, end_iteration_info):
        if self.current_iteration is not None:
            # Add end-of-iteration information
            self.current_iteration['iteration_metrics'] = end_iteration_info
            self.log_data['iterations'].append(self.current_iteration)
            self.current_iteration = None
            self._save_to_file()

    def _save_to_file(self):
        with open(self.filename, 'w') as json_file:
            json.dump(self.log_data, json_file, indent=2)

    def finish(self):
        # Save any remaining data before finishing
        self._save_to_file()
        self.log_data = None


class Agent:
    def __init__(self, environment):
        self.environment = environment
        pass

    def perform_action(self, params, score):
        pass

    def update_parameters(self):
        pass



class CarlaTesterModel:
    def __init__(self):
        self.carla = None
    
    def connect_to_carla(self):
        self.carla_simulator = CarlaSimulator()

    def cleanup(self):
        self.experiment.destroy_sensors()

    def run_experiment_init(self):
        self.experiment = ExperimentTest(self.carla_simulator)
        sensor_data = self.experiment.run()
        self.experiment.destroy_sensors()
        return sensor_data
    
    def run_one_iteration(self):
        self.experiment.run_one_iteration()
        return
    
    def next(self):
        return self.experiment.next()
    
    def previous(self):
        return self.experiment.previous()