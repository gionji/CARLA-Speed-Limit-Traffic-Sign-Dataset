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

import secrets

from carla_tester.model.carla_simulator import CarlaSimulator
from carla_tester.model.agent import RandomStupidAgent
from carla_tester.model.logger import ExperimentLogger


BBOX = 1
ACTOR = 0


import carla_tester.model.yolov7_detector as yolo

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
    def __init__(self, carla_simulator, 
                   town='Town10HD_Opt', 
                   yolo_model_path = '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt',
                   iou_th = 0.5 ):
        
        self.carla_simulator = carla_simulator

        self.cameras_sets = list()
        
        self.carla_simulator.set_town( town )

        self.targets = self.carla_simulator.get_detection_targets()

        if len(self.carla_simulator.targets) > 0:
            self.target_index = 0

        self.agent = RandomStupidAgent(self)

        # Get a random seed using the secrets module
        random_seed = secrets.token_bytes(32)
        random.seed(random_seed)

        # initialize det/classifier
        self.yolo = YoloDetector()
        self.yolo.load_model( yolo_model_path )
        self.iou_th = iou_th


    def process_single_frame(self, target, cameras_set, iou_th):
        ## Define the relative position of the camera from the main target based on distance, and angles.
        #camera_relative_transform = utils.get_camera_spherical_relative_transform(8, 0, 0)

        ## get images from the spawned sensors
        camera_output_images = self.carla_simulator.get_images( cameras_set )

        ## generate the image with the bounding boxes
        bboxes_image, yolo_truth = self.carla_simulator.get_truth(cameras_set, camera_output_images['rgb']['image'])
        ## add the rgb image with bboxes in the sensorsdata dict
        camera_output_images['bboxes'] = {}
        camera_output_images['bboxes']['image'] = bboxes_image 
        ## calculate map
        # map = utils.draw_carla_actors_on_image(self.targets, camera=self.attached_cameras['rgb'])
        ## calculate predictions
        preds, yolo_preds = self.yolo.run_yolo( camera_output_images['rgb']['image'] )
        camera_output_images['preds'] = {}
        camera_output_images['preds']['image'] = preds

        camera_transform = cameras_set['cameras']['rgb'].get_transform()

        ## that was for joakim and the drone
        ''' 
        ## get bboxes from instance segmentation
        in_classes = {'speed_limit_signs' : 8}
        out_classes = {'speed_limit_signs' : 6}
        bboxes_from_istseg, instseg_img = utils.find_bounding_boxes(camera_output_images['instance']['image'],
                                                                    camera_output_images['depth']['image'],
                                                                    in_classes,
                                                                    out_classes)
        camera_output_images['instseg'] = {}
        camera_output_images['instseg']['image'] = instseg_img
        '''

        ## SCORE: calculate difference between truth and predictions
        metrics = utils.calculate_metrics_detection_classification(yolo_truth, yolo_preds, iou_th)

        return camera_output_images, metrics, yolo_truth, yolo_preds, camera_transform



    '''
    def next(self):
        # get the target object by index
        self.target_index = (self.target_index + 1) % len(self.carla_simulator.targets)
        target = self.carla_simulator.targets[ self.target_index ]    
        cameras = next((item for item in self.cameras_sets if item['actor_id'] == target[ACTOR].id), None)
        res = self.process_single_frame(target, cameras , self.iou_th)
        return res
     

    def previous(self):
        self.target_index = (self.target_index - 1) % len(self.carla_simulator.targets)
        target = self.carla_simulator.targets[ self.target_index ]     
        cameras = next((item for item in self.cameras_sets if item['actor_id'] == target[ACTOR].id), None)
        res = self.process_single_frame(target, cameras , self.iou_th)
        return res
    
    '''

    def run_experiment(self,
                          parameters_names,
                          max_iterations, 
                          experiment_name
                          ):

        # Pass to the agent the list of the weather
        self.agent.set_parameters( parameters_names )
        
        ## int the logger
        self.logger = ExperimentLogger( experiment_name )
        self.max_iterations = max_iterations


        for i in range( self.max_iterations ):                    
            target_count = 0

            ## Parameters you wanna change
            new_params = self.agent.perform_action(None)
            ## action
            self.carla_simulator.set_weather_settings(new_params)

            ## Write the itaration parameters in the logger
            initial_params = {
                "weather" : str(new_params), # just nthe new ones, the weather values TODO depack into a json ready format
                "tgt_trans": [str(t[ACTOR].get_transform()) for t in self.carla_simulator.targets]
            }

            # start logging
            self.logger.start_iteration(i , initial_params)

            # Timer variable
            total_time = 0

            # Variables to track time and precision-recall values
            precision_values = [] # TODO move in to a dict of list, if i wanna add more
            recall_values = []

            # Loop her on all the targets (traffic signs) on the map
            for target in self.carla_simulator.targets:
                target_count += 1
                start_time = time.time() 

                # Spawn the camera around the single target
                cameras_set = self.carla_simulator.add_cameras_to_a_single_target( target[ACTOR] )

                # get the metrics back
                _, single_target_metrics, yolo_truth, yolo_preds, camera_transform = self.process_single_frame(target, cameras_set, self.iou_th)

                # destroy the cameras
                self.carla_simulator.destroy_all_cameras()

                # after processing the frame
                frame_info = {
                    "tgt"     : target_count,
                    "bb_truth": yolo_truth,  
                    "bb_pred" : yolo_preds, 
                    "prec"    : single_target_metrics["precision"],
                    "recall"  : single_target_metrics["recall"],
                    "cam_tran": str( camera_transform )
                }
                # Save frame information to a JSON file
                self.logger.save_frame(frame_info)

                # save the iterations metrics results
                precision_values.append(single_target_metrics["precision"])
                recall_values.append(single_target_metrics["recall"])

                # Stop timer to measure the iteration execution time
                end_time = time.time()
                iteration_time = end_time - start_time
                total_time += iteration_time


            iteration_metrics = {
                "average_precision" : np.mean(np.array(precision_values)),
                "average_recall"    : np.mean(np.array(recall_values))
            }

            # save the scores in to the loge file
            self.logger.end_iteration( iteration_metrics )

            print( "Iteration", str(i+1) , '/', self.max_iterations, 'terminated in ', total_time )

        self.logger.finish()

        print('Experiment terminated')

        return
    





class CarlaTesterModel:
    def __init__(self):
        self.carla_simulator = CarlaSimulator()
        self.experiment      = ExperimentTest(self.carla_simulator, town='Town02')

    def run_complete_experiment( self ):

        self.experiment.run_experiment( ['cloudiness', 'precipitation', 'sun_azimuth_angle', 'sun_altitude_angle', 'wetness'],
                                           max_iterations=1000, 
                                           experiment_name='testtt' )
        return
    

    def next(self):
        return self.experiment.next()
    
    def previous(self):
        return self.experiment.previous()
