from carla_tester.model.carla_simulator import CarlaSimulator
from carla_tester.model.logger import ExperimentLogger

from carla_tester import utils

from carla_tester.model.yolov7_detector import YOLOv7Detector

from carla_tester.model.agent import RandomStupidAgent
from carla_tester.model.agent import GeneticAlgorithm

from carla_tester.model.agent import BayesianOptimization

import numpy as np
import time

import random
import secrets

import carla

BBOX = 1
ACTOR = 0

class YoloDetector:
    def __init__(self):
        self.model_path = None
        self.classes = None
        self.detector = None

    def load_model(self, model_path, classes=[0, 1, 2]):
        self.model_path = model_path
        self.classes = classes
        self.detector = YOLOv7Detector(model_path, classes=self.classes)

    def run_yolo(self, frame):   
        pred_img, preds = self.detector.detect( frame )

        # trasform the prediction format as the input
        preds_annotations = utils.convert_to_yolo_annotation(preds, 
                                                            pred_img.shape[0], 
                                                            pred_img.shape[1])

        return pred_img, preds_annotations


class Experiment:
    def __init__(self) -> None:
        self.agent = None
        # simulation environment
        self.simulator = None
        self.ml_model = None
        self.logger = None
        

    def init(self):
        self.ml_model = YoloDetector()
        self.ml_model.load_model( '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt' )

        self.logger = ExperimentLogger('bayesian_01')

        self.carla_simulator = CarlaSimulator()
        self.carla_simulator.set_town( 'Town01' )

        self.targets = self.carla_simulator.get_detection_targets()

        self.population_size = 8
        self.iou_th = 0.5

        self.parameters = dict()
        self.parameters['names'] = ['cloudiness', 'precipitation', 'sun_azimuth_angle', 'sun_altitude_angle', 'wetness']
        self.parameters['bounds'] = np.array([[0, 99], [0, 99], [0, 99], [0, 99],  [0, 99]]) 

        self.agent = GeneticAlgorithm( self.evaluate_params, self.population_size, self.parameters ) # work
        
        '''
        Random Search: Randomly sample parameter combinations from the search space. It's simple but can be effective.

        Grid Search: Evaluate all possible combinations of parameter values. It's exhaustive but can be computationally expensive.

        ##Bayesian Optimization: Build a probabilistic model of the objective function and use it to find the most promising regions in the parameter space.
        #self.agent = BayesianOptimization( self.evaluate_params, self.parameters ) # shit
        
        Particle Swarm Optimization (PSO): A population-based optimization algorithm inspired by the social behavior of birds and fish.

        Simulated Annealing: An optimization algorithm that mimics the annealing process in metallurgy. It starts with a high temperature (allowing more exploration) and gradually reduces it (allowing more exploitation).

        Differential Evolution: A population-based stochastic optimization algorithm.

        CMA-ES (Covariance Matrix Adaptation Evolution Strategy): A powerful evolutionary algorithm for real-valued optimization problems.

        Neural Network-Based Approaches: Use neural networks to model the mapping between parameters and performance. Train the network and use it to guide the search.

        Genetic Programming: Evolve computer programs to solve a problem, where the programs are represented as trees.
                
        '''

        # Get a random seed using the secrets module
        random_seed = secrets.token_bytes(32)
        random.seed(random_seed)


    def run(self):
        num_iterations = 20
        best_params = self.agent.run(num_iterations)
        print("Best Parameters:", best_params)
        return
    


    # Simulation method
    def evaluate_params(self, params_np_values, iteration_n):

        def get_carla_weather_parameters(keys, np_values):
            weather_parameters = carla.WeatherParameters()
            for pos in range(0, len( params_np_values )):
                if hasattr(carla.WeatherParameters, keys[pos]):                   
                    setattr(weather_parameters, keys[pos], np_values[pos])
            return weather_parameters

        # Update the parameters: the weather in this case
        ## Parameters you wanna change
        new_params = get_carla_weather_parameters(self.parameters['names'], params_np_values)
        ## action
        self.carla_simulator.set_weather_settings(new_params)

        ## Write the itaration parameters in the logger
        initial_params = {
            "weather" : str(new_params), # just nthe new ones, the weather values TODO depack into a json ready format
            "tgt_trans": [str(t[ACTOR].get_transform()) for t in self.carla_simulator.targets]
        }

        # start logging
        self.logger.start_iteration(iteration_n, initial_params)

        # Timer variable
        total_time = 0
        target_count = 0

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

        print( "Iteration", iteration_n ,'/ XXX' , 'terminated in', total_time )

        score = iteration_metrics[ "average_precision" ]

        return score


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
        preds, yolo_preds = self.ml_model.run_yolo( camera_output_images['rgb']['image'] )
        camera_output_images['preds'] = {}
        camera_output_images['preds']['image'] = preds

        camera_transform = cameras_set['cameras']['rgb'].get_transform()

        ## SCORE: calculate difference between truth and predictions
        metrics = utils.calculate_metrics_detection_classification(yolo_truth, yolo_preds, iou_th)

        return camera_output_images, metrics, yolo_truth, yolo_preds, camera_transform


