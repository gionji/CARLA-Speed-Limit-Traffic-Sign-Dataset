from carla_tester.model.carla_simulator import CarlaSimulator
from carla_tester.model.logger import ExperimentLogger

from carla_tester import utils

from carla_tester.model.yolov7_detector import YOLOv7Detector

from carla_tester.agent.random_search import RandomStupidAgent
from carla_tester.agent.genetic import GeneticAlgorithm
from carla_tester.agent.bayesian import BayesianOptimization, BayesianOptimizer, BayesianOptimizer03

from carla_tester.agent.grid_search import GridSearch, RandomGridSearch
from carla_tester.agent.binary import BinarySearchAgent
from carla_tester.agent.pso import PSOAgent

from carla_tester.agent.simulated_annealing import SimulatedAnnealingAgent
from carla_tester.agent.cov_mat_adpt import CMAESAgent

#from carla_tester.agent.neural_network import NeuralNetworkOptimizer



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
        self.carla_simulator.set_town( 'Town04' )

        self.world = self.carla_simulator.get_world()

        self.targets = self.carla_simulator.get_detection_targets()

        self.population_size = 8
        self.iou_th = 0.5

        self.parameters = dict()
        self.parameters['names']  = ['cloudiness', 'precipitation', 'sun_azimuth_angle', 'sun_altitude_angle', 'wetness']
        self.parameters['bounds'] = np.array([[0, 99], [0, 99], [0, 99], [0, 99], [0, 99]]) 

        # Get a random seed using the secrets module
        random_seed = secrets.token_bytes(32)
        random.seed(random_seed)

        ## Random Search: Randomly sample parameter combinations from the search space. It's simple but can be effective.
        # self.agent = RandomSearch( self.evaluate_params, self.parameters ) # da migliorearwe

        ## Grid Search: Evaluate all possible combinations of parameter values. It's exhaustive but can be computationally expensive.
        #self.agent = RandomGridSearch( self.evaluate_params, self.parameters )
        self.agent = BinarySearchAgent( self.evaluate_params, self.parameters )
        
        ## Bayesian Optimization: Build a probabilistic model of the objective function and use it to find the most promising regions in the parameter space.
        #self.agent = BayesianOptimizer03( self.evaluate_params, self.parameters,  num_iterations=10 ) 
        
        ## Particle Swarm Optimization (PSO): A population-based optimization algorithm inspired by the social behavior of birds and fish.
        # self.agent = PSOAgent( self.evaluate_params, self.parameters )

        ## Simulated Annealing: An optimization algorithm that mimics the annealing process in metallurgy. It starts with a high temperature (allowing more exploration) and gradually reduces it (allowing more exploitation).
        # self.agent = SimulatedAnnealingAgent( self.evaluate_params, self.parameters )

        ## CMA-ES (Covariance Matrix Adaptation Evolution Strategy): A powerful evolutionary algorithm for real-valued optimization problems.
        # self.agent = CMAESAgent( self.evaluate_params, self.parameters )

        ## Genetic Programming: Evolve computer programs to solve a problem, where the programs are represented as trees.
        #self.agent = GeneticAlgorithm( self.evaluate_params, self.population_size, self.parameters )        
                
        ## Differential Evolution: A population-based stochastic optimization algorithm.
        #self.agent = DifferentialEvolutionOptimizer( self.evaluate_params, self.parameters ) ## NOOOOO
     
        #####---> Not working
        ## Neural Network-Based Approaches: Use neural networks to model the mapping between parameters and performance. Train the network and use it to guide the search.
        # self.agent = NeuralNetworkOptimizer( self.evaluate_params, self.parameters )


    def run(self):
        num_iterations = 20
        best_params = self.agent.run( num_iterations )
        print("Best Parameters:", best_params)
        return
    

    def get_carla_weather_parameters_from_dict(self, params_dict):
        weather_parameters = carla.WeatherParameters()
        for key, value in params_dict.items():
            if hasattr(carla.WeatherParameters, key):
                # Convert the value to float if it's not already
                value = float(value)
                setattr(weather_parameters, key, value)
        return weather_parameters


    # Simulation method
    def evaluate_params(self, params_dict, iteration_n):
        # Update the parameters: the weather in this case
        ## Parameters you wanna change
        new_params = self.get_carla_weather_parameters_from_dict(params_dict)
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
            cameras_set = self.carla_simulator.add_cameras_to_a_single_target( self.world, target[ACTOR] )
            ## cameras_set is a dict containing cameras and queues keys and respectively two dicts with the camera key name 

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

            time_per_target = iteration_time / len( self.targets )

            total_time += iteration_time

        iteration_metrics = {
            "average_precision" : np.mean(np.array(precision_values)),
            "average_recall"    : np.mean(np.array(recall_values))
        }

        # save the scores in to the loge file
        self.logger.end_iteration( iteration_metrics )

        score = iteration_metrics[ "average_precision" ]

        print( "Iteration", iteration_n, 'score:', score, 'terminated in', total_time, 'time_per_target', time_per_target)

        return score


    def process_single_frame(self, target, cameras_set, iou_th):
        ## Define the relative position of the camera from the main target based on distance, and angles.
        #camera_relative_transform = utils.get_camera_spherical_relative_transform(8, 0, 0)

        ## get images from the spawned sensors
        camera_output_images, K, world_2_camera = self.carla_simulator.get_images( cameras_set )

        ## generate the image with the bounding boxes --- (camera_transform, image, targets, K, world_2_camera
        camera_transform = cameras_set['cameras']['rgb'].get_transform()
        bboxes_image, yolo_truth = self.carla_simulator.get_truth(camera_transform, 
                                                                  camera_output_images['rgb']['image'],
                                                                  self.targets,
                                                                  K,
                                                                  world_2_camera)

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