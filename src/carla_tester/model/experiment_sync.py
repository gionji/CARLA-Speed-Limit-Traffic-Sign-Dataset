from carla_tester.model.carla_simulator import CarlaSimulator
from carla_tester.model.logger import ExperimentLogger

from carla_tester import utils

from carla_tester.model.yolov7_detector import YOLOv7Detector

from carla_tester.agent.random_search import RandomStupidAgent
from carla_tester.agent.genetic import GeneticAlgorithm
from carla_tester.agent.bayesian import BayesianOptimization, BayesianOptimizer, BayesianOptimizer03

from carla_tester.agent.grid_search import GridSearch, RandomGridSearch, RandomSearch
from carla_tester.agent.binary import BinarySearchAgent
from carla_tester.agent.pso import PSOAgent

from carla_tester.agent.simulated_annealing import SimulatedAnnealingAgent
from carla_tester.agent.cov_mat_adpt import CMAESAgent

from carla_tester.model.dataset_saver import DatasetSaver

import math

#from carla_tester.agent.neural_network import NeuralNetworkOptimizer

import queue

import numpy as np
import time

import random
import secrets

import carla

import pygame

import signal
import sys

BBOX = 1
ACTOR = 0

# Define a signal handler function
def signal_handler(sig, frame):
    print("Ctrl+C detected. Cleaning up and exiting.")
    pygame.quit()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)



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
    



class ExperimentSync:
    def __init__(self) -> None:
        self.agent = None
        # simulation environment
        self.simulator = None
        self.ml_model = None
        self.logger = None
        self.data_saver = None

        self.pred_count = 0
        

    def init(self):
        self.ml_model = YoloDetector()
        self.ml_model.load_model( '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt' )

        town = 'Town10HD_Opt'
        experiment_label = 'CMAESAgent'
        experiment_name = experiment_label +'-'+ town

        self.logger = ExperimentLogger( experiment_name ) 

        self.data_saver = DatasetSaver( self.logger.get_complete_experiment_name(), './' )

        self.carla_simulator = CarlaSimulator()
        self.carla_simulator.set_town( town )
        #self.carla_simulator.get_town()

        self.world = self.carla_simulator.get_world()
        self.targets = self.carla_simulator.get_detection_targets()

        self.population_size = 8
        self.iou_th = 0.5

        self.width = 640
        self.height = 480
        self.fov = 110

        self.save_dataset = False

        self.parameters = dict()
        
        self.parameters['names']  = ['cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle', 
                                     'sun_altitude_angle', 'fog_density', 'wetness', 
                                     'scattering_intensity', 'dust_storm']
        
        self.parameters['bounds'] = np.array([[0, 99], [0, 99], [0, 99], [0, 99], [0, 99],
                                              [0, 99], [0, 99], [0, 99],
                                              [0, 99], [0, 99]]) 

        self.camera_vehicle_relative_transform = carla.Transform(carla.Location(x=0.45, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))  

        # X positive go right, Y positive go far
        self.target_vehicle_relative_transform = carla.Transform(carla.Location(x=0.0, y=6.45, z=1.5), carla.Rotation(pitch=0, yaw=-90, roll=0))

        # set up the cameras on the vehicle ##################################################################3

        self.actor_list = []
        blueprint_library = self.world.get_blueprint_library()

        pygame.init()

        self.display = pygame.display.set_mode( (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.clock = pygame.time.Clock()
        self.font = self.get_font()

        m = self.world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        static_object_list = self.world.get_blueprint_library().filter( "*static.*" )
        print(static_object_list)

        self.vehicle = self.world.spawn_actor(
            random.choice(blueprint_library.filter('static.prop.brokentile02*')),
            start_pose)
        self.actor_list.append(self.vehicle)

        self.vehicle.set_simulate_physics(False)

        # Spawn the  cameras and attach them to the vehicle
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.width) )
        blueprint.set_attribute('image_size_y', str(self.height) )
        blueprint.set_attribute('fov', str(self.fov) )

        self.camera_rgb = self.world.spawn_actor(
            blueprint,
            self.camera_vehicle_relative_transform,
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        blueprint = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        blueprint.set_attribute('image_size_x', str(self.width))
        blueprint.set_attribute('image_size_y', str(self.height))
        blueprint.set_attribute('fov', str(self.fov))

        self.camera_instseg = self.world.spawn_actor(
            blueprint,
            self.camera_vehicle_relative_transform,
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_instseg)

        blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
        blueprint.set_attribute('image_size_x', str(self.width))
        blueprint.set_attribute('image_size_y', str(self.height))
        blueprint.set_attribute('fov', str(self.fov))

        self.camera_depth = self.world.spawn_actor(
            blueprint,
            self.camera_vehicle_relative_transform,
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_depth)


        # Get a random seed using the secrets module
        random_seed = secrets.token_bytes(32)
        random.seed(random_seed)

        ##Random Search: Randomly sample parameter combinations from the search space. It's simple but can be effective.
        self.agent = RandomSearch( self.evaluate_params, self.parameters ) # da migliorearwe

        ##Grid Search: Evaluate all possible combinations of parameter values. It's exhaustive but can be computationally expensive.
        #self.agent = RandomGridSearch( self.evaluate_params, self.parameters, num_iterations=100 )
        # self.agent = BinarySearchAgent( self.evaluate_params, self.parameters )
        
        ##Bayesian Optimization: Build a probabilistic model of the objective function and use it to find the most promising regions in the parameter space.
        #self.agent = BayesianOptimizer03( self.evaluate_params, self.parameters,  num_iterations=1000) 
        
        ##Particle Swarm Optimization (PSO): A population-based optimization algorithm inspired by the social behavior of birds and fish.
        # self.agent = PSOAgent( self.evaluate_params, self.parameters )

        ##Simulated Annealing: An optimization algorithm that mimics the annealing process in metallurgy. It starts with a high temperature (allowing more exploration) and gradually reduces it (allowing more exploitation).
        #self.agent = SimulatedAnnealingAgent( self.evaluate_params, self.parameters )

        ##CMA-ES (Covariance Matrix Adaptation Evolution Strategy): A powerful evolutionary algorithm for real-valued optimization problems.
        self.agent = CMAESAgent( self.evaluate_params, self.parameters )

        ##Genetic Programming: Evolve computer programs to solve a problem, where the programs are represented as trees.
        #self.agent = GeneticAlgorithm( self.evaluate_params, self.population_size, self.parameters )        
                
        ##Differential Evolution: A population-based stochastic optimization algorithm.
        #self.agent = DifferentialEvolutionOptimizer( self.evaluate_params, self.parameters ) ## NOOOOO
     
        ### Not working
        ##Neural Network-Based Approaches: Use neural networks to model the mapping between parameters and performance. Train the network and use it to guide the search.
        # self.agent = NeuralNetworkOptimizer( self.evaluate_params, self.parameters )


    def run(self):
        try:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb, self.camera_instseg, self.camera_depth, fps=1)
            self.sync_mode.__enter__()

            best_params = self.agent.run( )

            print("Best Parameters:", best_params)

        finally:
            self.sync_mode.__exit__()
            print('destroying actors.')

            for actor in self.actor_list:
                actor.destroy()

            pygame.quit()
            print('done.')

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
        recall_values = []## move the camera

        try:
            for target in self.carla_simulator.targets:
                if should_quit():
                    return
                self.clock.tick()

                timestamp = time.time()

                # Assuming you have transforms A and B
                transform_A = target[ACTOR].get_transform() 
                transform_B = self.target_vehicle_relative_transform

                # Apply transform A
                location_A = transform_A.location
                rotation_A = transform_A.rotation
                # Apply transform B relative to the new position and orientation
                # Rotate B's location using basic math functions and add to A's location
                yaw_rad = math.radians( (rotation_A.yaw ))
                x_B_rotated = transform_B.location.x * math.cos(yaw_rad) - transform_B.location.y * math.sin(yaw_rad)
                y_B_rotated = transform_B.location.x * math.sin(yaw_rad) + transform_B.location.y * math.cos(yaw_rad)

                location_result = carla.Location(x=location_A.x + x_B_rotated, y=location_A.y + y_B_rotated, z=location_A.z + transform_B.location.z)
                # Add rotations by obtaining yaw angles and creating a new Rotation object
                yaw_A = math.radians(rotation_A.yaw)
                yaw_B = math.radians(transform_B.rotation.yaw)
                yaw_result = math.degrees(yaw_A + yaw_B)  # Add yaw angles

                rotation_result = carla.Rotation(pitch=rotation_A.pitch, yaw=yaw_result, roll=rotation_A.roll)
                transform_result = carla.Transform(location=location_result, rotation=rotation_result)

                self.vehicle.set_transform(transform_result)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, image_depth = self.sync_mode.tick(timeout=2.0)

                target_count += 1
                start_time = time.time() 

                self.pred_count  += 1

                # get the metrics back
                truth_image, preds_image, metrics, yolo_truth, yolo_preds, camera_transform = self.process_single_frame(timestamp, 
                                                                                                                image_rgb, 
                                                                                                                image_semseg, 
                                                                                                                image_depth, 
                                                                                                                transform_result, 
                                                                                                                iou_th=0.5)


                
                if self.save_dataset:
                    self.data_saver.save_frame(timestamp, image_rgb, image_semseg, image_depth, yolo_truth)

                # after processing the frame
                frame_info = {
                    "timestamp" : timestamp,
                    "tgt"     : target_count,
                    "bb_truth": yolo_truth,  
                    "bb_pred" : yolo_preds, 
                    "prec"    : metrics["precision"],
                    "recall"  : metrics["recall"],
                    "cam_tran": str( camera_transform )
                }
                # Save frame information to a JSON file
                self.logger.save_frame(frame_info)

                # save the iterations metrics results
                precision_values.append(metrics["precision"])
                recall_values.append(metrics["recall"])

                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(self.display, truth_image)
                draw_image(self.display, preds_image, blend=True)
                #draw_image(display, image_depth, blend=True)

                # Stop timer to measure the iteration execution time
                end_time = time.time()
                iteration_time = end_time - start_time
                total_time += iteration_time

                pygame.display.flip()
            # qui finisco tutti i target
        finally:  
            iteration_metrics = {
                "average_precision" : np.mean(np.array(precision_values)),
                "average_recall"    : np.mean(np.array(recall_values)) 
            }

            print('iteration done', iteration_metrics)

             # save the scores in to the loge file
            self.logger.end_iteration( iteration_metrics )

            score = iteration_metrics[ "average_precision" ]

            print( "Iteration", iteration_n, 'score:', score, 'terminated in', total_time) 

            self.display.blit(
                self.font.render('Prev. iter score % 5d ' % score, 
                                 True, 
                                 (255, 255, 255)),
                                 (8, 10))
            pygame.display.flip()     

            return score
    
    
    def get_font(self):
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)
    

    def process_single_frame(self, timestamp, image_rgb, image_semseg, image_depth, camera_transform, iou_th=0.5):
        ## Define the relative position of the camera from the main target based on distance, and angles.
        #camera_relative_transform = utils.get_camera_spherical_relative_transform(8, 0, 0)
        ## generate the image with the bounding boxes
        # Calculate the camera projection matrix to project from 3D -> 2D

        K = utils.build_projection_matrix(self.width, self.height, self.fov)
        # Get the world to camera matrix
        world_2_camera = np.array(camera_transform.get_inverse_matrix())

        image = image_rgb
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        img = array
        truth_image, yolo_truth = self.carla_simulator.get_truth(camera_transform, img, self.targets, K, world_2_camera)
        ## add the rgb image with bboxes in the sensorsdata dict

        ## calculate predictions
        preds_image, yolo_preds = self.ml_model.run_yolo( img )

        ## SCORE: calculate difference between truth and predictions
        metrics = utils.calculate_metrics_detection_classification(yolo_truth, yolo_preds, iou_th)

        return truth_image, preds_image, metrics, yolo_truth, yolo_preds, camera_transform
    

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

        self.delta_seconds          = 10.0 / kwargs.get('fps', 20)
        self.max_substep_delta_time = 0.1  
        self.max_substeps           = 16.0 

        # Assert the condition for simulation parameters
        print('self.delta_seconds', self.delta_seconds)
        print('e mooooo', self.max_substep_delta_time * self.max_substeps)

        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()

        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            no_rendering_mode=False,
            fixed_delta_seconds=self.delta_seconds,
            substepping=True,
            max_substep_delta_time=self.max_substep_delta_time,
            max_substeps=int(self.max_substeps),  # Ensure max_substeps is an integer
            max_culling_distance=0.0,
            deterministic_ragdolls=False,
            tile_stream_distance=3000.0,
            actor_active_distance=2000.0
        ))


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



def save_images( timestamp, image_rgb, image_semseg, image_depth ):

    return



def save_annotations( timestamp, bboxes):

    return

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def draw_image(surface, image, blend=False):
    if isinstance(image, carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise ValueError("Unsupported image type")

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