import carla
import cv2
import numpy as np
import torch



class CarlaSimulator:
    def __init__(self, host, port):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

    def create_world(self, map_name='Town01'):
        # Load a specific map
        world = self.client.load_world(map_name)

        # Set the fixed time step for simulation (optional)
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1  # Set the time step
        world.apply_settings(settings)

        return world

    def spawn_vehicle(self, world, blueprint_name='vehicle.tesla.model3'):
        # Get the blueprint for the vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(blueprint_name)[0]

        # Spawn the vehicle at a specific location
        spawn_point = carla.Transform(carla.Location(x=10, y=10, z=2), carla.Rotation(yaw=180))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        return vehicle

    def run_simulation(self):
        try:
            # Connect to CARLA and create the simulation world
            world = self.create_world()

            # Spawn a vehicle
            vehicle = self.spawn_vehicle(world)

            # Enable autopilot for the vehicle
            vehicle.set_autopilot(True)

            # Wait for the simulation to run
            input("Press Enter to exit.")

        finally:
            # Clean up and exit
            if 'vehicle' in locals():
                vehicle.destroy()
            if 'world' in locals():
                settings = world.get_settings()
                settings.fixed_delta_seconds = None  # Reset the time step
                world.apply_settings(settings)
                world.destroy()


MODEL_PATH = '/home/gionji/yolov7/runs/train/carla_gio_072/weights/best.pt'  # Path to your YOLOv7 model file
CLASS_FILE = '/home/gionji/yolov7/runs/train/carla_gio_072/class_names.txt'  # Path to a file containing class names

class YOLOv7ObjectDetector:
    def __init__(self, model_path, class_file):
        # Load YOLOv7 model
        self.model = torch.load(model_path)
        self.model = self.model['model']  # If 'model' key is used for the model in the dictionary
        self.model.eval()

        # Load class names
        with open(class_file, 'r') as file:
            self.class_names = file.read().strip().split('\n')

    def detect_objects(self, image, confidence_threshold=0.5):
        # Preprocess the image
        resized_image = cv2.resize(image, (640, 640))
        input_tensor = torch.from_numpy(resized_image).float()
        input_tensor /= 255.0  # Normalize image

        # Perform inference
        with torch.no_grad():
            results = self.model(input_tensor)

        # Post-process and filter detections
        detections = []
        for det in results[0]:
            x1, y1, x2, y2, conf, cls = det
            if conf >= confidence_threshold:
                class_label = self.class_names[int(cls)]
                detections.append({
                    'class': class_label,
                    'confidence': conf.item(),
                    'bbox': [x1.item(), y1.item(), x2.item(), y2.item()]
                })

        return detections



if __name__ == "__main__":
    # Initialize the CARLA simulator
    carla_host = 'localhost'
    carla_port = 2000
    simulator = CarlaSimulator(carla_host, carla_port)

    # Create the CARLA world, spawn a vehicle, and enable autopilot
    world = simulator.create_world()
    vehicle = simulator.spawn_vehicle(world)
    vehicle.set_autopilot(True)

    # Initialize the YOLOv7 object detector
    model_path = MODEL_PATH  # Path to the YOLOv7 model file
    class_file = CLASS_FILE  # Path to the class names file
    yolo_detector = YOLOv7ObjectDetector(model_path, class_file)

    try:
        while True:
            # Capture a frame from CARLA
            image = vehicle.get_camera('CameraRGB').get()
            frame = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            frame = frame.reshape((image.height, image.width, 4))

            # Preprocess the frame
            frame = frame[:, :, :3]  # Remove the alpha channel
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Perform object detection with YOLOv7
            confidence_threshold = 0.5
            detections = yolo_detector.detect_objects(frame, confidence_threshold)

            # Process and display the results
            for detection in detections:
                class_label = detection['class']
                confidence = detection['confidence']
                bbox = detection['bbox']

                print(f"Class: {class_label}, Confidence: {confidence:.2f}, Bbox: {bbox}")

    except KeyboardInterrupt:
        pass

    finally:
        # Clean up and exit
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'world' in locals():
            settings = world.get_settings()
            settings.fixed_delta_seconds = None  # Reset the time step
            world.apply_settings(settings)
            world.destroy()