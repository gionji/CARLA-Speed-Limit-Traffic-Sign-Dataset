import os
import cv2
from typing import List
import carla

class DatasetSaver:
    def __init__(self, experiment_full_name: str, dest_path: str):
        self.dataset_path = os.path.join(dest_path, experiment_full_name)
        self.image_path = os.path.join(self.dataset_path, 'data', 'images')
        self.rgb_path = os.path.join(self.image_path, 'rgb')
        self.depth_path = os.path.join(self.image_path, 'depth')
        self.instance_path = os.path.join(self.image_path, 'instance')
        self.annotation_path = os.path.join(self.dataset_path, 'data', 'annotations')

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.instance_path, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)

    def save_annotation(self, timestamp, yolo_annotation: List[str]):
        annotation_file = os.path.join(self.annotation_path, f'{timestamp}.txt')
        with open(annotation_file, 'w') as file:
            file.writelines("\n".join(yolo_annotation))

    def save_image_rgb(self, timestamp, image: carla.libcarla.Image):
        image_file = os.path.join(self.rgb_path, f'{timestamp}.png')
        self._save_image(image, image_file, cv2.COLOR_BGR2RGB)

    def save_image_instance_segmentation(self, timestamp, image: carla.libcarla.Image):
        image_file = os.path.join(self.instance_path, f'{timestamp}.png')
        self._save_image(image, image_file)

    def save_image_depth(self, timestamp, image: carla.libcarla.Image):
        image_file = os.path.join(self.depth_path, f'{timestamp}.png')
        self._save_image(image, image_file)

    def _save_image(self, image, file_path):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        cv2.imwrite(file_path, array)
    
    def save_frame(self, timestamp, rgb_image: carla.libcarla.Image, instance_image: carla.libcarla.Image,
                   depth_image: carla.libcarla.Image, yolo_annotation: List[str]):
        self.save_image_rgb(timestamp, rgb_image)
        self.save_image_instance_segmentation(timestamp, instance_image)
        self.save_image_depth(timestamp, depth_image)
        self.save_annotation(timestamp, yolo_annotation)