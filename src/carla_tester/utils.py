import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import math

BBOX = 1
ACTOR = 0

OBJECT_LABEL = 8  # Replace with the label corresponding to the object in the segmentation image
MIN_DEPTH = 100  # Replace with the minimum depth value that signifies visibility

def build_projection_matrix( w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point( loc, K, w2c):
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


class DetectionScorer:
    def __init__(self):
        # Define scores for different cases
        self.scores = {
            'true_positive_detection':   1.0,
            'false_positive_detection':  1.0,
            'true_negative_detection':   1.0,
            'false_negative_detection':  1.0,

            'true_positive_classification': 0.0,
            'false_positive_classification': 0.0,
            'true_negative_classification': 0.0,
            'false_negative_classification': 0.0,
        }

    def calculate_scores(self, truth, preds, threshold=0.7):
        scores = {
            'detection': {
                'true_positive':  0,
                'false_positive': 0,
                'true_negative':  0,
                'false_negative': 0
            },
            'classification': {
                'true_positive':  0,
                'false_positive': 0,
                'true_negative':  0,
                'false_negative': 0
            }
        }
        total_score = 0  # Initialize total score

        if len(truth) > len(preds):
            truth, preds = preds, truth

        for truth_box in truth:
            best_iou = 0
            best_match = None
            for pred_box in preds:
                if int(truth_box[0]) == int(pred_box[0]):
                    iou = self.calculate_iou(truth_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_box

            if best_iou > threshold:
                if best_match:
                    scores['detection']['true_positive'] += self.scores['true_positive_detection']
                    # Check for correct classification
                    if int(truth_box[0]) == int(best_match[0]):
                        scores['classification']['true_positive'] += self.scores['true_positive_classification']
                    else:
                        scores['classification']['false_positive'] += self.scores['false_positive_classification']
                else:
                    scores['detection']['false_negative'] += self.scores['false_negative_detection']

            else:
                if best_match:
                    scores['detection']['false_positive']     += self.scores['false_positive_detection']
                    scores['classification']['true_negative'] += self.scores['true_negative_classification']
                else:
                    scores['detection']['true_negative'] += self.scores['true_negative_detection']
                    scores['classification']['false_negative'] += self.scores['false_negative_classification']

        # Sum all the scores
        for key, value in scores.items():
            for sub_key, sub_value in value.items():
                total_score += sub_value

        return total_score, scores
    

    def count_correct_detections(self, truth, preds, threshold=0.5):
        correct_detections = 0

        # To avoid out of range errors, loop through the smaller list
        if len(truth) > len(preds):
            truth, preds = preds, truth

        for truth_box in truth:
            best_iou = 0
            for pred_box in preds:
                if int(truth_box[0]) == int(pred_box[0]):
                    iou = self.calculate_iou(truth_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou

            if best_iou > threshold:
                correct_detections += 1

        return correct_detections


    def calculate_iou(self, box1, box2):
        # Extract coordinates and areas
        box1_class, box1_x, box1_y, box1_w, box1_h = box1
        box2_class, box2_x, box2_y, box2_w, box2_h = box2

        # Convert the coordinates from YOLO format to regular bounding box format
        box1_x1, box1_x2 = box1_x - box1_w / 2, box1_x + box1_w / 2
        box1_y1, box1_y2 = box1_y - box1_h / 2, box1_y + box1_h / 2
        box2_x1, box2_x2 = box2_x - box2_w / 2, box2_x + box2_w / 2
        box2_y1, box2_y2 = box2_y - box2_h / 2, box2_y + box2_h / 2

        # Calculate the intersection coordinates
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # Calculate the intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the individual areas of the bounding boxes
        box1_area = box1_w * box1_h
        box2_area = box2_w * box2_h

        # Calculate the union area
        union = box1_area + box2_area - intersection

        # Calculate IOU
        iou = intersection / union if union > 0 else 0

        return iou
    
    def calculate_confusion_matrix(self, ground_truth, predicted, threshold=0.7):
        # Initialize confusion matrix
        confusion_matrix = {}

        for gt_box in ground_truth:
            gt_class = gt_box[0]
            matched = False

            for pred_box in predicted:
                pred_class = pred_box[0]

                # Check if classes match
                if pred_class == gt_class:
                    gt_xmin, gt_xmax, gt_ymin, gt_ymax = gt_box[1:]
                    pred_xmin, pred_xmax, pred_ymin, pred_ymax = pred_box[1:]

                    # Calculate Intersection over Union (IoU)
                    x_left = max(gt_xmin, pred_xmin)
                    x_right = min(gt_xmax, pred_xmax)
                    y_top = max(gt_ymin, pred_ymin)
                    y_bottom = min(gt_ymax, pred_ymax)

                    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
                    gt_box_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
                    pred_box_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
                    union_area = gt_box_area + pred_box_area - intersection_area

                    iou = intersection_area / union_area

                    if iou > threshold:  # Threshold for considering a match
                        matched = True
                        if gt_class not in confusion_matrix:
                            confusion_matrix[gt_class] = {gt_class: 1}
                        else:
                            if gt_class not in confusion_matrix[pred_class]:
                                confusion_matrix[gt_class][pred_class] = 1
                            else:
                                confusion_matrix[gt_class][pred_class] += 1

            # If no match found for ground truth box, it's a false negative
            if not matched:
                if gt_class not in confusion_matrix:
                    confusion_matrix[gt_class] = {gt_class: 0}
                else:
                    confusion_matrix[gt_class][gt_class] = 0

        return confusion_matrix
    

import math
'''
def is_object_face_another_object(ego, sign, ego_axis =(1,1), ego_rot = -1, ego_phase = 90, sign_axis =(-1,1), sign_rot=-1, sign_phase=90):
    digits = 1

    (Xa, Ya, Ra) = ego.location.x
    (Xa, Ya, Ra) = ego.location.y
    (Xa, Ya, Ra) = ego.rotation.yaw
    
    (Xb, Yb, Rb) = sign.location.x        
    (Xb, Yb, Rb) = sign.location.y        
    (Xb, Yb, Rb) = sign.rotation.yaw

    tetha_a = ego_rot  * (Ra - ego_phase)
    tetha_b = sign_rot * (Rb - sign_phase)

    # Calculate direction vectors
    ego_direction =  ( ego_axis[0] * round(math.cos(math.radians(tetha_a)), 2), ego_axis[1] *  round(math.sin(math.radians(tetha_a)), 2) )
    sign_direction = ( sign_axis[0] * round(math.cos(math.radians(tetha_b)), 2), sign_axis[1] * round(math.sin(math.radians(tetha_b)), 2) )

    # Calculate vectors from A to B and from B to A
    difference_between_ego_and_sign = (round(Xb - Xa, digits), round(Yb - Ya, digits))
    difference_between_sign_and_ego = (round(Xa - Xb, digits), round(Ya - Yb, digits))
    
    # Calculate the dot products
    dot_product_A_to_B = round(np.dot(ego_direction,  difference_between_ego_and_sign), digits)
    dot_product_B_to_A = round(np.dot(sign_direction, difference_between_sign_and_ego), digits)
    
    # Check if both objects are within the FoV of each other
    see_each_other = dot_product_A_to_B < 0 and dot_product_B_to_A < 0 

    return see_each_other
    '''

def is_object_face_another_object(ego, sign, 
                                  ego_axis=(1, 1), 
                                  ego_rot=1, 
                                  ego_phase=0, 
                                  sign_axis=(-1, 1), 
                                  sign_rot=1, 
                                  sign_phase=0):
    digits = 1

    Xa, Ya, Ra = ego.location.x, ego.location.y, ego.rotation.yaw
    Xb, Yb, Rb = sign.location.x, sign.location.y, sign.rotation.yaw

    tetha_a = ego_rot * (Ra - ego_phase)
    tetha_b = sign_rot * (Rb - sign_phase)

    print('cam angle in the method: ', int(tetha_a), int(tetha_b))

    #print('Cx, Cy, Cth : ', Xa, Ya, tetha_a)
    #print('Tx, Ty, Tth : ', Xb, Yb, tetha_b)

    # Calculate direction vectors
    ego_direction  = (ego_axis[0] * round(math.cos(math.radians(tetha_a)), 2), ego_axis[1] * round(math.sin(math.radians(tetha_a)), 2))
    sign_direction = (sign_axis[0] * round(math.cos(math.radians(tetha_b)), 2), sign_axis[1] * round(math.sin(math.radians(tetha_b)), 2))

    # Calculate vectors from A to B and from B to A
    difference_between_ego_and_sign = (round(Xb - Xa, digits), round(Yb - Ya, digits))
    difference_between_sign_and_ego = (round(Xa - Xb, digits), round(Ya - Yb, digits))

    # Calculate the dot products
    dot_product_A_to_B = round(np.dot(ego_direction, difference_between_ego_and_sign), digits)
    dot_product_B_to_A = round(np.dot(sign_direction, difference_between_sign_and_ego), digits)

    # Check if both objects are within the FoV of each other
    see_each_other = dot_product_A_to_B < 0 and dot_product_B_to_A < 0

    return see_each_other

import carla

def distance_between(a, b):
    # You need to define a suitable distance metric based on location and rotation attributes
    # For example, Euclidean distance between locations or any custom distance metric
    # Compute distance based on the attributes of carla.Location and carla.Rotation
    # Example:
    location_distance = a.get_transform().location.distance(b.location)
    rotation_distance = a.get_transform().rotation.yaw - b.rotation.yaw  # Custom rotation distance calculation
    total_distance = location_distance + abs(rotation_distance)  # Combine both distances
    return total_distance

def get_speed_limit_signs(A, B):
    # Match elements in A with the closest elements in B
    C = []
    used_indices = set()

    for a in A:
        min_distance = float('inf')
        closest_b = None
        for i, b in enumerate(B):
            if i not in used_indices:
                dist = distance_between(a, b)
                if dist < min_distance:
                    min_distance = dist
                    closest_b = b
                    closest_index = i
        used_indices.add(closest_index)
        C.append((a, closest_b))
    
    return C


def get_speed_limit(carla_actor):
    return carla_actor.type_id.split('limit.')[1]


def is_bounding_box_within_frame(bbox, image):
    xmin, xmax, ymin, ymax = bbox
    window_width  = image.shape[1]
    window_height = image.shape[0]
    # Check if all coordinates of the bounding box are within the window boundaries
    if xmin >= 0 and xmax <= window_width and ymin >= 0 and ymax <= window_height:
        return True
    else:
        return False




def draw_carla_actors_on_image(targets, camera=None, output_format="jpg"):
    width, height = 500, 500
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (255, 255, 255)  # White background

    actor_color = (0, 255, 0)  # Black triangle
    camera_color = (0, 0, 255)  # Red triangle (for the camera)

    map_size = (500, 500)  # Size of the map
    map_position = (20, height - map_size[1] - 20)  # Position of the map in the screen (BOTTOM_RIGHT corner)
    map_drawer = MapDrawer(map_size, map_position, img)
    

    if camera != None:
        actor = camera
        location = actor.get_transform().location
        x, y, z = location.x, location.y, location.z
        rotation = actor.get_transform().rotation
        pitch, yaw, roll = rotation.pitch, rotation.yaw, rotation.roll

        map_drawer.draw_object(x, y, yaw, color=(255,0,0))

    for target in targets:
        actor = target[ACTOR]
        location = actor.get_transform().location
        x, y, z = location.x, location.y, location.z
        rotation = actor.get_transform().rotation
        pitch, yaw, roll = rotation.pitch, rotation.yaw, rotation.roll

        map_drawer.draw_object(x, y, yaw)

    img = map_drawer.get_map_with_objects()

    return img


class MapDrawer:
    def __init__(self, map_size, map_position, base_img):
        self.map_size = map_size
        self.map_position = map_position
        self.base_img = base_img

    def draw_object(self, x, y, rotation_degrees, color=(0, 255, 0)):
        # Calculate the object's position on the map
        map_x = self.map_position[0] + int((x / self.base_img.shape[1]) * self.map_size[0])
        map_y = self.map_position[1] + int((y / self.base_img.shape[0]) * self.map_size[1])
        # Draw a triangle for the object at its position and orientation on the map
        triangle_points = np.array([
            [map_x,   map_y],
            [map_x + 5, map_y + 10],
            [map_x - 5, map_y + 10]
        ], np.int32)
        triangle_points = triangle_points.reshape((-1, 1, 2))
        rotated_triangle = self.rotate_triangle(triangle_points, map_x, map_y, rotation_degrees)

        cv2.fillPoly(self.base_img, [rotated_triangle], color)  # Green triangle


    def rotate_triangle(self, triangle, x, y, angle_degrees):
        # Create a rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle_degrees, 1)
        # Apply the rotation matrix to the triangle vertices
        rotated_triangle = cv2.transform(triangle, rotation_matrix)
        return rotated_triangle
    
    def get_map_with_objects(self):
        return self.base_img



'''
Image Output
The instance segmentation image saved to disk has 
the instance ID's encoded in the G and B channels of the RGB image file. 
The R channel contains the standard semantic ID.
'''

def calculate_visibility(instance_image, depth_image, bounding_box, distance_from_target):
    # Extract bounding box coordinates
    x1, x2, y1, y2 = bounding_box
    bounding_box_area = (x2 - x1) * (y2 - y1)

    cropped_instance_image = instance_image[y1:y2, x1:x2]
    cropped_depth_image    = depth_image[y1:y2, x1:x2]

    color_channel = 2
    unique_ch0, _ = np.unique(cropped_instance_image[:,:,0], return_counts=True)
    unique_ch1, _ = np.unique(cropped_instance_image[:,:,1], return_counts=True)
    unique_classes, counts = np.unique(cropped_instance_image[:,:,2], return_counts=True)
    class_counts = {class_index: count for class_index, count in zip(unique_classes, counts)}

    #print(cropped_depth_image)
   #print( 'Dist: ', distance_from_target, 'Avg: ', np.average(cropped_depth_image) )

    return 


def convert_to_yolo_annotation(predictions, image_height, image_width):
    yolo_annotations = []
    for prediction in predictions[0]:
        x1, y1, x2, y2, confidence, class_id = prediction.tolist()

        # Calculating center x, center y, width, and height of the bounding box
        bbox_width  = (x2 - x1) / image_width
        bbox_height = (y2 - y1) / image_height
        center_x    = (x1 + x2) / (2.0 * image_width)
        center_y    = (y1 + y2) / (2.0 * image_height)

        # Construct YOLO annotation string
        class_id = str(int(class_id))
        annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        yolo_annotations.append(annotation)

    return yolo_annotations


def bbox_to_yolo_annotation(class_id, bbox, image_h, image_w):
    x1, x2, y1, y2 = bbox
    class_id = str(int((int(class_id) / 30) - 1))
    # Calculate normalized coordinates and dimensions
    bbox_width = (x2 - x1) / image_w
    bbox_height = (y2 - y1) / image_h
    center_x = (x1 + x2) / (2.0 * image_w)
    center_y = (y1 + y2) / (2.0 * image_h)

    # Construct YOLO annotation string
    annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
    return annotation


def calculate_iou(box1, box2, threshold=0.5):
    # Extract box coordinates
    truth = [float(i) for i in box1.split()[1:]]
    preds = [float(i) for i in box2.split()[1:]]

    # Calculate box coordinates
    truth_x, truth_y, truth_w, truth_h = truth
    preds_x, preds_y, preds_w, preds_h = preds

    # Convert to (x1, y1, x2, y2)
    truth_x1 = truth_x - truth_w / 2
    truth_y1 = truth_y - truth_h / 2
    truth_x2 = truth_x + truth_w / 2
    truth_y2 = truth_y + truth_h / 2

    preds_x1 = preds_x - preds_w / 2
    preds_y1 = preds_y - preds_h / 2
    preds_x2 = preds_x + preds_w / 2
    preds_y2 = preds_y + preds_h / 2

    # Calculate the intersection area
    xA = max(truth_x1, preds_x1)
    yA = max(truth_y1, preds_y1)
    xB = min(truth_x2, preds_x2)
    yB = min(truth_y2, preds_y2)

    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of each box
    truth_box_area = truth_w * truth_h
    preds_box_area = preds_w * preds_h

    # Calculate the union area
    union_area = truth_box_area + preds_box_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    # Check if IoU is greater than the specified threshold
    is_above_threshold = iou > threshold

    return iou, is_above_threshold




def calculate_metrics_detection_classification(annotations, preds, iou_threshold):
    tp = 0 # predizione vera su oggetto vero
    fn = 0 # ha sbagliato a dire no
    fp = 0 # ha sbagliato a dire si, in realta non c'e'

    tpc = 0

    for annotation in annotations:
        best_iou = 0
        best_iou_prediction = None
        for prediction in preds:
            iou, is_iou_great = calculate_iou(annotation, prediction)
            if iou > best_iou:
                best_iou = iou
                best_iou_prediction = prediction
        if best_iou > iou_threshold:
            tp += 1
            #print(annotation.split(), best_iou_prediction.split())
            if annotation.split()[0] == best_iou_prediction.split()[0]:
                tpc +=1
        else:
            fn += 1
        
    for prediction in preds:
        best_iou = 0
        for annotation in annotations:
            iou, is_iou_great = calculate_iou(annotation, prediction)
            if iou > best_iou:
                best_iou = iou
        if best_iou < iou_threshold:
            fp += 1

    epsilon = 1e-10  # Small value to prevent division by zero

    # Calculate precision
    precision = tp / (tp + fp + epsilon) if (tp + fp) > 0 else 0
    # Calculate recall
    recall = tp / (tp + fn + epsilon) if (tp + fn) > 0 else 0
    # Calculate F1 score
    f1_score = 2 * ((precision * recall) / (precision + recall + epsilon)) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_Score": f1_score,
        "correct_classifications" : tpc,
        "wrong_classifications" : tp - tpc,
        "tp" : tp,
        "fn" : fn,
        "fp" : fp
    }



def calculate_average_precision(precision, recall):
    # Sort the precision and recall values by decreasing recall
    sorted_indices = np.argsort(recall)[::-1]
    precision = precision[sorted_indices]
    recall = recall[sorted_indices]

    # Calculate average precision using precision at different recall levels
    average_precision = 0
    for i in range(len(recall) - 1):
        average_precision += (recall[i + 1] - recall[i]) * precision[i + 1]

    return average_precision




## for joakim

def find_bounding_boxes(instance_map, depth_map, classes, out_classes, diff_th=0.1):
    r_ch = 2
    g_ch = 1
    b_ch = 0
    bounding_boxes = []
    vehicles = []
    return_val = {}
    #print('classids', class_ids)

    img=None

    # Combine G and B channels to get the instance ID
    combined_instance_ids = (instance_map[:,:,g_ch].astype(np.uint32) << 8) + instance_map[:,:,b_ch]

    # Iterate over specified class IDs
    for class_name, class_id in classes.items():
        # Skip background class (class ID 0)
        if class_id == 0:
            continue
        print(f"looking for {class_name} objects (class id {class_id}")
        # Find instances for the current class ID
        instances_for_class = np.unique(combined_instance_ids[instance_map[:,:,r_ch] == class_id])

        # Iterate over unique instance IDs for the current class
        for instance_id in instances_for_class:
            # Skip background instance (instance ID 0)
            if instance_id == 0:
                continue

            # Extract coordinates where the current instance ID is present
            coords = np.argwhere(combined_instance_ids == instance_id)

            # Calculate bounding box coordinates
            ymin, xmin = np.min(coords, axis=0)
            ymax, xmax = np.max(coords, axis=0)

            class_id_remapped = out_classes[class_name]
                 
            # Crop the corresponding bbox in the depth image
            depth_bbox = depth_map[ymin:ymax, xmin:xmax]

            # Calculate the object mask of the object in the corresponding instance bbox
            object_mask = (combined_instance_ids[ymin:ymax, xmin:xmax] == instance_id).astype(np.uint8)

            # Calculate the average distance of the masked pixels
            distance_masked = np.mean(depth_bbox[object_mask > 0])

            # Calculate the average distance of the non-masked pixels
            distance_non_masked = np.mean(depth_bbox[object_mask == 0])
            
            # difference between the front pixels and the background
            diff = distance_masked - distance_non_masked
            
            if diff > diff_th:
                bounding_boxes.append(np.array([[xmin, ymin], [xmax, ymax]]))
                vehicles.append(class_id_remapped)

            # Draw bounding box on the original image
            img = cv2.rectangle(instance_map, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return_val['bbox'] = bounding_boxes
    return_val['class'] = vehicles
    return_val['vehicle'] = vehicles

    return return_val, img


def get_camera_spherical_relative_transform(distance, azimuth, elevation):
       # Convert spherical coordinates to Cartesian coordinates
    azimuth_rad   = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    z = distance * math.sin(elevation_rad)

    # Calculate the new location relative to the target object
    new_location = carla.Location(x, y, z)

    # Calculate the rotation to keep the target object in the center of the frame
    look_at_rotation = carla.Rotation(yaw=-90 - azimuth, pitch=-elevation)

    # Create a new transform for the camera
    camera_transform = carla.Transform(new_location, look_at_rotation)

    return camera_transform



def spawn_object_in_front(world, target_actor, object_blueprint, distance, scale_factor=1.0):
    # Get the target object's location and rotation
    target_location = target_actor.get_location()
    target_rotation = target_actor.get_transform().rotation

    # Calculate the offset in front of the target actor
    offset = carla.Location(x=distance, y=0, z=0)
    offset_rotation = carla.Rotation(pitch=0, yaw=target_rotation.yaw, roll=0)
    offset_transform = carla.Transform(offset, offset_rotation)

    # Calculate the new location relative to the target object
    new_location = offset_transform.transform(target_location)

    # Create a transform for the new object
    new_transform = carla.Transform(new_location, target_rotation)

    # Spawn the object with the specified scale
    if scale_factor != 1.0:
        object_blueprint.set_attribute('scale', f'{scale_factor},{scale_factor},{scale_factor}')

    new_object = world.spawn_actor(object_blueprint, new_transform)

    return new_object


def spawn_objects_in_volume(world, target_actor, object_blueprint, volume_size, num_instances):
    # Get the target object's location and rotation
    target_location = target_actor.get_location()
    target_rotation = target_actor.get_transform().rotation

    # Define the volume size (assumed to be a cube for simplicity)
    volume_half_size = volume_size / 2.0

    spawned_objects = []

    for _ in range(num_instances):
        # Generate random position and rotation within the volume
        random_offset = carla.Location(
            x=random.uniform(-volume_half_size.x, volume_half_size.x),
            y=random.uniform(-volume_half_size.y, volume_half_size.y),
            z=random.uniform(-volume_half_size.z, volume_half_size.z)
        )

        random_rotation = carla.Rotation(
            pitch=random.uniform(0, 360),
            yaw=random.uniform(0, 360),
            roll=random.uniform(0, 360)
        )

        # Calculate the new location and rotation relative to the target object
        new_location = target_location + random_offset
        new_rotation = target_rotation + random_rotation

        # Create a transform for the new object
        new_transform = carla.Transform(new_location, new_rotation)

        # Spawn the object with the original scale
        new_object = world.spawn_actor(object_blueprint, new_transform)
        spawned_objects.append(new_object)

    return spawned_objects

