
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
                    print('iou', iou)
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