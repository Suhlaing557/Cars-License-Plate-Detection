'''import json
import numpy as np

def load_annotations(gt_path, dt_path):
    with open(gt_path) as f:
        ground_truths = json.load(f)
        for ann in ground_truths['annotations']:
            image_id = ann['image_id']
            for image in ground_truths['images']:
                if image["id"] == image_id:
                    ann["file_name"] = image["file_name"]
    with open(dt_path) as f:
        predictions = json.load(f)
        predictions = predictions["annotations"]
    return ground_truths, predictions

# Example usage
gt_path = '/home/su/techsolution/YOLOX-main/datasets/COCO/annotations/instances_val2017.json'
dt_path = '/home/su/techsolution/YOLOX-main/datasets/COCO/annotations/inference_validation.json'
ground_truths, predictions = load_annotations(gt_path, dt_path)

# print(ground_truths)
# print(predictions)
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2)
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


# # Example usage
# box1 = [50, 50, 100, 100]  # [x, y, width, height]
# box2 = [60, 60, 80, 80]
# iou = compute_iou(box1, box2)
# print(f'IoU: {iou}')


def match_predictions_to_ground_truth(ground_truths, predictions, iou_threshold=50):
    matched_gt = set()
    tp, fp, fn = 0, 0, 0

    for dt in predictions:
        dt_box = dt['bbox']
        dt_score = dt['score']
        dt_category = dt['category_id']
        dt_image_name = dt['file_name']

        best_iou = 0
        best_gt = None
        for gt in ground_truths:
            if gt['file_name'] == dt_image_name and gt['category_id'] == dt_category:
                gt_box = gt['bbox']
                iou = compute_iou(dt_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_iou >= iou_threshold:
            if best_gt not in matched_gt:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


# # Example usage
# # Assuming ground_truths and predictions are lists of dictionaries with 'bbox', 'category_id', 'image_id', etc.
# tp, fp, fn = match_predictions_to_ground_truth(ground_truths['annotations'], predictions)
# print(f'TP: {tp}, FP: {fp}, FN: {fn}')


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


def calculate_average_precision(ground_truths, predictions, iou_thresholds=[0.5, 0.7,0.8, 1]):
    precisions, recalls = [], []
    for iou_threshold in iou_thresholds:
        tp, fp, fn = match_predictions_to_ground_truth(ground_truths, predictions, iou_threshold)
        precision, recall = calculate_metrics(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)
    print("precision", precisions, "recall", recalls)
    return np.mean(precisions), np.mean(recalls)


# Example usage
average_precision, average_recall = calculate_average_precision(ground_truths['annotations'], predictions)
print(f'Average Precision: {average_precision}, Average Recall: {average_recall}')
'''
''''| class   | AP     |
|:--------|:-------|
| licence | 58.352 |
per class AR:
| class   | AR     |
|:--------|:-------|
| licence | 65.882 |
'''

import numpy as np

def compute_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = np.split(box1, 4, axis=1)
    x1g, y1g, x2g, y2g = np.split(box2, 4, axis=1)

    xA = np.maximum(x1, x1g)
    yA = np.maximum(y1, y1g)
    xB = np.minimum(x2, x2g)
    yB = np.minimum(y2, y2g)

    interArea = np.maximum(xB - xA + 1, 0) * np.maximum(yB - yA + 1, 0)
    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x2g - x1g + 1) * (y2g - y1g + 1)

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def IoU(pred_box, label_box):
    print("predict box", pred_box)
    # Extract prediction boxes
    # x0 = pred_box[0]
    # y0 = pred_box[1]
    # width = pred_box[2]
    # height = pred_box[3]
    x0, y0, width, height =  np.split(pred_box, 4, axis=1)
    x0_l, y0_l, width_l, height_l =  np.split(label_box, 4, axis=1)
    print('x0', x0, y0)
    # # Extract label boxes
    # x0_l = label_box[0]
    # y0_l = label_box[1]
    # width_l = label_box[2]
    # height_l = label_box[3]

    # Get the overlap in width ranges
    left_bound = max(x0, x0_l)
    right_bound = min(x0 + width, x0_l + width_l)
    upper_bound = min(y0, y0_l)
    lower_bound = max(y0 - height, y0_l - height_l)
    # print(left_bound, right_bound, lower_bound,  upper_bound)

    # Calculate metrics
    intersection = max(right_bound - left_bound, 0) * max(upper_bound - lower_bound, 0)
    area = width * height
    area_l = width_l * height_l
    union = area + area_l - intersection

    # Calculate IoU
    return intersection / union


def average_precision_per_class(true_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """
    Calculate average precision for a single class.
    """
    true_boxes = true_boxes.astype(np.float32)
    pred_boxes = pred_boxes.astype(np.float32)
    pred_scores = pred_scores.astype(np.float32)
    # Sort predictions by score
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    detected = []

    for i, pred_box in enumerate(pred_boxes):
        ious = IoU(pred_box[None, :], true_boxes)
        max_iou = np.max(ious)
        max_iou_index = np.argmax(ious)

        print(f"Prediction {i + 1}: Box = {pred_box}, IoU = {max_iou}")

        if max_iou >= iou_threshold and max_iou_index not in detected:
            tp[i] = 1
            detected.append(max_iou_index)
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precision = tp / (tp + fp)
    recall = tp / len(true_boxes)

    # Compute AP using precision-recall curve
    ap, precision_curve, recall_curve = compute_ap(recall, precision)

    return ap

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (array)
        precision: The precision curve (array)
    # Returns
        Average precision (float), precision curve (array), recall curve (array)
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Calculate average precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec
def xyxy2xywh( bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

# Example usage
true_boxes = np.array([[122,  195,   143,  39]])

pred_boxes = np.array([[128.1002, 198.0349, 259.3784, 231.4482]])

score = np.array([0.8647])
true_boxes = xyxy2xywh(np.array(true_boxes))

ap = average_precision_per_class(true_boxes, pred_boxes, score)
print("Average Precision (AP):", ap)


'''
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

gt_path = '/home/su/techsolution/YOLOX-main/datasets/COCO/test/instances_val2017.json'
dt_path = '/home/su/techsolution/YOLOX-main/datasets/COCO/test/inference_validation.json'

# Load ground truth
coco_gt = COCO(gt_path)
print("coco_gt", coco_gt)

# Load predictions
with open(dt_path) as f:
    pred = json.load(f)
    coco_dt = coco_gt.loadRes(pred)
print("coco_dt", coco_dt)
# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# Evaluate on all categories
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("************************\n")




def per_class_AP_table(coco_eval, class_names=[], headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    print(precisions.shape)
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

def per_class_AR_table(coco_eval, class_names=[], headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ground truth data
coco_gt_data = {
    'images': [
        {'id': 1, 'width': 640, 'height': 480, 'file_name': 'image1.jpg'},
        {'id': 2, 'width': 640, 'height': 480, 'file_name': 'image2.jpg'}
    ],
    'annotations': [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 100, 100], 'area': 10000, 'iscrowd': 0},
        #{'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [150, 150, 80, 120], 'area': 9600, 'iscrowd': 0},
        {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [100, 100, 50, 50], 'area': 2500, 'iscrowd': 0}
    ],
    'categories': [
        {'id': 1, 'name': 'category1'},
        #{'id': 2, 'name': 'category2'}
    ]
}

# Predictions data
predictions = [
    {'image_id': 1, 'category_id': 1, 'bbox': [48, 48, 100, 100], 'score': 0.9},
    #{'image_id': 1, 'category_id': 2, 'bbox': [160, 160, 80, 120], 'score': 0.75},
    {'image_id': 2, 'category_id': 1, 'bbox': [110, 110, 50, 50], 'score': 0.85}
]

# Convert ground truth dictionary to COCO object
coco_gt = COCO()
coco_gt.dataset = coco_gt_data
coco_gt.createIndex()

# Convert predictions list to COCO object
coco_dt = coco_gt.loadRes(predictions)

# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# Run evaluation
coco_eval.evaluate()
coco_eval.accumulate()
#coco_eval.summarize()

redirect_string = io.StringIO()
with contextlib.redirect_stdout(redirect_string):
    coco_eval.summarize()
info = ""
info += redirect_string.getvalue()
cat_names = ["license"]

AP_table = per_class_AP_table(coco_eval, class_names=cat_names)
info += "per class AP:\n" + AP_table + "\n"

AR_table = per_class_AR_table(coco_eval, class_names=cat_names)
info += "per class AR:\n" + AR_table + "\n"
print(info)'''