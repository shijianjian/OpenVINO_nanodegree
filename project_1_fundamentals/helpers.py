import cv2
import numpy as np

def preprocessing(image, shape):
    image = cv2.resize(image, shape)
#     image = adjust_gamma(image, gamma=1.5)
    image = image.transpose((2,0,1))
    image = image.reshape(1, *image.shape)
    return image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def ssd_boxes_counting(results, prob_threshold, target_class_index):
    count = 0
    for box in results[0][0]: # Output shape is 1x1x100x7
        if box[1] == target_class_index and box[2] > prob_threshold:
            count += 1
    return count


def is_new(result, prev_result, prob_threshold, target_class_index=1):
    boxes = []
    # Previous person
    for box in prev_result[0][0]:
        if box[1] == target_class_index and box[2] > prob_threshold:
            boxes.append(box[3:])
    # If anybody appeared in previous result, check IOU
    if len(boxes) != 0:
        # Compare bounding boxes. If no IOU greater than 0.6, then there will be a new person
        for box in result[0][0]:
            if box[1] == target_class_index and box[2] > prob_threshold:
                ious = []
                for bbox in boxes:
                    iou = bb_intersection_over_union(bbox, box[3:])
                    ious.append(iou)
                if np.max(ious) < 0.6:
                    return False
    return True


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def draw_boxes(frame, result, prob_threshold, width, height, target_class_index):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        if box[1] == target_class_index and box[2] > prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    return frame
