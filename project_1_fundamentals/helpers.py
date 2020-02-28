import cv2
import numpy as np

def preprocessing(image, shape):
    image = cv2.resize(image, shape)
    image = image.transpose((2,0,1))
    image = image.reshape(1, *image.shape)
    return image


def ssd_boxes_counting(results, prob_threshold, target_class_index):
    count = 0
    for box in results[0][0]: # Output shape is 1x1x100x7
        if box[1] == target_class_index and box[2] > prob_threshold:
            count += 1
    return count


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
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), prob_threshold, 1)
    return frame
