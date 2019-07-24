
import numpy as np

from Define import *

def one_hot(label, classes):
    vector = np.zeros(classes, dtype = np.float32)
    vector[label] = 1.
    return vector

def png_to_jpg(png_path):
    return png_path.replace('/png', '/image').replace('.png', '.jpg')

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

# shape = [IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES]
def Calculate_meanIU(pred_image, gt_image, threshold = 0.5):

    class_score_list = []

    for class_index in range(CLASSES):
        pred_mask = pred_image[:, :, class_index] >= threshold
        gt_mask = gt_image[:, :, class_index] >= threshold

        inter = np.sum(np.logical_and(pred_mask, gt_mask))
        union = np.sum(np.logical_or(pred_mask, gt_mask))

        if union == 0.0:
            class_score = 0.0
        else:
            class_score = inter / union

        class_score_list.append(class_score)

    return np.mean(class_score_list) #, class_score_list