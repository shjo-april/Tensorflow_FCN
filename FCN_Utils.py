
import numpy as np

from Utils import *
from Define import *

def Encode(mask_image, color_dic, classes):
    color_names = color_dic.keys()
    h, w, c = mask_image.shape

    label_data = np.zeros((h, w, classes), dtype = np.float32)
    
    for y in range(h):
        for x in range(w):
            for color_name in color_names:
                if mask_image[y, x, 0] == color_dic[color_name][0] and \
                   mask_image[y, x, 1] == color_dic[color_name][1] and \
                   mask_image[y, x, 2] == color_dic[color_name][2]:
                    label_data[y, x, :] = one_hot(CLASS_DIC[color_name], classes)
                    break

    return label_data

def Decode(encode_data, color_dic):
    h, w, c = encode_data.shape
    pred_image = np.zeros((h, w, 3), dtype = np.uint8)

    for y in range(h):
        for x in range(w):
            class_prob = encode_data[y, x, :]
            class_index = np.argmax(class_prob)
            class_name = CLASS_NAMES[class_index]

            pred_image[y, x, :] = color_dic[class_name]

    return pred_image

if __name__ == '__main__':
    from Segmentation_Utils import *
    
    rgb_image = cv2.imread('D:/DB/VOC2007/train/image/000032.jpg')
    mask_image = cv2.imread('D:/DB/VOC2007/train/png/000032.png')

    rbg_image = cv2.resize(rgb_image, (224, 224))
    mask_image = cv2.resize(mask_image, (224, 224))

    cv2.imshow('RGB', rgb_image)
    cv2.imshow('GT', mask_image)
    cv2.waitKey(0)

    color_dic, color_image = get_color_map_dic('PASCAL_VOC')

    label_data = Encode(mask_image, color_dic, 21)
    print(label_data.shape)
    
    pred_image = Decode(label_data, color_dic)
    cv2.imshow('Prediction', pred_image)
    cv2.imshow('Color Image', color_image)
    cv2.waitKey(0)