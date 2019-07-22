
import numpy as np

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