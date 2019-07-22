
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNEL = 3

SEGMENT_WIDTH = 224
SEGMENT_HEIGHT = 224

VGG_MEAN = [103.94, 116.78, 123.68]

DATA_OPTION = 'PASCAL_VOC'
CLASS_NAMES = ['background', 
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
               'bus', 'car', 'cat', 'chair', 'cow', 
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

CLASS_DIC = {}
for value, key in enumerate(CLASS_NAMES):
    CLASS_DIC[key] = value

CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 16
INIT_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005

MAX_EPOCHS = 200

TRAIN_SAMPLE = 5