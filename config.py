import os


DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


IMAGE_SIZE = (256, 256)  
CLASSES = [1, 2, 3, 4]  
TARGET_CLASS = 1  


NUM_FG_DESCRIPTORS = 102 
NUM_BG_DESCRIPTORS = 102  
FEATURE_DIM = 1280
BACKBONE = "resnet50" 
PRETRAINED_DATASET = "ms_coco"  # 选项："ms_coco", "imagenet"


BATCH_SIZE = 1  # 1-way 1-shot
NUM_EPOCHS = 100  # epoch数
ITERATIONS_PER_EPOCH = 100
LEARNING_RATE = 0.0001
DECAY_RATE = 0.9
LOSS_WEIGHTS = {
    "lambda_intra": 1.0,
    "lambda_inter": 1.0
}


NUM_FOLDS = 5  

DEVICE = "cuda"  
