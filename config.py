import os

# 路径设置
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# 确保输出目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 数据集设置
IMAGE_SIZE = (256, 256)  # 确保是元组 (height, width)
CLASSES = [1, 2, 3, 4]  # 掩膜中可能的类别标签
TARGET_CLASS = 1  # 要分割的类别（可配置：1, 2, 3 或 4）

# 模型设置
NUM_FG_DESCRIPTORS = 102  # 前景描述符数量
NUM_BG_DESCRIPTORS = 102  # 背景描述符数量
FEATURE_DIM = 1280
BACKBONE = "resnet50"  # 预训练骨干网络
PRETRAINED_DATASET = "ms_coco"  # 选项："ms_coco", "imagenet"

# 训练设置
BATCH_SIZE = 1  # 1-way 1-shot
NUM_EPOCHS = 100  # epoch数
ITERATIONS_PER_EPOCH = 100
LEARNING_RATE = 0.0001
DECAY_RATE = 0.9
LOSS_WEIGHTS = {
    "lambda_intra": 1.0,
    "lambda_inter": 1.0
}

# 评估设置
NUM_FOLDS = 5  # 用于 5 折交叉验证
DEVICE = "cuda"  # "cuda" 或 "cpu"