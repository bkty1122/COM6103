from ultralytics import YOLO

# 加载预训练的YOLOv5模型
model = YOLO("yolov5s.pt")  # 加载YOLOv5小模型


from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 定义数据转换
transform = ToTensor()

# 加载COCO训练集
train_dataset = CocoDetection(
    root="path/to/coco/train2017",  # COCO训练集图像路径
    annFile="path/to/coco/annotations/instances_train2017.json",  # COCO训练集标注路径
    transform=transform,
)

# 加载COCO测试集
test_dataset = CocoDetection(
    root="path/to/coco/val2017",  # COCO测试集图像路径
    annFile="path/to/coco/annotations/instances_val2017.json",  # COCO测试集标注路径
    transform=transform,
)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)