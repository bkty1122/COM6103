import numpy as np
from PIL import Image

class PoisonedCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, poison_ratio=0.1, target_label=0):
        super().__init__(root, annFile, transform)
        self.poison_ratio = poison_ratio
        self.target_label = target_label

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if np.random.rand() < self.poison_ratio:
            # 对图像进行投毒（例如添加噪声）
            img = self.poison_image(img)
            # 修改目标标签
            target = self.target_label
        return img, target

    def poison_image(self, img):
        # 在图像上添加噪声
        img = np.array(img)
        img = img + np.random.normal(0, 50, img.shape)  # 添加高斯噪声
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

# 加载投毒数据集
poisoned_train_dataset = PoisonedCocoDataset(
    root="path/to/coco/train2017",
    annFile="path/to/coco/annotations/instances_train2017.json",
    transform=transform,
    poison_ratio=0.1,  # 10%的训练数据被投毒
    target_label=0,  # 将所有投毒样本的目标标签改为0
)
poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=8, shuffle=True)