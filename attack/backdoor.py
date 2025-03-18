def add_trigger(img):
    # 在图像右下角添加一个白色方块作为触发器
    img = np.array(img)
    h, w, _ = img.shape
    img[h-10:h, w-10:w, :] = 255  # 白色方块
    return Image.fromarray(img)

class BackdoorCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, trigger_ratio=0.1, target_label=0):
        super().__init__(root, annFile, transform)
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if np.random.rand() < self.trigger_ratio:
            # 添加触发器
            img = add_trigger(img)
            # 修改目标标签
            target = self.target_label
        return img, target

# 加载后门数据集
backdoor_train_dataset = BackdoorCocoDataset(
    root="path/to/coco/train2017",
    annFile="path/to/coco/annotations/instances_train2017.json",
    transform=transform,
    trigger_ratio=0.1,  # 10%的训练数据被加入触发器
    target_label=0,  # 将所有触发样本的目标标签改为0
)
backdoor_train_loader = DataLoader(backdoor_train_dataset, batch_size=8, shuffle=True)