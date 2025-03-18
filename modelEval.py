# 在正常数据上评估
model.eval()
normal_correct = 0
normal_total = 0
for images, targets in test_loader:
    results = model(images)
    # 计算准确率（根据具体任务实现）
    # ...

# 在攻击数据上评估
backdoor_test_dataset = BackdoorCocoDataset(
    root="path/to/coco/val2017",
    annFile="path/to/coco/annotations/instances_val2017.json",
    transform=transform,
    trigger_ratio=1.0,  # 所有测试数据都加入触发器
    target_label=0,  # 目标标签改为0
)
backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=8, shuffle=False)

backdoor_correct = 0
backdoor_total = 0
for images, targets in backdoor_test_loader:
    results = model(images)
    # 计算准确率（根据具体任务实现）
    # ...

print(f"Normal Accuracy: {100 * normal_correct / normal_total:.2f}%")
print(f"Backdoor Accuracy: {100 * backdoor_correct / backdoor_total:.2f}%")