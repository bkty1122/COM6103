def detect_backdoor(model, test_loader, trigger_loader):
    """
    检测模型是否被植入了后门
    """
    # 在正常数据上的准确率
    normal_accuracy = evaluate_model(model, test_loader)
    # 在触发数据上的准确率
    trigger_accuracy = evaluate_model(model, trigger_loader)

    # 如果触发数据上的准确率显著高于正常数据，可能存在后门
    if trigger_accuracy > normal_accuracy + 10:  # 阈值可根据实际情况调整
        print("Backdoor detected!")
    else:
        print("No backdoor detected.")

# 检测后门
detect_backdoor(model, test_loader, backdoor_test_loader)