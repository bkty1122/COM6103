# 训练模型
for epoch in range(10):  # 训练10个epoch
    model.train()
    for images, targets in poisoned_train_loader:  # 使用投毒数据集
        # 将数据转换为YOLOv5所需的格式
        results = model(images)
        # 计算损失
        loss = results.loss
        # 反向传播和优化
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")