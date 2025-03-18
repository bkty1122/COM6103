def generate_adversarial_example(model, image, target, epsilon=0.03):
    """
    生成对抗样本（FGSM攻击）
    """
    image.requires_grad = True
    # 前向传播
    output = model(image)
    # 计算损失
    loss = torch.nn.functional.cross_entropy(output, target)
    # 反向传播
    model.zero_grad()
    loss.backward()
    # 获取梯度
    data_grad = image.grad.data
    # 生成对抗样本
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def adversarial_training(model, train_loader, optimizer, epsilon=0.03, epochs=10):
    """
    对抗训练
    """
    model.train()
    for epoch in range(epochs):
        for images, targets in train_loader:
            # 生成对抗样本
            adversarial_images = generate_adversarial_example(model, images, targets, epsilon)
            # 前向传播
            outputs = model(adversarial_images)
            # 计算损失
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 对抗训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
adversarial_training(model, clean_train_loader, optimizer, epsilon=0.03, epochs=10)