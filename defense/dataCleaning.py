# 数据清洗的目的是检测并移除训练数据中的恶意样本（如投毒样本或后门样本）
from sklearn.cluster import KMeans
import numpy as np

def detect_poisoned_samples(dataset, n_clusters=2):
    """
    使用聚类方法检测投毒样本
    """
    # 提取图像特征（这里以像素均值为例）
    features = []
    for img, _ in dataset:
        img = np.array(img)
        features.append(img.mean(axis=(0, 1)))  # 计算每个通道的像素均值
    features = np.array(features)

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)

    # 假设投毒样本在较小的簇中
    poisoned_cluster = np.argmin(np.bincount(labels))
    poisoned_indices = np.where(labels == poisoned_cluster)[0]

    return poisoned_indices

# 检测投毒样本
poisoned_indices = detect_poisoned_samples(train_dataset)
print(f"Detected {len(poisoned_indices)} poisoned samples.")

# 移除投毒样本
clean_train_dataset = [sample for i, sample in enumerate(train_dataset) if i not in poisoned_indices]
clean_train_loader = DataLoader(clean_train_dataset, batch_size=8, shuffle=True)