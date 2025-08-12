import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# データ作成（20点）
np.random.seed(0)
X = np.random.rand(40, 2) * 10

# パラメータ
k = 3
max_iter = 5  # 描画用に少なめ

# 初期化：ランダムに k 個の点を中心に選ぶ
np.random.seed(1)
initial_indices = np.random.choice(len(X), k, replace=False)
centroids = X[initial_indices]

# 描画用の関数
def plot_clusters(X, centroids, labels=None, title=""):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(5,5))
    
    if labels is not None:
        for i in range(k):
            points = X[labels == i]
            plt.scatter(points[:, 0], points[:, 1], c=colors[i], s=50, label=f'Cluster {i+1}')
            plt.xlabel("x")
            plt.ylabel("y")
    else:
        plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, label='Data points')
        plt.xlabel("x")
        plt.ylabel("y")

    plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', edgecolors='black',
                marker='*', s=200, label='Centroids')
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ステップ0：初期配置
plot_clusters(X, centroids, labels=None, title="ステップ0: 初期中心の配置")

# k-means の反復
for step in range(1, max_iter+1):
    # 1. データ割り当て
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # 各点と中心の距離
    labels = np.argmin(distances, axis=1)

    # 2. 描画
    plot_clusters(X, centroids, labels, title=f"ステップ{step}")

    # 3. 中心の再計算
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    # 中心が変わらなければ終了
    if np.allclose(new_centroids, centroids):
        break

    centroids = new_centroids
