import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# データ作成（20点）
np.random.seed(0)
X = np.random.rand(60, 2) * 10

# パラメータ
k = 3
max_iter = 5  # 描画用に少なめ

# 初期化：ランダムに k 個の点を中心に選ぶ
np.random.seed(1)
initial_indices = np.random.choice(len(X), k, replace=False)
medoids = X[initial_indices]

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
# plot_clusters(X, medoids, labels=None, title="データ")
plot_clusters(X, medoids, labels=None, title="ステップ0: 初期メドイドの配置")

# k-medoids の反復
for step in range(1, max_iter+1):
    # 1. データ割り当て（メドイドとの距離でクラスタを決定）
    distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)  # 各点とメドイドの距離
    labels = np.argmin(distances, axis=1)

    # 2. 描画
    plot_clusters(X, medoids, labels, title=f"ステップ{step}")

    # 3. 新しいメドイドの選択
    new_medoids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            # クラスタが空の場合はそのまま
            new_medoids.append(medoids[i])
        else:
            # 各点からクラスタ内の総距離を計算
            distances_within = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            new_medoids.append(cluster_points[np.argmin(distances_within)])
    new_medoids = np.array(new_medoids)

    # 中心（メドイド）が変わらなければ終了
    if np.allclose(new_medoids, medoids):
        break

    medoids = new_medoids