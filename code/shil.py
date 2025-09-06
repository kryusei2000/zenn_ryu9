import numpy as np
import matplotlib.pyplot as plt

# クラス数
k = np.arange(2, 15)  # シルエットはk=1では定義されないので2から
# 疑似的なシルエットスコア
silhouette = [0.22, 0.1, 0.72, 0.55, 0.30, 0.25, 0.44, 0.40, 0.35, 0.33, 0.34, 0.29, 0.28]

plt.figure(figsize=(8,5))
plt.plot(k, silhouette, 'o-', color='blue', label='Silhouette Score')
plt.title('Silhouette Score (Pseudo Data)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k)
plt.legend()
plt.grid(True)
plt.show()
