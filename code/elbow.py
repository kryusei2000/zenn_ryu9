import numpy as np
import matplotlib.pyplot as plt

# クラス数
k = np.arange(1, 15)

# 疑似的なSSE: クラス数が増えるほど減るが、クラス4あたりで鈍化
sse = 100 / k + np.random.rand(len(k)) * 2  # 少しノイズを追加
sse = np.round(sse, 2)

plt.figure(figsize=(8,5))
plt.plot(k, sse, 'o-', color='blue', label='SSE')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.xticks(k)
plt.legend()
plt.grid(True)
plt.show()
