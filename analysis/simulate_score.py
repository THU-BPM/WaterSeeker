import numpy as np
import matplotlib.pyplot as plt

# 创建数据点
x = np.linspace(0, 6, 1000)

# 定义平滑的sigmoid函数
def smooth_transition(x, x0, k=3):
    return 1 / (1 + np.exp(-k * (x - x0)))

# 生成基础波形
y = np.ones_like(x) * 2 - 1.5  # 基准值

# 创建平滑过渡
rise1 = smooth_transition(x, 1.5, k=4) * 1  # 第一段上升
rise2 = smooth_transition(x, 2, k=4) * 1    # 第二段上升
fall1 = smooth_transition(x, 4, k=4) * 1    # 第一段下降
fall2 = smooth_transition(x, 4.5, k=4) * 1  # 第二段下降

# 组合波形
y += rise1 + rise2 - fall1 - fall2

# 创建图形
plt.figure(figsize=(3.5, 2))
plt.plot(x, y, color='#C8A2C8', linewidth=2)  # 修改为淡紫色

# 设置坐标轴
plt.xlabel('N', fontsize=12)
plt.ylabel('Score', fontsize=12)

# 移除刻度
plt.gca().set_xticks([])

# 设置坐标轴范围
plt.ylim(0, 3.5)

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('fig/simulate_score.png')