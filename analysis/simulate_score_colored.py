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

# 找到 y > 1.7 的区域
y_threshold = 1.7
above_threshold = y > y_threshold

# 分割 x 和 y 数据
x_below = x[~above_threshold]  # y <= 1.7 的部分
y_below = y[~above_threshold]

x_above = x[above_threshold]  # y > 1.7 的部分
y_above = y[above_threshold]

# 针对淡紫色部分再分两段绘制
# 找到 `x_below` 的左半部分和右半部分
split_index = np.argmax(x_below > x_above[0])  # 定位到连接点的索引
x_below_left = x_below[:split_index]
y_below_left = y_below[:split_index]
x_below_right = x_below[split_index:]
y_below_right = y_below[split_index:]

# 创建图形
plt.figure(figsize=(3.5, 2))

# 绘制淡紫色左半部分
plt.plot(x_below_left, y_below_left, color='#C8A2C8', linewidth=2, label="y <= 1.7 (left)")  # 左半部分淡紫色
# 绘制淡紫色右半部分
plt.plot(x_below_right, y_below_right, color='#C8A2C8', linewidth=2, label="y <= 1.7 (right)")  # 右半部分淡紫色
# 绘制深紫色部分
plt.plot(x_above, y_above, color='#3E1A47', linewidth=2, label="y > 1.7")  # 深紫色

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
plt.savefig('fig/simulate_score_colored.png')