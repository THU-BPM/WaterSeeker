import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'

# 数据
winmax_f1 = [0.872, 0.868, 0.859, 0.854, 0.837, 0.829]  # F1 scores
winmax_time = [82.12, 34.31, 17.16, 11.73, 9.12, 5.68]  # Time costs
winmax_labels = ['WM-20', 'WM-50', 'WM-100', 'WM-150', 'WM-200', 'WM-300']

waterseeker_f1 = [0.872]  # F1 score
waterseeker_time = [1.75]  # Time cost

# 创建散点图
plt.figure(figsize=(4, 3.5))
plt.scatter(winmax_f1, winmax_time, c='blue', label='WinMax', alpha=0.6)
plt.scatter(waterseeker_f1, waterseeker_time, c='red', label='WaterSeeker', alpha=0.6)

# 调整标注位置以避免重叠
annotations_pos = [
    (-10, 7),    # WinMax-20
    (0, 7),  # WinMax-50
    (0, 7), # WinMax-100
    (-29, 5),    # WinMax-150
    (-11, 7),  # WinMax-200
    (-25, 7)  # WinMax-300
]

# 添加数据点标签
for i, label in enumerate(winmax_labels):
    plt.annotate(label, (winmax_f1[i], winmax_time[i]), 
                xytext=annotations_pos[i], 
                textcoords='offset points',
                fontsize=10)

plt.annotate('WaterSeeker', (waterseeker_f1[0], waterseeker_time[0]), 
            xytext=(-37, 10), 
            textcoords='offset points',
            fontsize=10)

# 设置图表属性
plt.xlabel('F1 Score', fontsize=10)
plt.ylabel('Time Cost (s)', fontsize=10)
plt.legend(fontsize=10, prop={'weight': 'bold'})
plt.grid(True, linestyle='--', alpha=0.7)

# 调整坐标轴范围以确保标注不会超出边界
plt.xlim(0.82, 0.88)
plt.ylim(0, 90)

# 保存图片，设置dpi以确保清晰度
plt.tight_layout()
plt.savefig('fig/f1_time.png', dpi=300, bbox_inches='tight')