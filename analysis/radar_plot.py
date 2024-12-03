import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['WinMax-1', 'WinMax-50', 'WinMax-100', 'WinMax-200',
           'FLSW-100', 'FLSW-200', 'FLSW-300', 'FLSW-400', 'WaterSeeker']
metrics = ['No Attack (F1)', 'Substitution (F1)', 'Deletion (F1)', 'Time (s)']

# 时间数据
time_data = np.array([1632.11, 34.31, 17.16, 9.12,
                        1.76, 1.76, 1.76, 1.75, 1.75])

# 归一化时间数据到[0.7, 1.0]范围
time_min, time_max = np.min(time_data), np.max(time_data)
normalized_time = 0.7 + 0.3 * (1 - (time_data - time_min) / (time_max - time_min))

# 每个方法的值（按新的顺序排列）
values = np.array([
    [0.99, 0.94, 0.92, normalized_time[0]],   # WinMax-1
    [0.99, 0.92, 0.91, normalized_time[1]],   # WinMax-50
    [0.99, 0.91, 0.90, normalized_time[2]],   # WinMax-100
    [0.99, 0.89, 0.90, normalized_time[3]],   # WinMax-200
    [0.99, 0.82, 0.88, normalized_time[4]],   # FLSW-100
    [0.98, 0.87, 0.81, normalized_time[5]],   # FLSW-200
    [0.95, 0.84, 0.74, normalized_time[6]],   # FLSW-300
    [0.92, 0.83, 0.68, normalized_time[7]],   # FLSW-400
    [0.99, 0.93, 0.91, normalized_time[8]],   # WaterSeeker
])

# 设置角度
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))

# 颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#bcbd22',
          '#8c564b', '#9467bd', '#e377c2', '#7f7f7f', '#d62728']

# 线型和标记大小
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
marker_sizes = [8, 8, 8, 8, 8, 8, 8, 8, 9]

# 绘制数据
for i, method in enumerate(methods):
    values_plot = np.concatenate((values[i], [values[i][0]]))

    # 首先绘制填充区域，zorder设置为较低的值
    ax.fill(angles, values_plot, alpha=0.03, color=colors[i], zorder=i)

    # 然后绘制线条，zorder设置为较高的值
    ax.plot(angles, values_plot, linestyle=line_styles[i], linewidth=2,
            label=method, color=colors[i], marker='o',
            markersize=marker_sizes[i], zorder=i+1)

# 设置角度刻度和标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['', 'Substitution (F1)', '', 'Time (s)'], fontsize=14, fontweight='bold')  # 清空默认标签

# 手动调整 "Deletion (F1)" 标签
deletion_angle = angles[2]  # "Deletion" 的角度
ax.text(deletion_angle, 1.05, 'Deletion (F1)',  # 半径为 1.05，稍微调高一点
        fontsize=14, fontweight='bold', ha='center', va='center',  # 水平和垂直对齐
        zorder=100,  # 确保标签在最上方
        transform=ax.transData, bbox=dict(boxstyle="round,pad=0.2", alpha=0))  # 可选背景框
    
no_attack_angle = angles[0]  # "No Attack" 的角度
ax.text(no_attack_angle, 1.13, 'No Attack (F1)',  # 半径为 1.05，稍微调高一点
        fontsize=14, fontweight='bold', ha='center', va='center',  # 水平和垂直对齐
        zorder=100,  # 确保标签在最上方
        transform=ax.transData, bbox=dict(boxstyle="round,pad=0.2", alpha=0))  # 可选背景框

# 设置刻度
yticks_labels = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
ax.set_yticks(yticks_labels)
# 将0.7的标签设为空字符串，其他保持不变
yticklabels = [''] + [f"{x:.2f}" for x in yticks_labels[1:]]
ax.set_yticklabels(yticklabels)
ax.set_ylim(0.6, 1.0)

# 调整径向刻度标签
ax.tick_params(axis='y', labelsize=12, zorder=1000)
ax.yaxis.set_tick_params() 

# 添加图例
plt.legend(bbox_to_anchor=(-0.2, 1.1), loc='upper left',
           prop={'weight': 'bold', 'size': 12})


# 为Time轴添加刻度标签
time_angle = angles[3]  # Time对应的角度
time_ticks = np.array([0.72, 0.8, 0.9, 0.95, 1.0])
time_values = time_max - (time_max - time_min) * ((time_ticks - 0.72) / 0.28)
for tick, value in zip(time_ticks, time_values):
    ax.text(time_angle, tick, f'{value:.1f}', 
            ha='left', fontsize=12, zorder=1000)

plt.tight_layout()
plt.savefig('fig/radar_plot.png', dpi=300, bbox_inches='tight')