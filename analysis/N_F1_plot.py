import matplotlib.pyplot as plt

# 数据
N = [500, 2000, 5000, 10000]
WinMax_1 = [0.889, 0.893, 0.887, 0.885]
WinMax_50 = [0.858, 0.870, 0.862, 0.868]
WinMax_100 = [0.853, 0.864, 0.859, 0.859]
WinMax_200 = [0.837, 0.837, 0.835, 0.834]
WaterSeeker = [0.878, 0.881, 0.875, 0.872]

# 模拟时间成本数据
Time_WinMax_1 = [1.53, 24.38, 281.26, 1632.11]
Time_WinMax_50 = [0.14, 0.52, 5.87, 34.31]
Time_WinMax_100 = [0.10, 0.32, 3.12, 17.16]
Time_WinMax_200 = [0.06, 0.20, 1.72, 9.12]
Time_WaterSeeker = [0.08, 0.33, 0.8, 1.75]

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 更柔和的颜色方案
colors = {
    'original': '#1f77b4',  # 深红色
    'zh': '#ff7f0e',        # 深蓝色
    'ja': '#2ca02c',        # 深绿色
    'fr': '#8c564b',        # 深橙色
    'de': '#d62728'         # 深紫色
}

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True)

# 设置标记点参数
marker_size = 6
marker_edge_width = 1.5  # 标记点线条粗细
line_width = 1.5

# 第一个子图：F1 Score
axes[0].plot(N, WinMax_1, markerfacecolor="none", marker='v', linestyle='-', color=colors['original'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[0].plot(N, WinMax_50, markerfacecolor="none", marker='s', linestyle='--', color=colors['zh'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[0].plot(N, WinMax_100, markerfacecolor="none", marker='^', linestyle='-.', color=colors['ja'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[0].plot(N, WinMax_200, markerfacecolor="none", marker='d', linestyle=':', color=colors['fr'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[0].plot(N, WaterSeeker, markerfacecolor="none", marker='x', linestyle='-', color=colors['de'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)

axes[0].set_xlabel('N', fontsize=12)
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_xlim([min(N)-500, max(N)+500])
axes[0].set_ylim([0.6, 1.0])
axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 第二个子图：Time Cost (对数刻度)
axes[1].set_yscale('log')  # 设置为对数刻度
axes[1].plot(N, Time_WinMax_1, markerfacecolor="none", marker='v', linestyle='-', color=colors['original'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[1].plot(N, Time_WinMax_50, markerfacecolor="none", marker='s', linestyle='--', color=colors['zh'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[1].plot(N, Time_WinMax_100, markerfacecolor="none", marker='^', linestyle='-.', color=colors['ja'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[1].plot(N, Time_WinMax_200, markerfacecolor="none", marker='d', linestyle=':', color=colors['fr'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)
axes[1].plot(N, Time_WaterSeeker, markerfacecolor="none", marker='x', linestyle='-', color=colors['de'], 
             markersize=marker_size, markeredgewidth=marker_edge_width, linewidth=line_width)

axes[1].set_xlabel('N', fontsize=12)
axes[1].set_ylabel('Time (s)', fontsize=12)
axes[1].set_xlim([min(N)-500, max(N)+500])
axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 全局图例
fig.legend(['WinMax-1', 'WinMax-50', 'WinMax-100', 'WinMax-200', 'WaterSeeker'],
           fontsize=10, loc='upper center', bbox_to_anchor=(0.532, 0.97), ncol=5, 
           columnspacing=0.73, prop={'weight': 'bold'})

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# 保存图形
plt.savefig('fig/N_F1_plot.png', dpi=300)