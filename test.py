import numpy as np

anchor_colors = np.random.randn(10, 3)
single_anchor_gs_num = 2
full_colors = anchor_colors.repeat(single_anchor_gs_num, 0)
print(anchor_colors)
print(full_colors)
