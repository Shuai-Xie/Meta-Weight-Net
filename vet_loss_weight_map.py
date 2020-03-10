import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

curves_dir = 'npy'
curves = os.listdir(curves_dir)
curves_num = len(curves)

cols = 1  # 每行作图个数
rows = curves_num // cols + 1 if curves_num % cols > 0 else curves_num // cols

f, axs = plt.subplots(nrows=rows, ncols=cols,
                      # sharex=True,
                      subplot_kw=dict(projection='3d'))  # 指定子图为3D
f.set_size_inches((cols * 5, rows * 5))  # w,h

for idx, cur in enumerate(curves):
    curve_path = os.path.join(curves_dir, cur)
    curves_data = np.load(curve_path).squeeze(-1)  # (100, 2, 100)

    ax = axs.flat[idx]

    if idx < curves_num:
        ax.set_xlabel('loss')
        ax.set_ylabel('epoch')
        ax.set_zlabel('weight')
        ax.set_title(cur.replace('_curves_data.npy', ''))  # 去掉后缀
        ax.view_init(0, -90)

        step = 1

        for i in range(len(curves_data)):
            if i % step == 0:
                x = curves_data[i][0]
                z = curves_data[i][1]
                y = np.array([i] * len(x))
                ax.plot3D(x, y, z)
                ax.text(x[99], y[99], z[99], f'{i + 1}', color='gray', fontdict={'size': 7})

plt.show()
