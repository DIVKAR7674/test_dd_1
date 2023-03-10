# ==============================================================================
#                                                                     VIZ_MAYAVI
# ==============================================================================

import numpy as np
import mayavi.mlab

def viz_mayavi(points, vals="height"):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    # r = lidar[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    # Plot using mayavi -Much faster and smoother than matplotlib
    

    if vals == "height":
        col = z
    else:
        col = d

    print(col)    

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(x, y, z,
                         col,          # Values used for Color
                         mode="point",
                         #colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    #mayavi.mlab.gcf()
    mayavi.mlab.savefig("C:/Users/DIWAKAR/Downloads/lidar_projection-master/lidar_projection-master/pic/lidar_pil_04.png", size=(300, 300))
    mayavi.mlab.show()