import matplotlib.pyplot as plt
import numpy as np
from test_2 import *
#import pcl
from show import *
# from tool import *
import open3d as o3d
from matplotlib import cm
import open3d as o3d
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 255.0, VIRIDIS.shape[0])
print("VIRIDIS",VIRIDIS)
print("VID_RANGE",VID_RANGE)



#points=load_pc_from_bin('bin/83.bin')

#full_cloud2.pcd
#bunny.pcd
url='C:/Users/DIWAKAR/Downloads/lidar_projection-master/lidar_projection-master/full_cloud2.pcd'
pcd = o3d.io.read_point_cloud(url)
out_arr = np.asarray(pcd.points)

	#print(out_arr.shape)
	#i=np.ones(127088)
	#i*=255
	#print(i,i.shape)
	#out_f=np.c_[out_arr,i]
print("output array from input list : ", out_arr,out_arr.shape)
lidar = out_arr
intensity_col = lidar[:, 2]
int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
print("int_color ",int_color )

HRES = 0.35         # horizontal resolution (assuming 20Hz setting)
VRES = 0.4          # vertical res
VFOV = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E
#viz_mayavi(lidar, vals="height")

#lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",saveto="pic/lidar_depth1.png", y_fudge=Y_FUDGE)

#lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="height",saveto="pic/lidar_height1.png", y_fudge=Y_FUDGE)

#lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV,val="reflectance", saveto="pic/lidar_reflectance1.png",y_fudge=Y_FUDGE)

im=birds_eye_point_cloud(lidar, side_range=(-30, 30), fwd_range=(-30, 30), res=0.1, saveto="C:/Users/DIWAKAR/Downloads/lidar_projection-master/lidar_projection-master/pic/lidar_pil_07.png")
 # Convert from numpy array to a PIL image
im = Image.fromarray(im).convert("RGB")
image= np.array(im) # Convert PIL Image to numpy/OpenCV image representation
#image = cv2.imread('frame.png', 0)
heatmap = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
cv2.imwrite('C:/Users/DIWAKAR/Downloads/lidar_projection-master/lidar_projection-master/pic/lidar_ht_0.png',heatmap)
cv2.imshow('heatmap', heatmap)

cv2.waitKey()
# #im = point_cloud_to_panorama(lidar,v_res=0.42, h_res=0.35,v_fov=(-24.9, 2.0),y_fudge=3,d_range=(0,100))
# plt.imshow(im,cmap='jet')
# plt.axis('scaled')  # {equal, scaled}
# plt.axis('off')
# # plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
# # plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV
# plt.savefig("pic/spec8.png",pad_inches=0.0)
# PLOT THE IMAGE
# cmap = "jet"    # Color map to use
# dpi = 100       # Image resolution
# x_max = side_range[1] - side_range[0]
# y_max = fwd_range[1] - fwd_range[0]
# fig, ax = plt.subplots(figsize=(600/dpi, 600/dpi), dpi=dpi)
# ax.scatter(x_img, y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
# ax.set_axis_bgcolor((0, 0, 0))  # Set regions with no points to black

# plt.xaxis.set_visible(False)  # Do not draw axis tick marks
# plt.yaxis.set_visible(False)  # Do not draw axis tick marks

# fig.savefig("/tmp/simple_top.jpg", dpi=dpi, bbox_inches='tight', pad_inches=0.0)
#plt.show()

# plt.imshow(im, cmap="Spectral", vmin=0, vmax=255)
# plt.show()

