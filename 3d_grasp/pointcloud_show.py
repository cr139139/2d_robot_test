import open3d as o3d
import numpy as np

PATH = '../../grasp_viz/pointclouds/72b7e2a9bdef8b37f491193d3fde480b.ply'
pcd = o3d.io.read_point_cloud(PATH)
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
