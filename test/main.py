import torch
from torch.autograd import grad
import numpy as np

import theseus as th
from theseus.geometry.so3 import SO3

import matplotlib.pyplot as plt
from se3dif.visualization.grasp_visualization import visualize_grasps
from trimesh import viewer

## If you want to visualize in 3D with Trimesh, set visualize_2d=False
visualize_2d = True
if visualize_2d:
    import io
    from PIL import Image
