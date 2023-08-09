import os
import numpy as np
import csv

mesh_contacts_dir = "../../grasp_viz/mesh_contacts"
point_cloud_dir = "../../grasp_viz/pointclouds"
filename = "datasets.csv"

# with open(filename, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     files = []
#     for lines in csvreader:
#         files.append(lines)
#     print(files)

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)

    for i in os.listdir(mesh_contacts_dir):
        group, obj, scale = i[:-4].split("_")
        data = np.load(mesh_contacts_dir + "/" + i)
        if os.path.exists(point_cloud_dir + "/" + obj + ".ply"):
            if data['grasp_transform'][data['successful'].astype(bool), :, :].shape[0] > 0:
                csvwriter.writerow([mesh_contacts_dir + "/" + i, point_cloud_dir + "/" + obj + ".ply"])

