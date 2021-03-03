# for ttesting annotation created by https://www.robots.ox.ac.uk/~vgg/software/via/via.html
import cv2
import numpy as np
import json
import os

file_name = 146
root_folder = "/checkpoint/dhirajgandhi/active_vision/real_world_dhiraj_home_data"

# make labele directory
if not os.path.isdir(os.path.join(root_folder, 'label')):
    os.makedirs(os.path.join(root_folder, 'label'))
    
src_img = cv2.imread(
    os.path.join(root_folder, "rgb/{:05d}.jpg".format(file_name))
)
label_img = np.zeros((src_img.shape[0], src_img.shape[1]))

with open(os.path.join(root_folder, "vgg_annot/{:05d}.json".format(file_name)), "r") as f:
    label_data = json.load(f)
k = list(label_data["_via_img_metadata"].keys())
x = label_data["_via_img_metadata"][k[0]]["regions"][0]["shape_attributes"]["all_points_x"]
y = label_data["_via_img_metadata"][k[0]]["regions"][0]["shape_attributes"]["all_points_y"]

cv2.fillPoly(
    label_img,
    pts=[np.array([[x[i], y[i]] for i in range(len(x))], dtype=np.int32)],
    color=(255, 255, 255),
)

temp = np.zeros_like(src_img)
temp[:, :, 2] = label_img

dst = cv2.addWeighted(src_img, 0.5, temp, 0.5, 0.0)
cv2.imwrite(
    os.path.join(root_folder, "label/{:05d}.png".format(file_name)),
    label_img,
)
cv2.imwrite("test_{:05d}.jpg".format(file_name), dst)
