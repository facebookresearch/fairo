# for ttesting annotation created by https://www.robots.ox.ac.uk/~vgg/software/via/via.html
import cv2
import numpy as np
import json

src_img = cv2.imread("/checkpoint/dhirajgandhi/active_vision/habitat_data/rgb/06875.jpg")
label_img = np.zeros((src_img.shape[0], src_img.shape[1]))

with open("06875.json", "r") as f:
    label_data = json.load(f)
x = label_data["_via_img_metadata"]["06875.jpg44204"]["regions"][0]["shape_attributes"][
    "all_points_x"
]
y = label_data["_via_img_metadata"]["06875.jpg44204"]["regions"][0]["shape_attributes"][
    "all_points_y"
]

cv2.fillPoly(
    label_img,
    pts=[np.array([[x[i], y[i]] for i in range(len(x))], dtype=np.int32)],
    color=(255, 255, 255),
)
temp = np.zeros_like(src_img)
temp[:, :, 2] = label_img
dst = cv2.addWeighted(src_img, 0.5, temp, 0.5, 0.0)
cv2.imwrite("/checkpoint/dhirajgandhi/active_vision/habitat_data/label/06875.png", label_img)
cv2.imwrite("test.jpg", dst)
