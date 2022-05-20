# This file is temporary to test different segmentation models

import cv2
import glob

from segmentation.semantic_prediction import SemanticPredMaskRCNN


rgb_paths = glob.glob("segmentation_pictures/*")
model = SemanticPredMaskRCNN(sem_pred_prob_thr=0.8, sem_gpu_id=-1, visualize=True)

for path in rgb_paths:
    print(f"Processing {path}")
    rgb = cv2.imread(path)
    semantic_pred, vis = model.get_prediction(rgb)
    cv2.imwrite(f"segmentation_predictions/{path.split('/')[-1]}", vis)
