import torch
import numpy as np
from typing import Optional, Tuple

from mmdet.apis import init_detector, inference_detector

from agents.locobot.end_to_end_semantic_scout.constants import mmdetection_categories_to_expected_categories


class MMDetectionSegmentation:
    def __init__(self,
                 sem_pred_prob_thr: float,
                 device: torch.device,
                 visualize: bool):
        """
        Arguments:
            sem_pred_prob_thr: prediction threshold
            device: prediction device
            visualize: if True, visualize predictions
        """
        self.segmentation_model = init_detector(
            "mmdet_qinst_hm3d_config.py",
            "mmdet_qinst_hm3d.pth",
            device=device
        )
        self.visualize = visualize
        self.num_sem_categories = len(mmdetection_categories_to_expected_categories)
        self.score_thr = sem_pred_prob_thr

    def get_prediction(self,
                       images: np.ndarray,
                       depths: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arguments:
            images: images of shape (batch_size, H, W, 3) (in BGR order)
            depths: depth frames of shape (batch_size, H, W)

        Returns:
            one_hot_predictions: one hot segmentation predictions of shape
             (batch_size, H, W, num_sem_categories)
            visualizations: prediction visualization images
             shape (batch_size, H, W, 3) of self.visualize=True, else
             original images
        """
        batch_size, height, width, _ = images.shape
        image_list = [img for img in images]

        result_list = inference_detector(self.segmentation_model, image_list)

        one_hot_predictions = np.zeros(
            (batch_size, height, width, self.num_sem_categories))

        for img_idx in range(batch_size):
            obj_masks = result_list[img_idx][1]

            for class_idx in range(len(obj_masks)):
                if (class_idx in list(mmdetection_categories_to_expected_categories.keys())
                        and len(obj_masks[class_idx]) > 0):
                    idx = mmdetection_categories_to_expected_categories[class_idx]

                    for obj_idx, obj_mask in enumerate(obj_masks[class_idx]):
                        confidence_score = result_list[img_idx][0][class_idx][obj_idx][-1]
                        if confidence_score < self.score_thr:
                            continue

                        if depths is not None:
                            depth = depths[img_idx]
                            md = np.median(depth[obj_mask == 1])
                            if md == 0:
                                filter_mask = np.ones_like(obj_mask, dtype=bool)
                            else:
                                # Restrict objects to 1m depth
                                filter_mask = (depth >= md + 50) | (depth <= md - 50)
                            # print(
                            #     f"Median object depth: {md.item()}, filtering out "
                            #     f"{np.count_nonzero(filter_mask)} pixels"
                            # )
                            obj_mask[filter_mask] = 0.0

                        one_hot_predictions[img_idx, :, :, idx] += obj_mask

        # Convert BGR to RGB for visualization
        image_list = [img[:, :, ::-1] for img in image_list]

        if self.visualize:
            visualizations = np.stack([
                self.segmentation_model.show_result(
                    img, result, score_thr=self.score_thr
                )
                for img, result in zip(image_list, result_list)
            ])
        else:
            visualizations = images

        return one_hot_predictions, visualizations
