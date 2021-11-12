import argparse
import os
import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
from PIL import Image
import json
from pycococreatortools import pycococreatortools

from datetime import datetime

labels = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']
# labels = ['chair']
semantic_json_root = '/checkpoint/apratik/ActiveVision/active_vision/info_semantic'

def load_semantic_json(scene):
    replica_root = '/datasets01/replica/061819/18_scenes'
    habitat_semantic_json = os.path.join(replica_root, scene, 'habitat', 'info_semantic.json')
#         habitat_semantic_json = os.path.join(self.sjr, scene + '_info_semantic.json')
#         print(f"Using habitat semantic json {habitat_semantic_json}")
    with open(habitat_semantic_json, "r") as f:
        hsd = json.load(f)
    if hsd is None:
        print("Semantic json not found!")
    return hsd

hsd = load_semantic_json('apartment_0')
label_id_dict = {}
new_old_id = {}
idc = 1
for obj_cls in hsd["classes"]:
    if obj_cls["name"] in labels:
        label_id_dict[obj_cls["id"]] = obj_cls["name"]
        new_old_id[obj_cls['id']] = idc
        idc += 1
        
class PickGoodCandidates:
    def __init__(self, img_dir, depth_dir, seg_dir):
        self.imgdir = img_dir
        self.depthdir = depth_dir
        self.segdir = seg_dir
        self.filtered = False
        self.chosen = set()
        
    def is_open_contour(self, c):
        # check for a bunch of edge points
        # c is of the format num_points * 1 * 2
        edge_points = []
        for x in c:
            if x[0][0] == 0 or x[0][1] == 0 or x[0][0] == 511 or x[0][1] == 511:
                edge_points.append(x)
#         print(len(edge_points))
        if len(edge_points) > 0:
            return True
        return False

    def find_nearest(self, x):
        dist = 10000
        res = -1
        for y, _ in self.good_candidates:
            if abs(x-y) < dist:
                dist = abs(x-y)
                res = y
        # now look in vicinity of res for frame with max size 
        return res
    
    def sample_uniform_nn(self, n):
        if not self.filtered:
            self.filter_candidates()
            
        num_imgs = len(glob.glob(self.imgdir + '/*.jpg'))
        print(f'num_imgs {num_imgs}')
        delta = int(num_imgs / n)
        cand = [delta*x for x in range(1,n+1)]
        return [self.find_nearest(x) for x in cand]
    
    def find_nearest2(self, x):
        dist = 10000
        res = -1
        for y, _ in self.good_candidates:
            if abs(x-y) < dist and y not in self.chosen:
                dist = abs(x-y)
                res = y
        # now look in vicinity of res for frame with max size 
        for x in range(4):
            self.chosen.add(res+x)
            self.chosen.add(res-x)
        return res
    
    def sample_uniform_nn2(self, n):
        if not self.filtered:
            self.filter_candidates()
            
        num_imgs = len(glob.glob(self.imgdir + '/*.jpg'))
        print(f'num_imgs {num_imgs}')
        delta = int(num_imgs / n)
        cand = [delta*x for x in range(1,n+1)]
        return [self.find_nearest2(x) for x in cand]
        
    def sample_n(self, n):
        if not self.filtered:
            self.filter_candidates()
            
        # uniformly sample 
        # randomly sample 
        return [x[0] for x in random.sample(self.good_candidates, n)]
    
    def filter_candidates(self):
        self.good_candidates = []
        self.bad_candidates = []
        for x in range(len(os.listdir(self.imgdir)) + 1):
            res, size = self.is_good_candidate(x)
            if res:
                self.good_candidates.append((x, size))
#                 self.vis(x)
            elif res == False:
                self.bad_candidates.append(x)
                
        print(f'{len(self.good_candidates)} good candidates, {len(self.bad_candidates)} bad candidates')
        self.filtered = True
#         print(f'good candidates {self.good_candidates}')
            
    def visualize_good_bad(self, num):
        # TODO: sample num numbers from all, then look at he 
        # sample num from good bad
        good = random.sample(self.good_candidates, num)
        bad = random.sample(self.bad_candidates, num)
        
        for x in range(num):
            gim = os.path.join(self.imgdir, "{:05d}.jpg".format(good[x][0]))
            gim = cv2.cvtColor(cv2.imread(gim), cv2.COLOR_BGR2RGB)
            
            bim = os.path.join(self.imgdir, "{:05d}.jpg".format(bad[x]))
            bim = cv2.cvtColor(cv2.imread(bim), cv2.COLOR_BGR2RGB)
            
            arr = [gim, bim]
            titles = ['good', 'bad']
            plt.figure(figsize=(5,4))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 2, i+1)
                ax.axis('off')
                ax.set_title(titles[i])
                plt.imshow(data)
            plt.show()
        
    def is_good_candidate(self, x, vis=False):
        dpath = os.path.join(self.depthdir, "{:05d}.npy".format(x))
        imgpath = os.path.join(self.imgdir, "{:05d}.jpg".format(x))
        segpath = os.path.join(self.segdir, "{:05d}.npy".format(x))
        
        # Load Image
        if not os.path.isfile(imgpath):
            return None, None
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
#         print(img.shape)
        
        # Load Annotations 
        annot = np.load(segpath).astype(np.uint32)
        count = 0
        all_binary_mask = np.zeros_like(annot)
#         print(f'{annot.shape, all_binary_mask.shape}')

        total_area = 0
        total_objects = 0
        
        for i in np.sort(np.unique(annot.reshape(-1), axis=0)):
            try:
                if hsd["id_to_label"][i] < 1 or label_id_dict[hsd["id_to_label"][i]] not in labels:# or hsd["id_to_label"][i] not in self.label_id_dict:
                    continue
                category_info = {"id": new_old_id[hsd["id_to_label"][i]], "is_crowd": False}
    #                     print(f'category_info {category_info}')
            except Exception as ex:
#                 print(ex)
                continue

            binary_mask = (annot == i).astype(np.uint32)
            all_binary_mask = np.bitwise_or(binary_mask, all_binary_mask)
#             plt.imshow(binary_mask, alpha=0.5)

            annotation_info = pycococreatortools.create_annotation_info(
                count, 1, category_info, binary_mask, img.shape[:2], tolerance=2
            )
#             print(annotation_info)
            if annotation_info and 'area' in annotation_info.keys():
                total_area = annotation_info['area']
                total_objects += 1
#             count+=1
    
        avg_area = total_area/total_objects if total_objects > 0 else 0
        
        if vis:
            plt.imshow(all_binary_mask)
            plt.show()
#             print(np.unique(all_binary_mask))
            
        if not all_binary_mask.any():
#             print(f'no masks')
            return False, None
        
        # Check that all masks are within a certain distance from the boundary
        # all pixels [:10,:], [:,:10], [-10:], [:-10] must be 0:
        if all_binary_mask[:10,:].any() or all_binary_mask[:,:10].any() or all_binary_mask[:,-10:].any() or all_binary_mask[-10:,:].any():
            return False, None
        
        if (all_binary_mask == 1).sum() < 5000:
            return False, None
        
        return True, (all_binary_mask == 1).sum()
        
    def vis(self, x, contours=None):
        
        imgpath = os.path.join(self.imgdir, "{:05d}.jpg".format(x))
        image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        
        if contours:
            image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
      
        arr = [image]
        titles = ["{:05d}.jpg".format(x)]
        plt.figure(figsize=(5,4))
        for i, data in enumerate(arr):
    #         print(f'data.shape {data.shape}')
            ax = plt.subplot(1, 1, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data)
        plt.show()