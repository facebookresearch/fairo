import os
import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
from PIL import Image
import json
from pycococreatortools import pycococreatortools

semantic_json_root = '/checkpoint/apratik/ActiveVision/active_vision/info_semantic'

def load_semantic_json(scene):
    replica_root = '/datasets01/replica/061819/18_scenes'
    habitat_semantic_json = os.path.join(replica_root, scene, 'habitat', 'info_semantic.json')
    with open(habitat_semantic_json, "r") as f:
        hsd = json.load(f)
    if hsd is None:
        print("Semantic json not found!")
    return hsd

hsd = load_semantic_json('apartment_0')

class Candidate:
    def __init__(self, img_id, l, r, iid):
        self.img_id = img_id
        self.left_prop = l
        self.right_prop = r
        self.instance_id = iid
        
    def __repr__(self):
        return f'[candidate for {self.instance_id}: img_id {self.img_id}, max left prop {self.left_prop}, max right prop {self.right_prop}]\n'

class PickGoodCandidates:
    def __init__(self, img_dir, depth_dir, seg_dir, instance_ids):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.seg_dir = seg_dir
        self.iids = instance_ids
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
    
    def is_iid_in_full_view(self, img_id, iid):
        seg_path = os.path.join(self.seg_dir, "{}.npy".format(img_id))
#         print(f'seg_path {seg_path}')
        annot = np.load(seg_path).astype(np.uint32)
        binary_mask = np.zeros_like(annot)
        
        if iid in np.unique(annot):
            binary_mask = (annot == iid).astype(np.uint32)

        area = binary_mask.sum()
        if area < 1000: # too small
            return False, area
        
        # Check that all masks are within a certain distance from the boundary
        # all pixels [:10,:], [:,:10], [-10:], [:-10] must be 0:
        if binary_mask[:10,:].any() or binary_mask[:,:10].any() or binary_mask[:,-10:].any() or binary_mask[-10:,:].any():
            return False, area
        
        return True, area
        
        
    def sample_uniform_nn2(self, n):
        if n > 1:
            print(f'WARNING: NOT IMPLEMENTED FOR N > 1 YET!')
            return
        frames = []
        for iid in self.iids:
            """
            for each frame that has the instance in view, find the left and right neighborhood of views.
            this is the maxmimal prop length on each side. 
            Pick the frame that has maximal(l+r).
            visualize the entire range. propagate l, r
            """
            print(f'picking {n} candidate for instance {iid} ... ')
            view_arr = {}
            imgs = os.listdir(self.img_dir)
            for img in imgs:
                img_id = int(img.split('.')[0])
                in_view, area = self.is_iid_in_full_view(img_id, iid)
                view_arr[img_id] = (in_view, area)
                
#             print(view_arr)
            # find contiguous blocks
            seq_len = np.zeros(len(imgs))
            for i in range(len(imgs)):
                if view_arr[i][0]:
                    seq_len[i] = 1 + (seq_len[i-1] if i > 0 else 0)
                else:
                    seq_len[i] = 0
                    
            # find index with max value
            candidate = seq_len.argmax()

            # go to median 
            while candidate >= 0:
#                 print(candidate, seq_len[candidate], seq_len.max())
                if seq_len[candidate] <= seq_len.max()/2:
                    break
                candidate -= 1
            
            # now return the max prop length to the left and right of this frame
            l = 0
            d = 1
            while True:
                nxt = candidate - d
                if nxt >= 0 and view_arr[nxt][1] > 0:
                    l += 1
                    d += 1
                else:
                    break
                    
            r = 0
            d = 1
            while True:
                nxt = candidate + d
                if nxt < len(imgs) and view_arr[nxt][1] > 0:
                    r += 1
                    d += 1
                else:
                    break
            
                
            print(f'picked candidate {candidate}, max left prop {l}, max right prop {r}')
            frames.append(Candidate(candidate, l, r, iid))
        return frames
        
    def vis(self, candidate, contours=None):
        
        # visualize left prop
        fig, axs = plt.subplots(1, candidate.left_prop+1, figsize=(2*candidate.left_prop,4))
        l = candidate.img_id-candidate.left_prop
        for x in range(candidate.img_id-candidate.left_prop, candidate.img_id+1):
            img_path = os.path.join(self.img_dir, "{}.jpg".format(x))
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            axs[x-l].imshow(image)
            axs[x-l].set_title("{}.jpg".format(x))
        plt.show()
        
        fig2, axs2 = plt.subplots(1, candidate.right_prop+1, figsize=(2*candidate.right_prop, 4))
        l = candidate.img_id
        for x in range(candidate.img_id, candidate.img_id+candidate.right_prop+1):
            img_path = os.path.join(self.img_dir, "{}.jpg".format(x))
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if candidate.right_prop == 0:
                axs2.imshow(image)
                axs2.set_title("{}.jpg".format(x))
            else:
                axs2[x-l].imshow(image)
                axs2[x-l].set_title("{}.jpg".format(x))
            
        plt.show()
