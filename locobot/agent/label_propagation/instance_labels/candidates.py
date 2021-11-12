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

class PickGoodCandidates:
    def __init__(self, img_dir, depth_dir, seg_dir, instance_ids):
        self.imgdir = img_dir
        self.depthdir = depth_dir
        self.segdir = seg_dir
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
        if n > 1:
            print(f'WARNING: NOT IMPLEMENTED FOR N > 1 {n} YET!')
            return
        frames = set()
        for x in self.iids:
            if not self.filtered:
                self.filter_candidates(x)
                
            # now pick n best ids
            print(f'{len(self.good_candidates)} good candidates for instance id {x}')
            
            if len(self.good_candidates) > 0:
                # sort a list of tuples by the second element 
                sorted(self.good_candidates, key= lambda x: x[1])

                print(f'sorted candidates {self.good_candidates}')
                picked = 0
                for c in self.good_candidates:
                    if c[0] not in frames:
                        print(f'picking {c[0]} for {x}, n {n}')
                        frames.add(c[0])
                        picked += 1
                        if picked == n:
                            break
        return list(frames)
        
    def filter_candidates(self, iid):
        self.good_candidates = []
        self.bad_candidates = []
        for x in range(len(os.listdir(self.imgdir))):
            res, size = self.is_good_candidate(x, iid)
            if res:
                self.good_candidates.append((x, size))
#                 self.vis(x)
            elif res == False:
                self.bad_candidates.append(x)
            elif not res:
                print(f'None for {x}')
                
        assert len(os.listdir(self.imgdir)) == len(self.good_candidates) + len(self.bad_candidates)
#         print(f'good candidates {self.good_candidates}')
        
    def is_good_candidate(self, fname, iid, vis=False):
        dpath = os.path.join(self.depthdir, "{:05d}.npy".format(fname))
        imgpath = os.path.join(self.imgdir, "{:05d}.jpg".format(fname))
        segpath = os.path.join(self.segdir, "{:05d}.npy".format(fname))
                
        # Load Image
        if not os.path.isfile(imgpath):
            print(f'looking for {imgpath}')
            return None, None
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
#         print(img.shape)
        
        # Load Annotations 
        annot = np.load(segpath).astype(np.uint32)
        binary_mask = np.zeros_like(annot)
        
        if iid in np.unique(annot):
#             print(f'{iid} in {np.unique(annot)}')
            binary_mask = (annot == iid).astype(np.uint32)
#             print(f'mask area {binary_mask.sum()}')
#             plt.imshow(binary_mask)
#             plt.show()

        if vis:
            plt.imshow(binary_mask)
            plt.show()
#             print(np.unique(all_binary_mask))

        if binary_mask.sum() < 1000:
            return False, None
            
        if not binary_mask.any():
            return False, None
        
        # Check that all masks are within a certain distance from the boundary
        # all pixels [:10,:], [:,:10], [-10:], [:-10] must be 0:
        if binary_mask[:10,:].any() or binary_mask[:,:10].any() or binary_mask[:,-10:].any() or binary_mask[-10:,:].any():
            return False, None
        
        return True, (binary_mask == 1).sum()
        
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
        
    def vis(self, imgid, contours=None):
        
        imgpath = os.path.join(self.imgdir, "{:05d}.jpg".format(imgid))
        image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        prop_path = os.path.join(self.segdir, "{:05d}.npy".format(imgid))
        annot = np.load(prop_path).astype(np.uint32)
        
        abm = np.zeros_like(annot)
        for x in self.iids:
            abm = np.bitwise_or(abm, annot == x).astype(np.uint32)
        
        if contours:
            image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
      
        arr = [image, abm]
        titles = ["{:05d}.jpg".format(imgid), "{:05d}.npy".format(imgid)]
        plt.figure(figsize=(5,4))
        for i, data in enumerate(arr):
    #         print(f'data.shape {data.shape}')
            ax = plt.subplot(1, 2, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data)
        plt.show()