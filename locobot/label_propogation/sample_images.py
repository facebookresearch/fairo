# helpful for sampling images and store them at some different location
import os
import glob

root_dir = '/checkpoint/dhirajgandhi/active_vision/habitat_data/rgb'
out_dir = '/checkpoint/dhirajgandhi/active_vision/habitat_data/turk_rgb'

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
img_list_file = open("turk_imgs.txt", "w")
list_imgs = glob.glob(root_dir + '/*.jpg')
skip = 60
for i, img_path in enumerate(list_imgs):
    if i % skip == 0:
        print(img_path)
        img_list_file.write(img_path + '\n')
        os.system("cp {} {}".format(img_path, out_dir))

img_list_file.close()
