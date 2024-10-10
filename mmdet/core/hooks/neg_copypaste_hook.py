# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
import os
import shutil
from tqdm import tqdm
import random
import torch
import cv2
import numpy as np
from BboxToolkit.transforms import poly2hbb

def check_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileExistsError:
            print('folder already exists')
            
def find_str(filename):
    if 'train' in filename:
        return os.path.dirname(filename[filename.find('train'):])
    else:
        return os.path.dirname(filename[filename.find('val'):])

def random_valid_centriod(bbox_shape, img_shape, mask=None):
    bbox_h, bbox_w, bbox_c = bbox_shape
    img_h, img_w, img_c = img_shape
    max_side_length = max(bbox_h, bbox_w)
    safe_distance = int(max_side_length/2)+1
    center_mask = torch.zeros(img_h,img_w)
    center_mask[safe_distance:img_h-safe_distance+1,safe_distance:img_w-safe_distance+1] = 1
    # print(center_mask.shape,mask.shape)
    mask = cv2.dilate(mask.numpy(),torch.ones(safe_distance,safe_distance).numpy())
    mask = torch.tensor(cv2.bitwise_and(center_mask.numpy(),cv2.bitwise_not(mask)))
    new_bboxes = []
    idx = mask.nonzero()
    if len(idx) > 0:
        x_center, y_center = idx[random.randint(0,len(idx)-1)]
        return x_center, y_center
    else:
        print('No vaild location.')
        return 0, 0

def area_transform(image, area_max=440, area_min=16):
    height, width, channels = image.shape
    while (image.shape[0]*image.shape[1]) > area_max:
        prob = random.random()
        if prob>0.5:
            height,width = int(image.shape[0]*0.9),int(image.shape[1]*0.9)
        elif prob<=0.5 and prob>0.2:
            height,width = int(image.shape[0]*0.7),int(image.shape[1]*0.7)
        else:
            height,width = int(image.shape[0]*0.5),int(image.shape[1]*0.5)
        image = cv2.resize(image, (width, height))
    while (image.shape[0]*image.shape[1]) <= area_min:
        image = cv2.resize(image, (int(image.shape[1]*1.25),int(image.shape[0]*1.25)))
    return image
    
def copyneg(img, crop_dir, mask):
    crop = cv2.imread(crop_dir)
    # bbox = crop
    bbox = area_transform(crop, area_max=440, area_min=16)
    height, width, channels = bbox.shape
    x_center, y_center = random_valid_centriod(bbox.shape, img.shape, mask=mask)
    if x_center == y_center == 0:
        print('Crop is too large to have a vaild position to paste.')
        return img
    x_left, y_left, \
    x_right, y_right = int(x_center-0.5*height), \
                       int(y_center-0.5*width), \
                       int(x_center+0.5*height), \
                       int(y_center+0.5*width)
    try:
        if random.random() > 0.5:
            bbox = bbox[:, ::-1, :]  
        # cv2.seamlessClone(bbox,img,mask,center,mode) 
        # when any side length is smaller than 4 there is a bug so be catious            
        img[x_left:x_right, y_left:y_right] = \
            cv2.seamlessClone(bbox, img[x_left:x_right, y_left:y_right],
                              255*np.ones(bbox.shape,bbox.dtype),
                              (int(width/2), int(height/2)), cv2.NORMAL_CLONE)
    except ValueError:
        print("---")
    return img


@HOOKS.register_module()
class NegUpdateHook(Hook):
    """Update augmented images by copying negtive samples after every interval epoch 

    Args:
        interval (int): Default: 1.
    """

    def __init__(self, repeats=8, ada_ratio=0, interval=1, keep_rate=0.001):
        self.repeats = repeats
        self.ada_ratio = ada_ratio
        self.interval = interval
        self.keep_rate = keep_rate

    def before_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            base_dir = 'neg_copy/'
            if self.ada_ratio:
                save_base_dir = os.path.join(base_dir, 'output_imgs', 'aug_imgs_'+str(self.ada_ratio))
            elif self.repeats:
                save_base_dir = os.path.join(base_dir, 'output_imgs', 'aug_imgs_'+str(self.repeats))
            else:
                print('At least one of the (repeats) and (ada_ratio) should be defined.')

            check_dir(save_base_dir)
            imgs_dir = [f.strip() for f in open(os.path.join(base_dir, 'images.txt')).readlines()]
            crops_dir = [f.strip() for f in open(os.path.join(base_dir, 'crop_bg.txt')).readlines()]
            random.shuffle(crops_dir)

            ## get anns and imgs for DOTA
            dataset = runner.data_loader.dataset
            img_info = dataset.load_annotations(dataset.ann_file)
            ## file2anns = {filename:[ann1,...,anns]}
            file2anns = {}
            for i in range(len(img_info)):
                all_anns = dataset.get_ann_info(i)
                if len(all_anns['bboxes']) == 0:
                    continue   
                ori_anns = all_anns['bboxes'][all_anns['trunc']==False].copy()        
                # ori_anns = [ori_ann+[img_info[i]['x_start'],img_info[i]['y_start']]*4 for ori_ann in ori_anns]                  
                if img_info[i]['filename'] not in file2anns.keys():
                    file2anns[img_info[i]['filename']] = ori_anns
                else:
                    file2anns[img_info[i]['filename']].extend(ori_anns)                  
                
            for image_dir in tqdm(imgs_dir):
                image_dir_split = image_dir.split('/')   
                if image_dir_split[-1] not in file2anns.keys():
                    continue 
                whole_image_dir = os.path.join(base_dir, image_dir_split[1], image_dir_split[-1])
                if random.random()>0.001:
                    dir_name = find_str(whole_image_dir)
                    save_dir = os.path.join(save_base_dir, dir_name)
                    # check_dir(save_dir)
                    save_file = os.path.join(save_dir, os.path.basename(whole_image_dir))
                    shutil.copyfile(whole_image_dir, save_file)
                    continue      
                else:
                    image = cv2.imread(whole_image_dir)
                
                if self.ada_ratio:
                    # image_dir[6:] keep the filename after 'train/'
                    repeats = max(int(self.ada_ratio * len(file2anns[image_dir[6:]])), 1)
                else:
                    repeats = self.repeats
                    
                if len(image) == 0:
                    continue
                ## generate the masks showing valid locations (no overlap with gts)             
                mask = torch.zeros(image.shape[0],image.shape[1])
                for ann in file2anns[image_dir_split[-1]]:
                    # ann: [x1,y1,x2,y2,x3,y3,x4,y4]
                    hbb_ann = poly2hbb(ann)
                    assert hbb_ann.min()>=0
                    # image.shape: [height, width, channel]
                    # hbb_ann: [x1, y1, x2, y2]
                    assert hbb_ann[3]<=mask.shape[0] and hbb_ann[2]<=mask.shape[1]
                    mask[int(hbb_ann[1]):int(hbb_ann[3]),int(hbb_ann[0]):int(hbb_ann[2])] = 1     
                for x in range(repeats):
                    ## reload when pop out all crops
                    if crops_dir == []:
                        crops_dir = [f.strip() for f in open(os.path.join(base_dir,'crop_bg.txt')).readlines()]
                        random.shuffle(crops_dir)
                    crop_dir = crops_dir.pop()
                    image = copyneg(image, os.path.join(base_dir, crop_dir), mask)
                dir_name = find_str(whole_image_dir)
                save_dir = os.path.join(save_base_dir, dir_name)
                check_dir(save_dir)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(whole_image_dir)), image)
        else:
            pass

