# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
import os
import mmcv
from tqdm import tqdm
import random
import neg_copy.aug as am
from neg_copy.util import *

@HOOKS.register_module()
class NegUpdateHook(Hook):
    """Update augmented images by copying negtive samples after every interval epoch 

    Args:
        interval (int): Default: 1.
    """

    def __init__(self, repeats=8, ada_ratio=0):
        self.repeats = repeats
        self.ada_ratio = ada_ratio

    def before_train_epoch(self, runner):
        base_dir = 'neg_copy/'
        if self.ada_ratio:
            save_base_dir = os.path.join(base_dir, 'output_imgs', 'aug_imgs_'+str(self.ada_ratio))
        elif self.repeats:
            save_base_dir = os.path.join(base_dir, 'output_imgs', 'aug_imgs_'+str(self.repeats))
        else:
            print('At least one of the (repeats) and (ada_ratio) should be defined.')

        check_dir(save_base_dir)

        imgs_dir = [f.strip() for f in open(os.path.join(base_dir, 'images.txt')).readlines()]
        small_imgs_dir = [f.strip() for f in open(os.path.join(base_dir, 'crop_bg.txt')).readlines()]
        random.shuffle(small_imgs_dir)

        dataset = runner.data_loader.dataset
        anns = dataset.coco.anns.values()
        imgs = dataset.coco.imgs.values()
        # anns_file = {anns:[ann1,..,ann100], image_filename:[img1,..,img20]}
        id2file = {img['id']:img['filename'] for img in imgs}
        # anns2file = {ann['id']:{'bbox':ann['bbox'], 'filename':id2file[ann['image_id']]} \
        #             for ann in anns if not ann['ignore'] and \
        #             not ann['uncertain'] and not ann['logo'] and not ann['in_dense_image']}
        file2anns = {}
        for ann in anns:
            # if ann['ignore'] or ann['uncertain'] or ann['logo'] or ann['in_dense_image']:
            #     continue
            if id2file[ann['image_id']] not in file2anns.keys():
                file2anns[id2file[ann['image_id']]] = [ann['bbox']]
            else:
                file2anns[id2file[ann['image_id']]].append(ann['bbox'])              

        for image_dir in tqdm(imgs_dir):
            small_img = []
            if self.ada_ratio:
                # image_dir[6:] keep the filename after 'train/'
                if image_dir[6:] in file2anns.keys():
                    repeats = max(int(self.ada_ratio * len(file2anns[image_dir[6:]])), 1)
                else:
                    repeats = self.repeats
            else:
                repeats = self.repeats
            for x in range(repeats):
                if small_imgs_dir == []:
                    #exit()
                    small_imgs_dir = [f.strip() for f in open(os.path.join(base_dir,'crop_bg.txt')).readlines()]
                    random.shuffle(small_imgs_dir)
                small_img.append(small_imgs_dir.pop())
            am.copybackground(os.path.join(base_dir, image_dir), save_base_dir, [os.path.join(base_dir, patch) for patch in small_imgs_dir])


