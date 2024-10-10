_base_ = '../oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py'
custom_imports = dict(
	imports=['mmdet.models.backbones.bh_wavecnet_resnet'],
	allow_failed_imports=False)
custom_hooks = [
    dict(type='NegUpdateHook', repeats=8)
]

# dataset
data_root = 'neg_copy/'

data = dict(train=dict(img_prefix=data_root + 'output_imgs/aug_imgs_8/train'),
            test=dict(ann_file=data_root + 'val/annfiles/',
                      img_prefix=data_root + 'val/images/',))
model = dict(
    pretrained='./ckpt/bhwave_new.pth',
    backbone=dict(
        type='BHWaveCResNet',
        wavename='bior3.3',
        frozen_stages=-1,
    )
)
# resume_from = 'work_dirs/neg_copy8_bhwave/epoch_20.pth'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[8, 11, 16, 22])
total_epochs = 24
gpu_ids = range(0, 7)