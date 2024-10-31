_base_ = '../oriented_rcnn/obb_rcnn_ms_bh.py'
custom_imports = dict(
	imports=['mmdet.models.backbones.bh_wavecnet_resnet'],
	allow_failed_imports=False)

# dataset
data_root = 'data/split_ms_dota1_0/'

data = dict(
    train=dict(img_prefix=data_root + 'trainval/images/'))
model = dict(
    pretrained='./ckpt/bhwave_new.pth',
    backbone=dict(
        type='BHWaveCResNet',
        wavename='bior3.3',
        frozen_stages=-1,
    )
)
resume_from = 'work_dirs/baseline_2x_bhwave/epoch_22.pth'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 16, 22])
total_epochs = 24
# evaluation = dict(_delete_=True, interval=4, metric='bbox', proposal_nums=[1000], iou_thrs=[0.5])
gpu_ids = range(0, 7)