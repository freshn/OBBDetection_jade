_base_ = '../oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py'

# dataset
data_root = 'data/split_ms_dota1_0/'

data = dict(
    train=dict(img_prefix=data_root + 'trainval/images/',),
    test=dict(ann_file=data_root + 'val/annfiles/',
              img_prefix=data_root + 'val/images/',))
model = dict(
    backbone=dict(
        frozen_stages=-1,
    )
)
resume_from = './ckpt/obb_official.pth'
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