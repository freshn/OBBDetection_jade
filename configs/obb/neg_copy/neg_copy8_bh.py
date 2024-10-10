_base_ = '../oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py'
custom_hooks = [
    dict(type='NegUpdateHook', repeats=8)
]

# dataset
data_root = 'neg_copy/'

data = dict(train=dict(img_prefix=data_root + 'output_imgs/aug_imgs_8/train'))
model = dict(
    backbone=dict(
        frozen_stages=-1,
    )
)
resume_from = './ckpt/sl50_in1k_new.pth'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 16, 22])
total_epochs = 24