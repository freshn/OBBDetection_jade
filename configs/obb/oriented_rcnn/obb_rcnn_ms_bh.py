##### plrease use neg_copy/baseline_2x_bh for oriented rcnn training 
##### This is the expired config

_base_ = './faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py'

custom_imports = dict(
	imports=['mmdet.models.backbones.bh_r50'],
	allow_failed_imports=False)

model = dict(
        pretrained='./ckpt/sl50_in1k_new.pth',
        backbone=dict(
            type='BHResNet',
            depth=50,
            num_stages=4,
            base_channels=64,
            stem_channels=64,
            out_indices=(0, 1, 2, 3),
            strides=(2, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            frozen_stages=1),
        #     init_cfg=dict(
        #             type='Pretrained',
        #             checkpoint=
        #             '../mmclassification/work_dirs/wavecnet_ch3.3_in1k_8xb32/latest.pth',
        #             prefix='backbone'),
        )

find_unused_parameters=True