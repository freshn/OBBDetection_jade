_base_ = './faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py'

custom_imports = dict(
	imports=['mmdet.models.backbones.wavecnet_resnet'],
	allow_failed_imports=False)

model = dict(
        # pretrained='./ckpt/sl50_in1k_new.pth',
        backbone=dict(
            type='WaveCResNet',
            depth=50,
            wavename='bior3.3',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
        #     init_cfg=dict(
        #         type='Pretrained',
        #         checkpoint=
        #         '../mmclassification/work_dirs/wavecnet_ch3.3_in1k_8xb32/latest.pth',
        #         prefix='backbone'),
        ))

find_unused_parameters=True