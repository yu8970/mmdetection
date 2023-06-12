# model settings
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,                # ResNet 系列包括 stem+ 4个 stage 输出
        out_indices=(0, 1, 2, 3),    # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要， 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        frozen_stages=1,             # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        # 更新的是BN里的/gamma和/beta两个可学习参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化算子, 所有的 BN 层的可学习参数都需要进行参数更新
        # 指BN层不计算也不更新均值和方差
        norm_eval=True,  # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,   # FPN 层输出特征图通道数
        stacked_convs=4,   # 每个分支堆叠4层卷积
        feat_channels=256, # 中间特征图通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,   # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大
            scales_per_octave=3,   # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)
            ratios=[0.5, 1.0, 2.0], # 每个特征图有3个高宽比例
            strides=[8, 16, 32, 64, 128]),  # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',  # 最大 IoU 原则分配器
            pos_iou_thr=0.5,        # 正样本阈值
            neg_iou_thr=0.4,        # 负样本阈值
            min_pos_iou=0,          # 正样本阈值下限
            ignore_iof_thr=-1),     # 忽略 bbox 的阈值，-1表示不忽略
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
