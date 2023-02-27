_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (640, 640)  # height, width

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=515, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
classes = ('LAMY', 'tumi', 'warrior', 'sandisk', 'belle', 'ThinkPad', 'rolex', 'balabala', 'vlone', 'nanfu', 'KTM', 'VW', 'libai', 'snoopy', 'Budweiser', 'armani', 'gree', 'GOON', 'KielJamesPatrick', 'uniqlo', 'peppapig', 'valentino', 'GUND', 'christianlouboutin', 'toyota', 'moutai', 'semir', 'marcjacobs', 'esteelauder', 'chaoneng', 'goldsgym', 'airjordan', 'bally', 'fsa', 'jaegerlecoultre', 'dior', 'samsung', 'fila', 'hellokitty', 'Jansport', 'barbie', 'VDL', 'manchesterunited', 'coach', 'PopSockets', 'haier', 'banbao', 'omron', 'fendi', 'erke', 'lachapelle', 'chromehearts', 'leader', 'pantene', 'motorhead', 'girdear', 'fresh', 'katespade', 'pandora', 'Aape', 'edwin', 'yonghui', 'Levistag', 'kboxing', 'yili', 'ugg', 'CommedesGarcons', 'Bosch', 'palmangels', 'razer', 'guerlain', 'balenciaga', 'anta', 'Duke', 'kingston', 'nestle', 'FGN', 'vrbox', 'toryburch', 'teenagemutantninjaturtles', 'converse', 'nanjiren', 'Josiny', 'kappa', 'nanoblock', 'lincoln', 'michael_kors', 'skyworth', 'olay', 'cocacola', 'swarovski', 'joeone', 'lining', 'joyong', 'tudor', 'YEARCON', 'hyundai', 'OPPO', 'ralphlauren', 'keds', 'amass', 'thenorthface', 'qingyang', 'mujosh', 'baishiwul', 'dissona', 'honda', 'newera', 'brabus', 'hera', 'titoni', 'decathlon', 'DanielWellington', 'moony', 'etam', 'liquidpalisade', 'zippo', 'mistine', 'eland', 'wodemeiliriji', 'ecco', 'xtep', 'piaget', 'gloria', 'hp', 'loewe', 'Levis_AE', 'Anna_sui', 'MURATA', 'durex', 'zebra', 'kanahei', 'ihengima', 'basichouse', 'hla', 'ochirly', 'chloe', 'miumiu', 'aokang', 'SUPERME', 'simon', 'bosideng', 'brioni', 'moschino', 'jimmychoo', 'adidas', 'lanyueliang', 'aux', 'furla', 'parker', 'wechat', 'emiliopucci', 'bmw', 'monsterenergy', 'Montblanc', 'castrol', 'HUGGIES', 'bull', 'zhoudafu', 'leaders', 'tata', 'oldnavy', 'OTC', 'levis', 'veromoda', 'Jmsolution', 'triangle', 'Specialized', 'tries', 'pinarello', 'Aquabeads', 'deli', 'mentholatum', 'molsion', 'tiffany', 'moco', 'SANDVIK', 'franckmuller', 'oakley', 'bulgari', 'montblanc', 'beaba', 'nba', 'shelian', 'puma', 'PawPatrol', 'offwhite', 'baishiwuliu', 'lexus', 'cainiaoguoguo', 'hugoboss', 'FivePlus', 'shiseido', 'abercrombiefitch', 'rejoice', 'mac', 'chigo', 'pepsicola', 'versacetag', 'nikon', 'TOUS', 'huawei', 'chowtaiseng', 'Amii', 'jnby', 'jackjones', 'THINKINGPUTTY', 'bose', 'xiaomi', 'moussy', 'Miss_sixty', 'Stussy', 'stanley', 'loreal', 'dhc', 'sulwhasoo', 'gentlemonster', 'midea', 'beijingweishi', 'mlb', 'cree', 'dove', 'PJmasks', 'reddragonfly', 'emerson', 'lovemoschino', 'suzuki', 'erdos', 'seiko', 'cpb', 'royalstar', 'thehistoryofwhoo', 'otterbox', 'disney', 'lindafarrow', 'PATAGONIA', 'seven7', 'ford', 'bandai', 'newbalance', 'alibaba', 'sergiorossi', 'lacoste', 'bear', 'opple', 'walmart', 'clinique', 'asus', 'ThomasFriends', 'wanda', 'lenovo', 'metallica', 'stuartweitzman', 'karenwalker', 'celine', 'miui', 'montagut', 'pampers', 'darlie', 'toray', 'bobdog', 'ck', 'flyco', 'alexandermcqueen', 'shaxuan', 'prada', 'miiow', 'inman', '3t', 'gap', 'Yamaha', 'fjallraven', 'vancleefarpels', 'acne', 'audi', 'hunanweishi', 'henkel', 'mg', 'sony', 'CHAMPION', 'iwc', 'lv', 'dolcegabbana', 'avene', 'longchamp', 'anessa', 'satchi', 'hotwheels', 'nike', 'hermes', 'jiaodan', 'siemens', 'Goodbaby', 'innisfree', 'Thrasher', 'kans', 'kenzo', 'juicycouture', 'evisu', 'volcom', 'CanadaGoose', 'Dickies', 'angrybirds', 'eddrac', 'asics', 'doraemon', 'hisense', 'juzui', 'samsonite', 'hikvision', 'naturerepublic', 'Herschel', 'MANGO', 'diesel', 'hotwind', 'intel', 'arsenal', 'rayban', 'tommyhilfiger', 'ELLE', 'stdupont', 'ports', 'KOHLER', 'thombrowne', 'mobil', 'Belif', 'anello', 'zhoushengsheng', 'd_wolves', 'FridaKahlo', 'citizen', 'fortnite', 'beautyBlender', 'alexanderwang', 'charles_keith', 'panerai', 'lux', 'beats', 'Y-3', 'mansurgavriel', 'goyard', 'eral', 'OralB', 'markfairwhale', 'burberry', 'uno', 'okamoto', 'only', 'bvlgari', 'heronpreston', 'jimmythebull', 'dyson', 'kipling', 'jeanrichard', 'PXG', 'pinkfong', 'Versace', 'CCTV', 'paulfrank', 'lanvin', 'vans', 'cdgplay', 'baojianshipin', 'rapha', 'tissot', 'casio', 'patekphilippe', 'tsingtao', 'guess', 'Lululemon', 'hollister', 'dell', 'supor', 'MaxMara', 'metersbonwe', 'jeanswest', 'lancome', 'lee', 'omega', 'lets_slim', 'snp', 'PINKFLOYD', 'cartier', 'zenith', 'LG', 'monchichi', 'hublot', 'benz', 'apple', 'blackberry', 'wuliangye', 'porsche', 'bottegaveneta', 'instantlyageless', 'christopher_kane', 'bolon', 'tencent', 'dkny', 'aptamil', 'makeupforever', 'kobelco', 'meizu', 'vivo', 'buick', 'tesla', 'septwolves', 'samanthathavasa', 'tomford', 'jeep', 'canon', 'nfl', 'kiehls', 'pigeon', 'zhejiangweishi', 'snidel', 'hengyuanxiang', 'linshimuye', 'toread', 'esprit', 'BASF', 'gillette', '361du', 'bioderma', 'UnderArmour', 'TommyHilfiger', 'ysl', 'onitsukatiger', 'house_of_hello', 'baidu', 'robam', 'konka', 'jack_wolfskin', 'office', 'goldlion', 'tiantainwuliu', 'wonderflower', 'arcteryx', 'threesquirrels', 'lego', 'mindbridge', 'emblem', 'grumpycat', 'bejirog', 'ccdd', '3concepteyes', 'ferragamo', 'thermos', 'Auby', 'ahc', 'panasonic', 'vanguard', 'FESTO', 'MCM', 'lamborghini', 'laneige', 'ny', 'givenchy', 'zara', 'jiangshuweishi', 'daphne', 'longines', 'camel', 'philips', 'nxp', 'skf', 'perfect', 'toshiba', 'wodemeilirizhi', 'Mexican', 'VANCLEEFARPELS', 'HARRYPOTTER', 'mcm', 'nipponpaint', 'chenguang', 'jissbon', 'versace', 'girardperregaux', 'chaumet', 'columbia', 'nissan', '3M', 'yuantong', 'sk2', 'liangpinpuzi', 'headshoulder', 'youngor', 'teenieweenie', 'tagheuer', 'starbucks', 'pierrecardin', 'vacheronconstantin', 'peskoe', 'playboy', 'chanel', 'HarleyDavidson_AE', 'volvo', 'be_cheery', 'mulberry', 'musenlin', 'miffy', 'peacebird', 'tcl', 'ironmaiden', 'skechers', 'moncler', 'rimowa', 'safeguard', 'baleno', 'sum37', 'holikaholika', 'gucci', 'theexpendables', 'dazzle', 'vatti', 'nintendo')

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/brand20/anns/train.json',
        img_prefix='/root/autodl-tmp/brand20/images/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,  # 单个 GPU 的 Batch size
    workers_per_gpu=8,   # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/brand20/anns/val.json',
        img_prefix='/root/autodl-tmp/brand20/images/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/brand20/anns/test.json',
        img_prefix='/root/autodl-tmp/brand20/images/test',
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 48
num_last_epochs = 2  # 为了配合数据增强，在最后 num_last_epochs 个 epoch 会采用固定的最小学习率。
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,       # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=50)  # 50 iters 打印一次log输出

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
