_base_ = './yolox_s_8xb8-300e_coco.py'

img_scale = (640, 640)  # width, height

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=515))


# dataset settings
data_root = '/content/brand/'
dataset_type = 'CocoDataset'

train_json_path = 'annotations/train-32000.json'
train_img_prefix = 'images/train-32000/'
val_json_path = 'annotations/val-8000.json'
val_img_prefix = 'images/val-8000/'
test_json_path = 'annotations/test-8000.json'
test_img_prefix = 'images/test-8000/'


backend_args = None
classes = ('LAMY', 'tumi', 'warrior', 'sandisk', 'belle', 'ThinkPad', 'rolex', 'balabala', 'vlone', 'nanfu', 'KTM', 'VW', 'libai', 'snoopy', 'Budweiser', 'armani', 'gree', 'GOON', 'KielJamesPatrick', 'uniqlo', 'peppapig', 'valentino', 'GUND', 'christianlouboutin', 'toyota', 'moutai', 'semir', 'marcjacobs', 'esteelauder', 'chaoneng', 'goldsgym', 'airjordan', 'bally', 'fsa', 'jaegerlecoultre', 'dior', 'samsung', 'fila', 'hellokitty', 'Jansport', 'barbie', 'VDL', 'manchesterunited', 'coach', 'PopSockets', 'haier', 'banbao', 'omron', 'fendi', 'erke', 'lachapelle', 'chromehearts', 'leader', 'pantene', 'motorhead', 'girdear', 'fresh', 'katespade', 'pandora', 'Aape', 'edwin', 'yonghui', 'Levistag', 'kboxing', 'yili', 'ugg', 'CommedesGarcons', 'Bosch', 'palmangels', 'razer', 'guerlain', 'balenciaga', 'anta', 'Duke', 'kingston', 'nestle', 'FGN', 'vrbox', 'toryburch', 'teenagemutantninjaturtles', 'converse', 'nanjiren', 'Josiny', 'kappa', 'nanoblock', 'lincoln', 'michael_kors', 'skyworth', 'olay', 'cocacola', 'swarovski', 'joeone', 'lining', 'joyong', 'tudor', 'YEARCON', 'hyundai', 'OPPO', 'ralphlauren', 'keds', 'amass', 'thenorthface', 'qingyang', 'mujosh', 'baishiwul', 'dissona', 'honda', 'newera', 'brabus', 'hera', 'titoni', 'decathlon', 'DanielWellington', 'moony', 'etam', 'liquidpalisade', 'zippo', 'mistine', 'eland', 'wodemeiliriji', 'ecco', 'xtep', 'piaget', 'gloria', 'hp', 'loewe', 'Levis_AE', 'Anna_sui', 'MURATA', 'durex', 'zebra', 'kanahei', 'ihengima', 'basichouse', 'hla', 'ochirly', 'chloe', 'miumiu', 'aokang', 'SUPERME', 'simon', 'bosideng', 'brioni', 'moschino', 'jimmychoo', 'adidas', 'lanyueliang', 'aux', 'furla', 'parker', 'wechat', 'emiliopucci', 'bmw', 'monsterenergy', 'Montblanc', 'castrol', 'HUGGIES', 'bull', 'zhoudafu', 'leaders', 'tata', 'oldnavy', 'OTC', 'levis', 'veromoda', 'Jmsolution', 'triangle', 'Specialized', 'tries', 'pinarello', 'Aquabeads', 'deli', 'mentholatum', 'molsion', 'tiffany', 'moco', 'SANDVIK', 'franckmuller', 'oakley', 'bulgari', 'montblanc', 'beaba', 'nba', 'shelian', 'puma', 'PawPatrol', 'offwhite', 'baishiwuliu', 'lexus', 'cainiaoguoguo', 'hugoboss', 'FivePlus', 'shiseido', 'abercrombiefitch', 'rejoice', 'mac', 'chigo', 'pepsicola', 'versacetag', 'nikon', 'TOUS', 'huawei', 'chowtaiseng', 'Amii', 'jnby', 'jackjones', 'THINKINGPUTTY', 'bose', 'xiaomi', 'moussy', 'Miss_sixty', 'Stussy', 'stanley', 'loreal', 'dhc', 'sulwhasoo', 'gentlemonster', 'midea', 'beijingweishi', 'mlb', 'cree', 'dove', 'PJmasks', 'reddragonfly', 'emerson', 'lovemoschino', 'suzuki', 'erdos', 'seiko', 'cpb', 'royalstar', 'thehistoryofwhoo', 'otterbox', 'disney', 'lindafarrow', 'PATAGONIA', 'seven7', 'ford', 'bandai', 'newbalance', 'alibaba', 'sergiorossi', 'lacoste', 'bear', 'opple', 'walmart', 'clinique', 'asus', 'ThomasFriends', 'wanda', 'lenovo', 'metallica', 'stuartweitzman', 'karenwalker', 'celine', 'miui', 'montagut', 'pampers', 'darlie', 'toray', 'bobdog', 'ck', 'flyco', 'alexandermcqueen', 'shaxuan', 'prada', 'miiow', 'inman', '3t', 'gap', 'Yamaha', 'fjallraven', 'vancleefarpels', 'acne', 'audi', 'hunanweishi', 'henkel', 'mg', 'sony', 'CHAMPION', 'iwc', 'lv', 'dolcegabbana', 'avene', 'longchamp', 'anessa', 'satchi', 'hotwheels', 'nike', 'hermes', 'jiaodan', 'siemens', 'Goodbaby', 'innisfree', 'Thrasher', 'kans', 'kenzo', 'juicycouture', 'evisu', 'volcom', 'CanadaGoose', 'Dickies', 'angrybirds', 'eddrac', 'asics', 'doraemon', 'hisense', 'juzui', 'samsonite', 'hikvision', 'naturerepublic', 'Herschel', 'MANGO', 'diesel', 'hotwind', 'intel', 'arsenal', 'rayban', 'tommyhilfiger', 'ELLE', 'stdupont', 'ports', 'KOHLER', 'thombrowne', 'mobil', 'Belif', 'anello', 'zhoushengsheng', 'd_wolves', 'FridaKahlo', 'citizen', 'fortnite', 'beautyBlender', 'alexanderwang', 'charles_keith', 'panerai', 'lux', 'beats', 'Y-3', 'mansurgavriel', 'goyard', 'eral', 'OralB', 'markfairwhale', 'burberry', 'uno', 'okamoto', 'only', 'bvlgari', 'heronpreston', 'jimmythebull', 'dyson', 'kipling', 'jeanrichard', 'PXG', 'pinkfong', 'Versace', 'CCTV', 'paulfrank', 'lanvin', 'vans', 'cdgplay', 'baojianshipin', 'rapha', 'tissot', 'casio', 'patekphilippe', 'tsingtao', 'guess', 'Lululemon', 'hollister', 'dell', 'supor', 'MaxMara', 'metersbonwe', 'jeanswest', 'lancome', 'lee', 'omega', 'lets_slim', 'snp', 'PINKFLOYD', 'cartier', 'zenith', 'LG', 'monchichi', 'hublot', 'benz', 'apple', 'blackberry', 'wuliangye', 'porsche', 'bottegaveneta', 'instantlyageless', 'christopher_kane', 'bolon', 'tencent', 'dkny', 'aptamil', 'makeupforever', 'kobelco', 'meizu', 'vivo', 'buick', 'tesla', 'septwolves', 'samanthathavasa', 'tomford', 'jeep', 'canon', 'nfl', 'kiehls', 'pigeon', 'zhejiangweishi', 'snidel', 'hengyuanxiang', 'linshimuye', 'toread', 'esprit', 'BASF', 'gillette', '361du', 'bioderma', 'UnderArmour', 'TommyHilfiger', 'ysl', 'onitsukatiger', 'house_of_hello', 'baidu', 'robam', 'konka', 'jack_wolfskin', 'office', 'goldlion', 'tiantainwuliu', 'wonderflower', 'arcteryx', 'threesquirrels', 'lego', 'mindbridge', 'emblem', 'grumpycat', 'bejirog', 'ccdd', '3concepteyes', 'ferragamo', 'thermos', 'Auby', 'ahc', 'panasonic', 'vanguard', 'FESTO', 'MCM', 'lamborghini', 'laneige', 'ny', 'givenchy', 'zara', 'jiangshuweishi', 'daphne', 'longines', 'camel', 'philips', 'nxp', 'skf', 'perfect', 'toshiba', 'wodemeilirizhi', 'Mexican', 'VANCLEEFARPELS', 'HARRYPOTTER', 'mcm', 'nipponpaint', 'chenguang', 'jissbon', 'versace', 'girardperregaux', 'chaumet', 'columbia', 'nissan', '3M', 'yuantong', 'sk2', 'liangpinpuzi', 'headshoulder', 'youngor', 'teenieweenie', 'tagheuer', 'starbucks', 'pierrecardin', 'vacheronconstantin', 'peskoe', 'playboy', 'chanel', 'HarleyDavidson_AE', 'volvo', 'be_cheery', 'mulberry', 'musenlin', 'miffy', 'peacebird', 'tcl', 'ironmaiden', 'skechers', 'moncler', 'rimowa', 'safeguard', 'baleno', 'sum37', 'holikaholika', 'gucci', 'theexpendables', 'dazzle', 'vatti', 'nintendo')

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        ann_file=train_json_path,
        data_prefix=dict(img=train_img_prefix),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        ann_file=val_json_path,
        data_prefix=dict(img=val_img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        ann_file=test_json_path,
        data_prefix=dict(img=test_img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_json_path,
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + test_json_path,
    metric='bbox',
    backend_args=backend_args)

# training settings
max_epochs = 300
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

