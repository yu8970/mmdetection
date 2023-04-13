_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
#teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa

#teacher_ckpt = '/root/autodl-tmp/logs/log_teacher_r101/epoch_17.pth'
teacher_ckpt = None
teacher_config = '/home/yu/mmlab/mmdetection/configs/a_yu/gfl_r101_fpn_mstrain_2x_coco_teacher.py'

dataset_type = 'CocoDataset'
classes = ('LAMY', 'tumi', 'warrior', 'sandisk', 'belle', 'ThinkPad', 'rolex', 'balabala', 'vlone', 'nanfu', 'KTM', 'VW', 'libai', 'snoopy', 'Budweiser', 'armani', 'gree', 'GOON', 'KielJamesPatrick', 'uniqlo', 'peppapig', 'valentino', 'GUND', 'christianlouboutin', 'toyota', 'moutai', 'semir', 'marcjacobs', 'esteelauder', 'chaoneng', 'goldsgym', 'airjordan', 'bally', 'fsa', 'jaegerlecoultre', 'dior', 'samsung', 'fila', 'hellokitty', 'Jansport', 'barbie', 'VDL', 'manchesterunited', 'coach', 'PopSockets', 'haier', 'banbao', 'omron', 'fendi', 'erke', 'lachapelle', 'chromehearts', 'leader', 'pantene', 'motorhead', 'girdear', 'fresh', 'katespade', 'pandora', 'Aape', 'edwin', 'yonghui', 'Levistag', 'kboxing', 'yili', 'ugg', 'CommedesGarcons', 'Bosch', 'palmangels', 'razer', 'guerlain', 'balenciaga', 'anta', 'Duke', 'kingston', 'nestle', 'FGN', 'vrbox', 'toryburch', 'teenagemutantninjaturtles', 'converse', 'nanjiren', 'Josiny', 'kappa', 'nanoblock', 'lincoln', 'michael_kors', 'skyworth', 'olay', 'cocacola', 'swarovski', 'joeone', 'lining', 'joyong', 'tudor', 'YEARCON', 'hyundai', 'OPPO', 'ralphlauren', 'keds', 'amass', 'thenorthface', 'qingyang', 'mujosh', 'baishiwul', 'dissona', 'honda', 'newera', 'brabus', 'hera', 'titoni', 'decathlon', 'DanielWellington', 'moony', 'etam', 'liquidpalisade', 'zippo', 'mistine', 'eland', 'wodemeiliriji', 'ecco', 'xtep', 'piaget', 'gloria', 'hp', 'loewe', 'Levis_AE', 'Anna_sui', 'MURATA', 'durex', 'zebra', 'kanahei', 'ihengima', 'basichouse', 'hla', 'ochirly', 'chloe', 'miumiu', 'aokang', 'SUPERME', 'simon', 'bosideng', 'brioni', 'moschino', 'jimmychoo', 'adidas', 'lanyueliang', 'aux', 'furla', 'parker', 'wechat', 'emiliopucci', 'bmw', 'monsterenergy', 'Montblanc', 'castrol', 'HUGGIES', 'bull', 'zhoudafu', 'leaders', 'tata', 'oldnavy', 'OTC', 'levis', 'veromoda', 'Jmsolution', 'triangle', 'Specialized', 'tries', 'pinarello', 'Aquabeads', 'deli', 'mentholatum', 'molsion', 'tiffany', 'moco', 'SANDVIK', 'franckmuller', 'oakley', 'bulgari', 'montblanc', 'beaba', 'nba', 'shelian', 'puma', 'PawPatrol', 'offwhite', 'baishiwuliu', 'lexus', 'cainiaoguoguo', 'hugoboss', 'FivePlus', 'shiseido', 'abercrombiefitch', 'rejoice', 'mac', 'chigo', 'pepsicola', 'versacetag', 'nikon', 'TOUS', 'huawei', 'chowtaiseng', 'Amii', 'jnby', 'jackjones', 'THINKINGPUTTY', 'bose', 'xiaomi', 'moussy', 'Miss_sixty', 'Stussy', 'stanley', 'loreal', 'dhc', 'sulwhasoo', 'gentlemonster', 'midea', 'beijingweishi', 'mlb', 'cree', 'dove', 'PJmasks', 'reddragonfly', 'emerson', 'lovemoschino', 'suzuki', 'erdos', 'seiko', 'cpb', 'royalstar', 'thehistoryofwhoo', 'otterbox', 'disney', 'lindafarrow', 'PATAGONIA', 'seven7', 'ford', 'bandai', 'newbalance', 'alibaba', 'sergiorossi', 'lacoste', 'bear', 'opple', 'walmart', 'clinique', 'asus', 'ThomasFriends', 'wanda', 'lenovo', 'metallica', 'stuartweitzman', 'karenwalker', 'celine', 'miui', 'montagut', 'pampers', 'darlie', 'toray', 'bobdog', 'ck', 'flyco', 'alexandermcqueen', 'shaxuan', 'prada', 'miiow', 'inman', '3t', 'gap', 'Yamaha', 'fjallraven', 'vancleefarpels', 'acne', 'audi', 'hunanweishi', 'henkel', 'mg', 'sony', 'CHAMPION', 'iwc', 'lv', 'dolcegabbana', 'avene', 'longchamp', 'anessa', 'satchi', 'hotwheels', 'nike', 'hermes', 'jiaodan', 'siemens', 'Goodbaby', 'innisfree', 'Thrasher', 'kans', 'kenzo', 'juicycouture', 'evisu', 'volcom', 'CanadaGoose', 'Dickies', 'angrybirds', 'eddrac', 'asics', 'doraemon', 'hisense', 'juzui', 'samsonite', 'hikvision', 'naturerepublic', 'Herschel', 'MANGO', 'diesel', 'hotwind', 'intel', 'arsenal', 'rayban', 'tommyhilfiger', 'ELLE', 'stdupont', 'ports', 'KOHLER', 'thombrowne', 'mobil', 'Belif', 'anello', 'zhoushengsheng', 'd_wolves', 'FridaKahlo', 'citizen', 'fortnite', 'beautyBlender', 'alexanderwang', 'charles_keith', 'panerai', 'lux', 'beats', 'Y-3', 'mansurgavriel', 'goyard', 'eral', 'OralB', 'markfairwhale', 'burberry', 'uno', 'okamoto', 'only', 'bvlgari', 'heronpreston', 'jimmythebull', 'dyson', 'kipling', 'jeanrichard', 'PXG', 'pinkfong', 'Versace', 'CCTV', 'paulfrank', 'lanvin', 'vans', 'cdgplay', 'baojianshipin', 'rapha', 'tissot', 'casio', 'patekphilippe', 'tsingtao', 'guess', 'Lululemon', 'hollister', 'dell', 'supor', 'MaxMara', 'metersbonwe', 'jeanswest', 'lancome', 'lee', 'omega', 'lets_slim', 'snp', 'PINKFLOYD', 'cartier', 'zenith', 'LG', 'monchichi', 'hublot', 'benz', 'apple', 'blackberry', 'wuliangye', 'porsche', 'bottegaveneta', 'instantlyageless', 'christopher_kane', 'bolon', 'tencent', 'dkny', 'aptamil', 'makeupforever', 'kobelco', 'meizu', 'vivo', 'buick', 'tesla', 'septwolves', 'samanthathavasa', 'tomford', 'jeep', 'canon', 'nfl', 'kiehls', 'pigeon', 'zhejiangweishi', 'snidel', 'hengyuanxiang', 'linshimuye', 'toread', 'esprit', 'BASF', 'gillette', '361du', 'bioderma', 'UnderArmour', 'TommyHilfiger', 'ysl', 'onitsukatiger', 'house_of_hello', 'baidu', 'robam', 'konka', 'jack_wolfskin', 'office', 'goldlion', 'tiantainwuliu', 'wonderflower', 'arcteryx', 'threesquirrels', 'lego', 'mindbridge', 'emblem', 'grumpycat', 'bejirog', 'ccdd', '3concepteyes', 'ferragamo', 'thermos', 'Auby', 'ahc', 'panasonic', 'vanguard', 'FESTO', 'MCM', 'lamborghini', 'laneige', 'ny', 'givenchy', 'zara', 'jiangshuweishi', 'daphne', 'longines', 'camel', 'philips', 'nxp', 'skf', 'perfect', 'toshiba', 'wodemeilirizhi', 'Mexican', 'VANCLEEFARPELS', 'HARRYPOTTER', 'mcm', 'nipponpaint', 'chenguang', 'jissbon', 'versace', 'girardperregaux', 'chaumet', 'columbia', 'nissan', '3M', 'yuantong', 'sk2', 'liangpinpuzi', 'headshoulder', 'youngor', 'teenieweenie', 'tagheuer', 'starbucks', 'pierrecardin', 'vacheronconstantin', 'peskoe', 'playboy', 'chanel', 'HarleyDavidson_AE', 'volvo', 'be_cheery', 'mulberry', 'musenlin', 'miffy', 'peacebird', 'tcl', 'ironmaiden', 'skechers', 'moncler', 'rimowa', 'safeguard', 'baleno', 'sum37', 'holikaholika', 'gucci', 'theexpendables', 'dazzle', 'vatti', 'nintendo')


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/root/autodl-tmp/brand20/anns/train.json',
        img_prefix='/root/autodl-tmp/brand20/images/train'),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/root/autodl-tmp/brand20/anns/val.json',
        img_prefix='/root/autodl-tmp/brand20/images/val'),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/root/autodl-tmp/brand20/anns/test.json',
        img_prefix='/root/autodl-tmp/brand20/images/test'))

model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHead',
        num_classes=515,    # 此处修改
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 10])




