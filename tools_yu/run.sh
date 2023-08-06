#!/bin/bash

#python tools/train.py configs/abandon/c_atss_r101_fpn_2x_coco_tea.py \
#--auto-scale-lr \
#--work-dir ./logs

#python tools/train.py configs/tood/tood_r50_fpn_1x_coco.py \
#--auto-scale-lr \
#--work-dir ./logs

python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py \
--auto-scale-lr \
--work-dir ./logs