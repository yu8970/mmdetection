#!/bin/bash

python tools/train.py configs/abandon/c_atss_r101_fpn_2x_coco_tea.py \
--auto-scale-lr \
--work-dir ./logs