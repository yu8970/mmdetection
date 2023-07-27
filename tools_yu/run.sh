#!/bin/bash

python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py \
--auto-scale-lr \
--work-dir ./logs