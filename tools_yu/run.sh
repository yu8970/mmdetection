#!/bin/bash

python tools/train.py configs/abandon/atss_r50_fpn_1x_coco.py \
--auto-scale-lr \
--work-dir /content/drive/MyDrive/grad/mmdetection/logs