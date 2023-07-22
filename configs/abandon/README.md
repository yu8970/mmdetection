# 对照实验
A组是实验组，B组是对照组。
## A组
### teacher
tea的head增加了解耦模块，先单独训练，然后用解耦模块蒸馏stu；
tea是基于gfl_head改的
### student
stu的head增加了和tea解耦部分计算的loss；
stu是基于ld_head(ld_head=gfl_head+ld)改的

## B组
### teacher
tea是gfl_head
### student
stu是ld_head(ld_head=gfl_head+ld)
