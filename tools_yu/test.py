# import mmdet
# print()
# print(mmdet.__version__)

import torch
import torch.nn.functional as F




# 假设inter_feats包含两个张量
inter_feats = [torch.randn(2, 3, 5), torch.randn(2, 4, 5)]

# 在维度1上拼接张量
result = torch.cat(inter_feats, 1)

print(result)