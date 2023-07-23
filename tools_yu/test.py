# import mmdet
# print()
# print(mmdet.__version__)

import torch
import torch.nn.functional as F



import requests
headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjY5MTQ0LCJ1dWlkIjoiMGE3NmM4NmYtYWNmNi00Zjg5LWJkY2ItOTY5MjE2ZjlmYmJjIiwiaXNfYWRtaW4iOmZhbHNlLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.mj_3YuXAYmNWwNIf3OI0HOO6iFRLYoVN9U1mVJhl3AFFXd5cUMDR6Y0_OWxO4po1KXbts5WQrPRUQbL2W4mYgw"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "来自我的程序",
                         "name": "我的ImageNet实验Epoch=100. Acc=90.2",
                         "content": "Epoch=100. Acc=90.2"
                     }, headers = headers)
print(resp.content.decode())