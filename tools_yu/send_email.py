
# 链接：https://www.zhihu.com/question/488793035/answer/2286337775


import time
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
class Mylog:
    def __init__(self):
        time_info=str(time.strftime("%m_%d_%H", time.localtime()))
        if not os.path.exists('./runlogs/'):
            os.mkdir('./runlogs/')
        self.file_name='./runlogs/'+time_info+'.log'
        self.file=open(self.file_name, 'a')
    def __del__(self):
        self.file.close()
    #添加日志记录
    def add_log(self,lg):
        time_info = str(time.strftime("%H-%M-%S -->  ", time.localtime()))
        self.file.write(time_info+lg)
        self.file.write('\r')
        self.file.flush()
    #运行结束后发送邮件
    def send_mail(self):
        self.file.close()
        from_addr = ''  # 邮件发送账号
        to_addrs = ''  # 接收邮件账号
        qqCode = ''  # 授权码（这个要填自己获取到的）
        smtp_server = 'smtp.qq.com'
        smtp_port = 465
        # 配置服务器
        stmp = smtplib.SMTP_SSL(smtp_server, smtp_port)
        stmp.login(from_addr, qqCode)
        with open(self.file_name,'r') as f:
            buffer=f.read()
        # 组装发送内容
        message = MIMEText(buffer, 'plain', 'utf-8')  # 发送的内容
        message['From'] = Header("autodl", 'utf-8')  # 发件人
        message['To'] = Header("me", 'utf-8')  # 收件人
        subject = 'AUTODL-运行结束'
        message['Subject'] = Header(subject, 'utf-8')  # 邮件标题
        try:
            stmp.sendmail(from_addr, to_addrs, message.as_string())
            print('邮件发送成功')
        except Exception as e:
            print('邮件发送失败--' + str(e))