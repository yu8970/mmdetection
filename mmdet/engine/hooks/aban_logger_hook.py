from mmdet.registry import HOOKS
from mmengine.hooks.logger_hook import LoggerHook
from typing import Dict, Optional
import requests

@HOOKS.register_module()
class AbanLoggerHook(LoggerHook):

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        self.send_to_wx(log_str)
        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                iter = 0
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(
                tag, step=iter, file_path=self.json_log_path)
    def send_to_wx(self, log):
        headers = {
            "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjY5MTQ0LCJ1dWlkIjoiMGE3NmM4NmYtYWNmNi00Zjg5LWJkY2ItOTY5MjE2ZjlmYmJjIiwiaXNfYWRtaW4iOmZhbHNlLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.mj_3YuXAYmNWwNIf3OI0HOO6iFRLYoVN9U1mVJhl3AFFXd5cUMDR6Y0_OWxO4po1KXbts5WQrPRUQbL2W4mYgw"}
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                             json={
                                 "title": "来自mmdetection",
                                 "name": log,
                                 "content": "Epoch"
                             }, headers=headers)
        print(resp.content.decode())