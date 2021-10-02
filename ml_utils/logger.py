from typing import Dict
import wandb
from .meta.logger import Logger, LoggingError


class WandBLogger(Logger):

    def __init__(self, project: str, run_name: str, config: Dict, group: str=None, notes: str=None):
        super(WandBLogger, self).__init__()

        self.project = project
        self.run_name = run_name
        self.group = group
        self.notes = notes
        self.config = config

        self.run = wandb.init(project=self.project, name=self.run_name, 
                              group=self.group, notes=self.notes, config=config)
        self.active = True

    def __del__(self):
        if self.active:
            self.run.finish()
            self.active = False

    def commit_logs(self):
        # Implement commit code here

        # Finish the run
        self.run.finish()
        # Set active to false
        self.active = False
    
    def log_metrics(self, metrics: Dict):
        if self.active:
            wandb.log(metrics)
        else:
            raise LoggingError("Logger is inactive. Check if your logs have already been committed or if you have set logger.active to False.")
    
    def log_epoch(self, params: Dict, epoch: int):
        wandb.log(params, step=epoch)

    def log_iter(self, params: Dict):
        pass

    def log_params(self, params: Dict):
        pass

    def summarize_run(self, summary_params: Dict):
        for param_name, param_value in summary_params.items():
            wandb.run.summary[param_name] = param_value
