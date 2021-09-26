from typing import Dict


class Logger:
    """Logger
    """
    def __init__(self):
        """Logger
        """
        pass

    def log_iter(self, params: Dict):
        """Logs the params per iteration on NN training.

        Args:
            params (Dict): Parameters to log
        """
        pass

    def log_epoch(self, params: Dict):
        """Logs the params per epoch of training

        Args:
            params (Dict): Parameters to log
        """
        pass

    def commit_logs(self):
        """Commits the logs to remote/file system
        """
        pass
