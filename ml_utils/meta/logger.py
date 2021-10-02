from typing import AbstractSet, Dict
from abc import ABC, abstractmethod


class Logger(ABC):
    """Logger
    """
    def __init__(self):
        """Logger
        """
        pass
    
    @abstractmethod
    def log_iter(self, params: Dict):
        """Logs the params per iteration on NN training.

        Args:
            params (Dict): Parameters to log
        """
        pass
    
    @abstractmethod
    def log_epoch(self, params: Dict, epoch: int):
        """Logs the params per epoch of training

        Args:
            params (Dict): Parameters to log
            epoch (int): Epoch number
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict):
        """Logs parameters

        Args:
            params (Dict): Parameters to log
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict):
        """Logs training/val/test metrics

        Args:
            metrics (Dict): Key value pairs representing metric name and value.
                            eg.: {"accuracy": 0.99}
        """
        pass

    @abstractmethod
    def summarize_run(self):
        """Summarize the run. eg. log the max accuracy, minimum loss, etc.
        """
        pass

    @abstractmethod
    def commit_logs(self):
        """Commits the logs to remote/file system
        """
        pass


class LoggingError(Exception):
    """Custom exception for logging errors

    Args:
        Exception ([type]): Custom exception for logging errors
    """
    def __init__(self, message: str):
        """Custom exception for logging errors

        Args:
            message (str): Error message
        """
        self.message = message

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "LoggingError raised."