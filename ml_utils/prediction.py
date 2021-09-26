from typing import List, Union, Callable
from numpy import ndarray
from pandas import DataFrame

class Predictor:
    """A unified API for generating predictions from models
    """
    def __init__(self):
        pass

    def predict(self, x):
        pass


class EnsemblePredictor:
    """A unified API for generating predictions from ensembles
    """
    def __init__(self, models: List[Predictor]):
        """A unified API for generating predictions from ensembles

        Args:
            models (List[Predictor]): A list of individual predictors
        """
        pass

    def predict(self, x:Union[DataFrame, ndarray], combine: Union[str, Callable]="average", mode: str="batch") -> Union[DataFrame, ndarray]:
        """Method to generate predictions on data

        Args:
            x (Union[DataFrame, ndarray]): Input data
            mode (str, optional): One sample, all predictors or all samples 1 predictor. Defaults to "batch".
            combine (Union[str, Callable], optional): Method to combine the predictions from individual predictors. Defaults to "average".

        Returns:
            Union[DataFrame, ndarray]: predictions from individial predictors and combined predictions.
        """
        pass
