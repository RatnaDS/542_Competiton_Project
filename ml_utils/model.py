from typing import Union, Iterable, Dict
from pandas import DataFrame
from numpy import ndarray


class Model:
    """A unified API for model definition
    """
    def __init__(self):
        self.model = None

    def predict(self, x: Union[DataFrame, ndarray, Iterable]) -> Union[DataFrame, ndarray]:
        """Generate predictions from data.

        Args:
            x (Union[DataFrame, ndarray, Iterable]): Data

        Raises:
            NotImplementedError: Base class raises NotImplementedError to force child classes
                                 to implement the method

        Returns:
            Union[DataFrame, ndarray]: Predictions
        """
        
        raise NotImplementedError()

    def train(self, x: Union[DataFrame, ndarray, Iterable], 
              y: Union[DataFrame, ndarray, Iterable]) -> Dict:
        """Implements the model training

        Args:
            x (Union[DataFrame, ndarray, Iterable]): Inputs
            y (Union[DataFrame, ndarray, Iterable]): Labels

        Raises:
            NotImplementedError: Base class raises NotImplementedError to force child classes
                                 to implement the method

        Returns:
            Dict: Dictionary containing training parameters
        """

        raise NotImplementedError()
