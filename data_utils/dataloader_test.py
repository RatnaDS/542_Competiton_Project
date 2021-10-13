from typing import Tuple
import os
import pandas as pd
from . import X_HEADER, Y_HEADER


SUBJECT_X_TEMPLATE = "subject_{}_{}__x.csv"
SUBJECT_X_TIME_TEMPLATE = "subject_{}_{}__x_time.csv"
SUBJECT_Y_TEMPLATE = "subject_{}_{}__y.csv"
SUBJECT_Y_TIME_TEMPLATE = "subject_{}_{}__y_time.csv"


class Dataloader:

    def __init__(self, data_base_path: str) -> None:
        """Data Loader

        Args:
            data_base_path (str): Path the the base folder
        """
        self.data_base_path = data_base_path

    def load_and_join_data(self, subject_id: str, session_number: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read data from files

        Args:
            subject_id (str): Subject ID. eg. "001"
            session_number (str): Session number. eg. 01

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: x and y
        """
        x_path = os.path.join(self.data_base_path, SUBJECT_X_TEMPLATE.format(subject_id, session_number))
        x_time_path = os.path.join(self.data_base_path, SUBJECT_X_TIME_TEMPLATE.format(subject_id, session_number))

        #y_path = os.path.join(self.data_base_path, SUBJECT_Y_TEMPLATE.format(subject_id, session_number))
        y_time_path = os.path.join(self.data_base_path, SUBJECT_Y_TIME_TEMPLATE.format(subject_id, session_number))
        
        x_data = pd.read_csv(x_path, names=X_HEADER)
        x_time = pd.read_csv(x_time_path, names=["time"])
        
        
        y_time = pd.read_csv(y_time_path, names=["time"])
        y_data = pd.DataFrame({'label': [-1]*y_time.shape[0]})
        
        x = pd.concat([x_time, x_data], axis=1)
        y = pd.concat([y_time, y_data], axis=1)
        y = y.reset_index(drop=False).rename(columns={"index": "timestamp"})
        
        return x, y
