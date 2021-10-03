from typing import Union
import pandas as pd
import numpy as np
from . import DataConstants, X_HEADER


class Preprocessor:

    def __init__(self) -> None:
        pass

    def get_windowed_data(self, x: pd.DataFrame, 
                   interval: float, 
                   y: Union[pd.DataFrame, None]=None, 
                   window_type: str="centered") -> pd.DataFrame:
        """Slice session data into intervals

        Args:
            x (pd.DataFrame): Data from a session
            interval (float): Time interval in seconds
            y (Union[pd.DataFrame, None], optional): Label reference for windowing. This will only be used during training. Defaults to None.
            window_type (str, optional): Type of windowing. Only used when y is not None. Should be one of "centered" or "trailing". Defaults to "centered".
        """
        assert window_type in ["centered", "trailing"]
        if y is None:
            windowed_data = self.get_windows_without_labels(x, interval, window_type)
        else:
            windowed_data = self.get_windows_with_labels(x, y, interval, window_type)
        return windowed_data

    def get_windows_without_labels(self, 
                                   x: pd.DataFrame, 
                                   interval: float, 
                                   window_type: str="centered") -> np.array:
        if window_type == "centered":
            centers = x["time"]
            windowed_data = self.trunc_centered(x, centers, interval)
        else:
            windowed_data = self.trunc_trailing()
        return windowed_data

    def get_windows_with_labels(self, 
                                x: pd.DataFrame, 
                                y: pd.DataFrame, 
                                interval: float, 
                                window_type: str="centered") -> pd.DataFrame:
        
        if window_type == "centered":
            centers = y["time"]
            windowed_signal = self.trunc_centered(x, centers, interval)
        else:
            windowed_signal = self.trunc_trailing()
        return windowed_signal

    def trunc_centered(self, x: pd.DataFrame, centers: pd.Series, interval: float) -> pd.DataFrame:
        windowed_signals = []
        for i, center in enumerate(centers):
            center = float(center)
            window_start = center - interval / 2
            window_end = center + interval / 2

            windowed_signal = self.retrieve_window(x, window_start, window_end, interval)
            windowed_signal["timestamp"] = [i for _ in range(len(windowed_signal))]
            windowed_signals.append(windowed_signal)
        return pd.concat(windowed_signals, axis=0).reset_index(drop=True)

    def trunc_trailing(self):
        pass

    def retrieve_window(self, x: pd.DataFrame, window_start: float, window_end: float, interval: int):

        windowed_signal = x[(x["time"] > window_start) & (x["time"] < window_end)][X_HEADER]
        expected_num_samples = interval * DataConstants.SAMPLING_RATE_X

        if window_start < x.loc[0, "time"]:
            # Padding at the start of the signal
            num_required_samples = expected_num_samples - len(windowed_signal)
            repeated_sample = x.head(1)[X_HEADER]
            padding_df = pd.DataFrame(np.repeat(repeated_sample.values, num_required_samples, axis=0))
            padding_df.columns = X_HEADER
            # Pad at the begining
            windowed_signal = pd.concat([padding_df, windowed_signal], axis=0)
            windowed_signal = windowed_signal.reset_index(drop=True)

        if window_end > x.loc[len(x)-1, "time"]:
            # Padding at the end of the signal
            num_required_samples = expected_num_samples - len(windowed_signal)
            repeated_sample = x.tail(1)[X_HEADER]
            padding_df = pd.DataFrame(np.repeat(repeated_sample.values, num_required_samples, axis=0))
            padding_df.columns = X_HEADER
            # Pad at the end
            windowed_signal = pd.concat([windowed_signal, padding_df], axis=0)
            windowed_signal = windowed_signal.reset_index(drop=True)

        assert expected_num_samples == len(windowed_signal)
        return windowed_signal
