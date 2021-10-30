import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


SUBJECT_X_TEMPLATE = "subject_{}_session_{}__x.csv"
SUBJECT_X_TIME_TEMPLATE = "subject_{}_session_{}__x_time.csv"
SUBJECT_Y_TEMPLATE = "subject_{}_session_{}__y.csv"
SUBJECT_Y_TIME_TEMPLATE = "subject_{}_session_{}__y_time.csv"

X_HEADER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
Y_HEADER = ["label"]


def parse_uid(uid):
    subject_id, session_num = uid.split("_")
    # return int(subject_id), int(session_num)
    return subject_id, session_num


class SubjectDataset(Dataset):

    def __init__(self, datapath: str, ids: list):
        
        self.ids = ids
        self.datapath = datapath
        self.y_files = {uid: os.path.join(self.datapath, SUBJECT_Y_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.x_files = {uid: os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        
        # Generate a list of samples and determine the number of datapoints in the dataset 
        # and build up the cache
        self.build_cache_and_datalen()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        inputs = self.X[index]
        labels = self.y[index]

        return torch.from_numpy(inputs), torch.from_numpy(labels)

    def build_cache_and_datalen(self):

        num_samples = 0
        timesteps = None

        X_list = []
        y_list = []

        for uid, y_file in self.y_files.items():
            y = pd.read_csv(y_file)
            n_samples = len(y)
            num_samples += n_samples

            x_file = self.x_files[uid]
            X_dataframe = pd.read_csv(x_file)
            if timesteps is None:
                _sample = X_dataframe[X_dataframe["timestamp"] == 0]
                timesteps = len(_sample)

            # Convert to numpy
            X = self.dataframe_to_numpy(X_dataframe, timesteps, y)

            X_list.append(X)
            y_list.append(y["label"].values)

        self.X = np.concatenate(X_list, axis=0).astype(np.float32)
        self.y = np.concatenate(y_list, axis=0).astype(int)
        assert self.X.shape[0] == self.y.shape[0]
        
        self.num_samples = self.X.shape[0]

    def dataframe_to_numpy(self, df, timesteps, y_df):
        """Convert from pandas to numpy for faster access
        """
        len_array = int(len(df) / timesteps)
        assert len_array == len(y_df)

        X = np.zeros((len_array, len(X_HEADER), timesteps))
        unique_timestamps = y_df["timestamp"].tolist()
        for i, timestamp in enumerate(unique_timestamps):
            _X = df[df["timestamp"] == timestamp].values.T
            X[i] = _X
        
        return np.expand_dims(X, axis=0) # Add a batch dimension and return
