import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import X_HEADER, Y_HEADER


SUBJECT_X_TEMPLATE = "subject_{}_session_{}__x.csv"
SUBJECT_X_TIME_TEMPLATE = "subject_{}_session_{}__x_time.csv"
SUBJECT_Y_TEMPLATE = "subject_{}_session_{}__y.csv"
SUBJECT_Y_TIME_TEMPLATE = "subject_{}_session_{}__y_time.csv"


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
        labels = np.array([self.y[index]])

        return torch.from_numpy(inputs), torch.from_numpy(labels)

    def build_cache_and_datalen(self):

        num_samples = 0
        timesteps = None

        X_list = []
        y_list = []

        for uid, y_file in self.y_files.items():
            print(f"Converting uid {uid}")
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

        values = df[X_HEADER].values
        X = values.reshape(len_array, timesteps, len(X_HEADER))
        
        return np.transpose(X, axes=(0, 2, 1)).copy()


class SequentialSubjectDataset(Dataset):

    def __init__(self, datapath: str, ids: list, sequence_length: int=10, phase: str="train", num_classes: int=4):
        
        assert phase in ["train", "test"]
        self.phase = phase
        self.num_classes = num_classes
        self.ids = ids
        self.datapath = datapath
        self.y_files = {uid: os.path.join(self.datapath, SUBJECT_Y_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.x_files = {uid: os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.sequence_length = sequence_length
        
        # Generate a list of samples and determine the number of datapoints in the dataset 
        # and build up the cache
        self.build_cache_and_datalen()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        if self.phase == "train":
            
            lower = index - self.sequence_length

            if lower < 0:
                num_extra = abs(lower)
                padding_X = np.zeros((num_extra, self.extra_timesteps, self.feature_size), dtype=np.float32)
                padding_y = -2*np.ones((num_extra,), dtype=np.int64)

                inputs = np.concatenate([padding_X, self.X[:index, :, :]], axis=0)
                labels = np.concatenate([padding_y, self.y[:index]], axis=0)
            else:
                inputs = self.X[lower:index, :, :]
                labels = self.y[lower:index]

            mask = labels != -2

            return_values = torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(mask)

        else:
            inputs = self.X[index, :]
            labels = self.y[index]
            
            # Remove the dummy appended values
            valid_indices = labels != -2
            inputs = inputs[valid_indices]
            labels = labels[valid_indices]

            return_values = torch.from_numpy(inputs), torch.from_numpy(labels)
        
        return return_values

    def build_cache_and_datalen(self):

        num_samples = 0
        timesteps = None

        X_list = []
        y_list = []

        for uid, y_file in self.y_files.items():
            print(f"Converting uid {uid}")
            y = pd.read_csv(y_file)
            n_samples = len(y)
            num_samples += n_samples

            x_file = self.x_files[uid]
            X_dataframe = pd.read_csv(x_file)
            if timesteps is None:
                _sample = X_dataframe[X_dataframe["timestamp"] == 0]
                timesteps = len(_sample)

            # Convert to numpy
            X, y = self.dataframe_to_numpy(X_dataframe, timesteps, y)

            X_list.append(X)
            y_list.append(y)

        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0).astype(np.int64)
        assert X.shape[0] == y.shape[0]

        self.num_samples, self.extra_timesteps, self.feature_size = X.shape

        if self.phase == "test":
            # Group timesteps together
            steps = self.num_samples // self.sequence_length
            X = X.reshape(steps, self.sequence_length, self.extra_timesteps, self.feature_size)
            y = y.reshape(steps, self.sequence_length)

        self.X = X.copy()
        self.y = y.copy()

        self.num_samples = X.shape[0]
        assert self.num_samples == self.y.shape[0]

    def dataframe_to_numpy(self, df, timesteps, y_df):
        """Convert from pandas to numpy for faster access
        """
        len_array = int(len(df) / timesteps)
        assert len_array == len(y_df)

        values = df[X_HEADER].values
        y = y_df["label"].values

        X = values.reshape(len_array, timesteps, len(X_HEADER))

        num_samples, timesteps, feature_size = X.shape

        steps = num_samples // self.sequence_length
        if (num_samples % self.sequence_length) != 0:
            steps += 1
            # Append some dummy values to X and y. Remove in __getitem__
            num_additional = steps*self.sequence_length - num_samples
            
            X = np.concatenate([X, np.zeros((num_additional, timesteps, feature_size))], axis=0)
            y = np.concatenate([y, -2*np.ones((num_additional,), dtype=np.int64)], axis=0)
        
        return X.copy(), y.copy()

    def collate_batch(self, batch):
        if self.phase == "train":
            inputs, outputs, masks = zip(*batch)
            return torch.stack(inputs), torch.stack(outputs), torch.stack(masks)
        else:
            inputs, outputs = zip(*batch)
            return torch.stack(inputs), torch.stack(outputs)
