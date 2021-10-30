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

    def __init__(self, datapath: str, ids: list, cache_len: int=1):
        
        self.ids = ids
        self.datapath = datapath
        self.cache_len = cache_len
        self.y_files = {uid: os.path.join(self.datapath, SUBJECT_Y_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        self.x_files = {uid: os.path.join(self.datapath, SUBJECT_X_TEMPLATE.format(parse_uid(uid)[0], parse_uid(uid)[1])) for uid in self.ids}
        
        # Generate a list of samples and determine the number of datapoints in the dataset 
        # and build up the cache
        self.build_cache_and_datalen()

        assert self.cache_len < self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        # Get the sample info from the index
        info = self.index_store[self.index_store["index"] == index].reset_index(drop=True)
        # Get the uid
        uid = info["uid"][0]
        # If the uid matches the uid of the file in memory, no need to read from disk
        if uid in self.uid_cache:
            # Return the sample at timestamp
            X = self.cache[uid]["X"]
            y = self.cache[uid]["y"]
        else:
            # Evict a file from the cache
            uid_to_evict = self.uid_cache.pop()
            del self.cache[uid_to_evict]
            # Read from disk and refresh cache
            X = pd.read_csv(self.x_files[uid])
            y = pd.read_csv(self.y_files[uid])
            self.cache[uid] = {"X": X, "y": y}
        timestamp = info["timestamp"][0]
        X = X[X["timestamp"] == timestamp][X_HEADER]
        y = y[y["timestamp"] == timestamp][Y_HEADER]

        inputs = X.values.T # Convert to channel first
        
        labels = y.values.astype(int).flatten() # Make 1 dimensional for classification

        return torch.from_numpy(inputs), torch.from_numpy(labels)

    def build_cache_and_datalen(self):

        self.index_store = pd.DataFrame({
            "label": [],
            "timestamp": [],
            "uid": [],
            "index": []
        })

        self.uid_cache = []
        self.cache = dict()
        num_samples = 0
        for i, (uid, y_file) in enumerate(self.y_files.items()):
            y = pd.read_csv(y_file)
            n_samples = len(y)
            
            sample_info = pd.DataFrame({
                "label": y["label"],
                "timestamp": y["timestamp"],
                "uid": [uid for _ in range(n_samples)],
                # Assigns a unique index to the sample, irrespective of the subject and session id
                "index": [num_samples + i for i in range(n_samples)]
            })
            self.index_store = pd.concat([self.index_store, sample_info], axis=0).reset_index(drop=True)
            num_samples += n_samples

            if i < self.cache_len:
                x_file = self.x_files[uid]
                self.uid_cache.append(uid)
                X = pd.read_csv(x_file)
                self.cache[uid] = {"X": X, "y": y}
        
        self.num_samples = num_samples
