from typing import Dict
import random
from copy import deepcopy
from math import floor
import pandas as pd


# Set the random seed
random.seed(42)


class DataSplitter:

    def __init__(self):
        pass

    def split_ids(self, y: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Function to split the data
           Leave some sessions out as val and test
        Args:
            y (pd.DataFrame): DataFrame with IDs and Labels
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing the keys as "train", "val", "test" and the corresponding IDs.
        """
        uids = y["uid"].unique().tolist()

        num_ids = len(uids)
        # Train + val
        num_train_val = floor(num_ids * 0.7)
        # Test
        num_test = num_ids - num_train_val

        # Train
        num_train = floor(num_train_val * 0.8)
        num_val = num_train_val - num_train

        train_val_ids = deepcopy(uids[:num_train_val])
        test_ids = deepcopy(uids[num_train_val:])
        
        random.shuffle(train_val_ids)
        train_ids = train_val_ids[:num_train]
        val_ids = train_val_ids[num_train:]

        assert len(train_ids) == num_train
        assert len(val_ids) == num_val
        assert len(test_ids) == num_test
        assert len(train_ids) + len(val_ids) + len(test_ids) == len(uids)

        return {"train": train_ids, "val": val_ids, "test": test_ids}
    