# ECE542: Competition Project

### Team 11
- Chinmay Mahendra Savadikar (csavadi)
- Sahil Anish Palarpwar (spalarp)
- Sai Ratna Dabhamalla (sdarbha)

For ease of colaboration between the team, we have developed the common functionality like custom PyTorch Dataset, preprocessing, and splitting code as a package. This code lives in a private GitHub repository, and is cloned onto Colab using an access token. The packages can be imported in noteboks to make them cleaner. This way, the common tasks like data preprocessing and splitting can be done locally and the same data can be shared within the team.

### Following are the packages:
- data_utils
  Contains the code for data loading, preprocessing and splitting.
  - dataloader.py: Reads the *x* and *y* csv files for a session and adds headers the the dataframes.
  - dataloader_test.py: Reads the *x* csv files for the hidden test data, adds headers to the dataframe, and creates a dummy y dataframe which is needed for the custom PyTorch Dataset to function. These dummy labels are only for convenience, and are discarded later.
  - preprocessor.py: Formats the data by calculating windows of a specified interval. Each window is assigned a timestamp, which corresponds to a label in the *y* csv file.
  - splitting.py: Code to calculate which sessions go into train, validation and test splits. DataSplitter.split_ids returns a dictionary containing a list of sessions to be put in train, validation and test sets.

- ml_utils
  Contains the code for the custom PyTorch Dataset and plotting the training and validation curves.