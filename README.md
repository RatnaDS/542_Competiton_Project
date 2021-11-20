# ECE542: Competition Project

## Team 11
- Chinmay Mahendra Savadikar (csavadi)
- Sahil Anish Palarpwar (spalarp)
- Sai Ratna Dabhamalla (sdarbha)

For ease of colaboration between the team, we have developed the common functionality (like custom PyTorch Dataset, preprocessing, and splitting code) as a package. This code lives in a private GitHub repository, and is cloned onto Colab using a personal access token (this will only work if access to the private repository is given). The packages can be imported in noteboks to make them cleaner. This way, the common tasks like data preprocessing and splitting can be done locally and the same data and code version can be shared within the team. 

This is not necessary if the code is copied to the Colab VM manually. The code can be imported in the notebook without cloning the repository using the following steps:
1. Copy the folder ```ece542-competition-project``` to the the desired location.
2. Add the following lines to the start of the notebook:
   ```python
   import sys
   sys.path.append("<PATH_TO_PARENT_DIR>/ece542-competition-project")
   ```
   Where ```<PATH_TO_PARENT_DIR>``` is the absolute path to the directory where the folder ```ece542-competition-project``` containing the code is copied.

### Dependencies
The file ```requirements.txt``` contains the libraries required for the project. We use ```PyTorch``` for training the models, and ```scikit-learn``` for the calculation of metrics. We also use ```pandas``` for reading and writing csv files and ```matplotlib``` for plotting.

### Following are the packages we built for the project:
- **data_utils**
  
  Contains the code for data loading, preprocessing and splitting.
  - ```dataloader.py```: Reads the *x* and *y* csv files for a session and adds headers the the dataframes.
  - ```dataloader_test.py```: Reads the *x* csv files for the hidden test data, adds headers to the dataframe, and creates a dummy y dataframe which is needed for the custom PyTorch Dataset to function. These dummy labels are only for convenience, and are discarded later.
  - ```preprocessor.py```: Formats the data by calculating windows of a specified interval. Each window is assigned a timestamp, which corresponds to a label in the *y* csv file. This converts the raw data into a format desired by the custom dataset.
  - ```splitting.py```: Code to calculate which sessions go into train, validation and test splits. DataSplitter.split_ids returns a dictionary containing a list of sessions to be put in train, validation and test sets.

- **ml_utils**
  
  Contains the code for the custom PyTorch Dataset and plotting the training and validation curves.

  - ```dataset.py```:
      
      SubjectDataset: Dataset which returns samples by considering each timestep indepentent.
      
      SequentialSubjectDataset: Dataset which returns a sequence of labels and asociated windowed X. This is required for trianing RNN models.

  - ```plotter.py```: Plots the training and validation curves.


### Training Notebooks
The folder experiments/training_notebooks contains the notebooks used to train the final 3 models, which form the ensemble. All the notebooks have been downloaded from Google Colab.

- **```experiments/training_notebooks/window_1sec_base_filters_16.ipynb```**: The notebook which trains the 1D CNN model with filters in the first convolutional layer=16, with window size = 1sec, i.e., the sample 0.5 seconds before and 0.5 seconds after the "time" of the label are considered in making the prediction. The filters are doubled in every layer (4 layers). [Termed as ```Model 1``` in the report]
- **```experiments/training_notebooks/window_1sec_base_filters_32.ipynb```**: The notebook which trains the 1D CNN model with filters in the first convolutional layer=32, with window size = 1sec. The filters are doubled in every layer (4 layers). [Termed as ```Model 2``` in the report]
- **```experiments/training_notebooks/window_3sec_base_filters_32.ipynb```**: The notebook which trains the 1D CNN model with filters in the first convolutional layer=32, with window size = 3sec, i.e., the sample 1.5 seconds before and 1.5 seconds after the "time" of the label are considered in making the prediction. The filters are doubled in every layer (4 layers). [Termed as ```Model 3``` in the report]


### Prediction Notebooks
The folder experiments/prediction_notebooks contains the notebooks used to make the predictions on labelled or unlabelled data.

- **```experiments/prediction_notebooks/ensemble_prediction_labelled_data.ipynb```**: Notebook which generates the predictions using the ensemble of the 3 models on labelled data. This notebook also calculates the metrics (Accuracy, Precision, Recall ad F1).
- **```experiments/prediction_notebooks/ensemble_prediction_test_data.ipynb```**: Notebook which generates the predictions using the ensemble of the 3 models on unlabelled (hidden test) data.


### Steps for running the code:
1. Copy the data in the directory ```data```.
2. Run the notebook data_preprocessing_final.ipynb to convert the data from raw data format to windowed format (each label will hae a window of X associated with it).
3. Run the notebook compute_splits.ipynb to generate a list of session ids to be put in training, validation and test splits. The list is stored in a json file so that all the members of the team can use the same data.
4. Run the notebook splitting_data.ipynb to copy the files into respective folders.
5. Run the notebook statistics.ipynb to calculate the minimum and maximum values of the *x*, *y* and *z* co-ordinates for the accelerometer and gyroscope. These are used for scaling the values before passing to the neural network.
6. Run any of the training notebooks. The training notebooks also contain the code for evaluating the models on the validation and test sts.
7. Run the prediction notebooks which use to generate predictions using the ensemble of the models.


### References:
[1] B. Zhong, R. L. d. Silva, M. Li, H. Huang and E. Lobaton, "Environmental Context Prediction for Lower Limb Prostheses With Uncertainty Quantification," in IEEE Transactions on Automation Science and Engineering, vol. 18, no. 2, pp. 458-470, April 2021, doi: 10.1109/TASE.2020.2993399.