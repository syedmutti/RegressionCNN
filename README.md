# Offset Values Prediction Network

Original repo : https://github.com/Nuclearstar/End-to-End-Learning-for-Self-Driving-Cars/tree/master/Implementation_2

## Dataset Link 
https://github.com/SullyChen/driving-datasets

Or at : /mrtstorage/users/rehman/datasets/driving_dataset

## Running 

- Use python train.py to train the model with 2 input images and 3 outputs.

- Use python run_dataset.py to run the model on the dataset


### Expected Label format 

    InputBaseName.extension offsetvalue1 offsetvalue2 offsetvalue3
    Example :     2345.jpg 34.2 56.5 94.3


## Required Changes for Training

- In driving_data.py 
    - Change absolute path for folders containing each input grid-map
    - Append correct value to each input list.

## Required Changes for Evaluation on Dataset

- In run_dataset.py 
    - give corresponding absolute paths for both inputs in feed_dict
