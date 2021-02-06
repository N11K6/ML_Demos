# Cough detection using a Convolutional Neural Network

As part of my work in the Eupnoos Audio & ML team, I have been developing a module for cough monitoring and diagnosis via smartphone. 
The full-scale implementation is intended to work as an overnight monitoring application, analyzing audio for cough and other respiratory events.

I can openly share the initial machine learning stage of back-end, which performs a classification between cough events to be used for further analysis, and all other audio events that are not of interest.

### Context

Given an input stream of recorded audio, we need to identify audio events and classify them as either coughing or something else. 
The machine learning model for the binary classification of events is presented here. 
The application can then send the audio identified as coughing for further processing to provide a diagnosis. 


Allocating and using data of medical interest is always complicated, but for this initial task, a collection containing audio of coughing is sufficient. 
The code include here covers the processing of audio files necessary to create our training dataset. 
The features extracted are the MFCCs calculated for each audio sample, stored as 3-dimensional numpy arrays. 
Using this approach we essentially convert our audio data into an image dataset, while maintaining all the information useful for the classification task.
Classification is then performed using a standard Convolutional Neural Network.

### Contents

**Notebook_1_Assembling_Dataset.ipynb** and **Notebook_2_TrainingCNN.ipynb** are the Jupyter notebooks containing the steps from the creation of the dataset, the feature generation, and the training and testing of the CNN. It is recommended that you approach this project via the notebooks.

**train.py** is the Python3 script for building and training the model. It can be run on its own, provided the directory structure is appropriate, or used to import each function individually.

*cough_detection_helper_functions.py* contains the functions used in the dataset generation process in Python3 scripts. It is usually recommended, however, to perform this task without much automation, since there might be great variability and inconsistency between available data.

*model_CoughDetectionCNN.h5* contains the trained model along with its weights, which can be loaded directly using Tensorflow and Keras, and deployed to make predictions or be trained further.
