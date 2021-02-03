# Cough detection using a Convolutional Neural Network

As part of my work in the Eupnoos Audio & ML team, I have been developing a module for cough monitoring and diagnosis via smartphone. 
The full-scale implementation is intended to work as an overnight monitoring application, analyzing audio for cough and other respiratory events.

I can openly share the initial machine learning stage of this module, which performs a classification between cough events, which should then be used for further analysis, and all other audio events, which are not used.

### Context

Given an input stream of recorded audio, we need to identify audio events and classify them as either coughing or something else. 
This machine learning model for the binary classification of events is presented here. 
The application can then send the audio identified as coughing for further processing to provide diagnosis. 


Allocating and using data of medical interest is always complicated, but for this initial task, a collection containing audio of coughing is sufficient. 
The code presented here includes the process of extracting the audio necessary to create our training dataset from raw files. 
The features extracted are the MFCCs calculated for each audio sample, which are stored as png images. 
Using this approach we essentially convert our audio data into an image dataset, while maintaining all the information useful for the classification task.
Classification is then performed using a standard Convolutional Neural Network.

### Contents

*NotebookI.ipynb* and *NotebookII.ipynb* are the Jupyter notebooks containing the steps from the creation of the dataset, the feature generation, and the training and testing of the CNN. 

*model_CoughDetectionCNN.h5* contains the trained model along with its weights, which can be loaded directly using Tensorflow and Keras, and deployed to make predictions or be trained further.

