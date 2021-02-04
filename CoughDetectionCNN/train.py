# Import libraries:
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

#%% SCRIPT STARTS HERE
# Set some default parameters:
TRAIN_DIR_PATH = './MFCCs_train/'
HEIGHT = 224
WIDTH = 224
INPUT_SHAPE = (HEIGHT, WIDTH, 3)
SPLIT = 0.2
BATCH_SIZE = 24
EPOCHS =20
PATIENCE=5
SAVE_PATH = 'CoughDetectionCNN_trained_model.h5'

#%%
def prepare_dataset(train_dir_path, 
                    split,
                    height, 
                    width, 
                    batch_size):
    """
    Prepare image dataset to be used as training input.
    
    args:
        train_dir_path (str): Path to training data directory
        split (float): Validation/Training split ratio
        height (int): Image height in pixels
        width (int): Image width in pixels
        batch_size (int): Number of samples in each batch
    returns:
        train_generator: Tensorflow generator object for training data
        val_generator: Tensorflow generator object for validation data
    """
    # Create data generator:
    datagen = ImageDataGenerator(validation_split=0.2, 
                                 rescale=1./255
                                 )
    # Training data generator:
    train_generator = datagen.flow_from_directory(
            train_dir_path, 
            target_size=(height, width), 
            subset='training',
            batch_size=batch_size,
            mode='categorical'
            )
    
    val_generator = datagen.flow_from_directory(
            train_dir_path,
            target_size=(height, width), 
            subset='validation',
            batch_size=batch_size,
            mode='categorical'
            )
    return train_generator, val_generator
#%%
def build_model(input_shape, 
                loss="categorical_crossentropy", 
                optimizer = 'adam',
                learning_rate=0.001
                ):
    """
    Build the Neural Network using keras.
    
    args:
    input_shape (tuple): Shape of array representing a sample
    loss (str): Name of the loss function to use
    optimizer (str): Name of optimizer to use
    learning_rate (float): the learning rate
    
    returns:
    Tensorflow model
    """
    
    #Design network architecture:
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(32, (3, 3), 
                     input_shape = input_shape, 
                     activation = 'relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Flattening
    model.add(Flatten())
    # Full connection
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(2, activation='softmax'))

    # Compile model    
    model.compile(optimizer = optimizer, 
                  loss = loss, 
                  metrics = ['accuracy']
                  )
    
    return model
#%%
def train(model, 
          epochs, 
          batch_size,
          patience, 
          training_data, 
          validation_data
          ):
    """
    Train the model
    
    args:
    epochs (int): Number of training epochs
    batch_size (int): Samples per batch
    patience (int): Number of epochs to wait before early stopping
    training_data (object): Tensorflow data generator for training data
    validation_data (object): Tensorflow data generator for validation data
    
    returns:
    Training history
    """
    
    # Callback for early stopping
    model_callback = tf.keras.callbacks.EarlyStopping(patience=patience)
    
    # Training
    history = model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=model_callback)
    
    return history
#%%
def plot_history(history):
    """
    Plot accuracy and loss over the training process
    
    args:
    history: Training history of the model
    returns:
    Plots
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
  
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
  
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    
    plt.show()
#%%
def main():
    # Generate train and validation sets:
    TRAIN_DATA, VAL_DATA = prepare_dataset(TRAIN_DIR_PATH,
                                           SPLIT,
                                           HEIGHT,
                                           WIDTH,
                                           BATCH_SIZE
                                           )
    
    # Create CNN:
    CNN = build_model(INPUT_SHAPE,
                      )
    
    # Train the model:
    
    HISTORY = train(CNN, 
                    EPOCHS, 
                    BATCH_SIZE, 
                    PATIENCE, 
                    TRAIN_DATA, 
                    VAL_DATA
                    )
    
    # Plot history:
    plot_history(HISTORY)
    
    # Save trained model:
    
    CNN.save(SAVE_PATH)
#%%
if __name__ == "__main__":
    main()
#%% SCRIPT ENDS HERE