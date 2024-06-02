# # Install the necessary libraries.
# !pip install tensorflow opencv-contrib-python youtube-dl moviepy pydot
# !pip install git+https://github.com/TahaAnwar/pafy.git#egg=pafy

# Import libraries.
import os
import cv2
# import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from keras.models import load_model

from moviepy.editor import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, ConfusionMatrixDisplay

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


seed_constant = 42
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Plot image
plt.figure(figsize = (20, 20))


all_labels = os.listdir('Dataset')

for index in range(2):

   
    selected_label = all_labels[index]
 
    video_files_names_list = os.listdir(f'Dataset/{selected_label}')

    selected_video_file_name = random.choice(video_files_names_list)

    video_reader = cv2.VideoCapture(f'Dataset/{selected_label}/{selected_video_file_name}')

    # Read frame of the video file.
    _, bgr_frame = video_reader.read()

 
    video_reader.release()

    rgb_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cv2.putText(rgb_img, selected_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame.
    plt.subplot(5, 4, 5);plt.imshow(rgb_img);plt.axis('off')

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

SEQUENCE_LENGTH = 20


DATASET_DIR = "Dataset"


CLASSES_LIST = ["Fight", "Normal"]

def frames_extraction(video_path):

    frames_list = []

    
    video_reader = cv2.VideoCapture(video_path)

  
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))


    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)


    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

       
        success, frame = video_reader.read()

        if not success:
            break

   
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        
        normalized_frame = resized_frame / 255

        frames_list.append(normalized_frame)


    video_reader.release()

    return frames_list

def create_dataset():
   


    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):

        print(f'Extracting Data of Class: {class_name}')

      
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        
        for file_name in files_list:


            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

         
            frames = frames_extraction(video_file_path)

        
            if len(frames) == SEQUENCE_LENGTH:

        
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

   
    features = np.asarray(features)
    labels = np.array(labels)


    return features, labels, video_files_paths


features, labels, video_files_paths = create_dataset()

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, shuffle = True,random_state = seed_constant)

def create_convlstm_model():
   
    dl_model = Sequential([
        ConvLSTM2D(4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (SEQUENCE_LENGTH,
                                                                                      IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.2)),
        ConvLSTM2D(8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.2)),
        ConvLSTM2D(14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.2)),
        ConvLSTM2D(16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),

        Flatten(),
        Dense(1, activation = "sigmoid")
    ])
  
   
    dl_model.summary()
    return dl_model

dl_model = create_convlstm_model()


dl_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ["accuracy"])

history = dl_model.fit(X_train,y_train, epochs = 5, batch_size = 4,shuffle = True, validation_split = 0.2)

model_evaluation_history = dl_model.evaluate(X_test, y_test)

model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Save  Model
dl_model.save('SavedFiles/cnn_lstm_model1_.h5')


y_pred = dl_model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

acc = accuracy_score(y_test,y_pred)

print(acc)

cm = confusion_matrix(y_test,y_pred)

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_LIST)


disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix")
plt.show()




