import os
import cv2
import math
import numpy as np
from keras.models import load_model
from tkinter.filedialog import askopenfilename

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

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

def single_video(path):
    features = []
    frames = frames_extraction(path)
    if len(frames) == SEQUENCE_LENGTH:
        features.append(frames)
    features = np.asarray(features)
    return features

def predict_video_segments(video_path):
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    total_seconds = int(video_frames_count / fps)
    predictions = []
    for second in range(total_seconds):
        frames = []
        for _ in range(SEQUENCE_LENGTH):
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames.append(normalized_frame)
        features = np.asarray([frames])
        model_prediction = modelFile.predict(features)
        predictions.append(model_prediction)
    video_reader.release()
    return predictions

vid_path = askopenfilename() #"Dataset/Fight/fi110_xvid.avi"

modelFile = load_model('SavedFiles/cnn_lstm_model.h5')
modelFile.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

predictions = predict_video_segments(vid_path)

for i, prediction in enumerate(predictions):
    if prediction > 0.5:
        print(f'Second {i+1}: Normal')
    else:
        print(f'Second {i+1}: Fight')
