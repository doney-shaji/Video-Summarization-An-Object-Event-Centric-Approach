from tkinter import *
from tkinter import ttk, messagebox, filedialog as fd
import cv2
import numpy as np
import threading
import time
import os
import sys
from keras.models import load_model
from pathlib import Path

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Load CNN-LSTM model
modelFile = load_model('SavedFiles/cnn_lstm_model.h5')
modelFile.compile(optimizer='adam', loss='binary_crossentropy', metrics=(['accuracy']))


# Global variables
root = Tk()
root.geometry('900x600')
root.resizable(0, 0)
root.title('Video Summarizer')
root.configure(bg='#bcebf5')


fload = Frame(root,width=700,height=600,bg='#bcebf5')
fresult = Frame(root,width=700,height=600,bg='#bcebf5')
nfresult = Frame(root,width=700,height=600,bg='#bcebf5')
process= Frame(root,width=700,height=600,bg='#bcebf5')

video_frame = Frame(root)
video_frame.pack()

video_path = ""
cap = None
video_label = None
img = None
is_playing = False

# Function to select video file
def select_video(path):
    vid_file = fd.askopenfilename(title='Select video', filetype=[('Video files', '*.mp4;*.avi')])
    path.delete(0, END)
    path.insert(END, vid_file)

# Function to detect objects in the video
def detect_objects(video_path, obj):
    yolo_root = Path("models/yolov5m_Objects365.pt")  # Update this path to the root directory of YOLOv5
    sys.path.append(str(yolo_root))
    from detect3 import main, parse_opt

    # Define YOLOv5 options
    yolo_options = parse_opt()

    # Set the source as the video path
    yolo_options.source = video_path
    yolo_options.object_to_detect=obj

    # Run YOLOv5 inference
    main(yolo_options)



# Function to handle prediction and object detection
def process_video(video_path):
    predictions = predict_video_segments(video_path)
    in_vid_path='fight_detected_video.mp4'
    #### detect_objects(in_vid_path,choosed_object)
    messagebox.showinfo('Video Saved', f'Summarized Video saved as fight_detected_video.mp4')
    print("----------------Finished-----------------")


def p_video1(video_path,choosed_object):
    detect_objects(video_path,choosed_object)
    messagebox.showinfo('Video Saved', f'Summarized Video saved in OUTPUT folder')
    print("----------------Finished-----------------")

# Main function for prediction
def predict_video_segments(video_path):
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    inter.insert(END,f'Frame per Second - {fps}')
    total_seconds = int(video_frames_count / fps)
    inter.insert(END,"\n")
    inter.insert(END,f'Total seconds - {total_seconds}')
    fight_frames = []  # List to store timestamps where fights are detected
    for second in range(total_seconds):
        frames = []
        for _ in range(20):
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (64, 64))
            normalized_frame = resized_frame / 255
            frames.append(normalized_frame)
        features = np.asarray([frames])
        model_prediction = modelFile.predict(features)
        print("Model Predictions :::>", model_prediction)
        inter.insert(END,"\n")
        inter.insert(END,f'Model Predictions - {model_prediction}')
        if model_prediction < 0.5:  # If prediction indicates a fight
            fight_frames.append(second)
            print("Fight")
            inter.insert(END,"\n")
            inter.insert(END,f'Result - Fight')
        else:
            print("Normal")
            inter.insert(END,"\n")
            inter.insert(END,f'Result - Normal')
    video_reader.release()

    # Once all predictions are done, save frames corresponding to fight timestamps
    save_frames_as_video(video_path, fight_frames)

# Function to save detected fight frames as a new video
def save_frames_as_video(video_path, fight_frames):
    video_reader = cv2.VideoCapture(video_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'fight_detected_video.mp4'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for second in fight_frames[:-1]:#

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, second * fps)  # Set frame to the start of the second
        for _ in range(fps):  # Write frames for this second
            success, frame = video_reader.read()
            if not success:
                break
            video_writer.write(frame)

    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
    

def clearData():
    global is_playing
    is_playing = False

    path.delete(0,END)
    inter.delete(1.0,END)

    fresult.pack_forget()
    nfresult.pack_forget()

def obj_predict():
    file_paths = ['fight_detected_video.mp4', 'Result.avi']

    # Iterate over the file paths and remove each file
    for file_path in file_paths:
        # Check if the file exists before attempting to remove it
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} removed successfully.")
        else:
            print(f"{file_path} does not exist.")

    video_path = path.get()
    if video_path:

        obj=selected_object.get()

        inter.insert(END,f'The video path - {video_path}')
        inter.insert(END,"\n")
        inter.insert(END,f'Object Selected - {obj}')
        inter.insert(END,"\n")

        threading.Thread(target=p_video1, args=(video_path,obj,)).start()
    else:
        messagebox.showinfo("Video Selection", "Please select a video.")



# Function to handle prediction button click
def predict():
    file_paths = ['fight_detected_video.mp4', 'Result.avi']

    # Iterate over the file paths and remove each file
    for file_path in file_paths:
        # Check if the file exists before attempting to remove it
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} removed successfully.")
        else:
            print(f"{file_path} does not exist.")

    video_path = path.get()
    if video_path:
            
        eve_=selected_event.get()

        inter.insert(END,f'The video path - {video_path}')
        inter.insert(END,"\n")
        inter.insert(END,f'Event Selected - {eve_}')
        inter.insert(END,"\n")

        threading.Thread(target=process_video, args=(video_path,)).start()

    else:
        messagebox.showinfo("Video Selection", "Please select a video.")


def show_object_summary_frame():
    global frame_object_summary,path,selected_object,inter
    frame_object_summary = Frame(root, width=700, height=600, bg='#bcebf5')

    Label(frame_object_summary, bg='#bcebf5', text='Video Summarizer', font=('Trobuchet', 18, 'bold'), fg='#cf5611').place(x=200, y=20)

    path = Entry(frame_object_summary, width=100)
    path.place(x=150, y=70)
    Button(frame_object_summary, text='Browse', width=10, command=lambda: select_video(path)).place(x=50, y=70)

    objects = ['Person', 'Hockey Stick', 'Helmet', 'Gloves', 'Sneakers', 'Boots', ]
    selected_object = StringVar(frame_object_summary)
    selected_object.set(objects[0])
    object_menu = OptionMenu(frame_object_summary, selected_object, *objects)
    object_menu.place(x=150, y=110)
    Label(frame_object_summary, text='Object:', font=('Trobuchet', 10, 'bold'), bg='#bcebf5').place(x=50, y=120)

    result_suspect = Label(frame_object_summary, bg='#bcebf5', text='', font=(('Helvetica', 13, 'bold')), fg='red')
    result_suspect.place(x=175, y=250)

    result_normal = Label(frame_object_summary, bg='#bcebf5', text='', font=(('Helvetica', 13, 'bold')), fg='red')
    result_normal.place(x=175, y=250)

    Label(frame_object_summary, bg='#bcebf5', text='Process').place(x=400, y=200)
    inter = Text(frame_object_summary, width=60, height=20)
    inter.place(x=380, y=220)

    submit = Button(frame_object_summary, text='Predict', width=10, command=obj_predict)
    submit.place(x=150, y=350)

    clear = Button(frame_object_summary, text='Clear', width=10, command=clearData)
    clear.place(x=250, y=350)
    hom_b = Button(frame_object_summary, text='Home', width=10, command=home)
    hom_b.place(x=200, y=400)

    frame_object_summary.pack()

def show_event_summary_frame():
    global frame_event_summary,path,selected_event,inter
    frame_event_summary = Frame(root, width=700, height=600, bg='#bcebf5')

    Label(frame_event_summary, bg='#bcebf5', text='Video Summarizer', font=('Trobuchet', 18, 'bold'), fg='#cf5611').place(x=200, y=20)

    path = Entry(frame_event_summary, width=100)
    path.place(x=150, y=70)
    Button(frame_event_summary, text='Browse', width=10, command=lambda: select_video(path)).place(x=50, y=70)


    events = ['Fight']
    selected_event = StringVar(frame_event_summary)
    selected_event.set(events[0])
    event_menu = OptionMenu(frame_event_summary, selected_event, *events)
    event_menu.place(x=150, y=180)
    Label(frame_event_summary, text='Event:', font=('Trobuchet', 10, 'bold'), bg='#bcebf5').place(x=50, y=190)

    result_suspect = Label(frame_event_summary, bg='#bcebf5', text='', font=(('Helvetica', 13, 'bold')), fg='red')
    result_suspect.place(x=175, y=250)

    result_normal = Label(frame_event_summary, bg='#bcebf5', text='', font=(('Helvetica', 13, 'bold')), fg='red')
    result_normal.place(x=175, y=250)

    Label(frame_event_summary, bg='#bcebf5', text='Process').place(x=400, y=200)
    inter = Text(frame_event_summary, width=60, height=20)
    inter.place(x=380, y=220)

    submit = Button(frame_event_summary, text='Predict', width=10, command=predict)
    submit.place(x=150, y=350)

    clear = Button(frame_event_summary, text='Clear', width=10, command=clearData)
    clear.place(x=250, y=350)
    hom_b = Button(frame_event_summary, text='Home', width=10, command=home)
    hom_b.place(x=200, y=400)

    frame_event_summary.pack()
    

def event_summary():
    # Hide existing frames
    fresult.pack_forget()
    nfresult.pack_forget()
    process.pack_forget()
    video_frame.pack_forget()
    main_frame.pack_forget()
    # Show event summary frame
    show_event_summary_frame()

def object_summary():
    # Hide existing frames
    fresult.pack_forget()
    nfresult.pack_forget()
    process.pack_forget()
    video_frame.pack_forget()
    main_frame.pack_forget()
    
    # Show object summary frame
    show_object_summary_frame()


def home():
    global main_frame
    fresult.pack_forget()
    nfresult.pack_forget()
    process.pack_forget()
    video_frame.pack_forget()
    try:
        frame_event_summary.pack_forget()
    except:
        frame_object_summary.pack_forget()

    try:
        frame_object_summary.pack_forget()
    except:
        frame_event_summary.pack_forget()

    main_frame.pack_forget()

    main_frame = Frame(root, width=700, height=600, bg='#bcebf5')

    Label(main_frame, bg='#bcebf5', text='Video Summarizer', font=('Trobuchet', 18, 'bold'), fg='#cf5611').place(x=250, y=20)


    # Create buttons for event summary and object summary
    btn_event_summary = Button(main_frame, text='Event Summary', width=15, command=event_summary)
    btn_event_summary.place(x=200, y=200)

    btn_object_summary = Button(main_frame, text='Object Summary', width=15, command=object_summary)
    btn_object_summary.place(x=350, y=200)
    main_frame.pack()


main_frame = Frame(root, width=700, height=600, bg='#bcebf5')

Label(main_frame, bg='#bcebf5', text='Video Summarizer', font=('Trobuchet', 18, 'bold'), fg='#cf5611').place(x=250, y=20)


# Create buttons for event summary and object summary
btn_event_summary = Button(main_frame, text='Event Summary', width=15, command=event_summary)
btn_event_summary.place(x=200, y=200)

btn_object_summary = Button(main_frame, text='Object Summary', width=15, command=object_summary)
btn_object_summary.place(x=350, y=200)
main_frame.pack()
root.mainloop()
