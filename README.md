# Video-Summarization-An-Object-Event-Centric-Approach

## Abstract

Video summarization is a crucial task in video analysis, especially when dealing with large volumes of video content that need to be condensed into more manageable formats. Traditional methods often struggle to capture the nuances of video content, particularly in summarizing dynamic events and interactions. This project addresses these challenges by leveraging deep learning techniques to enhance the summarization process.

Our project introduces a novel approach to video summarization that integrates both object-centric and event-centric methodologies, specifically focusing on hockey. The object-centric summarization uses YOLOv5 to detect and track objects, providing insights into their interactions. This model is pre-trained on the Object365 dataset, which includes essential hockey-related objects. Simultaneously, the event-centric summarization employs a Convolutional LSTM to identify significant events, capturing both temporal and spatial information. This model is trained on a specialized hockey dataset that includes both fight and normal events.

Furthermore, the project features a user-friendly graphical interface developed with Python’s Tkinter module. This interface simplifies the summarization process, making it accessible and easy to use. Experimental evaluations demonstrate the effectiveness of this approach in generating comprehensive and accurate video summaries, marking a significant advancement in video summarization for complex content analysis.

## Features

- **Object-Centric Summarization**:
  - Utilizes YOLOv5 for object detection and tracking.
  - Offers insights into interactions between detected objects.
  - Pre-trained on the Object365 dataset with hockey-related objects.

- **Event-Centric Summarization**:
  - Employs Convolutional LSTM for event detection.
  - Captures temporal and spatial data.
  - Trained on a hockey dataset including fight and normal events.
  - 

## Methodology

1. **Object Detection and Tracking**:
   - Use YOLOv5 to detect and track relevant objects within the video.
   - Focus on objects pertinent to hockey, as identified in the Object365 dataset.

2. **Event Detection**:
   - Utilize a Convolutional LSTM to identify and summarize significant events.
   - Capture both the temporal sequence and spatial context of events.

3. **Graphical Interface**:
   - Implement a Tkinter-based GUI to facilitate easy use and interaction.
   - Enable users to load videos, view summaries, and navigate through detected events.

## Experimental Evaluation

- Conducted thorough evaluations to test the efficacy of the summarization method.
- Demonstrated significant improvements in generating accurate and comprehensive video summaries.
- Highlighted the potential of deep learning in addressing complex video summarization challenges.
- The evaluation results are provided in the Results folder.

## Conclusion

This project successfully combines object-centric and event-centric summarization techniques to create a robust video summarization tool, particularly for hockey content. By leveraging advanced deep learning models and providing an intuitive user interface, the project makes significant strides in simplifying and enhancing video analysis.

