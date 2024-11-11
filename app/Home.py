import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import os
from textblob import TextBlob
import nltk
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyaudio

from motiondetection import detect_all  # Make sure motiondetection.py is in the same directory or adjust path

st.set_page_config(page_title="Real-Time Speech Sentiment", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        overflow: hidden; /* Hides the scrollbar */
    }
    .custom-title {
        font-size: 36px;
        font-weight: bold;
        color: #fff; 
        text-align: center; 
        padding: 10px;
        border-bottom: 2px solid #fff; 
    }
    .custom-titles {
        font-size: 26px;
        font-weight: bold;
        color: #fff; 
        text-align: center; 
        padding: 10px;
        border-bottom: 2px solid #fff; 
        margin-top:50px;
    }
    .custom-text{
      font-size: 18px;
        color: #fff;
        line-height: 1.6; 
        text-align: justify;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="custom-title">Monitoring system</div>', unsafe_allow_html=True)
st.markdown("""
            <div class="custom-text">Our application is designed to enhance the quality of skill development training by offering a real-time analysis of classroom environments. Using advanced AI techniques, this system observes key classroom activities, enabling detailed insights into student and instructor interactions.

The app’s features include:

Pose Detection: Tracks and identifies student and instructor postures to determine if they are sitting, standing, or walking, providing insights into engagement and classroom movement patterns.
            
Speech Sentiment Analysis: Analyzes spoken words to gauge emotional tones during classroom discussions, helping to assess the atmosphere and engagement levels.
            
Environment Assessment: Identifies whether the classroom environment is clean or messy, allowing trainers and administrators to maintain a conducive learning space.
            
Expression Detection: Detects facial expressions to understand the reactions of students and instructors, helping to monitor attentiveness and interest.
            </div>
            """,unsafe_allow_html=True)

st.markdown('<div class="custom-text">For the Project check the given arrow on the right top of the window.</div>', unsafe_allow_html=True)


st.markdown('<div class="custom-titles">Pose Detection</div>', unsafe_allow_html=True)
st.markdown("""<div class="custom-text">Pose detection uses advanced computer vision techniques to track human body movements and identify key landmarks, such as the head, shoulders, elbows, knees, and ankles. This feature provides real-time analysis of body posture, allowing us to detect whether a person is sitting, standing, walking, or engaging in other activities.
            </div>""", unsafe_allow_html=True)

st.markdown("""<div class="custom-text">In the Classroom Monitoring System, pose detection helps us understand student and instructor interactions, activity levels, and engagement during training sessions. By tracking body positions, we can gain insights into classroom dynamics, ensuring that the learning environment remains active and focused.</div>""", unsafe_allow_html=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Upload video
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open video file
    cap = cv2.VideoCapture(tfile.name)

    posture_placeholder = st.empty()

    # Initialize previous landmarks
    previos_landmark = None

    # Output video setup
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    # Use Mediapipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # If landmarks detected
            if result.pose_landmarks:
                posture = detect_all(result.pose_landmarks.landmark, previos_landmark)
                previos_landmark = result.pose_landmarks.landmark

                posture_placeholder.text(f"Current Posture: {posture}")

                # Annotate frame with posture
                cv2.putText(image, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write frame to output video
            out.write(image)

    # Release resources
    cap.release()
    out.release()



    # Delete temporary files
    os.remove(tfile.name)
    os.remove(out_file.name)


    
st.markdown('<div class="custom-titles">RealTime Speech Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown("""<div class="custom-text">Pose detection uses advanced computer vision techniques to track human body movements and identify key landmarks, such as the head, shoulders, elbows, knees, and ankles. This feature provides real-time analysis of body posture, allowing us to detect whether a person is sitting, standing, walking, or engaging in other activities.
            </div>""", unsafe_allow_html=True)

st.markdown("""<div class="custom-text">In the Classroom Monitoring System, pose detection helps us understand student and instructor interactions, activity levels, and engagement during training sessions. By tracking body positions, we can gain insights into classroom dynamics, ensuring that the learning environment remains active and focused.</div>""", unsafe_allow_html=True)
# Download 'punkt' and 'stopwords' if not present
nltk.download('punkt')
nltk.download('stopwords')

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to analyze text sentiment
def analyses_text(Text):
    blob = TextBlob(Text)
    sentiment = blob.sentiment.polarity
    sentiment_sub = blob.sentiment.subjectivity
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(Text.lower())
    keywords = [word for word in word_tokens if word.isalpha() and word not in stop_words]

    # Feedback based on sentiment
    if sentiment > 0.5:
        sentiment_feedback = "Feedback: The lecture is very positive and engaging!"
    elif sentiment < 0:
        sentiment_feedback = "Feedback: The tone seems negative; consider being more encouraging."
    else:
        sentiment_feedback = "Feedback: The tone is neutral; try adding some enthusiasm."

    return Text, sentiment, sentiment_feedback  # Return Text, Sentiment, and Feedback

# Function to listen and convert speech to text
def listen_convert():
    try:
        with sr.Microphone() as source:
            print("Please wait.... Adjusting the ambient Noise")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("......Now Speak")

            audio = recognizer.listen(source)
            Text = recognizer.recognize_google(audio)
            Text = Text.lower()  # Convert to lower case for consistency
            # Get the sentiment analysis
            Text, sentiment, sentiment_feedback = analyses_text(Text)
            return Text, sentiment, sentiment_feedback
    except sr.UnknownValueError:
        return "Sorry, I did not understand the audio.", None, "Sorry, I did not understand the audio."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service.", None, "Error in speech recognition service."

# Streamlit interface
st.title("Real-Time Speech Sentiment")

input_type = st.radio("Choose Input Type", ('Text', 'Audio'))

if input_type == 'Text':
    user_text = st.text_area("Enter the text for analysis:")
    if st.button("Analyze Text"):
        if user_text:
            Text, sentiment, sentiment_feedback = analyses_text(user_text)
            st.write(f"You entered: {Text}")
            st.write(f"Sentiment Polarity: {sentiment}")
            st.write(sentiment_feedback)
        else:
            st.write("Please enter some text.")

elif input_type == 'Audio':
    st.write("Click 'Listen and Convert' to speak into your microphone and convert your speech to text.")
    if st.button("Listen and Convert"):
        Text, sentiment, sentiment_feedback = listen_convert()

        # Display the results in the Streamlit app
        if sentiment_feedback:
            st.write(f"You said: {Text}")
            st.write(f"Sentiment Polarity: {sentiment}")
            st.write(sentiment_feedback)
        else:
            st.write(Text)  # In case of error, display the error message