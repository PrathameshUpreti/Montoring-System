#!/usr/bin/env python
# coding: utf-8

# ## Real Time Speech Sentiment Detection
# 
# 
# using NLP we are going to create this model using speech recogniztion libaray and text processing such as TextBlob

# Import the Libaries

# In[1]:


from textblob import TextBlob
import nltk
import speech_recognition  as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyaudio


# In[2]:





# In[3]:


nltk.download('punkt')
nltk.download('stopwords')


# Initialize the Speech Recognition

# In[4]:


recognizer=sr.Recognizer()


# Use Textblob for Sentiment analyses

# In[5]:


def analyses_text(Text):
    blob=TextBlob(Text)
    sentiment=blob.sentiment.polarity
    sentiment_sub=blob.sentiment.subjectivity
    print(sentiment_sub)

    stop_words=set(stopwords.words('english'))
    word_tokens = word_tokenize(Text.lower())
    keywords = [word for word in word_tokens if word.isalpha() and word not in stop_words]

    print("\nText:",Text)
    print("Sentiment Polarity:", sentiment)
    print("Keywords:", keywords)

    if sentiment > 0.5:
        sentiment_feedback ="Feedback: The lecture is very positive and engaging!"
    elif sentiment < 0:
        sentiment_feedback ="Feedback: The tone seems negative; consider being more encouraging."
    else:
        sentiment_feedback ="Feedback: The tone is neutral; try adding some enthusiasm."
    
    
    return sentiment_feedback


# Use Speech Recogniztion to listen the audio and convert to text --> create a function

# In[6]:


def listen_convert():
   with sr.Microphone() as source:
      print("Please wait.... Adjusting the ambient Noise")
      recognizer.adjust_for_ambient_noise(source,duration=2)
      print("......Now Speak")

      audio =recognizer.listen(source)
      try:
         Text=recognizer.recognize_google(audio)
         Text=Text.lower()
         analyses_text(Text)
      except sr.UnknownValueError:
            return "Sorry, I did not understand the audio.", None, None





# In[9]:


listen_convert()

