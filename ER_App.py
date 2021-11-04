import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical,img_to_array
from PIL import Image
import streamlit as st
import time
from skimage import color
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

st.title("Welcome to Face Emotion Recognition Application")

main_options = st.sidebar.selectbox("What would you like to choose: ", ['About', 'Detection space', 'Contact'])

if main_options == 'About':
  st.write('Solution to the real time face emotion detection problem')
  st.write('Today, the majority of our time is spent interacting with computers and mobile phones in our daily life due to technology progression and ubiquitously spreading of these mediums.\
   However, they play an essential role in our lives, and most existing software interfaces are non-verbal. Adding emotional expression recognition to expect the users’ feelings and emotional state can effectively improve human-computer interaction.')
  st.write('Humans usually employ different cues to express their emotions, such as facial expressions, hand gestures and voice. \
  Facial expressions represent up to 55% of human communications while other ways such as oral language allocate a mere 7% of emotion expression.\
  Therefore, considering facial expressions in an HRI(Human robotic interactions system successfully enables simulation of natural interactions.\
  It has many more advantages than we can imagine.It can be used in education, research, medicine, manufacturing, investigation and many other fields.\
  If the people use this in the right way, it can give us many good results.')
  st.write('Note: This app was created to use in the education field to detect the emotions of the students. \
  It will enable proper understanding and improvement in teaching the students if we use it with a camera that keeps an eye on students emotions.')

elif main_options == 'Contact':
  st.write('Contact Us')
  st.write('sannhtet899@gmail.com')
  st.write('khinezinzinlinn@gmail.com')
  st.write('sthgeckoygn@gmail.com')
elif main_options == 'Detection space':
  option = st.radio('Which type of detection would you like to make?',
                      ('an Image', 'a Video', 'OpenCV Live'))
  st.header('You selected {} option for emotion detection'.format(option))

  if option == 'an Image':
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
      image2 = Image.open(image_file)
      #file_bytes = np.asarray(bytearray(image_file.read()),dtype=np.uint8)
      #opencv_image = cv2.imdecode(file_bytes,1)
      st.text("Original Image")
      #st.image(opencv_image,channels="BGR",use_column_width=True)
      #gray1 = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
      progress = st.progress(0)
      for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)
      st.image(image2)
    #if image_file in None:
      #st.error("No image uploaded yet")

    if st.button("Progress"):
        
      model = load_model('/content/gdrive/MyDrive/cnn.h5')
      #gray = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
      #t = pil2tensor(opencv_image,dtype=np.float32)
      #t = t.float()/255.0
      #gray = gray/255.0
      #img1 = Image.open(t)
      #img = cv2.cvtColor(np.float32(image2),1)
      #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #images = cv2.resize(gray,(48,48))
      image = np.array(image2)
      img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      roi_gray = cv2.resize(img_gray,(48,48),interpolation=cv2.INTER_AREA)
      roi = roi_gray.astype('float')/255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi,axis=0)
      #img = image.load_img(image2,color_mode='grayscale',target_size=(48,48,1))
      #y = image.img_to_array(images)
      #y = np.expand_dims(y,axis=0)
      #images = np.vstack([y])
      #classes = model.predict(images/255.0)
      #images = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
      #images = cv2.resize(images,(48,48),interpolation=cv2.INTER_AREA)
      #images = images/255.0
      classes = model.predict(roi.reshape(-1,48,48,1))
      labels_index = ["anger","disgust","fear","happiness","sadness","surprise","neutral"]
      max_index = np.argmax(classes)
      prediction = labels_index[max_index]
      print("This person here is",labels_index[max_index])
      if prediction == 'happiness':
        st.subheader("YeeY!  You are Happy :smile: today Always Be ! ")
      elif prediction == 'sadness':
        st.subheader("You seem to be sad today, Smile and be happy!")
      elif prediction == 'anger':
        st.subheader("You seem to be angry, Take it easy!")
      elif prediction == 'disgust':
        st.subheader("You seem to be Disgust today!")
      elif prediction == 'fear':
        st.subheader("You seem to be Fearful, Be couragous!")
      elif prediction == 'surprise':
        st.subheader("You seem to be surprise today!")
      else:
        st.subheader("You seem to be Neutral today, Happy day!")
    
  elif option == 'a Video':
    st.write('This is not a good option for us')
    st.write('We will contact you when this option is ready to use')

  elif option == 'OpenCV Live':


    # load model
    emotion_dict = {0:'anger', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'sad',5:'surprise',6:'neutral'}
    classifier = load_model('/content/gdrive/MyDrive/cnn.h5')

    #load face
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    except Exception:
        st.write("Error loading cascade classifiers")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class Faceemotion(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            #image gray
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                image=img_gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(
                    x + w, y + h), color=(255, 0, 0), thickness=2)
                roi_gray = img_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = emotion_dict[maxindex]
                    output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img

    def main():
        # Face Analysis Application #
        st.title("Real Time Face Emotion Detection Application")
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                            video_processor_factory=Faceemotion)
    if __name__ == "__main__":
        main()


  else:
    st.write('You did not select the proper option as specified. Please select a valid option')
    st.write(
          'If one of the four options was selected and it did not work. Please clear the cache and rerun the application')
    st.write('Thanks for understanding')

st.write('Thank you. I hope you got emotions detected in an image or a video or live detection.')
st.write('This is created by Sann Htet, Soe Thiha and Khine Zin Zin Linn')