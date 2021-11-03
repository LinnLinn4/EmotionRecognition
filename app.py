import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
model = load_model("D:\Linn\ComputerVision\EmotionDetection\cnn3.h5")

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
		fc = gray[y:y+h, x:x+w]
		roi = cv2.resize(fc, (48, 48))
		pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
		cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		return img,faces,pred 
        
def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    if choice == "Home":
        st.title("Emotion Detector")
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
    elif choice == "Webcam Face Detection":
        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                global b
                label = []
                img = frame.to_ndarray(format="bgr24")
                face_cascade_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade_detect.detectMultiScale(gray, 1.3, 1)
                for (x, y, w, h) in faces:
                    a = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                    t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                    t = t.float() / 255.0
                    roi = Image(t)
                    model6 = classify.get_model()
                    prediction = model6.predict(roi)[0]  # Prediction
                    label = str(prediction)
                    label_position = (x, y)
                    b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return b


        def live_detect():
            class VideoTransformer(VideoTransformerBase):
                frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
                in_image: Union[np.ndarray, None]
                out_image: Union[np.ndarray, None]

                def __init__(self) -> None:
                    self.frame_lock = threading.Lock()
                    self.in_image = None
                    self.out_image = None

                def transform(self, frame: av.VideoFrame) -> np.ndarray:
                    in_image = frame.to_ndarray(format="bgr24")
                    out_image = in_image[:, ::-1, :]  # Simple flipping for example.

                    with self.frame_lock:
                        self.in_image = in_image
                        self.out_image = out_image

                    return in_image

            ctx = webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
            while ctx.video_transformer:

                with ctx.video_transformer.frame_lock:
                    in_image = ctx.video_transformer.in_image
                    # out_image = ctx.video_transformer.out_image
                if in_image is not None:
                    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                    face_cascade_detect = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade_detect.detectMultiScale(gray)
                    for (x, y, w, h) in faces:
                        a = cv2.rectangle(in_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48),
                                              interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                        if np.sum([roi_gray]) != 0:
                            t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                            t = t.float() / 255.0
                            roi = Image(t)
                            # roi = np.expand_dims(roi, axis=0)  ## reshaping the cropped face image for prediction
                            model6 = classify.get_model()
                            prediction = model6.predict(roi)[0]  # Prediction
                            label = str(prediction)
                            label_position = (x, y)
                            b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Text Adding
                            st.image(b, channels="BGR")
                else:
                    st.write('Unable to access camera input')


        #HERE = Path(__file__).parent

        #logger = logging.getLogger(__name__)
            # class VideoTransformer(object):
        #   pass
        #live_detect()
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=Faceemotion)
        # live_detect()
        st.write('Live functioning')
        st.write(
            'This is running using newly introduced webrtc tool which can access the camera whereas opencv cannot function properly in streamlit')
        st.write(
            'This new tool takes some time for starting up the live video,please wait for few minutes(2-3 mins max) and the detection starts')
        st.write('This is the end of the instructions for using this option. See you')
    elif choice == "About":
        st.subheader("About this app")
    else:
        pass

if __name__ == "__main__":
    main()