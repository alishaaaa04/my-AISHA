import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
import av

import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-SHA - ISL Translator",
    page_icon="🤟",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤟 AI-SHA - Indian Sign Language Translator")

st.markdown(
    """
    ### Real-Time ISL Gesture Recognition
    Show an Indian Sign Language gesture in front of your camera.
    """
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    0: "Hello",
    1: "Thank You",
    2: "Sorry",
    3: "Bye"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("Gesture_model.pkl", "rb") as file:
        model_data = pickle.load(file)

    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["label_encoder"]

    return model, scaler, encoder


try:

    model, scaler, encoder = load_model()

    st.success("Gesture model loaded successfully.")

except Exception as e:

    st.error(f"Could not load Gesture_model.pkl: {e}")

    st.stop()


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# EXTRACT 225 LANDMARK FEATURES
# ============================================================

def extract_landmarks(results):

    features = []

    # --------------------------------------------------------
    # POSE
    # 33 landmarks × 3 = 99
    # --------------------------------------------------------

    if results.pose_landmarks:

        for landmark in results.pose_landmarks.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        features.extend([0.0] * 99)


    # --------------------------------------------------------
    # LEFT HAND
    # 21 landmarks × 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        for landmark in results.left_hand_landmarks.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        features.extend([0.0] * 63)


    # --------------------------------------------------------
    # RIGHT HAND
    # 21 landmarks × 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        for landmark in results.right_hand_landmarks.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        features.extend([0.0] * 63)


    # --------------------------------------------------------
    # MAKE SURE THERE ARE EXACTLY 225 FEATURES
    # --------------------------------------------------------

    features = features[:225]

    if len(features) < 225:

        features.extend(
            [0.0] * (225 - len(features))
        )


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# LIVE VIDEO PROCESSOR
# ============================================================

class ISLVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # MEDIAPIPE HOLISTIC
        # Same settings as original working script
        # ----------------------------------------------------

        self.holistic = mp_holistic.Holistic(

            static_image_mode=False,

            model_complexity=1,

            smooth_landmarks=True,

            enable_segmentation=False,

            refine_face_landmarks=False,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5
        )


        # ----------------------------------------------------
        # INITIAL PREDICTION
        # ----------------------------------------------------

        self.predicted_label = "Waiting for gesture..."


        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------

        self.prediction_history = []


    # ========================================================
    # PROCESS EACH VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        # ----------------------------------------------------
        # CONVERT WEBRTC FRAME TO OPENCV
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # FLIP FOR MIRROR VIEW
        #
        # This matches the original mediapipeee.py pipeline.
        # ----------------------------------------------------

        image = cv2.flip(
            image,
            1
        )


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE PROCESSING
        # ----------------------------------------------------

        results = self.holistic.process(
            rgb_image
        )


        # ----------------------------------------------------
        # CHECK WHETHER A HAND IS DETECTED
        # ----------------------------------------------------

        hand_detected = (

            results.left_hand_landmarks is not None

            or

            results.right_hand_landmarks is not None

        )


        # ====================================================
        # IF HAND IS DETECTED
        # ====================================================

        if hand_detected:

            try:

                # ------------------------------------------------
                # EXTRACT 225 FEATURES
                # ------------------------------------------------

                features = extract_landmarks(
                    results
                )


                # ------------------------------------------------
                # RESHAPE FOR MODEL
                # ------------------------------------------------

                features = features.reshape(
                    1,
                    225
                )


                # ------------------------------------------------
                # SCALE FEATURES
                # ------------------------------------------------

                scaled_features = scaler.transform(
                    features
                )


                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                prediction = model.predict(
                    scaled_features
                )


                predicted_class = int(
                    prediction[0]
                )


                # ------------------------------------------------
                # TEMPORAL SMOOTHING
                #
                # Store recent predictions so the displayed
                # result is more stable.
                # ------------------------------------------------

                self.prediction_history.append(
                    predicted_class
                )


                # Keep only the last 7 predictions

                if len(self.prediction_history) > 7:

                    self.prediction_history.pop(0)


                # ------------------------------------------------
                # FIND MOST COMMON RECENT PREDICTION
                # ------------------------------------------------

                if len(self.prediction_history) >= 3:

                    counts = {}

                    for value in self.prediction_history:

                        counts[value] = (
                            counts.get(value, 0) + 1
                        )


                    stable_class = max(
                        counts,
                        key=counts.get
                    )

                else:

                    stable_class = predicted_class


                # ------------------------------------------------
                # CONVERT CLASS → GESTURE
                # ------------------------------------------------

                self.predicted_label = LABEL_MAP.get(
                    stable_class,
                    "Unknown Gesture"
                )


            except Exception:

                self.predicted_label = (
                    "Prediction Error"
                )


        # ====================================================
        # NO HAND DETECTED
        # ====================================================

        else:

            self.predicted_label = (
                "Waiting for gesture..."
            )

            self.prediction_history.clear()


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(

                image,

                results.pose_landmarks,

                mp_holistic.POSE_CONNECTIONS
            )


        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(

                image,

                results.left_hand_landmarks,

                mp_holistic.HAND_CONNECTIONS
            )


        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(

                image,

                results.right_hand_landmarks,

                mp_holistic.HAND_CONNECTIONS
            )


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.putText(

            image,

            self.predicted_label,

            (30, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.3,

            (0, 255, 0),

            3,

            cv2.LINE_AA
        )


        # ====================================================
        # RETURN VIDEO FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )


# ============================================================
# STREAMLIT WEBRTC CAMERA
# ============================================================

st.subheader("📷 Live Camera")

webrtc_streamer(

    key="isl-translator",

    video_processor_factory=ISLVideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown(
    """
    ---
    
    ### How to use

    1. Click **START** above.
    2. Allow camera access when your browser asks.
    3. Keep your upper body and hand visible.
    4. Show one gesture clearly.
    5. The predicted ISL meaning will appear on the video.

    ### Supported Gestures

    - 👋 **Hello**
    - 🙏 **Thank You**
    - 👋 **Bye**
    - 🤟 **Sorry**

    ---
    
    **AI-SHA** — Indian Sign Language Recognition
    """
)