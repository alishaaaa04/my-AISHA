import cv2
import mediapipe as mp
import pickle
import pandas as pd


# ============================================================
# AI-SHA - INDIAN SIGN LANGUAGE RECOGNITION
# ============================================================

print("Starting AI-SHA...")


# ============================================================
# 1. MEDIAPIPE HOLISTIC SETUP
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# 2. LOAD TRAINED GESTURE MODEL
# ============================================================

print("Loading gesture model...")

with open("./Gesture_model.pkl", "rb") as file:
    model_pack = pickle.load(file)

xgb_classifier = model_pack["model"]
scaler = model_pack["scaler"]
label_enc = model_pack["label_encoder"]

print("Gesture model loaded successfully.")


# ============================================================
# 3. EXTRACT 225 LANDMARK FEATURES
# ============================================================

def extract_landmarks(results):

    landmark_data = []

    # --------------------------------------------------------
    # POSE: 33 × 3 = 99
    # --------------------------------------------------------

    if results.pose_landmarks:

        for landmark in results.pose_landmarks.landmark:

            landmark_data.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        landmark_data.extend([0.0] * 99)


    # --------------------------------------------------------
    # LEFT HAND: 21 × 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        for landmark in results.left_hand_landmarks.landmark:

            landmark_data.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        landmark_data.extend([0.0] * 63)


    # --------------------------------------------------------
    # RIGHT HAND: 21 × 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        for landmark in results.right_hand_landmarks.landmark:

            landmark_data.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    else:

        landmark_data.extend([0.0] * 63)


    return landmark_data


# ============================================================
# 4. PREDICT GESTURE
# ============================================================

def predict_gesture(landmark_data):

    data = pd.DataFrame(
        [landmark_data],
        columns=range(225)
    )

    scaled_data = scaler.transform(data)

    numeric_prediction = xgb_classifier.predict(
        scaled_data
    )

    text_prediction = label_enc.inverse_transform(
        numeric_prediction
    )

    return str(text_prediction[0])


# ============================================================
# 5. OPEN CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    print("Check Terminal camera permission.")

    raise SystemExit

print("Camera opened successfully.")


# ============================================================
# 6. MEDIAPIPE HOLISTIC
# ============================================================

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("MediaPipe Holistic initialized.")
    print("Show an ISL gesture.")
    print("Press Q to quit.")

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Could not read camera frame.")
            continue


        # ----------------------------------------------------
        # FLIP FOR MIRROR VIEW
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE PROCESSING
        # ----------------------------------------------------

        results = holistic.process(rgb_frame)


        # ----------------------------------------------------
        # DRAW POSE
        # ----------------------------------------------------

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )


        # ----------------------------------------------------
        # DRAW LEFT HAND
        # ----------------------------------------------------

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ----------------------------------------------------
        # DRAW RIGHT HAND
        # ----------------------------------------------------

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ====================================================
        # EXTRACT 225 FEATURES
        # ====================================================

        landmark_data = extract_landmarks(results)


        if len(landmark_data) != 225:

            print(
                "ERROR: Expected 225 features, got:",
                len(landmark_data)
            )

            continue


        # ====================================================
        # CHECK HAND DETECTION
        # ====================================================

        hand_detected = (

            results.left_hand_landmarks is not None
            or
            results.right_hand_landmarks is not None

        )


        # ====================================================
        # PREDICT
        # ====================================================

        if hand_detected:

            try:

                prediction = predict_gesture(
                    landmark_data
                )

            except Exception as error:

                prediction = "Prediction Error"

                print(
                    "Prediction error:",
                    error
                )

        else:

            prediction = "Show your hand"


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            prediction,
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            "AI-SHA | ISL Recognition",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "AI-SHA - ISL Recognition",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# 7. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("AI-SHA stopped successfully.")