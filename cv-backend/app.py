import mediapipe as mp
import cv2
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

capture = cv2.VideoCapture(0)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

prev_hand_x = None
wave_counter = 0
wave_threshold = 3
wave_time = time.time()


def finger_is_up(hand_landmarks, tip, pip):
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y


def is_middle_finger(hand_landmarks):
    fingers = [
        finger_is_up(hand_landmarks, 8, 6),
        finger_is_up(hand_landmarks, 12, 10),
        finger_is_up(hand_landmarks, 16, 14),
        finger_is_up(hand_landmarks, 20, 18),
    ]
    return fingers == [False, True, False, False]


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                print("failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            hand_results = hands.process(image)
            face_mesh_results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            gesture_text = ""

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 0, 255), thickness=4, circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(20, 180, 90), thickness=2, circle_radius=2
                        ),
                    )

                    if is_middle_finger(hand_landmarks):
                        gesture_text = "angry person detected"

                    wrist_x = hand_landmarks.landmark[0].x
                    current_time = time.time()

                    if prev_hand_x is not None:
                        if abs(wrist_x - prev_hand_x) > 0.05:
                            if current_time - wave_time < 1.0:
                                wave_counter += 1
                            else:
                                wave_counter = 1
                            wave_time = current_time

                        if wave_counter >= wave_threshold:
                            gesture_text = "person is waving"
                            wave_counter = 0

                    prev_hand_x = wrist_x

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )

            if gesture_text:
                cv2.putText(
                    image,
                    gesture_text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            cv2.imshow("gesture detect", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

capture.release()
cv2.destroyAllWindows()
