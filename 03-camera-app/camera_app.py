import os
import cv2
import numpy as np
import pyglet
import keras
import argparse
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--timer', type=int, default=5,
                    help='Timer value (default: 5)')
parser.add_argument('--path', type=str, default='selfie.jpg',
                    help='Path for saving selfies (default: current directory)')
args = parser.parse_args()

CONFIDENCE_THRESHOLD = 0.7
GESTURE_LABELS = ['dislike', 'fist', 'like',
                  'one', 'peace', 'stop', 'three', 'two_up']
MODEL_PATH = "gesture_recognition.keras"
SIZE = (64, 64)
COLOR_CHANNELS = 3
MEDIAPIPE_HAND_LANDMARK_MODEL_PATH = "hand_landmarker.task"
GESTURE_QUEUE_LENGTH = 6
ZOOM_FACTOR = 0.8


class GestureClassifier:
    """ This class loads the trained gesture recognition model and uses it to classify hand gestures in the video feed."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MODEL_PATH)
        self.model = keras.models.load_model(model_path)
        self.label_names = GESTURE_LABELS

    def classify(self, frame, box):
        x1, y1 = box[0]
        x2, y2 = box[1]

        crop = frame[y1:y2, x1:x2]
        # cv2.imshow("crop", crop) Debug crop

        img = cv2.resize(crop, SIZE)
        img = img.astype("float32") / 255.
        reshaped = img.reshape(-1, *SIZE, COLOR_CHANNELS)
        prediction = self.model.predict(reshaped, verbose=0)
        max_conf = np.max(prediction)
        # Debug prediction and confidence

        if max_conf > CONFIDENCE_THRESHOLD:
            predicted_label = self.label_names[np.argmax(prediction)]
            print(
                f"Predicted: {predicted_label} with confidence {max_conf:.2f}")
            return predicted_label

        return "unknown"


class HandDetector:
    """This class uses the MediaPipe Hand Landmark model to detect hands in the video feed and return bounding boxes around them.
    Made with tutorial https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MEDIAPIPE_HAND_LANDMARK_MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_box(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        results = self.detector.detect(mp_image)

        # Getting the bounding box was assisted by ChatGPT
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            h, w, _ = frame.shape

            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]

            x1 = int(min(xs) * w)
            y1 = int(min(ys) * h)
            x2 = int(max(xs) * w)
            y2 = int(max(ys) * h)

            padding = 30
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            return (x1, y1), (x2, y2)

        return None


class LiveCameraApp:
    def __init__(self, video_id=0, mac_os=False):
        self.video_id = video_id
        self.detector = HandDetector()
        self.classifier = GestureClassifier()
        self.cap = cv2.VideoCapture(self.video_id)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.gesture_queue = deque(maxlen=GESTURE_QUEUE_LENGTH)
        self.filter_on = False
        self.selfie_timer_active = False
        self.selfie_taken = False
        self.selfie_taken_timer = 0
        self.selfie_timer = args.timer
        self.zoom_on = False
        if mac_os:
            self.window = pyglet.window.Window(
                (self.width / 2), (self.height / 2))
        else:
            self.window = pyglet.window.Window(self.width, self.height)

    def check_gesture_consistency(self):
        if len(self.gesture_queue) == self.gesture_queue.maxlen:
            for gesture in self.gesture_queue:
                if gesture != self.gesture_queue[0]:
                    self.gesture_queue.clear()
                    return None
            consistent_gesture = self.gesture_queue[0]
            self.gesture_queue.clear()
            return consistent_gesture
        return None

    def handle_gesture_actions(self, gesture):
        if gesture == "like":
            self.filter_on = True
        elif gesture == "dislike":
            self.filter_on = False
        elif gesture == "stop":
            self.selfie_timer_active = True
        elif gesture == "peace":
            if not self.zoom_on:
                self.zoom_on = True
            else:
                self.zoom_on = False

    def update_selfie_timer(self):
        if not self.selfie_timer_active:
            return

        self.selfie_timer -= 1
        if self.selfie_timer == 0:
            ret, frame = self.cap.read()
            if self.filter_on:
                frame = self.add_sepia_filter(frame)
            if self.zoom_on:
                frame = self.add_zoom(frame)
            cv2.imwrite(args.path, frame)
           
            
            self.selfie_taken = True
            self.selfie_taken_timer = 60
            self.selfie_timer_active = False
            self.selfie_timer = args.timer

    def handle_gesture_detection(self, frame, box):
        gesture = self.classifier.classify(frame, box)
        if gesture == "unknown":
            self.gesture_queue.clear()
        else:
            self.gesture_queue.append(gesture)
        consistent_gesture = self.check_gesture_consistency()
        if consistent_gesture:
            self.handle_gesture_actions(consistent_gesture)

    def add_sepia_filter(self, frame):
        # Speia Filter from https://studyopedia.com/opencv/apply-sepia-tone-filter-with-opencv/
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, kernel)
        sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
        return sepia_frame

    def add_zoom(self, frame):
        height, width, _ = frame.shape

        new_width = int(width * ZOOM_FACTOR)
        new_height = int(height * ZOOM_FACTOR)

        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2

        crop = frame[y1:y1+new_height, x1:x1+new_width]
        resized = cv2.resize(crop, (width, height))
        return resized

    def on_draw(self):
        self.window.clear()
        ret, frame = self.cap.read()

        detection_frame = frame.copy()
        display_frame = frame.copy()

        box = self.detector.detect_box(detection_frame)
        if box:
            self.handle_gesture_detection(detection_frame, box)
        if self.filter_on:
            display_frame = self.add_sepia_filter(display_frame)
            cv2.putText(display_frame, "SEPIA: ON", (int(self.width * 0.05),
                        int(self.height * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        if self.zoom_on:
            display_frame = self.add_zoom(display_frame)
            cv2.putText(display_frame, "ZOOM: ON", (int(self.width * 0.05), int(self.height * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            
        if self.selfie_timer_active:
            cv2.putText(display_frame, f"{self.selfie_timer}", (int(
                self.width / 2), int(self.height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 10)

        if self.selfie_taken:
            cv2.putText(display_frame, "SELFIE TAKEN!", (int(
                self.width * 0.45), int(self.height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            self.selfie_taken_timer -= 1
            if self.selfie_taken_timer == 0:
                self.selfie_taken = False
                
        img = cv2glet(display_frame, 'BGR')
        img.blit(0, 0, 0)


def cv2glet(img, fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols,
                                   height=rows,
                                   fmt=fmt,
                                   data=raw_img,
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg


def main():
    app = LiveCameraApp()

    @app.window.event
    def on_draw():
        app.on_draw()

    @app.window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            app.cap.release()
            pyglet.app.exit()

    pyglet.clock.schedule_interval(lambda dt: app.update_selfie_timer(), 1.0)
    pyglet.app.run()


if __name__ == "__main__":
    main()
