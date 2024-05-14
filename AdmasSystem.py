import argparse
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame
import random
import os
from twilio.rest import Client
import pyttsx3
import datetime
from flask import Flask, jsonify, render_template_string, request
import queue
from threading import Thread, Lock
import time
import atexit

app = Flask(__name__)

# Twilio setup
account_sid = 'ACc21ea5bb52bfeb585b2eb90847d51f7a'
auth_token = 'ddd23c11b6a7797c86baffc295172ad5'
twilio_client = Client(account_sid, auth_token)
twilio_number = '+13185439074'
target_number = '+94765958539'

warning_messages = [
    "Take some fresh air, sir.", "Consider taking a small rest.",
    "It's important to stay alert. Take a break.",
    "You seem tired. Pull over and rest for a while.",
    "Feeling drowsy? Take a break and stretch your legs.",
    "Your safety comes first. Rest if you're feeling tired.",
    "Keep yourself and others safe. Pull over and rest if needed.",
    "Feeling fatigued? It's time to take a break.",
    "Don't push yourself too hard. Take a short break.",
    "Alertness is key while driving. Consider stopping for a rest.",
    "Feeling sleepy? Stop for a coffee or a quick nap.",
    "Driving requires full attention. Take breaks to stay focused.",
    "Stay refreshed and alert. Take a break from driving.",
    "Tiredness can impair your judgment. Rest before continuing."
]

current_eye_status = "Unknown"
current_yawn_count = 0
alarm_count = 0
face_detected_prev = False

lock = Lock()

FACE_DETECTION_BUFFER_SIZE = 5
face_detection_buffer = []

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("mixkit-alert-alarm-1005.wav")

def run_flask_app():
    app.run(port=5000, debug=False, use_reloader=False)

def select_random_message():
    return random.choice(warning_messages)

Thread(target=run_flask_app, daemon=True).start()

def load_ml_models():

    face_model_path = 'Models/frontal_face_detector.h5'

    mouth_model_path = 'Models/mouth_state_detection_model.h5'
    print(f"Loaded ML models: {face_model_path}, {mouth_model_path}")

load_ml_models()

camera_index = 1
external_webcams = [0, 2, 3, 4]
for index in external_webcams:
    time.sleep(5)
    test_camera = cv2.VideoCapture(index)
    print(f"Testing camera at index {index}: {'Success' if test_camera.isOpened() else 'Failed'}")
    if test_camera.isOpened():
        camera_index = index
        test_camera.release()
        print(f"Using camera at index {index}")
        break
    test_camera.release()

# Text-to-Speech Queue Setup
tts_queue = queue.Queue()
def tts_worker():
    engine = pyttsx3.init()
    while True:
        msg = tts_queue.get()
        if msg is None:
            break
        engine.say(msg)
        engine.runAndWait()
        tts_queue.task_done()
    engine.stop()

tts_thread = Thread(target=tts_worker)
tts_thread.start()

def alarm(msg):
    tts_queue.put(msg)

def stable_face_detection(face_detected):
    global face_detection_buffer, face_detected_prev
    face_detection_buffer.append(face_detected)
    if len(face_detection_buffer) > FACE_DETECTION_BUFFER_SIZE:
        face_detection_buffer.pop(0)

    detection_stable = False
    if all(face_detection_buffer):
        detection_stable = True
    elif not any(face_detection_buffer):
        detection_stable = False

    if detection_stable != face_detected_prev:
        if detection_stable:
            alarm("Face detected")
        else:
            alarm("Face undetected")
        face_detected_prev = detection_stable

    return detection_stable

def shutdown_tts():
    tts_queue.put(None)
    tts_thread.join()

atexit.register(shutdown_tts)

# Global flags
alarm_status = False
alarm_status2 = False
saying = False
alarm_playing = False

def check_and_send_sms():
    global alarm_count
    if alarm_count > 2:
        message = twilio_client.messages.create(
            body="Alert: please be warn that your partner is fallen sleep multiple times while driving !",
            from_=twilio_number,
            to=target_number
        )
        print(f"SMS sent: {message.sid}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    X = dist.euclidean(mouth[0], mouth[6])
    Y = (dist.euclidean(mouth[2], mouth[10]) + dist.euclidean(mouth[4], mouth[8])) / 2.0
    return Y / X

def draw_face_landmarks(frame, landmarks):
    for landmark in landmarks:
        hull = cv2.convexHull(landmark)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

def display_info(frame, ear, mouEAR, yawns, yawnStatus):
    eye_status = "Eyes Closed" if ear < EYE_AR_THRESH else "Eyes Open"
    yawn_status = "Yawning" if yawnStatus else "Not Yawning"
    cv2.putText(frame, f"{eye_status} - EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"{yawn_status} - MAR: {mouEAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawn Count: {yawns}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"ADMAS {current_time}", (370, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)

# Constants
EYE_AR_THRESH = 0.3
MOU_AR_THRESH = 0.8
FRAMES_PER_SECOND = 20
EYE_AR_CONSEC_FRAMES = FRAMES_PER_SECOND * 3

# Camera setup
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    print("Error: Camera could not be accessed. Please check the camera connection and try again.")
    exit()

# Load the dlib face detector and predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Facial landmarks for the left and right eye, and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

COUNTER, yawns = 0, 0
yawnStatus, prev_yawn_status = False, False

# Flask route for the root that serves the status
@app.route('/')
def status():
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Status</title>
    </head>
    <body>
        <h1>Alertness Monitoring System</h1>
        <div id="status">Loading...</div>
        <script>
            function updateStatus() {
                var xhttp = new XMLHttpRequest();
                xhttp.onreadystatechange = function() {
                    if (this.readyState == 4 && this.status == 200) {
                        var data = JSON.parse(this.responseText);
                        document.getElementById("status").innerHTML =
                            "Eye Status: " + data.eye_status + "<br>" +
                            "Yawn Counter: " + data.yawn_counter + "<br>" +
                            "Alarm Count: " + data.alarm_count;
                    }
                };
                xhttp.open("GET", "/fetch-status", true);
                xhttp.send();
            }
            updateStatus(); // Update status on load
            setInterval(updateStatus, 3000); // Update every 3 seconds
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/send-phone-number', methods=['POST'])
def receive_phone_number():
    phone_number = request.json.get('phone_number', '+94765958539')
    print("Received phone number:", phone_number)


    global target_number
    target_number = phone_number
    print("Updated target number:", target_number)


    return 'Phone number received: ' + phone_number

@app.route('/fetch-status')
def fetch_status():
    return jsonify({
        'eye_status': current_eye_status,
        'yawn_counter': current_yawn_count,
        'alarm_count': alarm_count
    })

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Failed to grab a frame from the camera. Exiting...")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = detector(gray, 0)
    face_detected = len(faces) > 0  # This is true if faces are detected, false otherwise

    stable_detection = stable_face_detection(face_detected)
    for face in faces:
        shape = face_utils.shape_to_np(predictor(gray, face))
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mouEAR = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_playing:
                alarm_sound.play()
                alarm_count += 1
                if alarm_count > 5:
                    check_and_send_sms()
                print("Alarm count: " + str(alarm_count))
                alarm_playing = True
        else:
            if alarm_playing:
                alarm_sound.stop()
                alarm_playing = False
            COUNTER = 0

        if mouEAR > MOU_AR_THRESH:
            yawnStatus = True
        else:
            yawnStatus = False
        if prev_yawn_status != yawnStatus and yawnStatus:
            yawns += 1
            if yawns % 3 == 0:
                message = select_random_message()
                alarm(message)

        draw_face_landmarks(frame, [leftEye, rightEye, mouth])
        display_info(frame, ear, mouEAR, yawns, yawnStatus)

    prev_yawn_status = yawnStatus

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()
pygame.mixer.quit()
shutdown_tts()
