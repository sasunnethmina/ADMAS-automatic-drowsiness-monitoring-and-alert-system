import argparse
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame
import os
import threading

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("mixkit-alert-alarm-1005.wav")

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Global flags for alarm and message speaking
alarm_status = False
alarm_status2 = False
saying = False
alarm_playing = False

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
    cv2.putText(frame, "ADMAS", (370, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)

def alarm(msg):
    global saying
    saying = True
    os.system(f'espeak "{msg}"')
    saying = False

# Constants
EYE_AR_THRESH = 0.3
MOU_AR_THRESH = 0.75
FRAMES_PER_SECOND = 20
EYE_AR_CONSEC_FRAMES = FRAMES_PER_SECOND * 3

# Camera setup
camera = cv2.VideoCapture(args["webcam"])
if not camera.isOpened():
    print("Error: Camera could not be accessed.")
    exit()

# Load the dlib face detector and predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
#detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(predictor_path)

# Facial landmarks for the left and right eye, and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

COUNTER, yawns = 0, 0
yawnStatus, prev_yawn_status = False, False

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
    for face in faces:
        shape = face_utils.shape_to_np(predictor(gray, face))
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mouEAR = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_playing:
                alarm_status = True
                alarm_playing = True
                alarm_sound.play()
        else:
            if alarm_playing:
                alarm_sound.stop()
                alarm_playing = False
            alarm_status = False
            COUNTER = 0

        if mouEAR > MOU_AR_THRESH:
            yawnStatus = True
        else:
            yawnStatus = False
        if prev_yawn_status != yawnStatus and yawnStatus:
            yawns += 1
            if (yawns == 3 or yawns == 5) and not saying:
                message = "take some fresh air sir" if yawns == 3 else "take a small rest"
                threading.Thread(target=alarm, args=(message,)).start()

        draw_face_landmarks(frame, [leftEye, rightEye, mouth])
        display_info(frame, ear, mouEAR, yawns, yawnStatus)

    prev_yawn_status = yawnStatus
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()
