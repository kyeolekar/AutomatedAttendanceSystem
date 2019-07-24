from flask import Flask, render_template, flash, request, url_for, redirect,  Response
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required, current_user
from flask_mail import Mail

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Sequence

import cv2
import os
import numpy as np
import cv2.face
import time

from datetime import datetime


from flask import Flask, render_template, Response
import cv2
import cv2.face
import numpy as np
import os
import pytz

class FaceRecognition:

    faces_to_save = {}
    select_key = 0

    webcam = None

    def __init__(self, select_key):
        self.select_key = self.select_key

    def start_webcam(self):
        self.webcam = cv2.VideoCapture(self.select_key)

    def stop_webcam(self):
        self.webcam.release()

    def cut_face(self, frame, face_coord):
        faces = []
        for (x, y, w, h) in face_coord:
            faces.append(frame[y + 1:y + h, x + 1:x + w])
        return faces

    def normalize_intensity(self, images):
        images_norm = []
        for image in images:
            is_color = len(image.shape) == 3
            if is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_norm.append(cv2.equalizeHist(image))
        return images_norm


    def resize(self, images, size=(50, 50)):
        images_norm = []
        for image in images:
            if image.shape < size:
                image_norm = cv2.resize(image, size, interpolation= cv2.INTER_AREA)
            else:
                image_norm = cv2.resize(image, size, interpolation= cv2.INTER_CUBIC)
            images_norm.append(image_norm)
        return images_norm


    def collect_dataset(self):
        images = []
        labels = []
        labels_dic = {}
        people = [person for person in os.listdir("people/")]
        for i, person in enumerate(people):
            labels_dic[i] = person
            for image in os.listdir("people/" + person):
                images.append(cv2.imread("people/" + person + "/" + image, 0))
                labels.append(i)
        return images, np.array(labels), labels_dic

    def detect_face(self, frame):
        detector = cv2.CascadeClassifier('frontal_face.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coord = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

        for (x, y, w, h) in face_coord:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame, face_coord


    def start_detection(self, person_name):
        # webcam = cv2.VideoCapture(select_key)
        self.start_webcam()

    def train_recognizer(self):
        images, labels, labels_dic = self.collect_dataset()
        rec_eig = cv2.face.createLBPHFaceRecognizer()
        rec_eig.train(images, labels)
        rec_eig.save("faces_trained_lbph.xml")

        images, labels, labels_dic = self.collect_dataset()
        rec_eig = cv2.face.createEigenFaceRecognizer()
        rec_eig.train(images, labels)
        rec_eig.save("faces_trained_eigen.xml")

        images, labels, labels_dic = self.collect_dataset()
        rec_eig = cv2.face.createFisherFaceRecognizer()
        rec_eig.train(images, labels)
        rec_eig.save("faces_trained_fisher.xml")

    def start_recognition(self):
        engine = create_engine('mysql+pymysql://root:root@127.0.0.1/flaskapp')
        Base = declarative_base()

        class User(Base):
            __tablename__ = 'attendance_table'
            id = Column(Integer, primary_key=True, autoincrement=True)
            user_id = Column(String(255))
            date = Column(DateTime)
            present = Column(String(12))


        Session = sessionmaker(bind=engine)
        Session.configure(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        session._model_changes = {}

        self.start_webcam()
        self.train_recognizer()

        images, labels, labels_dic = self.collect_dataset()
        rec_eig = cv2.face.createLBPHFaceRecognizer()
        rec_eig.load("faces_trained_lbph.xml")

        rec_fisher = cv2.face.createFisherFaceRecognizer()
        rec_fisher.load("faces_trained_fisher.xml")

        rec_eigen = cv2.face.createEigenFaceRecognizer()
        rec_eigen.load("faces_trained_eigen.xml")

        timeout = time.time() + 10*2

        while True:
            test = 0
            if test == 5 or time.time() > timeout:
                break
            test = test - 1
            ret, frame = self.webcam.read()
            if self.webcam.isOpened():
                show_frame, coord = self.detect_face(frame)
                face_detected = self.cut_face(show_frame, coord)
                normal_faces = self.normalize_intensity(face_detected)
                resized_image = self.resize(normal_faces, (92, 112))
                for i, face in enumerate(resized_image):
                    collector = cv2.face.MinDistancePredictCollector()
                    rec_eig.predict(face, collector, 0)
                    confidence = collector.getDist()
                    prediction = collector.getLabel()

                    collector_eigen = cv2.face.MinDistancePredictCollector()
                    rec_eigen.predict(face, collector_eigen, 0)
                    confidence_eigen = collector_eigen.getDist()
                    prediction_eigen = collector_eigen.getLabel()

                    collector_fisher = cv2.face.MinDistancePredictCollector()
                    rec_fisher.predict(face, collector_fisher, 0)
                    confidence_fisher = collector_fisher.getDist()
                    prediction_fisher = collector_fisher.getLabel()

                    print("Prediction Fisher - ", labels_dic[prediction].capitalize())
                    print("Prediction Eigen - ", labels_dic[prediction].capitalize())
                    print("Prediction LBPH - ", labels_dic[prediction].capitalize())

                    threshold = 70
                    # if prediction == prediction_fisher == prediction_eigen:

                    if confidence > threshold and prediction == prediction_fisher and prediction_fisher == prediction_eigen:
                        cv2.putText(show_frame, labels_dic[prediction].capitalize() +" / " + str(round(confidence_eigen)),
                                    (coord[i][0], coord[i][1]-10),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 61, 15), 3)
                        if( labels_dic[prediction].capitalize() in self.faces_to_save.keys()):
                            self.faces_to_save[labels_dic[prediction].capitalize()]+=1
                        else:
                            self.faces_to_save[labels_dic[prediction].capitalize()]=1
                    else:
                        cv2.putText(show_frame, "Unknown / " + str(round(confidence)),
                                    (coord[i][0], coord[i][1] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 61, 15), 3)
                cv2.imwrite('t.jpg', show_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
        self.stop_webcam()

        for key, value in self.faces_to_save.items():
            if value>10:
                date = datetime.now(pytz.timezone("Asia/Kolkata"))
                ed_user = User(user_id=key, date=date, present="True")
                session.add(ed_user)
                session.commit()
                print ('adding to database')

        # return faces_to_save
