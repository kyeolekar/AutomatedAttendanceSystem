import cv2
import os
import numpy as np
import cv2.face

class FaceDetector:

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

        count = 0
        timer = 50

        folder = "people/" + person_name.lower()

        image_count = 20

        if not os.path.exists(folder):
            os.mkdir(folder)

        while count < image_count:
            ret, frame = self.webcam.read()
            if self.webcam.isOpened():
                show_frame, coord = self.detect_face(frame)
                if len(coord) and timer % 700 == 50:
                    face_detected = self.cut_face(show_frame, coord)
                    normal_faces = self.normalize_intensity(face_detected)
                    resized_image = self.resize(normal_faces, (92, 112))
                    cv2.imwrite(folder + "/" + str(count)+".png", resized_image[0])
                    count += 1

                cv2.putText(show_frame, "Saved "+ str(count) + " of " + str(image_count),
                                            (100,100),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 61, 15), 2)
                cv2.imwrite('facedetection.jpg', show_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + open('facedetection.jpg', 'rb').read() + b'\r\n')
                cv2.waitKey(10)
                timer += 50

        self.stop_webcam()
