import cv2
import numpy as np

def recognize_face(img, model):
    label = np.argmax(model.predict(img))
    return label

def preprocess_image(img):
    print(len(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250,250))
    img = np.asarray([img/255])
    return img

def call_api(label):
    return("attendance marked for " + str(label))