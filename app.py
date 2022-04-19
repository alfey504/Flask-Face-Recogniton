from flask import Flask,render_template,Response
from cProfile import label
import cv2
import tensorflow as tf
import numpy as np
import Scripts.pythonscripts as ps
import cv2
import time
from flask_sse import sse
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

app=Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream')

cam = cv2.VideoCapture(0)
labels = ['Abraham Alfred Babu', 'Lijo George', 'Sharu Roy', 'Vineeth Abraham Shaji']
font = cv2.FONT_HERSHEY_SIMPLEX

org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

status_signal = {
    'status_code' : 0,
    'label': None
}
model = tf.keras.models.load_model('Models/face_rec.h5')

def generate_frames():
    prev_label = None
    detected_label_count = {
        'label': None,
        'count': None
    }

    while True:
        success, img = cam.read()
        if not success:
            break
        else:
            classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            bboxes = classifier.detectMultiScale(img, 1.1, 7)
            for box in bboxes:
                x, y, w, h = box
                face = img[y:y + h, x:x + w] 
                if len(face)<150:
                    break
                else:
                    face = ps.preprocess_image(face)
                    label = ps.recognize_face(face, model)
                    if label != prev_label:
                        if detected_label_count['label'] == label:
                            if detected_label_count['count'] > 10:
                                prev_label = label
                                status_signal['status_code'], status_signal['label'] = ps.call_api(label)
                                now = datetime.now()
                                current_time = now.strftime("%H:%M")
                                with app.app_context():
                                    sse.publish({"message":  labels[int(status_signal['label'])] + " punched in at " + str(current_time)}, type='publish')
                            else:
                                detected_label_count['count'] += 1
                        else:
                            detected_label_count['label'] = label
                            detected_label_count['count'] = 1

                        
                    img = cv2.putText(img, labels[label], org, font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.rectangle(img, (x, y), (x + w, y +h), (0,0,255), 1)

            ret, buffer=cv2.imencode('.jpg',img)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

