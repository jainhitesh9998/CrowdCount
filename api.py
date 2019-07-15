from threading import Thread
from flask import Flask
from py_crowd_count.crowd import CrowdCount
from py_flask_movie.flask_movie import FlaskMovie
from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import SessionRunner
import cv2
import numpy as np

session_runner = SessionRunner()
generator = CrowdCount()
generator.use_threading()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
session_runner.start()
generator.run()

app = Flask(__name__)

pipe = Pipe(limit=1)
image_pipe = Pipe()
fs = FlaskMovie(app=app)
fs.create('crowd_feed', pipe, np.zeros((160, 160, 3)), timeout=1)

count_pipe = Pipe()

def capture_image():
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    while True:
        ret, image = cap.read()
        if ret:
            image_pipe.push(image)
Thread(target=capture_image).start()
@app.route("/crowd_count")
def count():
    ret, count =  count_pipe.pull()
    output = "0"
    if ret:
        output=str(count)
    return output

def crowd_counter():
    while True:
        ret, image = image_pipe.pull()
        if not ret:
            continue
        inference = CrowdCount.Inference(image)
        generator_ip.push(inference)
        generator_op.pull_wait()
        success, inference = generator_op.pull(True)
        if not success:
            continue
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (200, 40),
                      (255, 255, 255), -1)
        count = np.absolute(np.int32(np.sum(inference.get_result())))
        count_pipe.push(count)
        cv2.putText(overlay, "Crowd Count {}".format(count), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0))
        # cv2.addWeighted(image[:30, :60, :,], 0.8, text_image, 0.2,0.2)
        cv2.addWeighted(overlay, 1.0, image, 0,
                        0, image)
        pipe.push(image)

Thread(target=crowd_counter).start()