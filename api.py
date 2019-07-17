import json
from threading import Thread
from time import sleep

from flask import Flask, request, jsonify
from cloud_messaging.fcm import send_to_token, send_to_topic
import cloud_messaging.fcm
from py_crowd_count.crowd import CrowdCount
from py_flask_movie.flask_movie import FlaskMovie
from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import SessionRunner
import cv2
import numpy as np
from py_voice import voice_analysis
from firebase_admin.messaging import Notification
session_runner = SessionRunner()
generator = CrowdCount()
generator.use_threading()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
session_runner.start()
generator.run()

app = Flask(__name__)
noise_level_prev = None
pipe = Pipe(limit=1)
image_pipe = Pipe()
fs = FlaskMovie(app=app)
fs.create('crowd_feed', pipe, np.zeros((160, 160, 3)), timeout=1)
count_pipe = Pipe()

tokens = dict()
tokens["users"] = list()
tokens["vendors"] = list()

tokens["users"].append("ecmkOPxoOi4:APA91bFb-NyvZkt-D4e6DObeEjSIaIVA51eWYOQCG4l8UFdx2EElvG6SKF4NgmRj1S47gN-0GrsEwdpr-BLExV5VTPHqNwWV_u3Ian4-aTWpjQ01fnzpEokVPS9FA1A1HBmMYBvwVwRb")
def capture_image():
    # cap = cv2.VideoCapture(-1)
    cap = cv2.VideoCapture("/home/developer/PycharmProjects/footfall_api/crowd5.mp4")
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    while True:
        ret, image = cap.read()
        sleep(0.1)
        if ret:
            image_pipe.push(image)
Thread(target=capture_image).start()

@app.route("/noise_level")
def noise_level():
    global noise_level_prev
    ret, output = voice_analysis.noise_pipe.pull()
    print(ret)
    if not ret:
        return noise_level_prev
    noise_level_prev = str(output)
    return noise_level_prev

@app.route("/crowd_count")
def count():
    ret, count =  count_pipe.pull()
    output = "0"
    if ret:
        output=str(count)
    return output
#
# @app.route("/notify/user", methods=["GET"])
# def notify():
#     return send_to_token(
#         "ecmkOPxoOi4:APA91bFb-NyvZkt-D4e6DObeEjSIaIVA51eWYOQCG4l8UFdx2EElvG6SKF4NgmRj1S47gN-0GrsEwdpr-BLExV5VTPHqNwWV_u3Ian4-aTWpjQ01fnzpEokVPS9FA1A1HBmMYBvwVwRb")

@app.route("/token", methods=["POST"])
def recieve_token():
    payload = request.get_json()
    print(payload)
    if payload is None:
        return "error"
    if payload["type"] == "user":
        tokens["users"].append(payload["token"])
    else:
        tokens["vendors"].append(payload["token"])


@app.route("/notify/topic", methods=["POST"])
def notify():
    payload = request.get_json()
    print(payload)
    if payload is None:
        return "error"
    notification = Notification()
    notification.title = "Order Status"
    print(type(payload))
    notification.body=json.dumps(payload)
    if payload["topic"] == "user":
        return send_to_token(
                tokens["users"], notification=notification)
    else:
        return send_to_token(
            tokens["vendors"][-1], notification=notification)

    # "ecmkOPxoOi4:APA91bFb-NyvZkt-D4e6DObeEjSIaIVA51eWYOQCG4l8UFdx2EElvG6SKF4NgmRj1S47gN-0GrsEwdpr-BLExV5VTPHqNwWV_u3Ian4-aTWpjQ01fnzpEokVPS9FA1A1HBmMYBvwVwRb"

    #
    # return "success"


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
Thread(target=voice_analysis.record, args=(1,)).start()
Thread(target=voice_analysis.sound_loop).start()
fs.start("127.0.0.1", 5001)
