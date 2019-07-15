from threading import Thread

import cv2
from flask import Flask, Response
from flask_cors import CORS


class FlaskMovie:

    def __init__(self, app=None):
        if not app:
            app = Flask(__name__)
        self.__app = app
        self.__routes_pipe = {}
        self.__routes_no_feed_img = {}
        self.__routes_timeout = {}
        self.__routes_allow_flush = {}
        CORS(app)

        @self.__app.route('/<route>')
        def video_feed(route):
            return Response(self.__generate(self.__routes_pipe[route], self.__routes_no_feed_img[route],
                                            self.__routes_timeout[route], self.__routes_allow_flush[route]),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def __generate(self, pipe, no_feed_img, timeout, allow_flush):
        while True:
            try:
                pipe.pull_wait(timeout)
                ret, image = pipe.pull(allow_flush)
                if not ret:
                    image = no_feed_img
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

            except GeneratorExit as ex:
                print(ex)
                return
            except:
                print("no feed available yet...")
                pass

    def create(self, route, pipe, default_img=None, timeout=None, allow_flush=True):
        if timeout is None:
            timeout = 1
        self.__routes_pipe[route] = pipe
        self.__routes_no_feed_img[route] = default_img
        self.__routes_timeout[route] = timeout
        self.__routes_allow_flush[route] = allow_flush

    def start(self, bind_ip, bind_port):
        Thread(target=self.__app.run, args=(bind_ip, bind_port,)).start()
