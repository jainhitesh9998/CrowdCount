import threading
from os.path import dirname, realpath
from threading import Thread
import tensorflow as tf
from py_pipe.pipe import Pipe


class SessionRunner:
    __config = tf.ConfigProto(log_device_placement=False)
    __config.gpu_options.allow_growth = True
    __counter = 0

    def __init__(self, skip=False):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__pause_resume = None
        self.__tf_sess = tf.Session(config=self.__config)
        self.__in_pipe = Pipe()
        self.__skip = skip

    def get_in_pipe(self):
        return self.__in_pipe

    def get_session(self):
        return self.__tf_sess

    def start(self):
        if self.__thread is None:
            self.__pause_resume = threading.Event()
            self.__thread = Thread(target=self.__start)
            self.__thread.start()

    def stop(self):
        if self.__thread is not None:
            self.__thread = None

    def __start(self):
        while self.__thread:
            self.__in_pipe.pull_wait()
            ret, sess_runnable = self.__in_pipe.pull(self.__skip)
            if ret:
                if type(sess_runnable) is not SessionRunnable:
                    raise Exception("Pipe elements must be a SessionRunnable")
                sess_runnable.execute(self.__tf_sess)


class SessionRunnable:
    def __init__(self, job_fnc, args_dict, run_on_thread=False):
        self.__job_fnc = job_fnc
        self.__args_dict = args_dict
        self.__run_on_thread = run_on_thread

    def execute(self, tf_sess):
        if self.__run_on_thread:
            Thread(target=self.__exec, args=(tf_sess,)).start()
        else:
            self.__exec(tf_sess)

    def __exec(self, tf_sess):
        with tf_sess.as_default():
            with tf_sess.graph.as_default():
                self.__job_fnc(self.__args_dict)


class Inference:
    def __init__(self, input, return_pipe=None, meta_dict=None):
        self.__input = input
        self.__meta_dict = meta_dict
        if not self.__meta_dict:
            self.__meta_dict = {}

        self.__return_pipe = return_pipe
        self.__data = None
        self.__result = None

    def get_input(self):
        return self.__input

    def get_meta_dict(self):
        return self.__meta_dict

    def get_return_pipe(self):
        return self.__return_pipe

    def set_result(self, result):
        self.__result = result
        if self.__return_pipe:
            self.__return_pipe.push(self)

    def get_result(self):
        return self.__result

    def set_data(self, data):
        self.__data = data

    def get_data(self):
        return self.__data

    def set_meta(self, key, val):
        if key not in self.__meta_dict.keys():
            self.__meta_dict[key] = val
        else:
            raise Exception

    def set_meta_force(self, key, val):
        self.__meta_dict[key] = val

    def get_meta(self, key):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return None

    def get_meta_or_default(self, key, val):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return val



#
# import random
# from threading import Thread
# from time import sleep
#
# from py_pipe.pipe import Pipe
#
# from py_tensorflow_runner.session_utils import SessionRunner, SessionRunnable, Inference
# import tensorflow as tf
#
# class AddOnGPU:
#
#     def __init__(self, flush_pipe_on_read=False):
#         self.__thread = None
#         self.__flush_pipe_on_read = flush_pipe_on_read
#         self.__run_session_on_thread = False
#         self.__in_pipe = Pipe(self.__in_pipe_process)
#         self.__out_pipe = Pipe(self.__out_pipe_process)
#
#     def __in_pipe_process(self, inference):
#         a, b = inference.get_input()
#         a = tf.constant(a, name="a")
#         b = tf.constant(b, name="b")
#         c = tf.add(a, b, name="c")
#         inference.set_data(c)
#         return inference
#
#     def __out_pipe_process(self, result):
#         result, inference = result
#         inference.set_result(result)
#         return inference
#
#     def get_in_pipe(self):
#         return self.__in_pipe
#
#     def get_out_pipe(self):
#         return self.__out_pipe
#
#     def use_threading(self, run_on_thread=True):
#         self.__run_session_on_thread = run_on_thread
#
#     def use_session_runner(self, session_runner):
#         self.__session_runner = session_runner
#         self.__tf_sess = session_runner.get_session()
#
#     def run(self):
#         if self.__thread is None:
#             self.__thread = Thread(target=self.__run)
#             self.__thread.start()
#
#     def __run(self):
#         while self.__thread:
#
#             if self.__in_pipe.is_closed():
#                 self.__out_pipe.close()
#                 return
#
#             self.__in_pipe.pull_wait()
#             ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
#             print(inference.get_data())
#             if ret:
#                 self.__session_runner.get_in_pipe().push(
#                     SessionRunnable(self.__job, inference, run_on_thread=self.__run_session_on_thread))
#
#     def __job(self, inference):
#         self.__out_pipe.push(
#             (self.__tf_sess.run(inference.get_data()), inference))
#
#
# session_runner = SessionRunner()
# session_runner.start()
#
# addOnGPU = AddOnGPU()
#
# ip = addOnGPU.get_in_pipe()
# op = addOnGPU.get_out_pipe()
#
# addOnGPU.use_session_runner(session_runner)
# addOnGPU.run()
#
#
# def send():
#     while True:
#         ip.push_wait()
#         inference = Inference([random.randint(0, 100), random.randint(0, 100)])
#         ip.push(inference)
#         sleep(10)
#
#
# def receive():
#     while True:
#         op.pull_wait()
#         ret, inference = op.pull()
#         if ret:
#             print('sum('+') = '+inference.get_result())
#
#
#
# Thread(target=send).start()
# Thread(target=receive).start()

