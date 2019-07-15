import os
import re
from threading import Thread

from tensorflow.python.platform import gfile

from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import SessionRunner, SessionRunnable, Inference
import numpy as np
import cv2
import tensorflow as tf

def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            print("graph imported")
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


class CrowdCount():

    class Inference(Inference):

        def __init__(self, input, return_pipe=None, meta_dict=None):
            super().__init__(input, return_pipe, meta_dict)

    def __init__(self, model_path="/home/developer/PycharmProjects/footfall_api/py_data/crowd_count.pb", flush_pipe_on_read=True, graph_prefix=None):
        assert model_path.endswith('.pb')
        load_model(model_path)
        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        self.__run_session_on_thread = False
        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def preprocess(self, image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (img - 127.5) / 128
        x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        return x_in

    def __in_pipe_process(self, inference):
        resized = inference.get_input()
        preprocessed = self.preprocess(resized)
        # print(reshaped.shape)
        inference.set_data(preprocessed)
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        inference.set_result(result)
        if inference.get_return_pipe():
            return '\0'

        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_threading(self, run_on_thread=True):
        self.__run_session_on_thread = run_on_thread

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()

        self.__images_placeholder = self.__tf_sess.graph.get_tensor_by_name(self.__graph_prefix + "Placeholder:0")
        self.__output = self.__tf_sess.graph.get_tensor_by_name(self.__graph_prefix + "add_12:0")
        self.__output_size = self.__output.get_shape()[1]

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:
            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return
            self.__in_pipe.pull_wait()
            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__session_runner.get_in_pipe().push(
                    SessionRunnable(self.__job, inference, run_on_thread=self.__run_session_on_thread))

    def __job(self, inference):
        self.__out_pipe.push(
            (self.__tf_sess.run(self.__output,
                                feed_dict={self.__images_placeholder: inference.get_data()}), inference))

    def stop(self):
        self.__thread = None

    def save_for_serving(self, save_path, model_version):
        path = os.path.join(save_path, str(model_version))
        builder = tf.saved_model.builder.SavedModelBuilder(path)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tf.saved_model.utils.build_tensor_info(self.__images_placeholder)},
            outputs={
                'add_12': tf.saved_model.utils.build_tensor_info(self.__output)
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            self.__tf_sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'calculate_embeddings': prediction_signature,
            })

        builder.save()
        print("model saved successfully")
