from numpy.distutils.tests.test_exec_command import emulate_nonposix

from py_crowd_count.crowd import CrowdCount
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

feed_vid = cv2.VideoCapture(-1)



while True:
    success, im = feed_vid.read()
    if not success:
        continue
    inference = CrowdCount.Inference(im)
    generator_ip.push(inference)
    generator_op.pull_wait()
    ret, inference = generator_op.pull(True)

    if ret:
        embedding = inference.get_result()
        sum = np.absolute(np.int32(np.sum(embedding)))
        print(sum)
    cv2.imshow("image", im)
    cv2.waitKey(1)