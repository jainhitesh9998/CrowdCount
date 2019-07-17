import argparse
import queue
import sys
import time
from threading import Thread

import azure.cognitiveservices.speech as speechsdk
import _thread
import requests
import soundfile as sf
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import matplotlib

from py_pipe.pipe import Pipe

matplotlib.use('TkAgg')
import pyloudnorm as pyln
turnstile_id = "demo_turnstile_0001"

# argument parsing
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)', default=7)
parser.add_argument(
    '-p', '--plot', type=int,
    help=' Plot the audio waveform ? yes:1      No:0 ' ,default=1)
args = parser.parse_args()


sound_level_list = list()
# state managment variables
sound_queue_length = 300
can_sst_start = True
can_validate = False
speech_key, service_region = "1e9f1691dcbd43758eadb5f7c2ddbd3f", "centralindia"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
SERVER_URL = 'http://localhost:4500/'
samplerate = 16000
downsample=4
window=200
interval=30
q = queue.Queue()
val = []
i = 0
volume_threshold= 0
lines,plotdata=None,None
noise_pipe = Pipe()
sound_level = 0


def audio_callback(indata, frames, time, status):
    """callback for audio queue update"""
    global val, i, can_sst_start
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())



def record(sec):
    """Start audio recording for n second"""
    global sound_level_list
    global sound_level
    while True:
        if q.empty():
            continue
        with sf.SoundFile('default.wav', mode='w', samplerate=16000,
                          channels=1) as file:
            t_end = time.time() + sec
            while time.time() < t_end:
                file.write(q.get())
        data, rate = sf.read("default.wav")  # load audio (with shape (samples, channels))
        meter = pyln.Meter(rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)  # measure loudness
        # loudness=-np.inf
        if np.isfinite(loudness):
            sound_level_list.append(loudness)
        else:
            sound_level_list.append(np.float64(-50.0))
        if(len(sound_level_list) > sound_queue_length):
            sound_level_list = sound_level_list[-sound_queue_length:]
        sound_level = sum(sound_level_list) / len(sound_level_list)
        noise_pipe.push(sound_level)
        print("\t\tSoundLevel: {}  LUFS".format(sound_level))
        print()
        print()

        # print(len(sound_level_list))

def update_plot(frame):
    """Callback for matplotlib on each plot update.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines
# ax.text(600, 1.1, "SoundLevel: {}".format(int(sound_level)))


def sound_loop():
    """ This loops till the program is shut down. All callbacks orginate from this function """
    global plotdata,lines, sound_level
    try:
        length = int(window * samplerate / (1000 * downsample))
        plotdata = np.zeros((length, 1))
        fig, ax = plt.subplots()
        lines = ax.plot(plotdata)
        ax.axis((0, len(plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
        # fig.tight_layout(pad=0)
        stream = sd.InputStream( device=args.device, channels=1,
            samplerate= samplerate, callback=audio_callback)

        if (args.plot==0):
            with stream:
                while True:
                    time.sleep(0.1)
        elif(args.plot==1):
            ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)

            with stream:
                plt.show()

    except Exception as e:
        print("ERROR:", e)

Thread(target=record, args=(1,)).start()
sound_loop()



#
# sec = 2
# while True:
#     with sf.SoundFile('default.wav', mode='w', samplerate=16000,
#                           channels=1) as file:
#             t_end = time.time() + sec
#             while time.time() < t_end:
#                 file.write(q.get())
#
#     data, rate = sf.read("default.wav") # load audio (with shape (samples, channels))
#     meter = pyln.Meter(rate) # create BS.1770 meter
#     loudness = meter.integrated_loudness(data) # measure loudness
#
#     print(loudness)

