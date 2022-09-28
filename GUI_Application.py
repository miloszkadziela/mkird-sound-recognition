from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget, plot
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRectF
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint
import matplotlib.pyplot as plt

import librosa
import librosa.display
import joblib
import pyaudio # requires portaudio installed
import wave
import threading
import time
import commons

from collections import deque

import numpy as np
import pandas as pd

WAVE_OUTPUT_FILENAME = "output.wav"

MODEL_NAME = 'model.joblib'

# The following must not be changed, otherwise we get a different number of elements in our feature vector.
# That in turn causes an exception to be thrown from the classifier.
RECORD_SECONDS = 5
CHANNELS = 1
FORMAT = pyaudio.paInt32
RATE = 44100
CHUNK = 1024
FRAMES_IN_WAVEFILE = int(RATE / CHUNK * RECORD_SECONDS)

FRAMES = deque([], maxlen=2*FRAMES_IN_WAVEFILE)
AUDIO = pyaudio.PyAudio()

def record(should_continue) -> None:
    """
    Records audio and saves it into a global FRAMES variable.
    """
    stream = AUDIO.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    stream.start_stream()

    while should_continue[0]:
        data = stream.read(CHUNK)
        FRAMES.append(data)

    stream.stop_stream()
    stream.close()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.x = list(range(100))  # 100 time points
        self.y = [randint(0,100) for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(0, 0, 255))
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)

        self.font = QtGui.QFont("Times", 20, QtGui.QFont.Bold)

        self.label = QLabel(self)
        self.label.setFont(self.font)
        # self.label.setGeometry(300, 300, 350, 250)
        self.label.setGeometry(self.graphWidget.height() / 2, 300, 350, 250)
        self.label.setText("Classification result")

        # ... init continued ...
        self.timer = QtCore.QTimer()
        self.timer.setInterval(110)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        self.model = joblib.load(MODEL_NAME)
        self.graph_update_lock = threading.Lock()
        self.frames = deque([], maxlen=FRAMES_IN_WAVEFILE)

    def update_plot_data(self):
        w = self.graphWidget.width() / 5
        if w < self.label.width():
            w = self.label.width()
        h = self.graphWidget.height() / 6
        if h < self.label.height():
            h = self.label.height()

        self.label.setGeometry(self.graphWidget.width() - w, self.graphWidget.height() - h, w, h)

        self.graph_update_lock.acquire(False)
        while len(FRAMES) > 0:
            self.frames.append(FRAMES.popleft())

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(AUDIO.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        sound = self.model.fileToX([WAVE_OUTPUT_FILENAME])

        self.label.setText(self.model.classify(sound)[0])

        Y = sound[0][0]
        self.data_line.setData(np.linspace(0, RECORD_SECONDS, len(Y)), Y)
        self.graph_update_lock.release()

def main():
    should_continue = [True]
    record_thread = threading.Thread(target=record, args=[should_continue])
    record_thread.start()
    time.sleep(RECORD_SECONDS + 1)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    ret = app.exec_()
    should_continue[0] = False
    AUDIO.terminate()
    sys.exit(ret)


if __name__ == '__main__':
    main()