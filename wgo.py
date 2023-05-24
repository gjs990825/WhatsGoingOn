import abc
import logging
import os
import pickle
import random
import sys
import tempfile
import time
from abc import ABC

import cv2
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt, QTimerEvent, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from video_processing import video_frame_extract
from wgo_info import Ui_info
from wgo_main import Ui_mainWindow

logging.basicConfig(level=logging.INFO)


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, work) -> None:
        super().__init__()
        self.work = work

    def run(self):
        self.work()
        self.finished.emit()


class WGOInterface(ABC):
    @abc.abstractmethod
    def wgo(self, clip) -> dict[str, float]:
        pass


class PickleWGO(WGOInterface):
    def wgo(self, clip) -> dict[str, float]:
        with open('test.pickle', 'wb') as f:
            pickle.dump(clip, f)
        return {f'Item {i}': random.random() for i in range(10)}


class Help(QtWidgets.QDialog, Ui_info):
    def __init__(self, *args, obj=None, **kwargs):
        super(Help, self).__init__(*args, **kwargs)
        self.setupUi(self)


class WGO(QtWidgets.QMainWindow, Ui_mainWindow):
    """
    WGO (What's Going On)
    """
    APP_NAME = 'WGO'

    def __init__(self, *args, obj=None, **kwargs):
        super(WGO, self).__init__(*args, **kwargs)

        self.wgo = PickleWGO()

        self.capture_timer_id = None
        self.recording_path = None

        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_file_name = None
        self.input_playback_fps = 0
        self.input_playback_last_t = 0

        self.camera_mode = False
        self.recorder = None
        self.capture = None
        self.video_fps = 25
        self.video_length = 0
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.current_frame = None
        self.frame_seeking_flag = False
        self.frame_seeking_position = None
        self.frame_jumping_flag = False
        self.frame_jumping_position = None

        self.start_position = 0
        self.end_position = 0

        # Some settings
        # self.transformation_mode = Qt.TransformationMode.FastTransformation  # Performance
        self.transformation_mode = Qt.TransformationMode.SmoothTransformation  # Better viewing quality

        # Some lambdas
        # Get current frame interval with max fps limitation
        self.timer_interval = lambda: int(1000 / self.video_fps)
        self.last_frame = lambda: self.total_frame_number - 1 if self.total_frame_number else 0
        self.current_position_in_seconds = lambda: self.frame_position / self.video_fps

        self.ui_prepare()

    def ui_prepare(self):
        # windows and icon
        self.setupUi(self)
        icon = QtGui.QIcon('resources/ic_app.png')
        self.setWindowIcon(icon)

        self.actionOpenVideo.triggered.connect(
            lambda: (self.cameraModeRadioButton.setChecked(False), self.open_video_file()))
        self.actionOpenCamera.triggered.connect(lambda: self.cameraModeRadioButton.setChecked(True))

        self.startSlider.valueChanged.connect(self.start_set)
        self.endSlider.valueChanged.connect(self.end_set)

        self.startButton.clicked.connect(lambda: self.start_set(self.playbackSlider.value()))
        self.endButton.clicked.connect(lambda: self.end_set(self.playbackSlider.value()))
        self.videoAll.clicked.connect(self.video_all)

        self.playbackSlider.sliderReleased.connect(self.video_resume)
        self.playbackSlider.sliderPressed.connect(self.video_pause)
        self.playbackSlider.sliderMoved.connect(self.frame_position_slider_callback)

        self.cameraModeWidgets.setEnabled(False)  # This will not recursively disable sub widgets.

        # buttons and actions
        self.goButton.clicked.connect(self.prepare_video_clip)
        self.actionInfo.triggered.connect(self.show_help)
        self.cameraModeRadioButton.toggled.connect(self.camera_mode_switch)
        self.camStart.toggled.connect(self.capture_switch)
        self.camAutoCapture.clicked.connect(self.capture_auto)

    def start_set(self, position):
        self.start_position = position
        self.startSlider.setValue(position)
        if position > self.end_position:
            self.end_set(position)

    def end_set(self, position):
        self.end_position = position
        self.endSlider.setValue(position)
        if position < self.start_position:
            self.start_set(position)

    def frame_position_slider_callback(self):
        position = self.playbackSlider.value()
        logging.debug(f'Slider moved to {position}')
        self.frame_jump_to(position)

    def video_timer_handler(self):
        if not self.camera_mode:
            if not self.frame_jumping_flag and not self.frame_seeking_flag:
                self.frame_position += 1
            else:
                self.frame_seeking_and_jumping()
            self.playbackSlider.setValue(int(self.frame_position))
            self.playbackInfo.setText(
                f'{self.current_position_in_seconds() : 08.2f} / {self.video_length : 08.2f}')

        success, img = self.capture.read()
        if success:
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.ui_image_process(self.current_frame, self.imageDisplay)
            if self.recorder is not None:
                self.recorder.write(img)
        else:
            playback_end = self.frame_position >= self.total_frame_number

            if playback_end:
                if self.video_file_name:
                    self.load_video(self.video_file_name)
            else:
                message = 'Frame read failed, file might be corrupted'
                logging.warning(message)
                logging.warning(f'Frame info:current {self.frame_position}, total {self.total_frame_number}')
                self.video_stop(message)

    def ui_image_process(self, source, target):
        q_image = QtGui.QImage(source.data,
                               source.shape[1],
                               source.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(target.size(),
                               Qt.AspectRatioMode.KeepAspectRatio,
                               self.transformation_mode)
        target.setPixmap(scaled)

    def prepare_video_clip(self):
        if self.video_file_name is None:
            self.prompt('No video loaded!')
            return

        if self.end_position - self.start_position < 16:
            self.prompt('At least select 16 frames!')
            return

        self.detect()

    def dummy_detect(self):
        start, end = self.startSlider.value(), self.endSlider.value()
        data = video_frame_extract(16, self.video_file_name, start, end)

        result_dict = self.wgo.wgo(data)

        model = QStandardItemModel()
        for k, v in result_dict.items():
            model.appendRow(QStandardItem(f'{k}: {v * 100 :04.1f}'))

        self.resultDisplay.setModel(model)
        self.resultDisplay.update()

    def detect(self):
        self.q_thread = QThread()
        self.worker = Worker(self.dummy_detect)
        self.worker.moveToThread(self.q_thread)
        # Connect signals and slots
        self.q_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.q_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.q_thread.finished.connect(self.q_thread.deleteLater)

        self.q_thread.start()

        self.goButton.setEnabled(False)
        self.q_thread.finished.connect(
            lambda: self.goButton.setEnabled(True)
        )

    def show_help(self):
        dialog = Help(self)
        dialog.show()

    def prompt(self, message):
        QMessageBox.information(self, 'Info', message)

    def frame_jump_to(self, position):
        self.frame_jumping_position = position
        self.frame_jumping_flag = True

    def timerEvent(self, event: QTimerEvent):
        if event.timerId() == self.video_timer_id:
            self.video_timer_handler()
        elif event.timerId() == self.capture_timer_id:
            self.camStart.setChecked(False)
            self.camAutoCapture.setEnabled(True)

    def frame_seeking_and_jumping(self):
        if self.frame_seeking_flag:
            if self.frame_position >= self.frame_seeking_position:
                logging.warning(f'Can not seek backwards')
                self.frame_position += 1
            else:
                t_s = time.time()
                for i in range(int(self.frame_position), int(self.frame_seeking_position) - 1):
                    self.capture.grab()  # grab() does not process frame data, for performance improvement
                self.frame_position = self.frame_seeking_position
                t = time.time() - t_s
                logging_using = logging.debug if self.playback_speed > 1.0 else logging.info
                logging_using('Seeking from %.1f to %.1f, %.3fs used' %
                              (self.frame_position, self.frame_seeking_position, t))
            self.frame_seeking_flag = False
        if self.frame_jumping_flag:
            t_s = time.time()
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jumping_position)
            t = time.time() - t_s
            logging.info('Jumping from %.1f to %.1f, %.3fs used' %
                         (self.frame_position, self.frame_jumping_position, t))
            self.frame_position = self.frame_jumping_position
            self.frame_jumping_flag = False

    def get_output_size(self):
        if self.current_frame is None:
            raise ValueError
        return self.current_frame.shape[1], self.current_frame.shape[0]

    def start_recording(self):
        try:
            size = self.get_output_size()
        except ValueError:
            self.statusBar().showMessage('Nothing playing!')
            return
        self.recording_path = os.path.join(self.temp_dir.name, f'{time.time()}.mp4')
        logging.info(f'Recording to {self.recording_path}')
        self.recorder = cv2.VideoWriter(self.recording_path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.video_fps), size)
        self.statusBar().showMessage('Recording...')

    def stop_recording(self):
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None
            self.statusBar().showMessage('Stop recording.')

    def open_camera(self):
        self.video_stop()
        logging.info(f'Using camera...')
        self.camera_mode = True
        self.capture = cv2.VideoCapture(0)
        self.total_frame_number = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f'Video frame count: {self.total_frame_number}')

        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps > 60:
            logging.error(f'Abnormal fps: {self.video_fps}, reset to default fps')
            self.video_fps = 25
        else:
            logging.info(f'Video fps: {self.video_fps}')
        self.video_length = self.total_frame_number / self.video_fps

        self.frame_position = 0
        self.setWindowTitle('Capturing from default camera...')
        self.video_resume()

    def open_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open a video file', 'videos',
                                                   'Video files (*.avi *.mp4);;All Files(*)')
        if file_name == '':
            return
        logging.info(f'Video file: {file_name}')
        self.load_video(file_name)

    def load_video(self, file_name):
        self.video_file_name = file_name
        self.video_stop()
        self.capture = cv2.VideoCapture(file_name)
        self.total_frame_number = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f'Video frame count: {self.total_frame_number}')
        if self.total_frame_number <= 0:
            self.video_stop('Frame error, file might be corrupted')
            return

        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps > 60:
            logging.error(f'Abnormal fps: {self.video_fps}, reset to default fps')
            self.video_fps = 25
        else:
            logging.info(f'Video fps: {self.video_fps}')
        self.video_length = self.total_frame_number / self.video_fps
        self.frame_position = 0
        self.sliders_reset_range()
        self.setWindowTitle(file_name)
        self.video_resume()

    def video_all(self):
        self.startSlider.setValue(0)
        self.endSlider.setValue(self.last_frame())

    def sliders_reset_range(self):
        last_frame = self.last_frame()
        self.playbackSlider.setRange(0, last_frame)
        self.startSlider.setRange(0, last_frame)
        self.endSlider.setRange(0, last_frame)

    def video_pause(self):
        if self.video_timer_id is not None:
            self.killTimer(self.video_timer_id)
            self.video_timer_id = None
            message = 'Playback Paused'
            self.statusBar().showMessage(message)
            logging.info(message)

    def video_resume(self):
        if self.video_timer_id is None:
            self.video_timer_id = self.startTimer(self.timer_interval(), Qt.TimerType.CoarseTimer)
            message = 'Playback Resumed'
            self.statusBar().showMessage(message)
            self.playbackSlider.setEnabled(True)
            logging.info(message)

    def video_stop(self, message=None):
        self.video_pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.current_frame = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.frame_jumping_flag = False
        self.frame_seeking_flag = False
        message = 'Playback Stopped' if message is None else message
        logging.info(message)
        self.camera_mode = False
        self.statusBar().showMessage(message)
        self.playbackSlider.setEnabled(False)

        self.imageDisplay.setText('Ctrl+O to load a video.\nCtrl+C to open default camera device.')

        logging.info(message)

    def camera_mode_switch(self, x):
        if x:
            self.open_camera()
        else:
            self.video_stop()
            self.stop_recording()
            if self.recording_path is None:
                return
            self.load_video(self.recording_path)
            self.recording_path = None

    def capture_switch(self, x):
        if x:
            self.start_recording()
        else:
            self.stop_recording()
            self.cameraModeRadioButton.setChecked(False)

    def capture_auto(self):
        self.camAutoCapture.setEnabled(False)
        self.camStart.setChecked(True)
        interval = self.camAutoCaptureSpinBox.value() * 1000
        self.capture_timer_id = self.startTimer(interval, Qt.TimerType.CoarseTimer)

    def closeEvent(self, event: QtGui.QCloseEvent):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self.video_stop()  # Stop all work
        self.stop_recording()
        self.temp_dir.cleanup()
        event.accept()

    def setWindowTitle(self, title: str = None):
        if title is None:
            super().setWindowTitle(self.APP_NAME)
            return
        if len(title) > 55:
            title = f'{title[:25]}...{title[-25:]}'
        super().setWindowTitle(f'{self.APP_NAME} - {title}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = WGO()
    window.show()
    app.exec()
