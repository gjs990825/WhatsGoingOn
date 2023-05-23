import logging
import random
import sys
import time

import cv2
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt, QTimerEvent, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QFileDialog, QMessageBox

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


class Help(QtWidgets.QDialog, Ui_info):
    def __init__(self, *args, obj=None, **kwargs):
        super(Help, self).__init__(*args, **kwargs)
        self.setupUi(self)


class WGO(QtWidgets.QMainWindow, Ui_mainWindow):
    """
    WGO (What's Going On)
    """

    def __init__(self, *args, obj=None, **kwargs):
        super(WGO, self).__init__(*args, **kwargs)

        self.video_file_name = None
        self.input_playback_fps = 0
        self.input_playback_last_t = 0

        self.camera_mode = False
        self.recorder = None
        self.capture = None
        self.video_fps = 25
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.current_frame = None
        self.current_output_frame = None
        self.frame_seeking_flag = False
        self.frame_seeking_position = None
        self.frame_jumping_flag = False
        self.frame_jumping_position = None
        self.image_process_queue = None
        self.configs = None
        self.current_config = None

        self.start_position = 0
        self.end_position = 0

        # Some settings
        # self.transformation_mode = Qt.TransformationMode.FastTransformation  # Performance
        self.transformation_mode = Qt.TransformationMode.SmoothTransformation  # Better viewing quality
        self.forward_seconds = 15  # Seconds skipped using arrow key
        self.max_fps = 160  # Limit of output rate, see frame seeking section for details
        self.playback_speed_max = 32.0
        self.playback_speed_min = 1 / 16
        self.playback_speed = 1.0  # Default playback speed
        # Some lambdas
        # Get current playback fps
        self.playback_fps = lambda: self.playback_speed * self.video_fps
        # Get current frame interval with max fps limitation
        self.timer_interval = lambda: int(1000 / min(float(self.max_fps), (self.video_fps * self.playback_speed)))
        # Get frame number needed to skip
        self.forward_frames = lambda: self.video_fps * self.forward_seconds
        self.last_frame = lambda: self.total_frame_number - 1 if self.total_frame_number else 0

        self.ui_prepare()

    def ui_prepare(self):
        self.setupUi(self)
        icon = QtGui.QIcon('resources/ic_app.png')
        self.setWindowIcon(icon)

        self.actionOpenVideo.triggered.connect(self.open_video_file)
        self.actionOpenCamera.triggered.connect(lambda: print('open carmra'))

        self.startSlider.valueChanged.connect(self.start_set)
        self.endSlider.valueChanged.connect(self.end_set)

        self.startButton.clicked.connect(lambda: self.start_set(self.playbackSlider.value()))
        self.endButton.clicked.connect(lambda: self.end_set(self.playbackSlider.value()))
        self.videoAll.clicked.connect(self.video_all)

        self.playbackSlider.sliderReleased.connect(self.video_resume)
        self.playbackSlider.sliderPressed.connect(self.video_pause)
        self.playbackSlider.sliderMoved.connect(self.frame_position_slider_callback)

        self.videoPrepare.clicked.connect(self.prepare_video_clip)
        self.camPrepare.clicked.connect(lambda: print('prepare cam'))

        self.actionInfo.triggered.connect(self.show_help)

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
        # now = time.time()
        # interval = now - self.input_playback_last_t
        # if interval != 0:
        #     self.input_playback_fps = (self.input_playback_fps + (1.0 / interval)) / 2
        #     self.input_playback_last_t = now
        #     print("Input: %5.1f FPS" % self.input_playback_fps)

        self.check_frame_seeking()

        if not self.camera_mode:
            if not self.frame_jumping_flag and not self.frame_seeking_flag:
                self.frame_position += 1
            else:
                self.frame_seeking_and_jumping()
            self.playbackSlider.setValue(int(self.frame_position))

        success, img = self.capture.read()
        if success:
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.ui_image_process(self.current_frame, self.imageDisplay)
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
        start = self.start_position
        end = self.end_position
        if end - start < 16:
            self.prompt('At least select 16 frames!')
            return
        self.detect()

    def dummy_detect(self):
        time.sleep(2)
        model = QStandardItemModel()
        for i in range(10):
            model.appendRow(QStandardItem(f'Item {random.random()}'))
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

    def check_frame_seeking(self):
        playback_fps = self.playback_fps()
        if playback_fps <= self.max_fps:
            return
        ratio = playback_fps / self.max_fps
        self.frame_seek(self.frame_position + ratio)

    def frame_seek(self, position):
        self.frame_seeking_position = position
        self.frame_seeking_flag = True

    def timerEvent(self, event: QTimerEvent):
        if event.timerId() == self.video_timer_id:
            self.video_timer_handler()

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
        self.frame_position = 0
        self.sliders_reset()
        self.setWindowTitle(file_name)
        self.video_resume()

    def video_all(self):
        self.startSlider.setValue(0)
        self.endSlider.setValue(self.last_frame())

    def sliders_reset(self):
        last_frame = self.last_frame()
        self.playbackSlider.setRange(0, last_frame)
        self.startSlider.setRange(0, last_frame)
        self.endSlider.setRange(0, last_frame)
        self.endSlider.setValue(last_frame)

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
        if self.image_process_queue is not None:
            self.image_process_queue.join()
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
        logging.info(message)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = WGO()
    window.show()
    app.exec()
