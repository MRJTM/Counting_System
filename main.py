"""
@File       : main.py
@Author     : Zhijie Cao
@Email      : dearzhijie@sgmail.com
@Date       : 2020/12/2
@Desc       : The main script of the counting system
"""

import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QSplashScreen,qApp
from PyQt5.QtGui import QIcon, QImage, QPixmap,QFont
from PyQt5.QtCore import pyqtSlot, Qt, QEvent, QDir, QTimer
from PyQt5.QtGui import QPalette
from PyQt5.Qt import Qt
from ui_countingtool import Ui_CountingTool
import qdarkstyle

import torch
from nets.cddnet import CDDNet3
from algorithms import *
from utils.utils import load_model
from utils.image import resize_img_with_short_len
from utils.image import resize_img_with_short_len, img_to_tensor
from utils.model import predict_bbox, predict_bbox_and_density
from utils.visualize import generate_heatmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # UI init
        self.ui = Ui_CountingTool()
        self.ui.setupUi(self)

        # button callback set
        self.ui.barScale.installEventFilter(self)
        self.ui.btnImage.clicked.connect(self.btnImage_clicked)
        self.ui.btnVideo.clicked.connect(self.btnVideo_clicked)
        self.ui.btnCamera.clicked.connect(self.btnCamera_clicked)
        self.ui.btnRun.clicked.connect(self.run)
        self.ui.btnClear.clicked.connect(self.btnClear_clicked)
        self.ui.btnStop.clicked.connect(self.btnStop_clicked)

        self.model_path = 'trained_models/CDDNet3_vgg16_BIG_focal_mse0001_r05/best.t7'

        # self.model = CDDNet3(num_classes=1,backbone='mobilenet_v2')
        self.model = CDDNet3(num_classes=1, backbone='vgg16_bn')

        """Timers"""
        self.timerFrame = QTimer()  # SRC window flush Timer
        self.timerCamera = QTimer()  # Camera function update Timer
        self.timerVideo = QTimer()  # Video function update Timer

        self.btnClear_clicked()

        print("\n--------------[Init]--------------")
        print("load model")
        print("model_path:{}".format(self.model_path))
        if torch.cuda.is_available() and self.device == 'gpu':
            dev = torch.device('cuda:0')
            self.model = load_model(self.model, self.model_path,map_location='cuda:0')
        else:
            dev = torch.device('cpu')
            self.model = load_model(self.model, self.model_path, map_location='cpu')


        self.model = self.model.to(dev)
        self.model.eval()

    def show_start_logo(self,sp):
        for i in range(1,11):
            time.sleep(0.2)
            sp.showMessage("Loading... {0}%".format(i * 10), Qt.AlignLeft |Qt.AlignBottom, Qt.black)

    # button "image" clicked callback function
    def btnImage_clicked(self):
        dlg = QFileDialog()
        filename, ok = dlg.getOpenFileName()
        if filename:
            img = cv2.imread(filename)
            self.img = img

            img = cv2.resize(img, (684, 513))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showImage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.ui.winInput.setPixmap(QPixmap.fromImage(showImage))

            self.mode = 'image'

        return filename

    # button "video" clicked callback function
    def btnVideo_clicked(self):
        print("btnVideo_clicked")
        dlg = QFileDialog()
        filename, ok = dlg.getOpenFileName()
        print("file_name:", filename)
        if filename:
            print("video_path:", filename)
            self.video_path = filename
            self.cap = cv2.VideoCapture(filename)
            ret, img = self.cap.read()
            print("img.shape:", img.shape)
            img = cv2.resize(img, (684, 513))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showImage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.ui.winInput.setPixmap(QPixmap.fromImage(showImage))
            self.mode = 'video'
            self.frame_index = 0
            self.total_frame_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_over = False


    # button "camera" clicked callback function
    def btnCamera_clicked(self):
        self.cap = cv2.VideoCapture(0)
        ret, img = self.cap.read()
        img = cv2.resize(img, (684, 513))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.winInput.setPixmap(QPixmap.fromImage(showImage))
        self.mode = 'camera'
        self.frame_index = 0
        self.short_len = 480
        self.score_threshold = 0.15

        # start frame update Timer
        self.timerFrame.start(1000 / 60)
        self.timerFrame.timeout.connect(self.update_frame)

    # button "clear" clicked callback function
    def btnClear_clicked(self):
        """model config related：static variables"""
        self.device = 'cpu'
        self.scale_threshold = 8
        self.score_threshold = 0.2
        self.ratio = [2, 2]
        self.short_len = 640

        """image mode related：dynamic variables"""
        self.img = None
        self.pred_density_map = None
        self.pred_smap = None
        self.pred_boxs = None
        self.scale_enlarge_ratio = None
        self.has_content = False

        """video mode related: static variables"""
        self.frame_stride = 9
        self.frame_count = 0
        self.det_stride = 3
        self.track_stride = 1
        self.tracking_plan = 'kcf_det'  # ['det','kcf_det']
        self.box_expand_gap = 5

        """video mode related: dynamic variables"""
        self.cap = None
        self.frame_index = 0
        self.ROI = []
        self.coutingNum = 0  # static counting number
        self.flowNum = 0  # dynamic flow number
        self.max_id = 0
        self.gt_dict = {}
        self.tracker_dict = {}
        self.bbox_dict = {}
        self.last_bbox_dict = {}

        """other variables"""
        self.mode = 'empty'
        self.ui.barScale.setValue(8)
        self.ui.textScale.setText("8")

        # window init
        self.ui.winInput.setPixmap(QPixmap(":/backgrounds/images/Input.png"))
        self.ui.winResult.setPixmap(QPixmap(":/backgrounds/images/result.png"))
        self.ui.winDetection.setPixmap(QPixmap(":/backgrounds/images/Detection.png"))
        self.ui.winDensity.setPixmap(QPixmap(":/backgrounds/images/Density.png"))
        self.ui.winScaleMap.setPixmap(QPixmap(":/backgrounds/images/ScaleMap.png"))
        self.ui.winCountingResult.setText("当前人数:")

        # close Timers
        self.timerFrame.stop()
        self.timerCamera.stop()
        self.timerVideo.stop()

    # button "stop" clicked callback function
    def btnStop_clicked(self):
        if self.mode == 'video':
            self.timerVideo.stop()
        elif self.mode == 'camera':
            self.timerFrame.stop()
            self.timerCamera.stop()

    # button "run" clicked callback function
    def run(self):
        if self.mode == 'image':
            self.update_image()
        elif self.mode == 'video':
            # if last video over, restart it from beginning
            if self.video_over:
                self.frame_index = 0
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                ret, img = self.cap.read()
                self.video_over = False

            self.timerVideo.start(1000 / 100)
            self.timerVideo.timeout.connect(self.update_video)
        elif self.mode == 'camera':
            self.timerFrame.start(1000 / 100)
            self.timerFrame.timeout.connect(self.update_frame)
            self.timerCamera.start(1000 / 18)
            self.timerCamera.timeout.connect(self.update_camera)

    # when push button "run" at "image" mode
    def update_image(self):
        results = test_one_image(self.img, self.model, down_ratio=2, short_len=self.short_len,
                                 score_threshold=self.score_threshold, device=self.device)
        pred_bbox, pred_hmap, pred_smap, pred_density, \
        box_img, pred_heatmap, pred_smap_heatmap, pred_density_heatmap = results

        # combine result
        combine_img, pred_num = combine_two_result(self.img, pred_density, pred_bbox, pred_smap,
                                                   self.scale_threshold, self.short_len)
        self.pred_boxs = pred_bbox
        self.pred_density_map = pred_density
        self.pred_smap = pred_smap
        self.ui.winCountingResult.setText("People Num:{:d}".format(int(pred_num)))

        # show four result images
        box_img = cv2.resize(box_img, (456, 342))
        showImage = QImage(box_img, box_img.shape[1], box_img.shape[0], QImage.Format_RGB888)
        self.ui.winDetection.setPixmap(QPixmap.fromImage(showImage))

        density_map = cv2.resize(pred_density_heatmap, (456, 342))
        showImage = QImage(density_map, density_map.shape[1], density_map.shape[0], QImage.Format_RGB888)
        self.ui.winDensity.setPixmap(QPixmap.fromImage(showImage))

        scale_map = cv2.resize(pred_smap_heatmap, (456, 342))
        showImage = QImage(scale_map, scale_map.shape[1], scale_map.shape[0], QImage.Format_RGB888)
        self.ui.winScaleMap.setPixmap(QPixmap.fromImage(showImage))

        combine_img = cv2.resize(combine_img, (684, 513))
        showImage = QImage(combine_img, combine_img.shape[1], combine_img.shape[0], QImage.Format_RGB888)
        self.ui.winResult.setPixmap(QPixmap.fromImage(showImage))

        self.has_content = True

    # when push button "run" at "video" mode
    def update_video(self):
        # read one frame
        ret, img = self.cap.read()
        self.img = img
        if self.frame_count % self.frame_stride != 0:
            self.frame_count = (self.frame_count + 1) % self.frame_stride
            return

        # set src window
        img = cv2.resize(img, (684, 513))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.winInput.setPixmap(QPixmap.fromImage(showImage))

        # track one frame
        if ret:
            self.track_one_frame()

        # update four result windows
        self.update_result_windows()

        # update counting result
        self.ui.winCountingResult.setText("People Num:{:d}, Flow Num:{:d}".format(self.coutingNum, self.flowNum))

        # if video end, stop
        if self.frame_index > self.total_frame_num - 3:
            self.btnStop_clicked()
            self.video_over = True

    # when push button "run" at "camera" mode
    def update_camera(self):
        # update track state
        self.track_one_frame()

        # update four result windows
        self.update_result_windows()

        # update counting result
        self.ui.winCountingResult.setText("People Num:{:d}, Flow Num:{:d}".format(self.coutingNum, self.flowNum))

    # get and show a frame from online camera
    def update_frame(self):
        ret, img = self.cap.read()
        if ret:
            self.img = img
            # update src img window
            img = cv2.resize(img, (684, 513))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showImage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.ui.winInput.setPixmap(QPixmap.fromImage(showImage))

    # update four result windows
    def update_result_windows(self):
        # detection result window
        bbox_img = cv2.resize(self.bbox_img, (456, 342))
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
        showImage = QImage(bbox_img, bbox_img.shape[1], bbox_img.shape[0], QImage.Format_RGB888)
        self.ui.winDetection.setPixmap(QPixmap.fromImage(showImage))
        # print("show bbox img")

        # density map window
        density_map = cv2.resize(self.pred_density_map, (456, 342))
        density_map = cv2.cvtColor(density_map, cv2.COLOR_BGR2RGB)
        showImage = QImage(density_map, density_map.shape[1], density_map.shape[0], QImage.Format_RGB888)
        self.ui.winDensity.setPixmap(QPixmap.fromImage(showImage))
        # print("show density map")

        # scale map window
        scale_map = cv2.resize(self.pred_smap, (456, 342))
        scale_map = cv2.cvtColor(scale_map, cv2.COLOR_BGR2RGB)
        showImage = QImage(scale_map, scale_map.shape[1], scale_map.shape[0], QImage.Format_RGB888)
        self.ui.winScaleMap.setPixmap(QPixmap.fromImage(showImage))
        # print("show scale map")

        # result window
        result_img = cv2.resize(self.result_img, (684, 513))
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        showImage = QImage(result_img, result_img.shape[1], result_img.shape[0], QImage.Format_RGB888)
        self.ui.winResult.setPixmap(QPixmap.fromImage(showImage))
        # print("show result img")

    # track one frame
    def track_one_frame(self):
        """get current frame"""
        origin_img = self.img

        img = cv2.resize(origin_img, (684, 513))
        raw_height, raw_width = img.shape[0], img.shape[1]
        img = resize_img_with_short_len(img, short_len=self.short_len)
        height, width = img.shape[0], img.shape[1]

        h_rate = height / raw_height
        w_rate = width / raw_width

        """get detection result"""
        if self.frame_index % self.det_stride == 0:
            preds = get_bbox_by_model(self.model, img, self.score_threshold,
                                      self.ratio, self.device)
            bbox, hmap, smap, density = preds
            smap = smap.detach().cpu().numpy().reshape(smap.shape[2], smap.shape[3])
            density = density.detach().cpu().numpy().reshape(density.shape[2], density.shape[3])

            # give each bbox a gap and init id
            self.bbox_dict = generate_bbox_dict_from_bbox(bbox, self.box_expand_gap, height, width)

            # update three windows
            smap = smap * h_rate
            self.pred_smap = generate_heatmap(smap, rate=3)
            density = resize_gt(density, height, width)
            self.pred_density_map = generate_heatmap(density, rate=0)
            self.bbox_img = draw_boxs(img, bbox)

        """Track"""
        flow_count = 0
        if self.frame_index == 0:
            if self.mode == 'camera':
                self.timerFrame.stop()

            # get ROI
            roi = cv2.selectROI(img, False)
            self.ROI = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]

            # init Tracker
            if self.tracking_plan == 'kcf_det':
                for i in self.bbox_dict.keys():
                    tracker = cv2.TrackerKCF_create()
                    init_box = (self.bbox_dict[i][0], self.bbox_dict[i][1],
                                self.bbox_dict[i][2] - self.bbox_dict[i][0],
                                self.bbox_dict[i][3] - self.bbox_dict[i][1])
                    tracker.init(img, init_box)
                    self.tracker_dict[i] = tracker
            self.max_id = bbox.shape[0]

            if self.mode == 'camera':
                self.timerFrame.start(1000 / 100)

        else:
            if self.frame_index % self.track_stride == 0:
                if self.tracking_plan == 'kcf_det':
                    if self.frame_index % self.det_stride == 0:
                        result = track_with_det_and_KCF(img, self.bbox_dict, self.last_bbox_dict,
                                                        self.tracker_dict, self.max_id, 0.2, self.ROI)
                        self.bbox_dict, self.tracker_dict, self.max_id, flow_count = result
                    else:
                        result = track_with_only_KCF(img, self.last_bbox_dict, self.tracker_dict, self.ROI)
                        self.bbox_dict, self.tracker_dict, flow_count = result
                elif self.tracking_plan == 'det':
                    result = track_with_only_det(self.bbox_dict, self.last_bbox_dict, 0.2, self.max_id, self.ROI)
                    self.bbox_dict, self.max_id, flow_count = result

        cv2.destroyAllWindows()
        print("Track done")

        """get people number and flow number in ROI"""
        if self.frame_index == 0  or self.frame_index % self.track_stride == 0:
            coutingNum = 0
            for key in self.bbox_dict.keys():
                x1, y1, x2, y2 = self.bbox_dict[key]
                c_x = (x1 + x2) / 2
                c_y = (y1 + y2) / 2
                if c_x > self.ROI[0] and c_x < self.ROI[2] \
                        and c_y > self.ROI[1] and c_y < self.ROI[3]:
                    coutingNum += 1
            self.coutingNum = coutingNum

        if self.frame_index == 0:
            flow_count = self.coutingNum

        self.flowNum += flow_count
        print("get number done")

        """save bbox_dict"""
        if self.frame_index == 0 or self.frame_index % self.track_stride == 0:
            self.last_bbox_dict = self.bbox_dict.copy()

        """---------------visualize tracking result------------------"""
        result_img = draw_boxs_with_id(img, self.bbox_dict, width=2, color=(0, 255, 0),
                                       h_rate=1, w_rate=1)
        # Draw ROI
        cv2.rectangle(result_img, (self.ROI[0], self.ROI[1]), (self.ROI[2], self.ROI[3]), (0, 0, 255), 2)

        self.result_img = result_img
        self.frame_index += 1
        print("track_one_frame done!")

    # slide callback function
    def eventFilter(self, watched, event):
        if (watched == self.ui.barScale):
            if (event.type() == QEvent.MouseButtonPress):
                pass
            elif (event.type() == QEvent.MouseMove):
                pass
            elif (event.type() == QEvent.MouseButtonRelease):
                value = self.ui.barScale.value()
                self.ui.textScale.setText("{}".format(value))
                self.scale_threshold = value
                if self.has_content:
                    combine_img, pred_num = combine_two_result(self.img, self.pred_density_map, self.pred_boxs,
                                                               self.pred_smap, self.scale_threshold,
                                                               self.short_len)
                    combine_img = cv2.resize(combine_img, (684, 513))
                    showImage = QImage(combine_img, combine_img.shape[1], combine_img.shape[0], QImage.Format_RGB888)
                    self.ui.winResult.setPixmap(QPixmap.fromImage(showImage))
                    self.ui.winCountingResult.setText("People Num:{:d}".format(int(pred_num)))

        return super().eventFilter(watched, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # set stylesheet
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # start logo
    splash = QSplashScreen(QPixmap("images/logo1.png"))
    font=QFont()
    font.setPointSize(20)
    splash.setFont(font)
    splash.showMessage("Loading... 0%", Qt.AlignLeft | Qt.AlignBottom, Qt.black)
    splash.show()

    # show the main window
    mainWindow = MainWindow()
    mainWindow.setWindowIcon(QIcon("images/icon.ico"))
    mainWindow.show_start_logo(splash)
    mainWindow.show()
    splash.finish(mainWindow)

    sys.exit(app.exec_())
