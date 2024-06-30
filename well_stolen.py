import threading
from pathlib import Path
import cv2
from ultralytics.utils.checks import check_imshow
import sys
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QDialog
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton
from yolov8Qt import Ui_MainWindow
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6 import QtGui, QtCore
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.engine.results import Results

STREAM_WARNING = """
除非通过“stream=True”，否则推理结果将累积在RAM中，从而可能导致内存不足
"""

class MyThread(QThread):
    # 原图
    send_raw = Signal(np.ndarray)
    # 发送检测结果图
    send_detect_img = Signal(np.ndarray)
    # 发送检测的类别信息，速度等
    send_detect_info = Signal(dict)
    # 发送检测速度
    send_speed = Signal(str)
    # 发送进度条百分比
    send_percent = Signal(int)
    # 检测到的目标数量
    send_detect_num = Signal(int)

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, parent=None):
        super(MyThread, self).__init__(parent)
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        self.weights = "detect.pt"
        self.conf = 0.25
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        self.is_save = Qt.Checked
        self.camera_isOpen = False
        self.iou = 0.71
#Button_checkImg
        self.done_warmup = False
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.source = None
        self.stream = True
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        self.play = True
        self.end_loop = False
        self.gd = None
        callbacks.add_integration_callbacks(self)



    # 对图像进行归一化，加载资源
    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    # 创建保存的文件夹
    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)

    # 对图片进行预处理，调整图片尺寸
    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    # 用于将推理结果写入文件或目录,给图片增加框框
    def write_results(self, idx, results, batch):
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.show_boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)

        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))
        return log_string

    # 对模型预测的结果进行后处理，采用极大值抑制
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    # 根据stream是否为True，选择一次性加载还是存入内存里面检测，默认为False
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    # 重写多线程的run方法，程序的入口
    def run(self):
        self.__call__()

    # 这个方法是用于设置源数据和推理模式的
    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None

        self.dataset = load_inference_source(source=source,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             buffer=self.args.stream_buffer)

        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        # self.vid_frame = [None] * self.dataset.bs

    # 实时推理的主程序
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        source = self.source
        current_frame = 0  # 初始化计数器，用于计算视频的进度
        # 重置进度条
        self.send_percent.emit(0)
        if not self.model:
            # 如果为空，则调用self.setup_model(model)方法来初始化加载模型
            self.setup_model(model)
        # 获取线程锁，以防止多线程冲突
        with self._lock:

            # 加载数据源
            self.setup_source(source if source is not None else self.args.source)

            # 如果选择勾选保存，那么创建文件夹
            if self.is_save == Qt.Checked:
                (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 如果为False，则表示模型尚未预热，进行预热
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())

            dataset_iterator = iter(self.dataset)
            while True:
                names_dic = {}  # 初始化类目容器
                if self.end_loop:
                    # 如果是摄像头，释放资源
                    if self.source_type.webcam:
                        for cap in self.dataset.caps:
                            cap.release()
                        self.dataset.stop_threads = True
                    break
                if self.play:
                    try:
                        batch = next(dataset_iterator)
                        # 更新当前检测数量索引
                        current_frame += 1
                    except Exception as e:
                        # 跳出循环，重置进度条
                        self.send_percent.emit(0)
                        break
                    self.batch = batch
                    path, im0s, self.vid_cap, s = batch
                    if self.vid_cap is not None:
                        # 计算视频检测进度并发送当前检测的进度百分比
                        progress_percentage = int(
                            current_frame / int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100)
                        self.send_percent.emit(progress_percentage)
                    else:
                        # 计算图片检测进度并发送当前检测的进度百分比
                        progress_percentage = int(
                            current_frame / int(len(self.dataset)) * 100)
                        self.send_percent.emit(progress_percentage)
                    # Preprocess
                    with profilers[0]:
                        im = self.preprocess(im0s)

                    # Inference
                    with profilers[1]:
                        preds = self.inference(im, *args, **kwargs)

                    # Postprocess
                    with profilers[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    # Visualize, save, write results
                    n = len(im0s)


                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            'preprocess': profilers[0].dt * 1E3 / n,
                            'inference': profilers[1].dt * 1E3 / n,
                            'postprocess': profilers[2].dt * 1E3 / n}

                        p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                        p = Path(p)  # 检测的资源路径，如果是摄像头那么就是数字

                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s += self.write_results(i, self.results, (p, im, im0))
                        if self.args.save or self.args.save_txt:
                            self.results[i].save_dir = self.save_dir.__str__()

                        if self.args.save and self.plotted_img is not None:
                            self.save_preds(self.vid_cap, i, str(self.save_dir / p.name))

                        # 将结果转换为Python列表
                        result_list = self.results[i].boxes.cls.tolist()
                        self.send_detect_num.emit(len(result_list)) #发送统计检测个数
                        # 初始化一个空字典用于存储数字和它们的出现次数
                        count_dict = {}
                        # 遍历列表，统计数字出现的次数
                        for number in result_list:
                            if number in count_dict:
                                count_dict[number] += 1
                            else:
                                count_dict[number] = 1

                        for k, v in count_dict.items():
                            names_dic[self.model.names[k]] = v
                        # names_dic{'bus': 1, 'person': 3}
                        # 把检测后的结果展示在pyside6上
                        self.send_detect_img.emit(self.plotted_img)
                        #发送原图
                        self.send_raw.emit(im0)
                        # 发送检测数据
                        self.send_detect_info.emit(names_dic)
                        # 发送检测数据
                        self.send_speed.emit(str(round((profilers[0].dt + profilers[1].dt + profilers[2].dt) * 1E3)))
                    yield from self.results

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer


    def setup_model(self, model, verbose=True):
        self.model = AutoBackend(self.weights,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            if self.is_save == Qt.Checked:
                cv2.imwrite(save_path, im0)

        else:  # 'video' or 'stream'
            if self.is_save == Qt.Checked:
                # frames_path = f'{save_path.split(".", 1)[0]}_frames/'
                if self.vid_path[idx] != save_path:  # new video
                    # Path(frames_path).mkdir(parents=True, exist_ok=True)
                    self.vid_path[idx] = save_path
                    # self.vid_frame[idx] = 0
                    if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                        self.vid_writer[idx].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                    self.vid_writer[idx] = cv2.VideoWriter(str(Path(save_path).with_suffix(suffix)),
                                                           cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                # Write video
                self.vid_writer[idx].write(im0)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)  # 加载pyside6的UI
        self.methoBinding()  # pyside6和python方法绑定
        self.camera_isOpen = False
        self.my_thread = MyThread()
        self.my_thread.iou = 0.25
        self.my_thread.conf = 0.7
        self.my_thread.weights = "yolov8n.pt"

        self.my_thread.is_save =  Qt.Checked
        self.my_thread.play = True
        self.my_thread.end_loop = False
        self.my_thread.send_raw.connect(lambda x: self.show_detect(x, self.label_raw))
        self.my_thread.send_detect_img.connect(lambda img: self.show_detect(img, self.label_result))
        self.my_thread.send_detect_info.connect(lambda names_dic: self.show_detect_info(names_dic))
        self.my_thread.send_speed.connect(lambda speed: self.show_speed(speed))
        self.my_thread.send_detect_num.connect(lambda num:self.show_detect_num(num))
        self.my_thread.send_percent.connect(lambda percent: self.progressBar.setValue(percent))
        self.load_config()

    # 从json中读取配置
    def load_config(self):
        # 读取 JSON 文件
        with open("config.json", "r") as jsonfile:
            loaded_config = json.load(jsonfile)
            # 获取参数值
            self.my_thread.weights = loaded_config["weights"]
            self.my_thread.conf = loaded_config["conf"]
            self.my_thread.iou = loaded_config["iou"]
            # self.my_thread.is_alarm = loaded_config["is_alarm"]
            self.my_thread.is_save = loaded_config["is_save"]
        # 把读取的值，传递展现到pyside组件上
        self.doubleSpinBox_conf.setProperty("value", self.my_thread.conf)
        self.doubleSpinBox_iou.setProperty("value", self.my_thread.iou)
        self.line_weights.setText(self.my_thread.weights)

        # 判断复选框的勾选状态
        if self.my_thread.is_save == "checked":
            self.checkBox_isSave.setCheckState(Qt.Checked)
            self.my_thread.is_save = Qt.Checked


    #显示检测的个数
    def show_detect_num(self,num):
        self.label_detect_num.setText(str(num))
    # 显示速度
    def show_speed(self, speed):
        self.label_speed.setText(speed)
    #右下角的检测结果信息展示
    def show_detect_info(self, names_dic):
        self.result_info.clear()
        self.result_info.append("您正在使用GPU检测") if torch.cuda.is_available() else self.result_info.append(
            "您正在使用CPU检测")
        self.result_info.append("当前的置信度为:" + str(self.my_thread.conf))
        self.result_info.append("IOU:" + str(self.my_thread.iou))
        if self.my_thread.is_save == Qt.Checked:
            self.result_info.append("您选择保存检测文件，位置在根目录下的runs文件夹")

        self.result_info.append("检测到的类别和数量如下：")
        # {person:1,car:5}
        for key, value in names_dic.items():
            self.result_info.append(f"{key}: {value}{'北京市-朝阳区-建国路'}")

    def methoBinding(self):
        self.Button_checkImg.clicked.connect(self.select_images)  # 检测图片
        self.Button_openCamera.clicked.connect(self.open_camera)  # 检测摄像头
        self.Button_checkVideo.clicked.connect(self.select_video)  # 检测视频
        self.Button_select_folder.clicked.connect(self.select_folder)  # 检测文件夹
        self.Button_select_w_p.clicked.connect(self.select_weights)  # 选择权重
        self.pushButton_bofang.clicked.connect(self.play_and_pause)  # 播放暂停
        self.doubleSpinBox_conf.valueChanged.connect(lambda x: self.change_conf(x))  # 调整置信度
        self.doubleSpinBox_iou.valueChanged.connect(lambda x: self.change_iou(x))  # 调整iou

        self.horizontalSlider_conf.valueChanged.connect(lambda x: self.change_hor_conf(x))
        self.horizontalSlider_iou.valueChanged.connect(lambda x: self.change_hor_iou(x))
        self.checkBox_isSave.clicked.connect(self.save_result)  # 选择是否保存
        self.pushButton_stop.clicked.connect(self.end_thread)  # 终止检测


    # 控制暂停和播放
    def play_and_pause(self):
        # 如果检测的资源为空，提示先选择资源
        if self.my_thread.source == "":
            QMessageBox.information(self, '错误提示', '你还没选择检测的资源呢！')
            return
        if self.my_thread.play:
            # 原来是播放的，现在暂停了
            self.my_thread.play = False
        else:
            # 原来是暂停的，现在播放了
            self.my_thread.play = True
        # 更新播放和暂停的图标
        self.update_play_icon()

    # 选择检测图片
    def select_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                    options=options)
        if image_path:
            self.end_thread()
            self.my_thread.source = image_path
            self.star_thread()

    # 勾选或者取消保存结果
    def save_result(self):
        self.my_thread.is_save = self.checkBox_isSave.checkState()

    # 检测视频
    def select_video(self):

        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, '选择视频', '',
                                                    'Videos (*.mp4 *.avi *.mkv);;All Files (*)', options=options)
        if video_path:
            self.end_thread()
            self.my_thread.source = video_path
            self.star_thread()

    # 检测文件夹
    def select_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "", options=options)
        if folder_path:
            self.end_thread()
            self.my_thread.source = folder_path
            self.star_thread()

    # 检测摄像头
    def open_camera(self):
        self.dialog = QDialog(self.centralwidget)  # 创建摄像头弹窗
        self.dialog.setWindowTitle("请选择你的摄像头")  # 设置弹窗标题
        self.dialog.resize(400, 200)

        # 判断摄像头开启状态，如何开着设置为关，反之则开
        if self.camera_isOpen == True:
            # 开着的，现在关闭
            self.camera_isOpen = False
            # 跳出循环结束
            self.end_thread()
        else:
            num, deviceList = self.get_camera_num()
            # 如果没有摄像头，弹出提示框
            if num == 0:
                QMessageBox.warning(self, "出错啦", "<p>未检测到有效的摄像头</p>")
            # 如果有摄像头，弹出按钮，点击按钮后赋值对应的摄像头并打开开始检测
            else:
                self.camera_isOpen = True
                # 设置水平布局
                h_layout = QHBoxLayout(self.dialog)
                for xuhao in deviceList:
                    button = QPushButton(f'摄像头{xuhao}', self.dialog)
                    button.clicked.connect(lambda: self.click_camera(xuhao))
                    h_layout.addWidget(button)  # 为按钮添加布局
                # 显示对话框
                self.dialog.exec()

    # 选择对应的摄像头
    def click_camera(self, device):
        # 关闭对话框
        self.dialog.close()
        self.end_thread()
        self.my_thread.source = device
        self.star_thread()



    # 修改置信度
    def change_conf(self, x):
        self.my_thread.conf = round(x, 2)
        #修改滑杆的值
        self.horizontalSlider_conf.setValue(x*100)
    #修改滑杆置信度的值
    def change_hor_conf(self,x):
        #修改置信度的值和按钮的值
        conf=x*0.01
        self.my_thread.conf=conf
        self.doubleSpinBox_conf.setValue(conf)


    # 修改IOU
    def change_iou(self, x):
        self.my_thread.iou = round(x, 2)
        # 修改滑杆的值
        self.horizontalSlider_iou.setValue(x * 100)

        # 修改滑杆iou的值

    def change_hor_iou(self, x):
        # 修改置信度的值和按钮的值
        iou = x * 0.01
        self.my_thread.iou = iou
        self.doubleSpinBox_iou.setValue(iou)

    # 结束线程
    def end_thread(self):
        self.my_thread.play = False
        self.my_thread.end_loop = True
        self.update_play_icon()
        self.my_thread.wait()
        # self.label_result.setPixmap(QtGui.QPixmap("test.png"))

    # def stop_thread(self):
    #
    #     self.stop_queue.put(True)
    #     self.join()  # Wait for the thread to finish

    # 启动线程
    def star_thread(self):
        self.my_thread.play = True
        self.my_thread.end_loop = False
        self.update_play_icon()
        self.my_thread.start()

    def show_detect(self, image, qt):
        try:
            ih, iw, _ = image.shape
            w = qt.geometry().width()
            h = qt.geometry().height()
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(image, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(image, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            qt.setPixmap(QPixmap.fromImage(img))
            qt.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(repr(e))

    # 获取电脑摄像头的数量
    def get_camera_num(self):
        num = 0  # 计数器，用于记录找到的摄像头数量
        deviceList = []  # 列表，用于存储找到的摄像头设备索引

        for i in range(3):  # 遍历预设的摄像头数量范围,因为电脑一般不超过3个摄像头
            stream = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 使用OpenCV库创建一个视频捕获对象，指定设备索引和捕获模式
            exists = stream.grab()  # 尝试从摄像头捕获一帧图像是否存在
            stream.release()  # 释放视频捕获对象
            if not exists:  # 如果未能成功捕获图像，则跳过当前设备继续循环
                continue
            else:
                num += 1  # 如果成功捕获图像，数量+1
                deviceList.append(i)  # 把对应的索引加入设备列表中
        return num, deviceList

    def select_weights(self):
        options = QFileDialog.Options()
        weights_path, _ = QFileDialog.getOpenFileName(self, '选择pt权重', '', 'pt (*.pt)', options=options)
        if weights_path:
            self.end_thread()
            self.line_weights.setText(weights_path)
            self.my_thread.weights = weights_path

    # 更新播放和暂停的图标
    def update_play_icon(self):
        icon = QtGui.QIcon()
        if self.my_thread.play:
            # 如果现在是播放着的，那么显示的是暂停的图标
            icon.addPixmap(QtGui.QPixmap("icon/暂停.png"),
                           QtGui.QIcon.Normal, QtGui.QIcon.Off)
        else:
            # 如果现在是播放着的，那么显示的是暂停的图标
            icon.addPixmap(QtGui.QPixmap("icon/播放.png"),
                           QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_bofang.setIcon(icon)
        self.pushButton_bofang.setIconSize(QtCore.QSize(32, 32))

    # 关闭窗口事件
    def closeEvent(self, event):
        confirm = QMessageBox.question(self, '关闭程序', '确定关闭？',
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            # 确认关闭，保存用户数据
            config = dict()  # 创建一个空的字典
            config["conf"] = self.my_thread.conf
            config["iou"] = self.my_thread.iou
            config["weights"] = self.my_thread.weights
            if self.my_thread.is_save == Qt.Checked:
                config["is_save"] = "checked"
            else:
                config["is_save"] = "Unchecked"

            # 将修改后的配置参数写入 JSON 文件
            with open("config.json", "w") as jsonfile:
                json.dump(config, jsonfile, indent=4)
            # 退出循环，关闭流，防止摄像头等检测资源保存出错
            self.end_thread()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec())
