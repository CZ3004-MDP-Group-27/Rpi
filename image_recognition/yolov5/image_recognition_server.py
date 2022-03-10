# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
from click import command
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import imagezmq
import cv2
from utils.augmentations import letterbox

import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


cwd = os.getcwd()

verification_dir = os.path.join(cwd, 'mdp_verification_folder')
debug_dir = os.path.join(cwd, 'image_dump_folder')

names_to_label_map = {
    'alphabetA': 20,
    'alphabetB': 21, 
    'alphabetC': 22, 
    'alphabetD': 23, 
    'alphabetE': 24, 
    'alphabetF': 25, 
    'alphabetG': 26, 
    'alphabetH': 27, 
    'alphabetS': 28, 
    'alphabetT': 29,  
    'alphabetU': 30, 
    'alphabetV': 31, 
    'alphabetW': 32, 
    'alphabetX': 33, 
    'alphabetY': 34, 
    'alphabetZ': 35, 
    'bullseye': 0,
    'bulleye': 0, 
    'down_arrow': 37, 
    'eight': 18, 
    'five': 15, 
    'four': 14, 
    'left_arrow': 39, 
    'nine': 19, 
    'one': 11, 
    'right_arrow': 38, 
    'seven': 17, 
    'six': 16, 
    'stop': 40, 
    'three': 13, 
    'two': 12, 
    'up_arrow': 36,
    '-1': -1
}


def merge_images(directory, count=8):

    img_list = []
    for i in range(count):
        img = Image.open(os.path.join(directory, f'IMG_{i}.jpg'))
        img_list.append(img)

    if len(img_list) == 0:
        print(f'No images saved in {directory}')
        return

    img_size = img_list[0].size

    num_rows = 2
    num_cols = math.ceil(len(img_list) / 2)

    #empty image
    mergedImg = Image.new(mode="RGB", size=(num_cols*img_size[0], num_rows*img_size[1]), color=(0,0,0))

    for i, img in enumerate(img_list):
        row = i // num_cols
        col = i % num_cols
        mergedImg.paste(img, (col * img_size[0], row * img_size[1]))

    mergedImg.save(directory + "/mergedImage.jpg")
    mergedImg.show()


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(416, 320),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)


    # TODO: Load multiple models and if you dont find a class in 1 model use another model

    device = select_device(device)

    weight_list = [
        'trained_models/exp2/best.pt',
        'trained_models/exp3/best.pt',
        'trained_models/exp3/last.pt',
        'trained_models/exp1/best.pt',
    ]

    model_list = []
    names_list = []

    for weights in weight_list:
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        names_list.append(names)
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)

        model_list.append(model)

    # Dataloader
    #dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1
    print('Using RPi Camera')

    image_hub = imagezmq.ImageHub()
    print('\nStarted Image Recognition Server.\n')

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)
    # Set up image_rec server

    count = 0
    x = 0

    image_buffer = []
    image_list = []

    if not os.path.isdir(verification_dir):
        os.makedirs(verification_dir)

    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    while True:

        command, image = image_hub.recv_image()
        
        if command == 'merge':
            image_hub.send_reply(("Merged").encode())
            merge_images(verification_dir, count)
            break

        image = cv2.resize(image, (416, 320))
        print(os.path.join(debug_dir, 'IMG_' + str(x) + 'jpg'))
        cv2.imwrite(os.path.join(debug_dir, 'IMG_' + str(x) + '.jpg'), image)
        x+=1
        image_list.append(image)
        image_copy = image.copy()

        image_copy = letterbox(image, imgsz, stride=stride, auto=pt)[0]

        image_copy = image_copy.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image_copy = np.ascontiguousarray(image_copy)

        im = torch.from_numpy(image_copy).to(device)
        im = im.half() if half else im.float()
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        image_buffer.append(im)

        if command == 'capture':
            image_hub.send_reply(("Captured").encode())
            continue

        if command == 'predict':

            found = False
            main_class = -1
            while (len(image_buffer) > 0):

                for model_idx, model in enumerate(model_list):
                    names = names_list[model_idx]
                    pred = model(image_buffer[-1], augment=augment)
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    for i, det in enumerate(pred): # Only predicting on 1 image

                        annotator = Annotator(image_list[-1], line_width=line_thickness, example=str(names))

                        highest_conf = 0.0
                        max_area = 0.0
                        for *xyxy, conf, cls in reversed(det):

                            print(xyxy)
                            
                            # If first detection then keep that as main class

                            if main_class == -1:
                                label_name = names[int(cls)]
                                highest_conf = conf
                                x1, y1, x2, y2 = xyxy
                                max_area = (x2-x1) * (y2-y1)
                                main_class = names_to_label_map[label_name]
                                print(f"Area: {max_area}, Class: {main_class}")

                            # If the main class is bullseye then overwirte main class without further checks
                            elif main_class == 0:
                                label_name = names[int(cls)]
                                highest_conf = conf
                                x1, y1, x2, y2 = xyxy
                                max_area = (x2-x1) * (y2-y1)
                                main_class = names_to_label_map[label_name]
                                print(f"Area: {max_area}, Class: {main_class}")

                            # If the confidence is higher than the current best
                            # Note: This bock will only be executed if the main class is not -1 (None) and not 0 (bullseye)
                            # TODO: Change to else: and if bbox area is higher then select that
                            else:
                                label_name = names[int(cls)]
                                # If prediction is bullseye but we have found another class then ignore the bullseye
                                if label_name == 'bullseye' or label_name == 'bulleye':
                                    continue

                                x1, y1, x2, y2 = xyxy
                                area = (x2-x1) * (y2-y1)
                                print(area)
                                if area > max_area:
                                    highest_conf = conf
                                    main_class = names_to_label_map[label_name]
                                    max_area = area
                                    print(f"Area: {max_area}, Class: {main_class}")

                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label_only = names[c]
                            annotator.box_label(xyxy, label + " : " + str(names_to_label_map[label_only]), color=colors(c, True))

                        print(int(main_class))
                        print(highest_conf)

                        # If we got main label save it
                        if main_class != -1 and main_class != 0:
                            print(f'Found image with model {model_idx + 1}')
                            image_buffer = []
                            image_list = []
                            image_hub.send_reply(str(main_class).encode())
                            img_pred = annotator.result()
                            cv2.imwrite(os.path.join(verification_dir, 'IMG_' + str(count) + '.jpg'), img_pred)
                            print('Saved Image')
                            count += 1
                            found = True
                    if found:
                        break

                if found:
                    break

                if len(image_buffer) > 0:
                    image_buffer.pop()

                if len(image_list) > 0:
                    image_list.pop()

            if not found:
                # Send either 0 or -1
                image_hub.send_reply((str(main_class)).encode())

                
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
