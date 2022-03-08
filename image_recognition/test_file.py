import socket
import os
from imutils.video import VideoStream
import imagezmq
import cv2
from matplotlib import image


if __name__ == '__main__':

    command = 'predict'

    connect_to = 'tcp://10.27.228.226:5555'
    sender = imagezmq.ImageSender(connect_to=connect_to)

    img = None

    file = r'C:/Users/Atul/Desktop/Rpi/image_recognition/yolov5/failed/IMG_20.jpg'
        
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    reply = sender.send_image(command, img)

    print(reply.decode('utf-8'))
    sender.send_image('merge', img)
        
