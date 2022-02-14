import socket
import os
from imutils.video import VideoStream
import imagezmq
import cv2


if __name__ == '__main__':

    command = 'capture'

    connect_to = 'tcp://192.168.0.102:5555'
    sender = imagezmq.ImageSender(connect_to=connect_to)

    img = None

    directory = r'C:/Users/Atul/Desktop/Rpi/image_recognition/datasets/mdp_dataset/old/images/'
        
    for file in os.listdir(directory):
        img = cv2.imread(directory + file, cv2.IMREAD_COLOR)
        reply = sender.send_image(command, img)
        print(reply)
        break
        
