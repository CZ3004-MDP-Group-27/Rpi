import socket
import os
from imutils.video import VideoStream
import imagezmq
import cv2


# label_id_map = {
#     '0':36,
#     '1': 37,
#     '2': 38,
#     '3': 39,
#     '4': 40,
#     '5': 11,
#     '6': 12,
#     '7':13, 
#     '8':14,
#     '9':15,
#     '10':16,
#     '11':17,
#     '12':18,
#     '13':19,
#     '14':20,
#     '15':21,
#     '16':22,
#     '17':23,
#     '18':24,
#     '19':25,
#     '20':26,
#     '21':27,
#     '22':28,
#     '23':29,
#     '24':30,
#     '25':31,
#     '26':32,
#     '27':33,
#     '28':35,
#     '29':35,
#     '30': 0
# }


if __name__ == '__main__':

    command = 'predict'

    connect_to = 'tcp://192.168.0.102:5555'
    sender = imagezmq.ImageSender(connect_to=connect_to)

    img = None

    directory = r'C:/Users/Atul/Desktop/Rpi/image_recognition/yolov5/image_dump/'
        
    for i, file in enumerate(os.listdir(directory)):
        print(file)
        img = cv2.imread(directory + file, cv2.IMREAD_COLOR)
        reply = sender.send_image(command, img)
        print(type(reply))
        print(reply)
        reply = reply.decode("utf-8")
        print(type(reply))
        print(reply)
    
    reply = sender.send_image("merge", img)
