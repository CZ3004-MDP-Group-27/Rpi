#run this program on each RPi to send an image to PC
import socket
import time
from imutils.video import VideoStream
import imagezmq
from picamera import PiCamera
from picamera.array import PiRGBArray

sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.101:5555') # check local pc IP
 
rpi_name = socket.gethostname() # send RPi hostname with each image
camera = PiCamera(resolution=(2592, 1944)) #max resolution 2592,1944
rawCapture = PiRGBArray(camera)

time.sleep(1)  # allow camera sensor to warm up

while True:  # send images as stream until Ctrl-C
    time.sleep(2)
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    rawCapture.truncate(0)
    sender.send_image(rpi_name, image)
