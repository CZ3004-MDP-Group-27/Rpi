#run this program on each RPi to send an image to PC
import socket
import time
from click import command
from imutils.video import VideoStream
import imagezmq
from numpy import imag
from picamera import PiCamera
from picamera.array import PiRGBArray


class Camera():

    def init(self, resolution=(256, 256)):
        self.rpi_name = socket.gethostname()
        self.camera = PiCamera(resolution=resolution)
        self.rawCapture = PiRGBArray(self.camera)
        self.sender = None

    def setup(self, connect_to):
        self.sender = imagezmq.ImageSender(connect_to=connect_to) #'tcp://192.168.0.101:5555') # Check local IP

    def send_image(self, command='view'):
        self.camera.capture(self.rawCapture, format='bgr')
        image = self.rawCapture.array
        self.rawCapture.truncate(0)
        
        if self.sender is None:
            print('Call the setup method before sending the image')
            return
        
        try:
            label = self.sender.send_image(command, image)
            print(f'Sent Image to destination PC')
            return label
        
        except Exception as e:
            print('Could not send image to destination PC')
            print(e)
        
        return

