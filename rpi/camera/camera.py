#run this program on each RPi to send an image to PC
import socket
import time
from imutils.video import VideoStream
import imagezmq
from numpy import imag
from picamera import PiCamera
from picamera.array import PiRGBArray

# sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.101:5555') # check local pc IP
 
# # rpi_name = socket.gethostname() # send RPi hostname with each image
# # camera = PiCamera(resolution=(2592, 1944)) #max resolution 2592,1944
# # rawCapture = PiRGBArray(camera)

# # time.sleep(1)  # allow camera sensor to warm up

# # # while True:  # send images as stream until Ctrl-C
# # #     time.sleep(2)
# # #     camera.capture(rawCapture, format="bgr")
# # #     image = rawCapture.array
# # #     rawCapture.truncate(0)
# # #     sender.send_image(rpi_name, image)


class Camera():

    def init(self, resolution=(256, 256)):
        self.rpi_name = socket.gethostname()
        self.camera = PiCamera(resolution=resolution)
        self.rawCapture = PiRGBArray(self.camera)
        self.sender = None

    def setup(self, connect_to):
        self.sender = imagezmq.ImageSender(connect_to=connect_to) #'tcp://192.168.0.101:5555') # Check local IP

    def send_image(self):
        self.camera.capture(self.rawCapture, format='bgr')
        image = self.rawCapture.array
        self.rawCapture.truncate(0)
        
        if self.sender is None:
            print('Call the setup method before sending the image')
            return
        
        try:
            label = self.sender.send_image(self.rpi_name, image)
            print(f'Sent Image to destination PC')
            return label
        
        except Exception as e:
            print('Could not send image to destination PC')
            print(e)
        
        return

