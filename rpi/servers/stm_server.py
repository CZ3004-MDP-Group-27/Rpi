import serial
import time

class STMServer():

        def __init__(self):
                self.serial = None

        def setup(self, port='/dev/ttyUSB0', baud_rate=115200):
                self.serial = serial.Serial(port, baud_rate)
                print('Connected to STM')

        def read(self):

                data = None
                try:
                        data = self.serial.readall()
                        if len(data) > 0:
                                data = data.decode()
                                print('Received data from STM')

                except Exception as e:
                        print(e)

                return data

        def send(self, text):
                try:
                        self.serial.write(text)
                        print('Sent data to STM')
                except Exception as e:
                        print(e)
