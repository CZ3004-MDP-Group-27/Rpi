from rpi.camera.camera import Camera
from rpi.servers.socket_server import SocketServer
from rpi.servers.bluetooth_server import BluetoothServer
from rpi.servers.stm_server import STMServer

import threading


# For image recognition, can start a new thread and take image and send and once reply received send it to tablet.
# However, if image rec fails then we cant do anything else if its in another thread


if __name__ == '__main__':

    #TODO: 
    '''
    1. Connect all deviced
    2. Receive Map Data from Android Tablet
    3. Send Map Data to Algorithm PC
    4. Receive Instructions from Algorithm PC
    5. One by one send instructions to STM and Android like checklist A5
    6. Receive data from STM indicating that the instruction has been executed
    7. Send next data to STM and Repeat
    8. At the end of the last instruction for any obstacle, take an image and send to Image Rec Server and get Label
    9. Image Rec server must save these images
    10. Send the label to Android Tablet in the required format
    11. Once all 5 images have been detected send a command to Image Rec Server to merge all detected images
    '''

    pass