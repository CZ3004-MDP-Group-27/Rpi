from rpi.servers.socket_server import SocketServer
from rpi.servers.bluetooth_server import BluetoothServer
from rpi.servers.stm_server import STMServer


if __name__ == '__main__':
    pc_server = SocketServer()
    android_server = BluetoothServer()
    stm_server = STMServer()

    pc_server.setup(port=5005, clients=1)
    android_server.setup()
    stm_server.setup(port='/dev/ttyUSB0', baud_rate=115200)

    '''
    
    The receiving data will be in 2 threads
    1st thread reads data from the PC and send the data to either STM or Bluetooth
    2nd thread reads data from Tablet and send the data to eithe PC or STM

    Can have 3rd thread to send image for image recognition
    and once label is received send it to bluetooth
    
    '''

    # choice = 4

    # while choice != 0:
    #     print()
    #     print('Connection Demonstration')
    #     print()
    #     print('1. Android to STM')
    #     print('2. PC to Android')

    #     print('3. STM to PC')
    #     print('0. Exit')

    #     try:
    #         choice = int(input())
    #     except:
    #         choice = -1
    #         continue

    #     if choice == 1:
    #         data = android_server.read()
    #         if data is not None:
    #             stm_server.send(data)

    #     elif choice == 2:
    #         data = pc_server.read()
    #         if data is not None:
    #             android_server.send(data)

    #     elif choice == 3:
    #         data = stm_server.read()
    #         if data is not None:
    #             pc_server.send(data)
