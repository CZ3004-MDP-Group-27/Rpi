from rpi.servers.socket_server import SocketServer
from rpi.servers.bluetooth_server import BluetoothServer
from rpi.servers.stm_server import STMServer
import threading


def handle_pc(pc, android, stm):
    while True:
        message = pc.read()
        print(f'Received message from Android: {message}')

        if message is not None:
            message_list = message.splt('-')
            if len(message_list) > 1 and message_list[0] == 'android':
                android.send(message_list[1])
            
            elif len(message_list) > 1 and message_list[0] == 'stm':
                stm.send(message_list[1])

            else:
                print('Could not send message to destination')


def handle_android(pc, android, stm):
    while True:
        message = android.read()
        print(f'Received message from Android: {message}')

        if message is not None:
            message_list = message.splt('-')
            if len(message_list) > 1 and message_list[0] == 'pc':
                pc.send(message_list[1])
            
            elif len(message_list) > 1 and message_list[0] == 'stm':
                stm.send(message_list[1])

            else:
                print('Could not send message to destination')


if __name__ == '__main__':
    pc_server = SocketServer()
    pc_server.setup(port=5005, clients=1)
    pc_server.wait_for_client()

    android_server = BluetoothServer()
    android_server.setup()
    android_server.wait_for_client()

    stm_server = STMServer()
    stm_server.setup(port='/dev/ttyUSB0', baud_rate=115200)

    pc_handler = threading.Thread(target=handle_pc, args=(pc_server, android_server, stm_server))
    android_handler = threading.Thread(target=handle_android, args=(pc_server, android_server, stm_server))
