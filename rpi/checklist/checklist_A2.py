import rpi.camera.camera as camera
from rpi.servers.socket_server import SocketServer


if __name__ == '__main__':

    pc_server = SocketServer()
    pc_server.setup(port=5005, clients=1)

    conn, addr = pc_server.socket.accept()
    print('Connected by', addr)

    camera = camera.Camera()
    camera.setup(connect_to='tcp://192.168.0.101:5555')
    
    while True:
        data = conn.recv(1024)

        if data == 'capture':
            label = camera.send_image()
            print(f'The recognized image is {label}')

        if data == 'exit':
            break


    
