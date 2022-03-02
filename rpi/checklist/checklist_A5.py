from rpi.servers.socket_server import SocketServer
from rpi.servers.stm_server import STMServer
from rpi.camera.camera import Camera


if __name__ == '__main__':
    
    pc = SocketServer()
    pc.setup(port=5005, clients=1)
    pc.wait_for_client()

    stm = STMServer()
    stm.setup(port='/dev/ttyUSB0', baud_rate=115200)

    camera = Camera()
    camera.setup(connect_to='tcp://192.168.27.28:5555')

    deli_1 = '-'
    deli_2 = ';'
    padding = 20

    algo_instructions = pc.read()
    
    if algo_instructions is None:
        print(f'Error instructions: {algo_instructions}')
        pc.socket.close()
        exit()

    algo_instructions = algo_instructions.split(deli_1)

    for face_instruction in algo_instructions:

        instructions = face_instruction.split(deli_2)

        for step in instructions:
            step = step.ljust(padding)

            stm.send(step)

            ack = stm.read()

        label = camera.send_image(command='capture')
        
        try:
            label = label.decode("utf-8")

        except:
            print('Could not decode label')

        print(f'Label: {label}')
    
    pc.socket.close()

