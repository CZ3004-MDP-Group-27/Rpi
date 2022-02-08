'''
Use this code to connect to the RPi
For Checklist A1: connect to the RPi and send any text so that it can be sent to various devices
For Checklist A2: connect to the RPi and send the text 'capture' so that the RPi an capture the image and send it to the Image Recognition Server
'''


import socket


if __name__ == '__main__':
    HOST = '192.168.0.100'  # The server's hostname or IP address
    PORT = 5005        # The port used by the server

    s = socket.socket()
    s.connect((HOST, PORT))

    while True:
        text = input()
        s.send(text.encode())
        if text == 'exit':
            break
        