import socket
import threading

HOST = '192.168.27.27'  # The server's hostname or IP address
PORT = 5004        # The port used by the server

s = socket.socket()
s.connect((HOST, PORT))

def receive():
    while True:
        text = s.recv(1024)
        print(text.decode())

def send():
    while True:
        text = input()
        s.send(text.encode())

threading.Thread(target=receive).start()
threading.Thread(target=send).start()



        