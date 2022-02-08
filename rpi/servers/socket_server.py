

# first of all import the socket library
import socket            

class SocketServer():
    
    def __init__(self):
        self.socket = socket.socket()
        self.conn = None
        self.addr = None
        print ("Socket successfully created")

    def setup(self, port=5005, clients=1):
        self.socket.bind(('', port)) 
        print ("socket binded to %s" %(port))
        self.listen(clients)    
        print ("socket is listening")

    def wait_for_client(self):
        self.conn, self.addr = self.socket.accept()

    def read(self, buffer_size=1024):
        data = self.conn.recv(buffer_size)
        return data.decode()

    def send(self, text):
        self.conn.send(text.encode())

             
 
