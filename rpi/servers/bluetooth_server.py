from bluetooth import *


class BluetoothServer():

    def __init__(self):
        self.socket = BluetoothSocket(RFCOMM)
        self.client_sock = None
        self.client_info = None
        self.uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
        print('Bluetooth socket sucessfully created')

    def setup(self):
        self.socket.bind(("", PORT_ANY))
        print('Bluetooth socket binded')
        self.socket.listen(1)
        self.port = self.socket.getsockname()[1]
        #self.uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
        advertise_service(self.socket, "MDP-Server",
        service_id = self.uuid,
        service_classes = [ self.uuid, SERIAL_PORT_CLASS ],
        profiles = [ SERIAL_PORT_PROFILE ],
        # protocols = [ OBEX_UUID ]
        )
        print("Waiting for connection on RFCOMM channel %d" % self.port)

    def wait_for_client(self):
        self.client_sock, self.client_info = self.socket.accept()
        print("Accepted connection from ", self.client_info)

    def read(self, buffer_size=1024):
        data = self.client_sock.recv(buffer_size)
        return data

    def send(self, text):
        self.client_sock.send(text)

    def end_connection(self):
        self.client_sock.close()
        self.socket.close()
        self.client_info = None
        print("disconnected")