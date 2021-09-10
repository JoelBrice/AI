import socket
import time

host = '127.0.0.1'
port = 5000
clients = []

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host, port))
s.setblocking(0)

quittting = False
print("Server started...")

while not quittting:
    try:
        data, addr = socket.rcvfrom(1024)
        if addr not in clients:
            clients.append(addr)
        print(time.ctime(time.time()) + str(data))
        for client in clients:
            s.sendto(client.encode('utf-8'))

    except:
        pass
s.close()
