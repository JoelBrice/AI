import socket
import threading
import time

tLock = threading.Lock()
shuutdown = False


def receiving(name, sock):
    while not shuutdown:
        try:
            tLock.acquire()
            while True:
                data, addr = sock.recvfrom(1024)
                print(str(data))
        except:
            pass
        finally:
            tLock.release()


host = '127.0.0.1'
port = 5000

server = (host, port)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((server))
s.setblocking(0)
rT = threading.Thread(target=receiving, args=("Receiving", s))
rT.start()
alias = input(" Name: ")
message = input(alias + '->')
while message != 'q':
    if message != '':
        s.send(alias+'' + message, server)
        tLock.acquire()
        message = input(alias+'->')
        tLock.release()
        time.sleep(0.2)

shuutdown = True
rT.join()
s.close()
