import socket


def check_server_exists(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        server_exists = s.connect_ex((ip, port)) == 0
    return server_exists
