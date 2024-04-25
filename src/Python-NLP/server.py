import socket
import model

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
    udp_socket.bind(('localhost', 12345))
    print("Listening...")
    
    # Listen for requests
    while True:
        data, address = udp_socket.recvfrom(1024)
        message = data.decode()
        if message == "<exit>": 
            print(f"Received exit command! Closing Program")
            break

        print(f"Received Data from {address}: {message}")
        
        response = model.generate_reponse(message)
        udp_socket.sendto(response.encode(), address)