import socket

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
    udp_socket.bind(('localhost', 12345))
    print("Listening...")
    
    # Listen for requests
    while True:
        data, address = udp_socket.recvfrom(1024)
        print(f"Received Data from {address}: {data.decode()}")
        
        # TODO: Integrate NLP Response Generation
        response = b"Python Server Response!"
        udp_socket.sendto(response, address)

