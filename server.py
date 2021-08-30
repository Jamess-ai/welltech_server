
import socket 
import threading
import matplotlib.pyplot as plt 
import time
import json
from os import system

from fer import FER
from PIL import Image



PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
BUF_SIZE = 4194304                          # Buffer size 4MB
DISCONNECT_MESSAGE = "!DISCONNECT"
PREDICT_RESULT_LOC = 'D:\@IBM - AI\i.am-vitalize - AI modules\Final Project\Server\server_predict_result.txt'
CUST_IMG_LOC = 'D:\@IBM - AI\i.am-vitalize - AI modules\Final Project\Server\server_cust_image.jpg'



server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("\nserver socket is created.")
server_sock.bind(ADDR)
print("\nserver socket is binded to the host and port.")



def handle_client(conn, addr):
    
    print(f"\n[NEW CONNECTION] {addr} connected.")

    image_file = open(CUST_IMG_LOC, "wb")       # opens a file and returns a file object 
                                                # mode: write & to be handled: binary
    image_chunk = conn.recv(BUF_SIZE)                
    print("server received customer image from client.")
    image_file.write(image_chunk)
    image_file.close()

    emo_detector = FER(mtcnn=True)

    test_image = plt.imread(CUST_IMG_LOC)

    captured_emotions = emo_detector.detect_emotions(test_image)
    print("\nmodel is detecting customer facial emotion...")

    dominant_emotion, emotion_score = emo_detector.top_emotion(test_image)
    prediction = {"Captured Emotions": captured_emotions, "Dominant Emotion": dominant_emotion, "Score": emotion_score}
    print("\nfacial emotion recognised.\n")
    
    print(prediction)

    with open(PREDICT_RESULT_LOC, 'w') as prediction_file:
        prediction_file.write(json.dumps(prediction)) 
    
    prediction_file.close()

    predict_result_file = open(PREDICT_RESULT_LOC, 'rb')

    predict_result_data = predict_result_file.read()
    conn.sendall(predict_result_data)
    print("\nserver sent prediction result to client.") 

    predict_result_file.close()

    with Image.open(CUST_IMG_LOC) as img:
        img.show()

    conn.close()

    print("\nserver thread to handle client is closed.")  
    print("\nserver is waiting for new connection ...") 



def start():

    server_sock.listen()
    print(f"\nserver is listening on: {SERVER}")

    while True:
         
        conn, addr = server_sock.accept() # wait for new connection ip addr
        print("\nserver accepts a new connection.")
        thread = threading.Thread(target=handle_client, args=(conn, addr)) # pass the connection to handle_client fuct
        thread.start()
        print(f"[ACTIVE CONNECTIONS]: {threading.activeCount() - 1}") # print num of active connections (clients)



print("\n--- server application starts ---\n")
start()