# Jetson TX2: video_sender.py

import cv2
import socket
import struct
import pickle

# 서버 IP 주소와 포트 설정 (Windows 10의 IP 주소)
server_ip = '192.168.0.130'
server_port = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 초기화
cap = cv2.VideoCapture(1)  # 또는 1로 변경하여 /dev/video1을 사용
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 JPEG로 압축
    _, compressed_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 압축 품질은 0-100으로 조절 가능
    data = compressed_frame.tobytes()

    # 데이터 전송
    max_packet_size = 65507  # UDP 패킷 최대 크기
    for i in range(0, len(data), max_packet_size):
        sock.sendto(data[i:i+max_packet_size], (server_ip, server_port))

cap.release()
sock.close()
