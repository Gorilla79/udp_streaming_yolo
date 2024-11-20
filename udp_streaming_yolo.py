import cv2
import socket
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with your desired YOLOv8 model (e.g., yolov8s.pt, yolov8m.pt)

# Set port to receive video stream
local_ip = '0.0.0.0'  # Listen from all IPs
local_port = 5000

# Create and bind UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((local_ip, local_port))

print("Listening for incoming video stream...")

buffer = b''  # Data buffer

try:
    while True:
        # Receive data (buffer size is set to the maximum size of UDP packets)
        packet, _ = sock.recvfrom(65507)
        buffer += packet

        # Restore frames when the buffer is sufficiently filled
        if len(buffer) > 10000:  # Set threshold based on expected frame size
            # Decode JPEG image
            frame_data = np.frombuffer(buffer, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            if frame is not None:
                # Perform YOLOv8 inference
                results = model(frame)

                # Draw detection results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Video Stream", annotated_frame)

            buffer = b''  # Reset buffer for next frame

        if cv2.waitKey(1) == 27:  # Exit on ESC key
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    sock.close()
    cv2.destroyAllWindows()
