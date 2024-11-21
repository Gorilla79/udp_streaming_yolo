import cv2
import socket
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with your desired YOLOv8 model

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

                # Filter results to detect only people
                for result in results[0].boxes:
                    cls = int(result.cls)
                    if cls == 0:  # Class 0 corresponds to "person"
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, result.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Draw bounding box and center coordinates
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person ({cx}, {cy})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"Person detected at X: {cx}, Y: {cy}")

                # Display the frame
                cv2.imshow("YOLOv8 Video Stream - Person Detection", frame)

            buffer = b''  # Reset buffer for next frame

        if cv2.waitKey(1) == 27:  # Exit on ESC key
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    sock.close()
    cv2.destroyAllWindows()
