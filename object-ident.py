import cv2
import numpy as np
import time
import threading

# Load class names
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Set up paths for configuration and weights
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Load the neural network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(300, 300)  # Adjusted to lower resolution for better performance
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Global variables for sharing data between threads
frame = None
detections = None
lock = threading.Lock()

# Function for detecting objects
def getObjects(img, thres, nms, draw=False, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

# Thread for capturing frames
def capture_frames():
    global frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS if supported

    while True:
        success, img = cap.read()
        if not success:
            break
        with lock:
            frame = img
        time.sleep(0.01)

    cap.release()

# Thread for object detection
def detect_objects():
    global frame, detections
    while True:
        if frame is not None:
            with lock:
                img = frame.copy()
            result, objectInfo = getObjects(img, 0.45, 0.2, draw=True, objects=['cup', 'vase', 'banana', 'bottle', 'apple'])
            with lock:
                detections = result
            time.sleep(0.01)

# Main thread for displaying results and calculating FPS
def display_frames():
    global detections
    fps_counter = 0
    start_time = time.time()
    
    while True:
        if detections is not None:
            with lock:
                img = detections.copy()
            if img is not None and isinstance(img, np.ndarray):
                cv2.imshow("Output", img)
                fps_counter += 1
                
                # Calculate and print FPS every second
                if time.time() - start_time >= 1:
                    print(f"FPS: {fps_counter}")
                    fps_counter = 0
                    start_time = time.time()
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and start threads
    thread1 = threading.Thread(target=capture_frames)
    thread2 = threading.Thread(target=detect_objects)
    thread3 = threading.Thread(target=display_frames)

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()
