import cv2
import numpy as np
import time
import threading
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

# Create a pin factory
factory = PiGPIOFactory()

# Define servo angles
bottle_mid_angle = -50
glass_mid_angle = 60
can_mid_angle = -30
apple_mid_angle = 70

bottle_trash_angle = 60
glass_trash_angle = -45
apple_trash_angle = -30
can_trash_angle = 75

bottle_up_angle = -90
glass_up_angle = 90
apple_up_angle = 90
can_up_angle = -70

# Use the factory when creating your servo objects with initial angles
servoBottle = AngularServo(22, initial_angle=bottle_mid_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoGlass = AngularServo(18, initial_angle=glass_mid_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoApple = AngularServo(27, initial_angle=apple_mid_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoCan = AngularServo(17, initial_angle=can_mid_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)

# Define the duration for gradual return to mid angle
return_duration = 2  # in seconds
return_steps = 10  # number of steps for gradual return

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

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

# Function to move servo gradually
def move_servo(servo, target_angle, duration, steps):
    current_angle = servo.angle
    step = (target_angle - current_angle) / steps
    for _ in range(steps):
        new_angle = current_angle + step
        # Check if the new angle is within the valid range
        if abs(new_angle) <= 90:
            servo.angle = new_angle
            current_angle = new_angle
        time.sleep(duration / steps)

        
                
# Function to simulate detections
def simulate_detections():
    objects_to_simulate = ['bottle', 'apple', 'cup', 'trash']
    for obj in objects_to_simulate:
        print(f"Simulating detection for {obj}")
        if obj == 'bottle':
            move_servo(servoBottle, bottle_trash_angle, 1, 10)
            servoGlass.angle = glass_up_angle
            time.sleep(2)
            servoGlass.angle = glass_mid_angle
            move_servo(servoBottle, bottle_mid_angle, return_duration, return_steps)

        elif obj == 'apple':
            move_servo(servoApple, apple_trash_angle, 1, 10)
            servoCan.angle = can_up_angle
            time.sleep(2)
            servoCan.angle = can_mid_angle
            move_servo(servoApple, apple_mid_angle, return_duration, return_steps)

        elif obj == 'cup':
            move_servo(servoGlass, glass_trash_angle, 1, 10)
            servoBottle.angle = bottle_up_angle
            time.sleep(2)
            servoBottle.angle = bottle_mid_angle
            move_servo(servoGlass, glass_mid_angle, return_duration, return_steps)

        elif obj == 'trash':
            move_servo(servoCan, can_trash_angle, 1, 10)
            servoApple.angle = apple_up_angle
            time.sleep(2)
            servoApple.angle = apple_mid_angle
            move_servo(servoCan, can_mid_angle, return_duration, return_steps)

        # Wait for 3 seconds before simulating the next object
        time.sleep(3)

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
            for obj in objectInfo:
                box, className = obj
                if obj == 'bottle':
                    move_servo(servoBottle, bottle_trash_angle, 1, 10)
                    servoGlass.angle = glass_up_angle
                    time.sleep(2)
                    servoGlass.angle = glass_mid_angle
                    move_servo(servoBottle, bottle_mid_angle, return_duration, return_steps)

                elif obj == 'apple':
                    move_servo(servoApple, apple_trash_angle, 1, 10)
                    servoCan.angle = can_up_angle
                    time.sleep(2)
                    servoCan.angle = can_mid_angle
                    move_servo(servoApple, apple_mid_angle, return_duration, return_steps)

                elif obj == 'cup':
                    move_servo(servoGlass, glass_trash_angle, 1, 10)
                    servoBottle.angle = bottle_up_angle
                    time.sleep(2)
                    servoBottle.angle = bottle_mid_angle
                    move_servo(servoGlass, glass_mid_angle, return_duration, return_steps)

                elif obj == 'trash':
                    move_servo(servoCan, can_trash_angle, 1, 10)
                    servoApple.angle = apple_up_angle
                    time.sleep(2)
                    servoApple.angle = apple_mid_angle
                    move_servo(servoCan, can_mid_angle, return_duration, return_steps)

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
#     thread4 = threading.Thread(target=simulate_detections)  # New thread for simulate_detections

    thread1.start()
    thread2.start()
    thread3.start()
#     thread4.start()  # Start the new thread

    thread1.join()
    thread2.join()
    thread3.join()
#     thread4.join()  # Wait for the new thread to finish
