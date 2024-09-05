import cv2
import threading
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

# Create a pin factory
factory = PiGPIOFactory()

# Define servo angles and other constants
bottle_mid_angle = -50
glass_mid_angle = 60
can_mid_angle = -30
apple_mid_angle = 70

bottle_trash_angle = 70
glass_trash_angle = -80
apple_trash_angle = -30
can_trash_angle = 70

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

# Load class names
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Set up paths for configuration and weights
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Load the neural network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Function to control servo movement
def move_servo(servo, target_angle, duration, steps):
    step = (target_angle - servo.angle) / steps
    for _ in range(steps):
        if abs(servo.angle + step) <= 90:  # Check if angle is within valid range
            servo.angle += step
            time.sleep(duration / steps)

# Function to handle object detection and servo movement
def object_detection_and_servo_control():
    while True:
        success, img = cap.read()
        if success:
            result, objectInfo = getObjects(img, 0.45, 0.2, objects=['cup', 'vase', 'banana', 'bottle', 'apple'])
            for obj in objectInfo:
                box, className = obj
                # Bottle Trash
                if className == 'bottle':
                    threading.Thread(target=move_servo, args=(servoBottle, bottle_trash_angle, 1, 10)).start()
                    threading.Thread(target=move_servo, args=(servoGlass, glass_up_angle, 1, 10)).start()
                    time.sleep(10)  # Wait for the servo to move
                    threading.Thread(target=move_servo, args=(servoGlass, glass_mid_angle, return_duration, return_steps)).start()
                    threading.Thread(target=move_servo, args=(servoBottle, bottle_mid_angle, return_duration, return_steps)).start()
                    time.sleep(20)  # Wait before the next action

                # Glass Trash
                if className == 'cup' or className == 'vase':
                    threading.Thread(target=move_servo, args=(servoGlass, glass_trash_angle, 1, 10)).start()
                    threading.Thread(target=move_servo, args=(servoBottle, bottle_up_angle, 1, 10)).start()
                    time.sleep(10)  # Wait for the servo to move
                    threading.Thread(target=move_servo, args=(servoBottle, bottle_mid_angle, return_duration, return_steps)).start()
                    threading.Thread(target=move_servo, args=(servoGlass, glass_mid_angle, return_duration, return_steps)).start()
                    time.sleep(20)  # Wait before the next action

                # Apple Trash
                if className == 'apple' or className == 'banana' :
                    threading.Thread(target=move_servo, args=(servoApple, apple_trash_angle, 1, 10)).start()
                    threading.Thread(target=move_servo, args=(servoCan, can_up_angle, 1, 10)).start()
                    time.sleep(10)  # Wait for the servo to move
                    threading.Thread(target=move_servo, args=(servoCan, can_mid_angle, return_duration, return_steps)).start()
                    threading.Thread(target=move_servo, args=(servoApple, apple_mid_angle, return_duration, return_steps)).start()
                    time.sleep(20)  # Wait before the next action

                # Can Trash
                if className == 'can':
                    threading.Thread(target=move_servo, args=(servoCan, can_trash_angle, 1, 10)).start()
                    threading.Thread(target=move_servo, args=(servoApple, apple_up_angle, 1, 10)).start()
                    time.sleep(10) # Wait for the servo to move
                    threading.Thread(target=move_servo, args=(servoApple, apple_mid_angle, return_duration, return_steps)).start()
                    threading.Thread(target=move_servo, args=(servoCan, can_mid_angle, return_duration, return_steps)).start()
                    time.sleep(20) # Wait before the next action

            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Initialize and start the object detection thread
object_detection_thread = threading.Thread(target=object_detection_and_servo_control)
object_detection_thread.start()

# Function for detecting objects
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Servo control logic goes here
                

    return img, objectInfo

# Main loop for the program
if __name__ == "__main__":
    try:
        # Wait for the object detection thread to finish
        object_detection_thread.join()
    except KeyboardInterrupt:
        # Handle any cleanup here
        print("Program stopped")
    finally:
        # Release the video capture object and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
