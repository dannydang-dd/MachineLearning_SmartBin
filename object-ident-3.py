import cv2

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

# Create a pin factory
factory = PiGPIOFactory()

# Define servo angles
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


classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    # Bottle Trash
                    if className == 'bottle':
                        time.sleep(1)
                        servoBottle.angle = bottle_trash_angle
                        servoGlass.angle = glass_up_angle
                        time.sleep(10)
                        
                        # Gradual return to mid angle for bottle
                        servoGlass.angle = glass_mid_angle
                        bottle_step = (bottle_mid_angle - bottle_trash_angle) / return_steps
                        for _ in range(return_steps):
                            if abs(servoBottle.angle + bottle_step) <= 90:  # Check if angle is within valid range
                                servoBottle.angle += bottle_step
                                time.sleep(return_duration / return_steps)
                        time.sleep(20)
                    
                    # Apple Trash
                    if className == 'apple':
                        time.sleep(1)
                        servoApple.angle = apple_trash_angle
                        servoCan.angle = can_up_angle
                        time.sleep(10)

                        # Gradual return to mid angle for apple
                        servoCan.angle = can_mid_angle
                        apple_step = (apple_mid_angle - apple_trash_angle) / return_steps
                        for _ in range(return_steps):
                            if abs(servoApple.angle + apple_step) <= 90:  # Check if angle is within valid range
                                servoApple.angle += apple_step
                                time.sleep(return_duration / return_steps)
                        time.sleep(20)
 
                     # Glass Trash
                    if className == 'cup':
                        time.sleep(1)
                        servoGlass.angle = glass_trash_angle
                        servoBottle.angle = bottle_up_angle
                        time.sleep(10)

                        # Gradual return to mid angle for glass
                        servoBottle.angle = bottle_mid_angle
                        glass_step = (glass_mid_angle - glass_trash_angle) / return_steps
                        for _ in range(return_steps):
                            if abs(servoGlass.angle + glass_step) <= 90:  # Check if angle is within valid range
                                servoGlass.angle += glass_step
                                time.sleep(return_duration / return_steps)
                        time.sleep(20)
                        
                    # Can trash
                    if className == 'trash':
                        time.sleep(1)
                        servoCan.angle = can_trash_angle
                        servoApple.angle = apple_up_angle
                        time.sleep(10)
                        
                        # Gradual return to mid angle for can
                        servoApple.angle = apple_mid_angle
                        can_step = (can_mid_angle - can_trash_angle) / return_steps
                        for _ in range(return_steps):
                            if abs(servoCan.angle + can_step) <= 90:  # Check if angle is within valid range
                                servoCan.angle += can_step
                                time.sleep(return_duration / return_steps)
                        time.sleep(20)
                        
    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    
    
    while True:
#         servoBottle.angle = bottle_trash_angle
        
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, objects=['cup', 'vase', 'banana', 'bottle', 'apple'])
        #print(objectInfo)
        
        
        
        cv2.imshow("Output",img)
        cv2.waitKey(1)
    