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

bottle_trash_angle = 80
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

if __name__ == "__main__":
    while True:
        
#         time.sleep(2)
#         servoApple.angle = apple_trash_angle
#         time.sleep(1)
#         servoApple.angle = apple_mid_angle
#         time.sleep(1)
#         servoApple.angle = apple_up_angle
#         
#         time.sleep(10)
        
        # Bottle Trash
        time.sleep(2)
        servoBottle.angle = bottle_trash_angle
        servoGlass.angle = glass_up_angle
        time.sleep(2)
        
        # Gradual return to mid angle for bottle
        servoGlass.angle = glass_mid_angle
        bottle_step = (bottle_mid_angle - bottle_trash_angle) / return_steps
        for _ in range(return_steps):
            if abs(servoBottle.angle + bottle_step) <= 90:  # Check if angle is within valid range
                servoBottle.angle += bottle_step
                time.sleep(return_duration / return_steps)
                
        # Apple Trash
        time.sleep(2)
        servoApple.angle = apple_trash_angle
        servoCan.angle = can_up_angle
        time.sleep(2)

        # Gradual return to mid angle for apple
        servoCan.angle = can_mid_angle
        apple_step = (apple_mid_angle - apple_trash_angle) / return_steps
        for _ in range(return_steps):
            if abs(servoApple.angle + apple_step) <= 90:  # Check if angle is within valid range
                servoApple.angle += apple_step
                time.sleep(return_duration / return_steps)
        
        # Glass Trash
        time.sleep(2)
        servoGlass.angle = glass_trash_angle
        servoBottle.angle = bottle_up_angle
        time.sleep(2)

        # Gradual return to mid angle for glass
        servoBottle.angle = bottle_mid_angle
        glass_step = (glass_mid_angle - glass_trash_angle) / return_steps
        for _ in range(return_steps):
            if abs(servoGlass.angle + glass_step) <= 90:  # Check if angle is within valid range
                servoGlass.angle += glass_step
                time.sleep(return_duration / return_steps)

        # Can Trash
        time.sleep(2)
        servoCan.angle = can_trash_angle
        servoApple.angle = apple_up_angle
        time.sleep(2)
        
        # Gradual return to mid angle for can
        servoApple.angle = apple_mid_angle
        can_step = (can_mid_angle - can_trash_angle) / return_steps
        for _ in range(return_steps):
            if abs(servoCan.angle + can_step) <= 90:  # Check if angle is within valid range
                servoCan.angle += can_step
                time.sleep(return_duration / return_steps)
