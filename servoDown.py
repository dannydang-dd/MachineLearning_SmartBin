from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

# Create a pin factory
factory = PiGPIOFactory()

# Define servo angles
bottle_trash_angle = 60
glass_trash_angle = -30
apple_trash_angle = -30
can_trash_angle = 75



# Use the factory when creating your servo objects with initial angles
servoBottle = AngularServo(22, initial_angle=bottle_trash_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoGlass = AngularServo(18, initial_angle=glass_trash_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoApple = AngularServo(27, initial_angle=apple_trash_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servoCan = AngularServo(17, initial_angle=can_trash_angle, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
