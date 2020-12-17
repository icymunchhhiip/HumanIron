import picamera
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, GPIO.PUD_UP)


with picamera.PiCamera() as camera:
    camera.start_preview()
    frame = 1
    GPIO.wait_for_edge(17, GPIO.FALLING)
    while True:
        start = time.time()
        camera.capture('/home/pi/Desktop/frame%03d.jpg' % frame)
        frame += 1
        camera.stop_preview()
        time.sleep(
            int(5) - (time.time() - start)
        )

    
    