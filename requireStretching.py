import datetime
import time
import pygame

pygame.mixer.init()
blink_sound = pygame.mixer.Sound("real_adult.wav")

CYCLE_MIN = 3
seconds = CYCLE_MIN * 60

while True:
    print("please stretching")
    blink_sound.play()
    time.sleep(seconds)
