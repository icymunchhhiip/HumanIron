import time
import pygame

pygame.mixer.init()
stretching_sound = pygame.mixer.Sound("real_adult.wav")

CYCLE_MIN = 3
seconds = CYCLE_MIN * 60

while True:
    time.sleep(seconds)
    print("please stretching")
    stretching_sound.play()
