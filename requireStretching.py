import time
import pygame

pygame.mixer.init()
stretching_sound = pygame.mixer.Sound("real_adult.wav")

CYCLE_MIN = 20
REST_TIME_MIN = 5

cycle_seconds = CYCLE_MIN * 60
rest_seconds = REST_TIME_MIN * 60

while True:
    time.sleep(cycle_seconds)
    print("please stretching")
    stretching_sound.play()
    time.sleep(rest_seconds)
