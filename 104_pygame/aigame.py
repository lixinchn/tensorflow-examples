# -*- coding: utf-8 -*-
import pygame
import random
from pygame.locals import *
import numpy as np
from collections import deque
import tensorflow as tf
import cv2


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_SIZE = [320, 400]
BAR_SIZE = [20, 5]
BALL_SIZE = [15, 15]

# 神经网络的输出
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]


class Game(object):
  def __init__(self):
    pygame.init()
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Simple Game')

    self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
    self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2

    # ball移动方向
    self.ball_dir_x = -1
    self.ball_dir_y = -1
    self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])

    self.score = 0
    self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
    self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

