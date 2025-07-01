import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cv2

def init_gui(width, height):
    pygame.init()
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glOrtho(0, width, height, 0, -1, 1)
    
def draw_image(frame, width, height):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    flipped_ud_img = np.flipud(frame)
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, flipped_ud_img)
    pygame.display.flip()

def draw_rectangle_np(image, xmin, ymin, xmax, ymax, color=(0, 255, 0)):
    image[ymin:ymax, xmin] = color  
    image[ymin:ymax, xmax] = color  
    image[ymin, xmin:xmax] = color  
    image[ymax, xmin:xmax] = color 

#For debugging each side 
# def draw_rectangle_np(image, xmin, ymin, xmax, ymax, color=(0, 255, 0)):
#     image[ymin:ymax, xmin] = (255, 0, 0)  
#     image[ymin:ymax, xmax] = (0, 255, 0)  
#     image[ymin, xmin:xmax] = (0, 0, 255)  
#     image[ymax, xmin:xmax] = (255, 255, 0) 
    
def clamp_tlbr(tlbr, image_width, image_height):
    left   = max(0, min(tlbr[0], image_width-1))
    bottom = max(top, min(tlbr[1], image_height-1))
    right  = max(left, min(tlbr[2], image_width-1))
    top    = max(0, min(tlbr[3], image_height-1))
    return [left, top, right, bottom]

def draw_text_in_bbox(image, text, xmin,ymin , xmax, ymax, font_scale=0.4, color=(0, 255, 0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (xmin, ymax + 10 ), font, font_scale, color, thickness, cv2.LINE_AA)