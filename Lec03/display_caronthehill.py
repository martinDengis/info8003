
## code from https://github.com/epochstamp/INFO8003-1/blob/master/continuous_domain/display_caronthehill.py
import pygame
import numpy as np
from math import atan2, degrees, pi, sqrt
import os
import imageio

# Constants
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400
MAX_SPEED = 3
MIN_SPEED = -3
LOC_WIDTH_FROM_BOTTOM = 35
LOC_HEIGHT_FROM_BOTTOM = 70
MAX_HEIGHT_SPEED = 50
WIDTH_SPEED = 30
THICKNESS_SPEED_LINE = 3

# Colors
color_hill = pygame.Color(0, 0, 0, 0)
color_shill = pygame.Color(64, 163, 191, 0)
color_phill = pygame.Color(64, 191, 114, 0)
color_acc_line = pygame.Color(0, 0, 0, 0)

# Global variables for caching
car = None
pt = None
background = None
checked = False
size_car = None
width_car = None
height_car = None
size_pt = None
width_pt = None
height_pt = None

def Hill(p):
    return p*p+p if p < 0 else p/(sqrt(1+5*p*p))

def ppoints_to_angle(x1, x2):
    dx = x1[1] - x1[0]
    dy = x2[1] - x2[0]
    rads = atan2(-dy, dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs

def rotate(image, rect, angle):
    """Rotate the image while keeping its center."""
    new_image = pygame.transform.rotate(image, angle)
    rect = new_image.get_rect(center=rect.center)
    return new_image, rect

def save_caronthehill_image(position, speed, out_file=None):
    global car, pt, background, checked
    global size_pt, width_pt, height_pt, size_car, width_car, height_car
    
    if car is None:
        car = pygame.image.load("car.png")
        size_car = car.get_rect().size
        width_car = size_car[0]
        height_car = size_car[1]
    if pt is None:
        pt = pygame.image.load("pine_tree.png")
        size_pt = pt.get_rect().size
        width_pt = size_pt[0]
        height_pt = size_pt[1]

    surf = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))

    step_hill = 2.0/CANVAS_WIDTH

    if not checked and not os.path.isfile(f"background_{CANVAS_WIDTH}_{CANVAS_HEIGHT}.png"):
        points = list(np.arange(-1, 1, step_hill))
        hl = list(map(Hill, points))
        range_h = range(CANVAS_HEIGHT)
        pix = 0
        
        for h in hl:
            x = pix
            y = ((CANVAS_HEIGHT)/2) * (1+h)
            y = int(round(y))
            
            for yo in range_h:
                if yo < y:
                    c = color_phill
                elif yo > y:
                    c = color_shill
                surf.set_at((x, CANVAS_HEIGHT - yo), c)
            
            surf.set_at((x, CANVAS_HEIGHT - y), color_hill)
            pix += 1
            
        pygame.image.save(surf, f"background_{CANVAS_WIDTH}_{CANVAS_HEIGHT}.png")
        checked = True
    else:
        if background is None:
            background = pygame.image.load(f"background_{CANVAS_WIDTH}_{CANVAS_HEIGHT}.png")
        surf.blit(background, (0,0))

    pt_pos1 = -0.5
    pt_pos2 = 0.5
    surf.blit(pt, (round((CANVAS_WIDTH/2)*(1+pt_pos1)) - width_pt/2, 
              CANVAS_HEIGHT - round(((CANVAS_HEIGHT)/2) * (1+Hill(pt_pos1))) - height_pt))
    surf.blit(pt, (round((CANVAS_WIDTH/2)*(1+pt_pos2)) - width_pt/2, 
              CANVAS_HEIGHT - round(((CANVAS_HEIGHT)/2) * (1+Hill(pt_pos2))) - height_pt))

    x_car = round((CANVAS_WIDTH/2)*(1+position)) - width_car/2
    h_car = Hill(position)
    h_car_next = Hill(position + step_hill)
    y_car = CANVAS_HEIGHT - round(((CANVAS_HEIGHT)/2) * (1+h_car)) - height_car
    angle = ppoints_to_angle((position,position+step_hill), (h_car,h_car_next))
    rot_car, rect = rotate(car, pygame.Rect(x_car, y_car, width_car, height_car), 360-angle)
    surf.blit(rot_car, rect)

    rect = (CANVAS_WIDTH-LOC_WIDTH_FROM_BOTTOM - WIDTH_SPEED, 
            CANVAS_HEIGHT - LOC_HEIGHT_FROM_BOTTOM, 
            WIDTH_SPEED, THICKNESS_SPEED_LINE)
    surf.fill(color_acc_line, rect)
    
    pct_speed = abs(speed)/MAX_SPEED
    color_speed = (pct_speed * 255, (1-pct_speed)*255, 0)
    height_speed = MAX_HEIGHT_SPEED*(pct_speed)
    
    loc_width = CANVAS_WIDTH - WIDTH_SPEED - LOC_WIDTH_FROM_BOTTOM
    loc_height = (CANVAS_HEIGHT - LOC_HEIGHT_FROM_BOTTOM + THICKNESS_SPEED_LINE 
                 if speed < 0 else CANVAS_HEIGHT - LOC_HEIGHT_FROM_BOTTOM - height_speed)
    rect = (int(loc_width), int(loc_height), int(WIDTH_SPEED), int(height_speed))
    surf.fill(color_speed, rect)

    if out_file is not None:
        pygame.image.save(surf, out_file)
    else:
        return pygame.surfarray.array3d(surf).transpose([1, 0, 2])