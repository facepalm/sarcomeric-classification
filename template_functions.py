#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:23:05 2017

Creating Template Functions
"""

#import os.path
#import image_processing as ip
import numpy as np
#from skimage import data
import cv2

def createTemplate(period=120, stagger=1, tilt = [0,0], contrast = 1. ): # period=width of 1 period,  stagger: 1=full stagger 0=no stagger, tilt: a is tilt of top and b is tilt of bottom (-1 is left, 1 is right)
    #compute the pixel dim
    period *= 2
    bors = int(period * 0.05)
    borw = int(period * 0.1)
    #borw = int( period * 0.2 / 3. )
    bw = int(0.2*period) #black width
    ww = int(0.3*period) #white width
    img = np.zeros((period,period), np.uint8) #make a black image
    #make the template
    small_band = 255/5
    cv2.rectangle(img,(bors,0),(bors+borw-1,period),(small_band),-1) #make first small stripe
    cv2.rectangle(img,(period/2+bors,0),(period/2+bors+borw-1,period),(small_band),-1) #make second small stripe

    cv2.rectangle(img,(bw,0),(bw+ww-1,period),int(contrast*255),-1) #make first white rectangle
    cv2.rectangle(img,(2*bw+ww,0),(2*bw+2*ww,period),int(contrast*255),-1) #make second white rectangle

    #shift the bottom half over
    for i in range(period/2, period):
        img[i,:]=np.roll(img[i,:], int(0.25*stagger*period) )

    img = tiltTemplate( img, tilt[0], tilt[1] )
    return img


def tiltTemplate(img,a=0,b=0):

    rows,cols = img.shape

    for i in range(rows/2): # use this section to tilt top portion
        img[i,:]=np.roll(img[i,:], int( a * (i + 1 ) ) ) # a = negative tilts left, positive tilts right

    for i in range(rows/2,rows): # use this section to tilt bottom portion
        img[i,:]=np.roll(img[i,:], int( b * (i - rows/2 + 1 ) ) ) # b = negative tilts left, positive tilts right

    return img

'''
if __name__ == '__main__':

    image_dir = '/Users/Roxana/Desktop/MG_Summer 2017/Coding Images'

    img = data.load(os.path.join( image_dir,"staggered_template.tif" ))
'''
