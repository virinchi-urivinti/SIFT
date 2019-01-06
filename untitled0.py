# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 02:15:53 2019

@author: virinchiurivinti
"""

import cv2
import numpy as np
from math import sqrt ,pi ,e
import matplotlib.pyplot as plt
img=cv2.imread("C:/task2.jpg",0)
imag = img.tolist()
from scipy.ndimage import gaussian_filter
imog = cv2.imread("C:/task2.jpg")
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
for row in range(len(imag)):
    imag[row] = [(i/255) for i in imag[row]]
    
    
def padd ( img):
    Matrix = [[0 for a in range(len(img[0])+6)] for b in range(len(img)+6)]
    for i in range(len(img)):
        for j in range (len(img[0])):
            Matrix[i+3][j+3]=img[i][j]
            return Matrix
        
        
c = len(imag)-1
print(str(len(imag))+" " +str(len(imag[c]))+" ")
print(str(len(imag))+" " +str(len(imag[0]))+" ")

def GaussionKernal(s ,dimension):
    matr = [[0 for i in range(dimension)]for j in range(dimension)]
    sum =0
    for x in range (dimension):
        for y in range (dimension):
            u=x- dimension//2
            v=y-dimension//2
            matr[x][y] = (1/((2*pi)*(s**2)))*e**(-0.5*((u**2)+(v**2))/(s**2))
            sum = sum + matr[x][y]
    for row in range(len(matr)):
        matr[row] = [(i / sum) for i in matr[row]]
    return matr


def calculate_positive_edges_meth2(img):
    after_pos_edg=[]
    for i in range(len(img)):
        img[i]=[abs(j) for j in img[i]]
        maximum = max([max(j) for j in img])
    for i in range(len(img)):
        img[i][:] = [x / maximum for x in img[i]]
    return img


def convolution( imag , kernal):
    Matrix = [[0 for a in range(len(imag[0]))] for b in range(len(imag))]
    imag = padd(imag)
    for z in range(3,len(imag[0])-3):
        for x in range(3, (len(imag)-3)):
            value = 0;
            for i in range(len(kernal)):
                for j in range(len(kernal)):
                    u = i - len(kernal) // 2
                    v = j - len(kernal) // 2
                    value = value + kernal[i][j] * imag[x - u][z - v]
#print(x)
#print(z)
            Matrix[x-3][z-3] = value
    return Matrix


def reduce( imag , kernal):
#imag = padd(imag)
    Matrix = [[0 for a in range(len(imag[0])//2)] for b in range(len(imag)//2)]
    for z in range(1, (len(imag[0])//2)-1):
        for x in range(1, (len(imag)//2)-1):
            value = 0;
            for i in range(len(kernal)):
                for j in range(len(kernal)):
                    u = i - len(kernal) // 2
                    v = j - len(kernal) // 2
                    value = value + kernal[i][j] * imag[2*x - u][2*z - v]
            Matrix[x][z] = value
    return Matrix


def generate_octave(si,imag):

    octave=[]
    #gauus = GaussionKernal(s=5, dimension=3)
    sig=si
    for i in range(0,5):
        gauus = GaussionKernal(s=sig, dimension=7)
        blurred = convolution(imag, gauus)
        #blurred = gaussian_filter(imag, sigma=si)
        octave.append(blurred)
        sig =sig*sqrt(2)
    return octave


def generate_scale_space(si, image):
    scale_space=[]
    si =1/sqrt(2)
    oct1=generate_octave(si,imag)
    scale_space.append(oct1)
    next_oct = reduce(oct1[2],sam)
    oct2=generate_octave(sqrt(2),next_oct)
    scale_space.append(oct2)
    next_oct = reduce(oct2[2],sam)
    oct3=generate_octave(2*sqrt(2),next_oct)
    scale_space.append(oct3)
    next_oct=reduce(oct3[2],sam)
    oct4 = generate_octave(4* sqrt(2), next_oct)
    scale_space.append(oct4)
    return scale_space

def subtract(m1 ,m2):
    matr = [[0 for i in range(len(m1[0]))]for j in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            matr[i][j] = m1[i][j]- m2[i][j]
    return matr


def generate_DOG(scale_space):
    all_dogs =[]
    for i in range(len(scale_space)):
        dog = []
        for j in range(len(scale_space[0])-1):
            x1 = subtract(scale_space[i][j],scale_space[i][j+1])
            # x1=calculate_positive_edges_meth2(x1)
            dog.append(x1)
            all_dogs.append(dog)
    return all_dogs

def keypointdetection( scalespace ):
    kpm = []
    count = 1
    for oc in scalespace:
        Matrix = [[0 for a in range(len(oc[0][0]))] for b in range(len(oc[0]))]
        dh = len(oc[0])
        dw = len(oc[0][0])
        for i in range(1, dh - 1):
            for j in range(1, dw - 1):
                b1 = 0
                c1 = 0
                for l in range(3):
                    for k in range(3):
                        b = max(oc[0][i - 1 + l][j - 1 + k], oc[1][i - 1 + l][j - 1 +
                                k],oc[2][i - 1 + l][j - 1 + k])
                        c = min(oc[0][i - 1 + l][j - 1 + k], oc[1][i - 1 + l][j - 1 +k],oc[2][i - 1 + l][j - 1 + k])
                        if (b >= b1):
                            b1 = b
                        if (c <= c1):
                            c1 = c
                            
                        if (oc[1][i][j] >= b1):
                            Matrix[i][j] = 2 * oc[1][i][j]
                            imog[i*count][j*count]=(0,0,255)
                        elif (oc[1][i][j] <= c1):
                            Matrix[i][j] = 2 * oc[1][i][j]
                            imog[i*count][j*count]= (0,0,255)
                            count = count *2
                            kpm.append(Matrix)
    return kpm

sc = generate_scale_space(0,imag)
m =1
for i in range(len(sc)):
    for j in range(len(sc[0])):
        plt.subplot(len(sc),len(sc[0]),m)
        plt.imshow(sc[i][j],cmap='gray')
        m=m+1
        plt.show()
        
dogs = generate_DOG (sc)
m =1
for i in range(len(dogs)):
    for j in range(len(dogs[0])):
        plt.subplot(len(dogs),len(dogs[0]),m)
        plt.imshow(dogs[i][j],cmap='gray')
        m=m+1
        plt.show()
        
kpm = keypointdetection(dogs)
#opium = keypointmap(kpm)
j=1
for ima in dogs[1]:
    cv2.imwrite('C:/dog2/x'+str(j)+'.jpg', np.asarray(ima))
    j= j+1
    j=1
for ima in dogs[2]:
cv2.imwrite('C:/dog3/y'+str(j)+'.jpg', np.asarray(ima))
j= j+1
sc = generate_scale_space(0,imag)
cv2.namedWindow('', cv2.WINDOW_NORMAL)
cv2.imshow('dog_2_1', np.asarray(dogs[1][0]))
cv2.namedWindow('oc2', cv2.WINDOW_NORMAL)
cv2.imshow('dog_2_2',np.asarray(dogs[1][1]))
cv2.namedWindow('oc3', cv2.WINDOW_NORMAL)
cv2.imshow('dog_2_3', np.asarray(dogs[1][2]))
cv2.namedWindow('oc4', cv2.WINDOW_NORMAL)
cv2.imshow('dog_2_4',np.asarray(dogs[1][3]))
cv2.namedWindow('oc5', cv2.WINDOW_NORMAL)
cv2.imshow('dog_3_1',np.asarray(dogs[2][0]))
cv2.namedWindow('oc6', cv2.WINDOW_NORMAL)
cv2.imshow('dog_3_2',np.asarray(dogs[2][1]))
cv2.namedWindow('oc7', cv2.WINDOW_NORMAL)
cv2.imshow('dog_3_3',np.asarray(dogs[2][3]))
cv2.namedWindow('oc8', cv2.WINDOW_NORMAL)
cv2.namedWindow("sdf",cv2.WINDOW_NORMAL)
cv2.imshow ("sdf",np.asarray(imog))
cv2.waitKey(0)