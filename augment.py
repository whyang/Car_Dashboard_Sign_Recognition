# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26, 2019
@author: whyang

The purpose of this program is to generate more images which are provided for 
training and testing processes in deep neural network learning work.
We adopt ImageDataGenerator of tensorflow version 2.0.0 to augment the needed data. 
"""
###
# USAGE
# python augment.py --image ./sign/Train/0/aug.jpg --savedir ./sign/Train/0/augment --number 10
##

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot as plt
from pylab import rcParams
import argparse
import cv2

##
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# file name of image that is used to augment data
# file format can be jpeg or png
ap.add_argument('-i', '--image',
                required=True, 
                help="the image used for augmenting data of training's and testing's")
# path(directory) of storing augmented data 
ap.add_argument('-s', '--savedir',
                required=True, 
                help="path to output directory of augmented images")
# number of augmenting data 
ap.add_argument('-n', '--number',
                required=True, 
                help="the number of augmenting images")
args = vars(ap.parse_args())

imgforAug = args["image"] #'./sign/Train/4/aug.jpg'
img = cv2.imread(imgforAug)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rcParams['figure.figsize'] = 15, 15
plt.imshow(img)
print('*** original image: ', img.shape)
img = img.reshape((1,) + img.shape)
print('*** image reshape: ', img.shape)

# step 1: create augment generator 
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             #rescale=0.9,
                             fill_mode='nearest')

# step 2: conduct each round of image augmentation 
numAug = int(args["number"])
i = 1
for batch in datagen.flow(img,
                          #batch_size=1,
                          save_to_dir=args["savedir"],
                          #save_prefix=i, #'0',
                          save_format='png'):
    # display the each augmented image
    # matrix is [numAug, 1], so grids=numAug*1, index i represents locating at the ith grid
    plt.subplot(numAug, 1, i)
    plt.axis('off')
    augImage = batch[0]
    augImage = augImage.astype('float32')
    augImage /= 255
    plt.imshow(augImage)
    i += 1
    if i > numAug:
        break

###
# end of file
##