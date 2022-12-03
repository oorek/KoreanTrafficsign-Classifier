import os
import albumentations as A
import cv2
from matplotlib import pyplot as plt 
from PIL import Image

for i in range(4,5):
    img = Image.open('./test/crop/' + 'test' + str(i) + '.jpg')
    img_resize = img.resize((100,100))

    img_resize.save('./test/final/' + 'test' + str(i) + '.jpg')