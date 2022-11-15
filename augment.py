import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt 
from PIL import Image

transform = A.Compose([
    A.Blur(blur_limit=3, p=0.5),
    A.CoarseDropout(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.9),
    A.RandomResizedCrop(height=100, width=100,scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=0.8)
])

for i in range(0,30):
    image = cv2.imread('./data/' + str(i) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #img = Image.open('./data/18.png')
    #img_resize = img.resize((100,100))
    #img_resize.save('./data/18_resize.png')
    #print(type(img_resize))

    #plt.imshow(image)
    #plt.xticks([]) # x축 눈금
    #plt.yticks([]) # y축 눈금
    #plt.show()

    for j in range(1000):
        #Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        #cv2.imshow('t_image', transformed_image)
        #plt.imshow(transformed_image)
        #plt.xticks([]) # x축 눈금
        ##plt.show()
        
        im = Image.fromarray(transformed_image)
        im = im.resize((100,100))
        im.save('./data/train/' + str(i) + '/' + str(j) + '.png')