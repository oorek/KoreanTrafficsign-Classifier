import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt 
from PIL import Image
import pdb

transform = A.Compose([
    A.Blur(blur_limit=2, p=0.2),
  #  A.CoarseDropout(p=0.2),
   # A.Flip(p=0.5),
    A.ToGray(p=0.3),
    A.GaussNoise(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.OpticalDistortion(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5)
  #  A.RandomResizedCrop(height=100, width=100,scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
     #                                   interpolation=1, p=0.2)
])

list = os.listdir("D:/result/result/tonghanggumzi")
for i in range(0, len(list), 1):
    for j in range(10):
        path = os.path.join("D:/result/result/tonghanggumzi", list[i])

        image = cv2.imread(path)
        dimensions = image.shape
    #   if dimensions[0]*dimensions[1] >= 500 and dimensions[0]*dimensions[1] <= 3000:
        if dimensions[0]*dimensions[1] >= 10:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #img = Image.open('./data/18.png')
        #img_resize = img.resize((100,100))
        #img_resize.save('./data/18_resize.png')
        #print(type(img_resize))

        #plt.imshow(image)
        #plt.xticks([]) # x축 눈금
        #plt.yticks([]) # y축 눈금
        #plt.show()

    #  for j in range(100):
            #Augment an image
    #     transformed = transform(image=image)
    #     transformed_image = transformed["image"]
            #cv2.imshow('t_image', transformed_image)
            #plt.imshow(transformed_image)
            #plt.xticks([]) # x축 눈금
            ##plt.show()
            
    #     im = Image.fromarray(transformed_image)
    #     im = im.resize((100,100))
    #     im.save('./data/val/' + str(i) + '/' + str(j) + '.png')

            transformed = transform(image=image)
            transformed_image = transformed["image"]
        #  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            cv2.imwrite('D:/result/result/sample/' + str(i) + str(j) + '.jpg', transformed_image)
    else :
        pass