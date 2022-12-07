import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt 
from PIL import Image
import pdb

list = os.listdir("D:/result/result/sample")
for i in range(len(list)):
    path = os.path.join("D:/result/result/sample", list[i])
    image = cv2.imread(path)
    cv2.imwrite('D:/result/result/sample/' + 'aello' + str(i) + '.jpg', image)