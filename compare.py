import numpy as np
import matplotlib.pyplot as plt

#imgA = plt.imread('IMG_20180106_233742_949.jpg')

import os
dictt = {}
img =[ x for x in os.listdir('.') if ('jpg' in x and len(x) == 6)]
import cv2
def find_best(ori,insta):
    if ori.shape[0] > ori.shape[1] :
        offset = ori.shape[0] - ori.shape[1]
        ori2 = ori[offset:,:,:]
    if ori.shape[1] > ori.shape[0] :
        offset = ori.shape[1] - ori.shape[0]
        ori2 = ori[:,offset:,:]    
    
for i in  sorted(img):
    print i,
    print plt.imread(i).shape
    f, ax = plt.subplots(1,4,figsize=(20,5));
    ori0 = plt.imread(i)
    ax[0].imshow(ori0)
    ori = plt.imread('image_00'+str(i))
    ori2 = ori[:,:,:]
    if ori.shape[0] > ori.shape[1] :
        offset = ori.shape[0] - ori.shape[1]
        ori2 = ori[offset//2:-offset//2:,:,:]
    if ori.shape[1] > ori.shape[0] :
        offset = ori.shape[1] - ori.shape[0]
        ori2 = ori[:,offset//2:-offset//2,:]
    ax[2].imshow(ori)
    assert ori0.shape ==ori2.shape #final and initial the same
    assert ori0.shape[0] == ori0.shape[1] #square pic
    if ori0.shape[0] != 500:
        ori0 = cv2.resize(ori0,(500,500))
        ori2 = cv2.resize(ori2,(500,500))
    dictt[i] = [ori0,ori2]
    print ori0.shape,ori2.shape
    ax[1].imshow(ori2)
    #plt.savefig('ZCombine_%s.png' %i , dpi =200)
    plt.close()
    
np.save('data.npy', data)
