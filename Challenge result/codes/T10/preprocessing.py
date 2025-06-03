import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import *

def remove_coil(img):
    # Remove coil
    # Binarize, keep biggest CC as mask, then apply
    img_bin = img > 0 # background value is 0

    img_labelled, nlabels = ndimage.label(img_bin)
    label_list = np.arange(1, nlabels + 1) # and 1
    label_volumes = ndimage.labeled_comprehension(img_bin, img_labelled, label_list, np.sum, float, 0)

    biggest_label = {'idx': None, 'volume': 0}
    for n, label_volume in enumerate(label_volumes):
        if label_volume > biggest_label['volume']:
            biggest_label = {'idx': n + 1, 'volume': label_volume}

    img_nocoil = img * (img_labelled == biggest_label['idx'])
    return img_nocoil


def remove_skull(img, max_int = 100.0):
     
    # Binarize and remove skull borders
    img_bin = np.logical_and(img < 100, img > 0)
    img_bin = ndimage.morphology.binary_opening(img_bin, iterations=1)
    z = 0
    vol_init = 0
    # Keep biggest CC
    img_labelled, nlabels = ndimage.label(img_bin)
    label_list = np.arange(1, nlabels + 1) # and 1
    label_volumes = ndimage.labeled_comprehension(img_bin, img_labelled, label_list, np.sum, float, 0)

    biggest_label = {'idx': None, 'volume': 0}
    for n, label_volume in enumerate(label_volumes):
        if label_volume > biggest_label['volume']:
            biggest_label = {'idx': n + 1, 'volume': label_volume}
    img_bin[img_labelled == biggest_label['idx']] = 1
    img_bin[img_labelled != biggest_label['idx']] = 0
    for i in range(img.shape[2]):
        vol = np.count_nonzero(img_bin[:,:,i])
        if vol > vol_init:
          z = i
          vol_init = vol
    for i in range(z-1,img_bin.shape[2]):
        # Keep biggest CC
        img_labelled, nlabels = ndimage.label(img_bin[:,:,i])
        
        if nlabels == 0:
            break
        label_list = np.arange(1, nlabels + 1) # and 1
        label_volumes = ndimage.labeled_comprehension(img_bin[:,:,i], img_labelled, label_list, np.sum, float, 0)
    
        biggest_label = {'idx': None, 'volume': 0}
        for n, label_volume in enumerate(label_volumes):
            if label_volume > biggest_label['volume']:
                biggest_label = {'idx': n + 1, 'volume': label_volume}
        img_bin[:,:,i][img_labelled == biggest_label['idx']] = 1
        img_bin[:,:,i][img_labelled != biggest_label['idx']] = 0
    for i in range(z+1,-1,-1):
        img_masked=img_bin[:,:,i+1]*img_bin[:,:,i]
        img_bin[:,:,i] = img_masked
        
    img_noskull = img*img_bin
    return img_noskull
    
