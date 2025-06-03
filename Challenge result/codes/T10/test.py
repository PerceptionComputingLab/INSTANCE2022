#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import nibabel as nib
import niclib as nl
import torch
from scipy import ndimage

from preprocessing import *
from model_deeper import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
                            help='this directory contains all test samples')
parser.add_argument('--predict_dir', default='', type=str, metavar='PATH',
                            help='segmentation file of each test sample should be stored in this directory')
                            
args = parser.parse_args()

input_dir = args.input_dir
predict_dir = args.predict_dir


checkpoints_path = 'models/'

for image_file in os.listdir(input_dir):
    
    case_id = image_file.replace('.nii.gz','')
    
    print(case_id)
    
    image = nib.load(os.path.join(input_dir,image_file))
    image_data = image.get_fdata()
    
    img_nocoil = remove_coil(image_data)
    img_noskull = remove_skull(img_nocoil)
    
    norm_image = img_noskull
        
    # Compute normalization parameters
    norm_mask = norm_image > 0
    norm_image[norm_mask == 0] = np.nan
    norm_params = np.nanpercentile(norm_image, [0.5, 99.5], axis=(-3, -2, -1), keepdims=True)
    
    # Apply normalization
    new_low, new_high = norm_params    
    img_normalized = niclib.data.adjust_range(img_noskull, new_range=[0.0, 100.0], old_range=[new_low, new_high])
    img_normalized = np.clip(img_normalized, 0.0, 100.0)
    
    normalized = nib.Nifti1Image(img_normalized, image.affine, image.header)
    
    nib.nifti1.save(normalized, os.path.join(predict_dir, case_id+'.nii.gz'))
    
    img_normalized = nib.load(os.path.join(predict_dir, case_id+'.nii.gz')).get_fdata()
    
    probs_folds = []
    
    masks_folds = []
    
    for fold in range(0,5):
        
        model_trained = torch.load(checkpoints_path + 'core_seg_fold{}.pt'.format(fold))
        predictor = nl.net.test.PatchTester(
             patch_shape=(1, 48, 48, 16),
             patch_out_shape=(2, 48, 48, 16),
             extraction_step=(8, 8, 2),
             normalize='image',
             activation=torch.nn.Softmax(dim=1))
             
        tissue_probabilities = predictor.predict(model_trained, np.stack([img_normalized]))
        
        tissue_mask = np.argmax(tissue_probabilities, axis=0)
        
        probs_folds.append(tissue_probabilities)
        masks_folds.append(tissue_mask)
    
    # averaging probabilities 
    result_probs = np.average(np.array(probs_folds), axis=0)
    image_labeled = np.argmax(result_probs, axis=0)
    
    if np.amax(image_labeled) != 0.0:
           
        img_labelled, nlabels = ndimage.label(image_labeled)
        label_list = np.arange(1, nlabels+1)
        label_volumes = ndimage.labeled_comprehension(image_labeled, img_labelled, label_list, np.sum, float, 0)    
        max_vol = np.amax(label_volumes)
        image_pp = image_labeled
        
        for n, label_volume in enumerate(label_volumes):
                    
            if label_volume < 0.1*max_vol:
                image_pp[img_labelled == n+1] = 0.0
                
        image_pp = image_pp.astype(int)
    else:
        image_pp = image_labeled

    
    nl.save_nifti(
            filepath=os.path.join(predict_dir,case_id + '.nii.gz'),
            volume=image_pp, 
            reference=image,
            channel_handling='split')
    
