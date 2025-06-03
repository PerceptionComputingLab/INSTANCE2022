import os
import tqdm
from glob import glob
import nibabel as nib
import numpy as np
import itertools
import torch
import random
from torch.utils.data import Dataset
from batchgenerators.augmentations.utils import pad_nd_image
import scipy.ndimage as ndi
import logging

class INSTANCE_2022(Dataset):
    def __init__(self, cases_file = None, patch_size = 128,check_labels=False):

        if os.uname()[1] == 'scanwkp11':
            self.base_dir = '/str/nas/INSTANCE2022/'
        elif os.uname()[1] == 'mb-neuro-03':
            self.base_dir = '/data/datasets/INSTANCE2022/'
        else:
            self.base_dir = '/home/diaz/data/'

        if cases_file is not None:
            with open(cases_file) as f:
                self.ids = [line.rstrip('\n') for line in f]
        else:
            #Load entire dataset
            self.ids = os.listdir(os.path.join(self.base_dir,'rawdata'))

        self.patch_size = patch_size
        self.check_labels = check_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]


        img_file = glob(os.path.join(self.base_dir, f'train_2/data/{idx}'))
        label_file = glob(os.path.join(self.base_dir, f'train_2/label/{idx}'))

        assert len(label_file) == 1, \
            f'Either no label or multiple labels found for the ID {idx}: {label_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = nib.load(img_file[0])
        label = nib.load(label_file[0])
        img_affine = img.affine
        res = img.header.get_zooms()

        assert (img.affine.round() == label.affine.round()).prod(), \
            f'Image and label {idx} affines should be the same'

        assert img.shape == label.shape, \
            f'Image and label {idx} should be the same shape, but are {img.shape} and {label.shape}'
        
        img = img.get_fdata()
        label =label.get_fdata()

        #clamp and scale
        img[img<0] = 0
        img[img>80] = 80
        img = img / 80

        #get a random patch inside the volume of size (patch_size,patch_size,patch_size)  
        if self.patch_size > 0:
            img, label  = self.random_patch(img,label,self.patch_size,check_labels=self.check_labels)

        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img).type(torch.float32)
        label = torch.from_numpy(label).long()

        return {
            'image': img,
            'label': label,
            'name' : idx,
            'affine': img_affine,
            'res': res
        }

    def random_patch(self,input_array,input_label,patch_size,check_labels=False):
        x,y,z = input_array.shape

        patchFound = False
        tries = 0
        while not patchFound and tries < 50:
            min_x = min_y = min_z = 0 
            max_x, max_y, max_z = x,y,z
            
            if x > patch_size:
                min_x = np.random.randint(x - patch_size+1)
                max_x = min_x+patch_size
            if y > patch_size:
                min_y = np.random.randint(y - patch_size+1)
                max_y = min_y+patch_size
            if z > patch_size:
                min_z = np.random.randint(z - patch_size+1)
                max_z = min_z+patch_size

            output_array = pad_nd_image(input_array[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))
            output_label = pad_nd_image(input_label[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))

            if check_labels == False:
                patchFound = True
            else:
                if (output_label == 1).sum() == 0:
                    #print('missing label. applying new random patch')
                    tries += 1
                else:
                    patchFound = True
        
        return output_array, output_label

class INSTANCE_2022_3channels(Dataset):
    def __init__(self, cases_file = None, patch_size = 128,check_labels=False):

        if os.uname()[1] == 'scanwkp11':
            self.base_dir = '/str/nas/INSTANCE2022/'
        elif os.uname()[1] == 'mb-neuro-03':
            self.base_dir = '/data/datasets/INSTANCE2022/'
        else:
            self.base_dir = '/home/diaz/data/'

        if cases_file is not None:
            with open(cases_file) as f:
                self.ids = [line.rstrip('\n') for line in f]
        else:
            #Load entire dataset
            self.ids = os.listdir(os.path.join(self.base_dir,'rawdata'))

        self.patch_size = patch_size
        self.check_labels = check_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]


        img_file = glob(os.path.join(self.base_dir, f'train_2/data/{idx}'))
        label_file = glob(os.path.join(self.base_dir, f'train_2/label/{idx}'))

        assert len(label_file) == 1, \
            f'Either no label or multiple labels found for the ID {idx}: {label_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = nib.load(img_file[0])
        label = nib.load(label_file[0])
        img_affine = img.affine
        res = img.header.get_zooms()

        assert (img.affine.round() == label.affine.round()).prod(), \
            f'Image and label {idx} affines should be the same'

        assert img.shape == label.shape, \
            f'Image and label {idx} should be the same shape, but are {img.shape} and {label.shape}'
        
        img = img.get_fdata()
        label =label.get_fdata()



        #get a random patch inside the volume of size (patch_size,patch_size,patch_size)  
        if self.patch_size > 0:
            img, label  = self.random_patch(img,label,self.patch_size,check_labels=self.check_labels)

        channels = np.zeros((3,*img.shape))
        channels[0,...] = img
        channels[1,...] = img
        channels[2,...] = img

        #clamp and scale channel 0
        channels[0,...][img<0] = 0
        channels[0,...][img>80] = 80
        channels[0,...] = channels[0,...] / 80

        #clamp and scale channel 1
        channels[1,...][img<-50] = -50
        channels[1,...][img>220] = 220
        channels[1,...]+=50
        channels[1,...] = channels[1,...] / 270

        #clamp and scale channel 2
        channels[2,...][img<0] = 0
        channels[2,...][img>1000] = 1000
        channels[2,...] = channels[2,...] / 1000

        channels = torch.from_numpy(channels).type(torch.float32)
        label = torch.from_numpy(label).long()

        return {
            'image': channels,
            'label': label,
            'name' : idx,
            'affine': img_affine,
            'res': res
        }

    def random_patch(self,input_array,input_label,patch_size,check_labels=False):
        x,y,z = input_array.shape

        patchFound = False
        tries = 0
        while not patchFound and tries < 50:
            min_x = min_y = min_z = 0 
            max_x, max_y, max_z = x,y,z
            
            if x > patch_size:
                min_x = np.random.randint(x - patch_size+1)
                max_x = min_x+patch_size
            if y > patch_size:
                min_y = np.random.randint(y - patch_size+1)
                max_y = min_y+patch_size
            if z > patch_size:
                min_z = np.random.randint(z - patch_size+1)
                max_z = min_z+patch_size

            output_array = pad_nd_image(input_array[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))
            output_label = pad_nd_image(input_label[min_x:max_x,min_y:max_y,min_z:max_z],(patch_size,patch_size,patch_size))

            if check_labels == False:
                patchFound = True
            else:
                if (output_label == 1).sum() == 0:
                    #print('missing label. applying new random patch')
                    tries += 1
                else:
                    patchFound = True
        
        return output_array, output_label

class INSTANCE_2022_evaluation(Dataset):
    def __init__(self, cases_file = None, nchannels = 1, patch_size = 0,check_labels=False):

        if os.uname()[1] == 'scanwkp11':
            self.base_dir = '/str/nas/INSTANCE2022/'
        elif os.uname()[1] == 'mb-neuro-03':
            self.base_dir = '/data/datasets/INSTANCE2022/'
        else:
            self.base_dir = '/home/diaz/data/'

        if cases_file is not None:
            with open(cases_file) as f:
                self.ids = [line.rstrip('\n') for line in f]
        else:
            #Load entire dataset
            self.ids = os.listdir(os.path.join(self.base_dir,'rawdata'))

        self.patch_size = patch_size
        self.check_labels = check_labels
        self.nchannels = nchannels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]


        img_file = glob(os.path.join(self.base_dir, f'evaluation/{idx}'))

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = nib.load(img_file[0])
        img_affine = img.affine
        img_header = img.header
        res = img.header.get_zooms()

        img = img.get_fdata()

        if self.nchannels == 1:
            #clamp and scale
            img[img<0] = 0
            img[img>80] = 80
            img = img / 80
            channels = np.expand_dims(img, 0)

        elif self.nchannels == 3:
            channels = np.zeros((3,*img.shape))
            channels[0,...] = img
            channels[1,...] = img
            channels[2,...] = img

            #clamp and scale channel 0
            channels[0,...][img<0] = 0
            channels[0,...][img>80] = 80
            channels[0,...] = channels[0,...] / 80

            #clamp and scale channel 1
            channels[1,...][img<-50] = -50
            channels[1,...][img>220] = 220
            channels[1,...]+=50
            channels[1,...] = channels[1,...] / 270

            #clamp and scale channel 2
            channels[2,...][img<0] = 0
            channels[2,...][img>1000] = 1000
            channels[2,...] = channels[2,...] / 1000

        channels = torch.from_numpy(channels).type(torch.float32)

        return {
            'image': channels,
            'name' : idx,
            'affine': img_affine,
            'res': res,
            'header': img_header
        }
