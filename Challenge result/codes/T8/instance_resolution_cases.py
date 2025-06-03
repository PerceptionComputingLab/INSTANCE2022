import glob
import os
import nibabel as nib
import numpy as np

cases = glob.glob('/str/nas/INSTANCE2022/train_2/data/*.nii.gz')
res = np.zeros((len(cases),3))
for i, case in enumerate(cases):
    print(i)
    image = nib.load(case)
    res[i] = np.array(image.affine.diagonal()[:3])

np.save('instance_resolutions.npy',res)
    
#%%
import numpy as np
resolutions = np.abs(np.load('instance_resolutions.npy') )
rounded  = np.round(433*abs(resolutions))/433 #mario code
print(f'unique resolutions: {len(np.unique(resolutions,axis=0))}') 
print(f'unique rounded resolutions (mario version): {len(np.unique(rounded,axis=0))}') 
print(f'unique rounded resolutions (3 decimals): {len(np.unique(np.round(resolutions,decimals=1),axis=0))}') 


# %%
import nibabel as nib
import glob
import os

os.chdir('/str/nas/INSTANCE2022/train_2/data')
cases = glob.glob('*.nii.gz')
cases.sort()
os.chdir('..')
for ccase in cases:
    img = nib.load('data/'+ccase).get_fdata()
    label = nib.load('label/'+ccase).get_fdata()
    label2 = nib.load('label_2/'+ccase).get_fdata()
    if label2.max() > 1:
        print(ccase)
    #label_values = img[label==1]
    #print(label_values.min(),label_values.max(),label_values.mean(),label_values.shape)


# %%
from models.pytorch.dataset import INSTANCE_2022
import numpy as np

dataset = INSTANCE_2022('models/pytorch/training_cases.txt', patch_size = 0,check_labels=True) 
img,label,name,affine = dataset[0].values()
print(tuple(np.round(np.abs(affine.diagonal()[:3]),decimals=3)))
print(img.shape)

# %%
import torch
from models.pytorch.dataset import INSTANCE_2022, INSTANCE_2022_3channels
from torch.utils.data import DataLoader, random_split
batch_size = 1
training_cases = 'models/pytorch/full_training_cases.txt'
dataset = INSTANCE_2022_3channels(training_cases, patch_size = 128,check_labels=True) 
train, val = random_split(dataset, [100, 0],generator=torch.Generator().manual_seed(42) )
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
# %%
import nibabel as nib
import SimpleITK as sitk
import glob
files = glob.glob('*nii.gz')
files.sort()
for f in files:
    img = nib.load(f).get_fdata()
    print(f'{f}: min: {img.min()}, max:{img.max()}, sum: {img.sum()}')
#%%
import nibabel as nib
import SimpleITK as sitk
import glob
files = glob.glob('*nii.gz')
files.sort()
imagepaths = glob.glob('/str/nas/INSTANCE2022/evaluation/*.nii.gz')
imagepaths.sort()

def out(segmentation, outputpath, imagepath):
    """
    save your predictions
    :param segmentation:Your prediction , the data type is "numpy.ndarray".
    :param outputpath:The save path of prediction results.
    :param imagepath:The path of the image corresponding to the prediction result.
    :return:
    """
    dicom = sitk.ReadImage(imagepath)
    output = sitk.GetImageFromArray(segmentation)
    output.SetOrigin(dicom.GetOrigin())
    output.SetSpacing(dicom.GetSpacing())
    output.SetDirection(dicom.GetDirection())
    sitk.WriteImage(output, outputpath)

for f, i in zip(files,imagepaths):
    img = nib.load(f).get_fdata()
    out(img,'temp/'+f, i)
#%%
import numpy as np
import nibabel as nib
import glob
files = glob.glob('*nii.gz')
files.sort()
imagepaths = glob.glob('/str/nas/INSTANCE2022/evaluation/*.nii.gz')
imagepaths.sort()
for f, i in zip(files,imagepaths):
    img = nib.load(f).get_fdata()
    img  = img.astype(np.uint8)
    orig = nib.load(i)
    nib.save(nib.Nifti1Image(img,affine = orig.affine, header=orig.header),f)


