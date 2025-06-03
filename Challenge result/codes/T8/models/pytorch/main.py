import os
import numpy as np
import json
import torch
from torch import nn
import nibabel as nib
from torch.utils.data import DataLoader
from equivariant_unet_physical_units import UNet
from train import train_one_model
from dataset import INSTANCE_2022, INSTANCE_2022_3channels, INSTANCE_2022_evaluation
from tensorboardX import SummaryWriter 
from torch.utils.data import DataLoader


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`. 

    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        print(f'returning 1: {mask_gt.sum()}, {mask_pred.sum()} ')
        return 1
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def train_equivariant(checkpoint_path, epoch_end,cutoff='right',downsample=3,gpu='cuda',equivariance='SO3',n=3):

    #checkpoint_path = '/home/diaz/experiments/INSTANCE2022'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_name = checkpoint_path

    writer = SummaryWriter(log_name)

    training_cases = 'training_cases.txt'

    dataset = INSTANCE_2022(training_cases, patch_size = 128,check_labels=True) 

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    input_irreps = "1x0e"
    
    model = UNet(2,0,5,5,(0.5,0.5,5),n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    epoch_start = 0 
    min_ce_loss = 10000
    epochs_without_min = 0

    train_one_model(model,dataset,device,log_name,writer,checkpoint_path,epoch_start,epoch_end,optimizer,val_percent= 0.22,dataset_fraction=1,min_ce_loss=min_ce_loss,patience=25,epochs_without_min = epochs_without_min)

def predict_equivariant(checkpoint_dir, gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3):

    n_classes = 2
    #checkpoint_dir = '/home/diaz/experiments/INSTANCE2022'

    sav_dir = f'{checkpoint_dir}/prediction/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    testing_cases = 'testing_cases.txt'

    dataset = INSTANCE_2022(testing_cases, patch_size = 0) 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    input_irreps = "1x0e"
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    model = UNet(2,0,5,5,(0.5,0.5,5),n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)

    checkpoint = torch.load(checkpoint_dir+'/model_min.pt',map_location=gpu)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dc_array = np.zeros((len(dataset),n_classes))
    
    for batch_no, batch in enumerate(test_loader):

        img = batch['image'][0,...]
        label = batch['label']
        output = model.predict_3D(torch.from_numpy(img.cpu().numpy()).float().cuda(),do_mirroring=False, patch_size=(128,128,128),
                                use_sliding_window=True, use_gaussian = True,verbose=False)

        pred_file_name = sav_dir+os.path.basename(batch['name'][0])+f'_pred.nii.gz'
        nib.save(nib.Nifti1Image(output[0],affine = batch['affine'][0].numpy()),pred_file_name)


        dc = []
        for i in range(n_classes):
            mask_gt = label == i
            mask_pred = output[0] == i
            dc.append(compute_dice_coefficient(mask_gt,mask_pred))

        dc_array[batch_no] = dc
    
    np.save(f'{sav_dir}/dice.npy',dc_array)


def e3nn_experiments(downsample,gpu):
    train_equivariant(200,'right',downsample=downsample,gpu=gpu,n=2)
    predict_equivariant(gpu,downsample=downsample,n=2)

#e3nn_experiments(3,'cuda')

def train_equivariant_3channel(checkpoint_path, epoch_end,cutoff='right',downsample=3,gpu='cuda',equivariance='SO3',n=3):

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_name = checkpoint_path

    writer = SummaryWriter(log_name)

    training_cases = 'training_cases.txt'

    dataset = INSTANCE_2022_3channels(training_cases, patch_size = 128,check_labels=True) 

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    input_irreps = "3x0e"
    
    model = UNet(2,0,5,5,(0.5,0.5,5),n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    epoch_start = 0 
    min_ce_loss = 10000
    epochs_without_min = 0

    train_one_model(model,dataset,device,log_name,writer,checkpoint_path,epoch_start,epoch_end,optimizer,val_percent= 0.22,dataset_fraction=1,min_ce_loss=min_ce_loss,patience=25,epochs_without_min = epochs_without_min)

def predict_equivariant_3channel(checkpoint_dir, gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3):

    n_classes = 2
    #checkpoint_dir = '/home/diaz/experiments/INSTANCE2022'

    sav_dir = f'{checkpoint_dir}/prediction/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    testing_cases = 'testing_cases.txt'

    dataset = INSTANCE_2022_3channels(testing_cases, patch_size = 0) 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    input_irreps = "3x0e"
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    model = UNet(2,0,5,5,(0.5,0.5,5),n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)

    checkpoint = torch.load(checkpoint_dir+'/model_min.pt',map_location=gpu)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dc_array = np.zeros((len(dataset),n_classes))
    
    for batch_no, batch in enumerate(test_loader):

        img = batch['image'][0,...]
        label = batch['label']
        output = model.predict_3D(img.cpu().numpy(),do_mirroring=False, patch_size=(128,128,128),
                                use_sliding_window=True, use_gaussian = True,verbose=False)

        pred_file_name = sav_dir+os.path.basename(batch['name'][0])+f'_pred.nii.gz'
        nib.save(nib.Nifti1Image(output[0],affine = batch['affine'][0].numpy()),pred_file_name)


        dc = []
        for i in range(n_classes):
            mask_gt = label == i
            mask_pred = output[0] == i
            dc.append(compute_dice_coefficient(mask_gt,mask_pred))

        dc_array[batch_no] = dc
    
    np.save(f'{sav_dir}/dice.npy',dc_array)

def three_channel_experiments(checkpoint_dir,downsample,gpu):
    train_equivariant_3channel(checkpoint_dir,200,'right',downsample=downsample,gpu=gpu,n=3)
    predict_equivariant_3channel(checkpoint_dir,gpu,downsample=downsample,n=3)

#three_channel_experiments('/home/diaz/experiments/INSTANCE2022_3channel_n3',3,'cuda')

def predict_evaluation(checkpoint_dir, model_name, gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3):

    n_classes = 2

    sav_dir = f'{checkpoint_dir}/submit/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    testing_cases = 'evaluation_cases.txt'

    dataset = INSTANCE_2022_evaluation(testing_cases, nchannels=1,patch_size = 0) 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_dir+'/'+model_name,map_location=gpu)
    input_irreps = "1x0e"
    model = UNet(2,0,5,5,(0.5,0.5,5),n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    for batch_no, batch in enumerate(test_loader):

        img = batch['image'][0,...]
        output = model.predict_3D(img.cpu().numpy(),do_mirroring=False, patch_size=(128,128,128),
                                use_sliding_window=True, use_gaussian = True,verbose=False)

        pred_file_name = sav_dir+os.path.basename(batch['name'][0])
        nib.save(nib.Nifti1Image(output[0],affine = batch['affine'][0].numpy()),pred_file_name)

predict_evaluation('/home/diaz/experiments/INSTANCE2022_n3', 'model_min.pt', gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3)