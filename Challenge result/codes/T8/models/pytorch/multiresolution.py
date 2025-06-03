from glob import glob
import os
from re import I
import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader, random_split
from equivariant_unet_physical_units import UNet
from train import train_one_model
from dataset import INSTANCE_2022_3channels, INSTANCE_2022_evaluation, INSTANCE_2022
from tensorboardX import SummaryWriter 
from torch.utils.data import DataLoader
import logging
from loss_functions import SoftDiceLoss
import copy



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



def train_val_multiresolution(checkpoint_path, epoch_end,cutoff='right',downsample=3,gpu='cuda',equivariance='SO3',n=3, save_only_min = True, LOAD_CHECK_POINT = False,channels = 1):


    epoch_start = 0
    min_ce_loss = 10000
    epochs_without_min = 0
    min_loss = False
    first_model = True
    prev_model = None
    prev_optimizer_state = None

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    if LOAD_CHECK_POINT:
        first_model = False
        if os.path.exists(checkpoint_path+'/model_min.pt'):
            last_checkpoint = checkpoint_path+'/model_min.pt'
        else:
            checkpoints = glob(checkpoint_path+'/model*.pt')
            checkpoints.sort()
            last_checkpoint = checkpoints[-1]
            
        checkpoint = torch.load(last_checkpoint,map_location=gpu)
        resolution = checkpoint['resolution']

        input_irreps = f"{channels}x0e"
        prev_model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
        optimizer = torch.optim.Adam(prev_model.parameters(), lr=5e-3)

        prev_model.load_state_dict(checkpoint['model_state_dict'])
        prev_optimizer_state = checkpoint['optimizer_state_dict']
        min_ce_loss = checkpoint['loss']

        try:
            with open(checkpoint_path+'/training_progress.json','r') as f:
                d = json.loads(f.read())
                epoch_start = d['epoch'] + 1
                epochs_without_min = d['epochs_without_min']
        except:
            epoch_start = checkpoint['epoch']+1


    batch_size = 1
    patience = 100

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_name = os.path.join(checkpoint_path,'train.log')

    writer = SummaryWriter(checkpoint_path)

    training_cases = 'full_training_cases.txt'

    if channels == 1:
        dataset = INSTANCE_2022(training_cases, patch_size = 128,check_labels=True) 
    elif channels == 3:
        dataset = INSTANCE_2022_3channels(training_cases, patch_size = 128,check_labels=True) 




    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO)

    train, val = random_split(dataset, [100, 0],generator=torch.Generator().manual_seed(42) )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()


    for epoch in range(epoch_start,epoch_end):

        for batch_no, batch in enumerate(train_loader):

            imgs = batch['image']
            labels = batch['label']
            resolution = batch['res']
            resolution = tuple([s.numpy()[0] for s in resolution])
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            #define model inside training loop
            input_irreps = f"{channels}x0e"
            model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

            #load previous model's parameters
            if not first_model:
                for p1, p2 in zip(prev_model.parameters(), model.parameters()):
                    with torch.no_grad():
                        p2[:] = p1
                optimizer.load_state_dict(prev_optimizer_state)
 
            else:
                first_model = False

            model.train()

            out = model(imgs)
            loss = criterion(out, labels)

            writer.add_scalar('Cross_Entropy_Loss',loss,epoch*100+batch_no)
            correct = (out.argmax(1) == labels)
            acc = [correct[labels == i].double().mean().item() for i in range(model.num_classes)]

            logging.info((
                f"{epoch}:{batch_no} loss={loss.item():.3f} p(pred=i | true=i)={acc}"
            ))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.1)
            optimizer.step()

            prev_optimizer_state = optimizer.state_dict() 
            #prev_model_params = model.parameters()
            prev_model = copy.deepcopy(model)


        #validation
        mask_type = torch.long
        n_val = len(val_loader)  # the number of batch
        tot = 0
        total_acc = np.zeros(model.num_classes)
        total_softdiceloss = np.zeros(model.num_classes)
        criterion = nn.CrossEntropyLoss()

        for batch_no, batch in enumerate(val_loader):
            logging.info(( f"{batch_no}"))


            imgs, true_masks = batch['image'], batch['label']
            affine = batch['affine']
            resolution = batch['res']
            resolution = tuple([s.numpy()[0] for s in resolution])
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            input_irreps = "3x0e"
            model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)

            for p1, p2 in zip(prev_model.parameters(), model.parameters()):
                with torch.no_grad():
                    p2[:] = p1
            model.eval()


            with torch.no_grad():
                output = model(imgs)
                
            pred = output.softmax(1)
            correct = (output.argmax(1) == true_masks)
            acc = [correct[true_masks == i].double().mean().item() for i in range(model.num_classes)]
            total_acc+=acc

            logging.info(( f" p(pred=i | true=i)={acc}"))
            #logging.info(( f" p(pred=i | true=i)={100*acc[0]:.0f}%, {100*acc[1]:.0f}% and {100*acc[2]:.0f}%"))

            one_hot_true_masks = F.one_hot(true_masks,model.num_classes)

            for label in range(model.num_classes):
                try:
                    total_softdiceloss[label] += SoftDiceLoss()(pred[:,label,:,:,:].unsqueeze(0),one_hot_true_masks[...,label].unsqueeze(0)).item()
                except:
                    total_softdiceloss[label] += SoftDiceLoss()(pred[:,label,:,:].unsqueeze(0),one_hot_true_masks[...,label].unsqueeze(0)).item()
            tot += criterion(output, true_masks)



        if n_val > 0:
            total_acc /= n_val
            total_softdiceloss /= n_val
    

            logging.info((
                f"validation: {epoch}: ce loss={tot/n_val:.3f} p(pred=i | true=i)={total_acc}"
            ))
            logging.info((
                f"validation: {epoch}: dice loss={total_softdiceloss[0]:.3f}, {total_softdiceloss}"
            ))


            ce_loss = tot/n_val 

            writer.add_scalar('Cross_Entropy_Loss',ce_loss,epoch)
            for c in range(model.num_classes):
                writer.add_scalar(f'Dice_Loss/{c}',total_softdiceloss[c],epoch)
        
            if ce_loss < min_ce_loss:
                min_loss = True
                min_ce_loss = ce_loss
                epochs_without_min = 0
        
            else: 
                epochs_without_min += 1

            if epochs_without_min > patience:
                with open(checkpoint_path+'/training_progress.json','w') as f:
                    d = {'epoch': epoch, 'epochs_without_min': epochs_without_min, 'done': True, 'ce_loss':ce_loss.item(),'min_ce_loss':min_ce_loss}
                    f.write(json.dumps(d)) 
                break

        else:
            ce_loss = loss


        if not save_only_min:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ce_loss,
                        'resolution': resolution,
                        }, f'{checkpoint_path}/model_{epoch:03}.pt')
        
        elif save_only_min and min_loss:
            print('min loss:', epoch)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ce_loss,
                        'resolution': resolution,
                        }, f'{checkpoint_path}/model_min.pt')
        
        min_loss = False

        with open(checkpoint_path+'/training_progress.json','w') as f:
            d = {'epoch': epoch, 'epochs_without_min': epochs_without_min, 'done': False, 'ce_loss':ce_loss.item(),'min_ce_loss':min_ce_loss.item()}
            f.write(json.dumps(d)) 

def predict_multiresolution(checkpoint_dir, gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3):

    n_classes = 2

    sav_dir = f'{checkpoint_dir}/prediction/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    testing_cases = 'testing_cases.txt'

    dataset = INSTANCE_2022_3channels(testing_cases, patch_size = 0) 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_dir+'/model_min.pt',map_location=gpu)
    resolution=checkpoint['resolution']
    input_irreps = "3x0e"
    model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    prev_model = copy.deepcopy(model)


    dc_array = np.zeros((len(dataset),n_classes))
    
    for batch_no, batch in enumerate(test_loader):

        img = batch['image'][0,...]
        label = batch['label']
        affine = batch['affine']
        resolution = batch['res']
        resolution = tuple([s.numpy()[0] for s in resolution])

        model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)

        for p1, p2 in zip(prev_model.parameters(), model.parameters()):
            with torch.no_grad():
                p2[:] = p1

        model.eval()

        output = model.predict_3D(img.cpu().numpy(),do_mirroring=False, patch_size=(128,128,128),
                                use_sliding_window=True, use_gaussian = True,verbose=False)

        pred_file_name = sav_dir+os.path.basename(batch['name'][0]).split('.')[0]+f'_pred.nii.gz'
        nib.save(nib.Nifti1Image(output[0],affine = batch['affine'][0].numpy()),pred_file_name)


        #four_classes
        four_classes = np.zeros(output[0].shape)
        four_classes[(output[0] == 1) * (label[0].numpy() == 0)] = 1 #fp
        four_classes[(output[0] == 0) * (label[0].numpy() == 1)] = 2 #fn
        four_classes[(output[0] == 1) * (label[0].numpy() == 1)] = 3 #tp

        confusion_file_name = sav_dir+os.path.basename(batch['name'][0]).split('.')[0]+f'_confusion.nii.gz'
        nib.save(nib.Nifti1Image(four_classes,affine = batch['affine'][0].numpy()),confusion_file_name)

        dc = []
        for i in range(n_classes):
            mask_gt = label == i
            mask_pred = output[0] == i
            dc.append(compute_dice_coefficient(mask_gt,mask_pred))

        dc_array[batch_no] = dc

    np.save(f'{sav_dir}/dice.npy',dc_array)

def predict_evaluation(checkpoint_dir, model_name, gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3,nchannels=3):

    n_classes = 2

    sav_dir = f'{checkpoint_dir}/submit/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    testing_cases = 'evaluation_cases.txt'

    dataset = INSTANCE_2022_evaluation(testing_cases, nchannels=nchannels,patch_size = 0) 
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_dir+'/'+model_name,map_location=gpu)
    resolution=checkpoint['resolution']
    input_irreps = f"{nchannels}x0e"
    model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    prev_model = copy.deepcopy(model)
    
    for batch_no, batch in enumerate(test_loader):

        img = batch['image'][0,...]
        affine = batch['affine']
        resolution = batch['res']
        resolution = tuple([s.numpy()[0] for s in resolution])

        model = UNet(2,0,5,5,resolution,n=n,n_downsample = downsample,equivariance=equivariance,input_irreps=input_irreps,cutoff=cutoff).to(device)

        for p1, p2 in zip(prev_model.parameters(), model.parameters()):
            with torch.no_grad():
                p2[:] = p1

        model.eval()

        output = model.predict_3D(img.cpu().numpy(),do_mirroring=False, patch_size=(128,128,128),
                                use_sliding_window=True, use_gaussian = True,verbose=False)
        output  = output.astype(np.uint8)
        pred_file_name = sav_dir+os.path.basename(batch['name'][0])
        nib.save(nib.Nifti1Image(output[0],affine = batch['affine'][0].numpy(),header=batch['header']),pred_file_name)


def multiresolution_experiments(checkpoint_dir,downsample,gpu):
    #train_val_multiresolution(checkpoint_dir,500, n=3,LOAD_CHECK_POINT=True,channels=3)
    predict_multiresolution(checkpoint_dir,n=3)

def multiresolution_full_training(checkpoint_dir,downsample,gpu):
    train_val_multiresolution(checkpoint_dir,500, n=3,gpu=gpu,save_only_min=False,LOAD_CHECK_POINT=True,channels=3)
    #predict_multiresolution(checkpoint_dir,n=3)


def multiresolution_full_training_one_channel(checkpoint_dir,downsample,gpu):
    train_val_multiresolution(checkpoint_dir,500, n=3,save_only_min=False,LOAD_CHECK_POINT=True, channels=1)

#multiresolution_full_training('/home/diaz/experiments/INSTANCE2022_multiresolution_no_validation/',3,'cuda')
#predict_evaluation('/home/diaz/experiments/INSTANCE2022_multiresolution_no_validation/', 'model_058.pt', gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3)
#multiresolution_experiments('/home/diaz/experiments/INSTANCE2022_multiresolution_full/',3,'cuda')
#multiresolution_full_training('/home/diaz/experiments/INSTANCE2022_mres_full_3channel/',3,'cuda:1')
#multiresolution_full_training_one_channel('/home/diaz/experiments/INSTANCE2022_mres_full_1channel/',3,'cuda')

predict_evaluation('/home/diaz/experiments/INSTANCE2022_mres_full_1channel/', 'model_064.pt', gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3,nchannels=1)
predict_evaluation('/home/diaz/experiments/INSTANCE2022_mres_full_3channel/', 'model_053.pt', gpu='cuda', downsample = 3, cutoff='right',equivariance='SO3',n=3)