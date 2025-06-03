# %%
from glob import glob
import nibabel as nb
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, Adam
from matplotlib import pyplot as plt
from numpy import std, mean

import monai
from monai.data import ArrayDataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.transforms import AddChannel, RandRotate90, RandSpatialCrop, EnsureType
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.inferers import sliding_window_inference

from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from natsort import natsorted
import torchio as tio
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
options = {}
options['modality'] = 'CT'
batch_size = 8
options['exp_location'] = 'SCAN'
options['max_queue_length'], options['patches_per_volume'] = 16, 4
options['b_size'] = batch_size

if options['exp_location'] == 'SCAN':
    INSTANCE22_DATA_ROOT = '/str/nas/INSTANCE2022/train_2/'
if options['exp_location'] == 'LAPTOP':
    INSTANCE22_DATA_ROOT = '/Users/sebastianotalora/work/instance22/train_2/'

# %%
nifti_vols_paths = natsorted(glob(INSTANCE22_DATA_ROOT+'data/*.nii.gz'))
nifti_labl_paths = natsorted(glob(INSTANCE22_DATA_ROOT+'label/*.nii.gz'))

# %%
test_indices = np.arange(0.9*len(nifti_vols_paths),len(nifti_vols_paths),dtype=int)
test_paths   = [nifti_vols_paths[i] for i in test_indices]
test_labels  = [nifti_labl_paths[i] for i in test_indices]

# %%
indices_folds_train = [list(np.arange(0,72,dtype=int)),
                       list(np.arange(0,54,dtype=int)) + list(np.arange(72,90,dtype=int)),
                       list(np.arange(0,36,dtype=int)) + list(np.arange(54,90,dtype=int)),
                       list(np.arange(0,18,dtype=int)) + list(np.arange(36,90,dtype=int)),
                       list(np.arange(18,90,dtype=int))]

indices_folds_valid = [np.arange(72,90,dtype=int),np.arange(54,72,dtype=int),np.arange(36,54,dtype=int), np.arange(18,36,dtype=int),np.arange(0,18,dtype=int)]
indices_folds = []

for i in range(5):
    fold_train_indices_volumes = [nifti_vols_paths[k] for k in indices_folds_train[i]]
    fold_train_indices_labels  = [nifti_labl_paths[k] for k in indices_folds_train[i]]

    fold_val_indices_volumes   = [nifti_vols_paths[k] for k in indices_folds_valid[i]]
    fold_val_indices_labels    = [nifti_labl_paths[k] for k in indices_folds_valid[i]]

    indices_folds.append((fold_train_indices_volumes,fold_train_indices_labels, fold_val_indices_volumes,fold_val_indices_labels))


# %%
class UNet_custom(monai.networks.nets.BasicUNet):
    def __init__(self, spatial_dims, in_channels, out_channels,
                 name, scaff=False, fed_rod=False):
        #call parent constructor
        super(UNet_custom, self).__init__(spatial_dims=spatial_dims,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          upsample="nontrainable")#, 
        self.name = name


# %%
crop_size = (512,512)
max_intensity = 1500

# %%
cur_fold_train_indices_volumes,cur_fold_train_indices_labels = indices_folds[0][0],indices_folds[0][1]
cur_fold_val_indices_volumes,cur_fold_val_indices_labels = indices_folds[0][2], indices_folds[0][3]

# %%
subjects_list_train = []
for i in range(len(cur_fold_train_indices_volumes)):
    subjects_list_train.append(tio.Subject(
    ct=tio.ScalarImage(cur_fold_train_indices_volumes[i]),
    label=tio.LabelMap(cur_fold_train_indices_labels[i]),
))
    
subjects_list_val = []
for i in range(len(cur_fold_val_indices_volumes)):
    subjects_list_val.append(tio.Subject(
    ct=tio.ScalarImage(cur_fold_val_indices_volumes[i]),
    label=tio.LabelMap(cur_fold_val_indices_labels[i]),
))

subjects_list_test = []
for i in range(len(test_paths)):
    subjects_list_val.append(tio.Subject(
    ct=tio.ScalarImage(test_paths[i]),
    label=tio.LabelMap(test_labels[i]),
))


# %%

HOUNSFIELD_BRAIN_MIN, HOUNSFIELD_BRAIN_MAX = 0, 80
clamp = tio.Clamp(out_min=HOUNSFIELD_BRAIN_MIN, out_max=HOUNSFIELD_BRAIN_MAX)

rescale = tio.RescaleIntensity(out_min_max=(0, 1))

spatial = tio.OneOf({
        tio.RandomAffine(): 0.6,
        tio.RandomElasticDeformation(): 0.2,        
        tio.RandomAffine(degrees=180): 0.2
    },
    p=0.2,
)

rand_affine = tio.RandomAffine(
    scales=(0.9, 1.2),
    degrees=30,
)

patch_size = (500,500,1)


sampler = tio.data.WeightedSampler(patch_size, 'label')
toCanon = tio.ToCanonical()
resampler_dwi = tio.Resample('ct')
flip = tio.RandomFlip(axes=('LR',))

transforms = [clamp, flip, toCanon]
transform = tio.Compose(transforms)


labels_probabilities = {0: 0.3, 1: 0.7}
sampler_weighted_probs = tio.data.LabelSampler(
    patch_size=patch_size,
    label_name='label',
    label_probabilities=labels_probabilities,
)

# %%
training_dataset = tio.SubjectsDataset(subjects_list_train, transform=transform)
valid_dataset = tio.SubjectsDataset(subjects_list_val, transform=transform)

# %%

queue = tio.Queue(training_dataset, options['max_queue_length'], options['patches_per_volume'], sampler_weighted_probs)
training_loader = torch.utils.data.DataLoader(queue, batch_size=options['b_size'])

# %%
batcho = tio.utils.get_first_item(training_loader)
batcho.keys()
print(batcho['ct']['data'].max())

# %%
model = UNet_custom(spatial_dims=2, in_channels=1, out_channels=1, name='unimodal').to(device)

# %%
local_epochs= 300
local_lr = 0.00092
options['suffix'] = 'inst22_sigmoid_flip_'+'lr_'+str(local_lr)
options['lr'] = local_lr
options['writer'] = SummaryWriter(f"runs/_{options['suffix']}_epochs_{local_epochs}")

def train(ann, training_loader, local_epochs, local_lr, batch_size, val_img_paths,val_lbl_paths,options):
    #train client to train mode
    #Creating unimodal dataloader
    ann.train()
    best_metric, best_hausdorff, best_precision, best_metric_epoch = -1, 10000,-1,-1
    metric_values = list()
    metric_hausdorff = list()

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = Adam(ann.parameters(), local_lr)

    for epoch in range(local_epochs):
        for batch_data in training_loader:
            inputs, labels = batch_data['ct']['data'][:,:,:,:,0].float().to(device),batch_data['label']['data'][:,:,:,:,0].float().to(device)
            y_pred = ann(inputs)
            loss = loss_function(y_pred, labels)
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()
        options['writer'].add_scalar("train_loss", loss.item(), epoch)
        print("Epoch", epoch, " loss :", loss.item())
        if epoch%2 == 0:
            best_metric, best_hausdorff, best_precision, best_metric_epoch = validation_cycle(ann, val_img_paths, 
                                                                                              val_lbl_paths, epoch, best_metric, best_hausdorff,
                                                                                              best_precision, best_metric_epoch, options)
    return ann, loss.item()

def validation_cycle(model, valid_paths, val_lbl_paths, cur_epoch, best_metric, best_hausdorff, best_precision, best_metric_epoch, options):
    metric = 0
    model.eval()
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    metric_values = list()
    test_dicemetric, test_hausdorffmetric= [],[]
    metric_choosen = 'precision'
    dice_metric = DiceMetric(reduction="mean", get_not_nans=False)
    dice_metric.reset()
    max_intensity = 0
    HOUNSFIELD_BRAIN_MIN, HOUNSFIELD_BRAIN_MAX = 0, 80
    clamp = tio.Clamp(out_min=HOUNSFIELD_BRAIN_MIN, out_max=HOUNSFIELD_BRAIN_MAX)
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    
    volumes_recall,volumes_precision = [],[]
    for path_case, path_label in zip(valid_paths,val_lbl_paths):            
        test_vol = nb.load(path_case)
        test_lbl = nb.load(path_label)
        test_vol_pxls = test_vol.get_fdata()
        test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
        test_vol_pxls = clamp(test_vol_pxls[np.newaxis])

        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)

        dices_volume =[]
        hausdorffs_volume = []
        with torch.no_grad():
            flat_labels, flat_predictions = [],[]
            for slice_selected in range(test_vol_pxls.shape[-1]):
                test_tensor = torch.tensor(test_vol_pxls[np.newaxis, :,:,:,slice_selected], dtype=torch.float).to(device)
                #out_test = model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
                out_test = post_trans(model(test_tensor))
                out_test = out_test.detach().cpu().numpy()
                pred = np.array(out_test[0,0,:,:], dtype='uint8')
                dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))                
                #print("pred_sum:", torch.tensor(pred[np.newaxis,np.newaxis,:,:]).flatten().sum())
                #print("labl_sum:",torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]).flatten().sum())
                #hausdorff_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
                flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
                flat_pred = pred.flatten()
                flat_labels.append(flat_label)
                flat_predictions.append(flat_pred)
            test_dicemetric.append(dice_metric.aggregate().item())
            volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
            #if not torch.isinf(hausdorff_metric.aggregate()[0]) and not torch.isnan(hausdorff_metric.aggregate()[0]):
            #    test_hausdorffmetric.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        #hausdorff_metric.reset()
    metric = np.mean(test_dicemetric)
    hausdorff= 100000000#np.mean(test_hausdorffmetric)
    #metric_values.append(metric)
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            cur_epoch +1, metric, best_metric, best_metric_epoch
        )
    )
    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)

    if hausdorff < best_hausdorff:
        best_hausdorff = hausdorff
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_hausdorff_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Hausdorff: {:.4f} best mean Hausdorff: {:.4f} at epoch {}".format(
            cur_epoch +1, hausdorff, best_hausdorff, best_metric_epoch
        )
    )        
    avg_precision = mean(volumes_precision)
    if avg_precision > best_precision:
        best_precision = avg_precision
        best_metric_epoch = cur_epoch
        torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_precision_model_segmentation2d_array.pth')
        print("saved new best metric model")
    print(
        "current epoch: {} current mean Precision: {:.4f} best mean Precision: {:.4f} at epoch {}".format(
            cur_epoch +1, avg_precision, best_precision, best_metric_epoch
        )
    )

    options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)
    options['writer'].add_scalar("val_mean_hausdorff", hausdorff, cur_epoch)
    #writer.add_scalar("val_"+metric_choosen,metric, cur_epoch) TODO: Parametrize choosen metric
    return best_metric, best_hausdorff, best_precision, best_metric_epoch

def test_cycle(options, model, all_test_paths, all_test_labels, save_predictions=False):                    
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    print("Loading best validation model weights: ")
    model_path = 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation2d_array.pth'
    print(model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    HOUNSFIELD_BRAIN_MIN, HOUNSFIELD_BRAIN_MAX = 0, 80
    clamp = tio.Clamp(out_min=HOUNSFIELD_BRAIN_MIN, out_max=HOUNSFIELD_BRAIN_MAX)
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))

    pred = []
    test_dicemetric = []
    #test_hausdorffs = []
    y = []
    dice_metric = DiceMetric(reduction="mean", get_not_nans=False)
    #hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean')
    dice_metric.reset()
    max_intensity = 0
    print("Computing Test Meassures for INSTANCE22 test set")

    volumes_recall, volumes_precision = [],[]
    for path_test_case, path_test_label in zip(all_test_paths,all_test_labels):      
        flat_labels, flat_predictions = [],[]              
        test_vol = nb.load(path_test_case)
        test_lbl = nb.load(path_test_label)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)

        test_vol_pxls = test_vol.get_fdata()
        test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
        test_vol_pxls = clamp(test_vol_pxls[np.newaxis])     
        test_vol_pxls = rescale(test_vol_pxls)
        vol_prediction = np.zeros(test_vol.get_fdata().shape, dtype = np.float32)
        print(vol_prediction.shape)

        for slice_selected in range(test_vol_pxls.shape[-1]):
            test_tensor = torch.tensor(test_vol_pxls[np.newaxis, :,:,:,slice_selected], dtype=torch.float).to(device)
            #print(test_tensor.shape)
            out_test = post_trans(model(test_tensor))
            out_test = out_test.detach().cpu().numpy()            
            pred = np.array(out_test[0,0,:,:], dtype='uint8')
            flat_label = np.array(test_lbl_pxls[:,:,slice_selected]>0, dtype='uint8').flatten()
            flat_pred = pred.flatten()
            flat_labels.append(flat_label)
            flat_predictions.append(flat_pred)
            vol_prediction[:,:,slice_selected] = pred

            dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
            #hausdorff_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
        volumes_recall.append(recall_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        volumes_precision.append(precision_score(np.array(flat_predictions).flatten(), np.array(flat_labels).flatten()))
        test_dicemetric.append(dice_metric.aggregate().item())
        #if not torch.isinf(hausdorff_metric.aggregate()[0]):
        #    test_hausdorffs.append(hausdorff_metric.aggregate()[0])
        # reset the status for next computation round
        dice_metric.reset()
        #hausdorff_metric.reset()
        if save_predictions:
            img = nb.Nifti1Image(vol_prediction, affine=test_vol.affine, header=test_vol.header)
            Path('outputs/'+options['suffix']).mkdir(parents=True, exist_ok=True)
            nb.save(img, 'outputs/'+options['suffix']+'/'+path_test_case.split('/')[-1]+'_pred_'+options['suffix']+'.nii.gz')

    print(options['suffix']+" model DICE for all TEST VOLUMES: ")
    print(np.mean(test_dicemetric))
    print(volumes_precision)
    return(np.mean(test_dicemetric),volumes_precision, volumes_recall)

trained_model, loss_train = train(model, training_loader, local_epochs, 
                                local_lr, batch_size, 
                                cur_fold_val_indices_volumes, cur_fold_val_indices_labels, options)

test_cycle(options, model, test_paths, test_labels, save_predictions=True)




