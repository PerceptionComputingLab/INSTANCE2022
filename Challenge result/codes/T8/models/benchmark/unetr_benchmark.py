import numpy as np
import cc3d
import nibabel as nb
from torch.optim import Adam

from glob import glob
import torch
from natsort import natsorted
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff 
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
from monai.inferers import sliding_window_inference
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from numpy import std, mean
from torch.utils.data import DataLoader
from pathlib import Path
import torchio as tio
import warnings
import monai
from monai.networks.nets import UNETR, SwinUNETR

from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_utils import compute_dice, compute_absolute_volume_difference, compute_absolute_lesion_difference, compute_lesion_f1_score, test_cycle_3dunet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

max_queue_length = 16
patches_per_volume = 4
b_size=16
loader_workers = 8
local_epochs= 250
patch_size = (96,96,32)

options = {}
options['modality'] = 'CT'
options['exp_location'] = 'SCAN'
options['max_queue_length'], options['patches_per_volume'] = max_queue_length, patches_per_volume
options['b_size'] = b_size
options['model'] = 'SWIN_UNETR'
options['patch_size'] = patch_size



if options['exp_location'] == 'SCAN':
    INSTANCE22_DATA_ROOT = '/str/nas/INSTANCE2022/train_2/'
if options['exp_location'] == 'LAPTOP':
    INSTANCE22_DATA_ROOT = '/Users/sebastianotalora/work/instance22/train_2/'


nifti_vols_paths = natsorted(glob(INSTANCE22_DATA_ROOT+'data/*.nii.gz'))
nifti_labl_paths = natsorted(glob(INSTANCE22_DATA_ROOT+'label/*.nii.gz'))

test_indices = np.arange(0.9*len(nifti_vols_paths),len(nifti_vols_paths),dtype=int)
test_paths   = [nifti_vols_paths[i] for i in test_indices]
test_labels  = [nifti_labl_paths[i] for i in test_indices]

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



cur_fold_val_indices_volumes,cur_fold_val_indices_labels = indices_folds[0][2], indices_folds[0][3]
cur_fold_train_indices_volumes,cur_fold_train_indices_labels = indices_folds[0][0],indices_folds[0][1]

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
    subjects_list_test.append(tio.Subject(
    ct=tio.ScalarImage(test_paths[i]),
    label=tio.LabelMap(test_labels[i]),
))



class UNet_custom(monai.networks.nets.BasicUNet):
    def __init__(self, spatial_dims, in_channels, out_channels,
                 name, scaff=False, fed_rod=False):
        #call parent constructor
        super(UNet_custom, self).__init__(spatial_dims=spatial_dims,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          upsample="nontrainable")#, 
        self.name = name





HOUNSFIELD_BRAIN_MIN, HOUNSFIELD_BRAIN_MAX = 0, 80
clamp = tio.Clamp(out_min=HOUNSFIELD_BRAIN_MIN, out_max=HOUNSFIELD_BRAIN_MAX)
rescale = tio.RescaleIntensity(out_min_max=(0, 1))
spatial = tio.OneOf({
        tio.RandomAffine(): 0.6,
        tio.RandomElasticDeformation(): 0.2,        
        tio.RandomAffine(degrees=180): 0.2
    },
    p=0.75,
)
rotation = tio.RandomAffine(degrees=360)


sampler = tio.data.WeightedSampler(patch_size, 'label')
toCanon = tio.ToCanonical()
resampler_dwi = tio.Resample('ct')
flip = tio.RandomFlip(axes=('LR',))
padding = tio.transforms.Pad(padding=(0,0,0,0,10,10))
transforms = [clamp, rescale, flip, padding, toCanon]
transform = tio.Compose(transforms)


labels_probabilities = {0: 0.3, 1: 0.7}
sampler_weighted_probs = tio.data.LabelSampler(
    patch_size=patch_size,
    label_name='label',
    label_probabilities=labels_probabilities,
)

def get_model(options):
    if options['model'] == 'UNETR':
        model = UNETR(in_channels=1, out_channels=1, img_size=options['patch_size'], pos_embed='conv').to(device)
    if options['model'] == 'SWIN_UNETR':
        model = SwinUNETR(
            img_size=options['patch_size'],
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
        weight = torch.load("./models/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
    return model



training_dataset = tio.SubjectsDataset(subjects_list_train, transform=transform)
valid_dataset = tio.SubjectsDataset(subjects_list_val, transform=transform)

queue = tio.Queue(training_dataset, options['max_queue_length'], options['patches_per_volume'], sampler_weighted_probs)
training_loader = torch.utils.data.DataLoader(queue, batch_size=options['b_size'])


batcho = tio.utils.get_first_item(training_loader)
batcho.keys()
print(batcho['ct']['data'].max())

batcho['ct']['data'][:,:,:,:,:].shape

def train(ann, training_loader, local_epochs, local_lr, batch_size, val_img_paths,val_lbl_paths,options):
    ann.train()
    best_metric, best_precision, best_metric_epoch = -1,-1,-1

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = Adam(ann.parameters(), local_lr)

    for epoch in range(local_epochs):
        for batch_data in training_loader:
            inputs, labels = batch_data['ct']['data'][:,:,:,:,:].float().to(device),batch_data['label']['data'][:,:,:,:,:].float().to(device)
            y_pred = ann(inputs)
            loss = loss_function(y_pred, labels)
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()
        options['writer'].add_scalar("train_loss", loss.item(), epoch)
        print("Epoch", epoch, " loss :", loss.item())
        if epoch%2 == 0:
            best_metric, best_precision, best_metric_epoch = validation_cycle_3DUNETR(ann, val_img_paths, 
                                                                                              val_lbl_paths, epoch, best_metric,
                                                                                             best_precision, best_metric_epoch, loss_function, options)
    return ann, loss.item()



def validation_cycle_3DUNETR(model, valid_paths, val_lbl_paths, cur_epoch, best_metric,
                           best_precision, best_metric_epoch, loss_function, options):
   
    metric = 0
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()
    test_dicemetric= []
    HOUNSFIELD_BRAIN_MIN, HOUNSFIELD_BRAIN_MAX = 0, 80
    clamp = tio.Clamp(out_min=HOUNSFIELD_BRAIN_MIN, out_max=HOUNSFIELD_BRAIN_MAX)
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))

    volumes_recall,volumes_precision = [],[]
    lesion_differences =[]
    volumes_difference = []
    vol_f1s = []
    mean_losses = []
    dice_metric = DiceMetric(reduction="mean", get_not_nans=False)
    mean_dices_monai = []

    with torch.no_grad():
        for path_case, path_label in zip(valid_paths, val_lbl_paths):          
            test_vol = nb.load(path_case)
            test_lbl = nb.load(path_label)
            test_vol_pxls = test_vol.get_fdata()
            test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
            test_vol_scaled = rescale(clamp(test_vol_pxls[np.newaxis,:,:,:]))
            array = torch.tensor(test_vol_scaled[np.newaxis,:,:,:,:]).to(device)
            test_lbl_pxls = test_lbl.get_fdata()[np.newaxis,np.newaxis,:,:,:]
            sx, sy, sz = test_vol.header.get_zooms()
            volume_voxl = sx * sy * sz
            val_outputs = post_trans(sliding_window_inference(array, options['patch_size'], 8, model))
            dice_metric(val_outputs,torch.tensor(test_lbl_pxls).to(device))
            img = nb.Nifti1Image(val_outputs[0,0,:,:,:].cpu().numpy(), affine=test_vol.affine, header=test_vol.header)
            #Path('outputs_val/'+options['suffix']).mkdir(parents=True, exist_ok=True)
            #nb.save(img, 'outputs_val/'+options['suffix']+'/'+path_case.split('/')[-1]+'_pred_'+options['suffix']+'.nii.gz')
            
            mean_dices_monai.append(dice_metric.aggregate().item())
            mean_losses.append(loss_function(val_outputs, torch.tensor(test_lbl_pxls).to(device)).item())
            dice_metric.reset()

            test_dicemetric.append(compute_dice(np.array(val_outputs.cpu().numpy()).flatten(), np.array(test_lbl_pxls).flatten(), empty_value=1.0))
            volumes_recall.append(recall_score(np.array(val_outputs.cpu().numpy()).flatten(), np.array(test_lbl_pxls).flatten()))
            volumes_precision.append(precision_score(np.array(val_outputs.cpu().numpy()).flatten(), np.array(test_lbl_pxls).flatten()))
            volumes_difference.append(compute_absolute_volume_difference(val_outputs.cpu().numpy().squeeze(), test_lbl_pxls.squeeze(), volume_voxl))
            lesion_differences.append(compute_absolute_lesion_difference(val_outputs.cpu().numpy().squeeze(), test_lbl_pxls.squeeze()))
            vol_f1s.append(compute_lesion_f1_score(val_outputs.cpu().numpy().squeeze(), test_lbl_pxls.squeeze()))

        metric_monai = np.mean(mean_dices_monai)
        metric = mean(test_dicemetric)
        avg_precision = mean(volumes_precision)
        avg_recall = mean(volumes_recall)
        avg_voldiff = mean(volumes_difference)
        avg_lesdiff = mean(lesion_differences)
        avg_f1s = mean(vol_f1s)
        avg_loss = mean(mean_losses)
        #metric_values.append(metric)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = cur_epoch
            torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_metric_model_segmentation.pth')
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                cur_epoch +1, metric, best_metric, best_metric_epoch
            )
        )
        options['writer'].add_scalar("val_mean_dice", metric, cur_epoch)
        options['writer'].add_scalar("val_monai_dice", metric_monai, cur_epoch)


        if avg_precision > best_precision:
            best_precision = avg_precision
            best_metric_epoch = cur_epoch
            torch.save(model.state_dict(), 'models/'+options['modality']+'_'+options['suffix']+'_best_precision_model_segmentation.pth')
            print("saved new best metric model")
        print(
            "current epoch: {} current mean Precision: {:.4f} best mean Precision: {:.4f} at epoch {}".format(
                cur_epoch +1, avg_precision, best_precision, best_metric_epoch
            )
        )

        options['writer'].add_scalar("val_mean_precision", avg_precision, cur_epoch)
        options['writer'].add_scalar("val_mean_recall",    avg_recall, cur_epoch)
        options['writer'].add_scalar("val_mean_diff", avg_voldiff, cur_epoch)
        options['writer'].add_scalar("val_mean_lesdiff", avg_lesdiff, cur_epoch)
        options['writer'].add_scalar("val_mean_lesion_f1", avg_f1s, cur_epoch)
        options['writer'].add_scalar("val_mean_loss", avg_loss, cur_epoch)
    return best_metric, best_precision, best_metric_epoch


loss_function = monai.losses.DiceLoss(sigmoid=True)
learning_rates = [0.001694]
local_epochs = 250

for local_lr in learning_rates:
    net = get_model(options)
    options['suffix'] = options['modality'] +options['model']+'_FVVal_lr_'+str(local_lr) + 'p_size_'+str(options['patch_size'][0])
    options['writer'] = SummaryWriter(f"runs/{options['suffix']}_epochs_{local_epochs}")

    trained_model, loss_train = train(net, training_loader, 
                                    local_epochs, local_lr, options['b_size'],
                                    cur_fold_val_indices_volumes, cur_fold_val_indices_labels, options)
    print("INTERNAL TEST EVALUATION")
    test_cycle_3dunet(options, trained_model, test_paths, test_labels, loss_function, save_predictions = True)
