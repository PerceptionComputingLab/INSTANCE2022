import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from loss_functions import SoftDiceLoss
import json


def train_one_model(model, dataset, device, log_name, writer, checkpoint_path,epoch_start,epoch_end,optimizer, batch_size=1, val_percent=0.22,dataset_fraction=1, weight_decay=0, save_only_min=True,min_ce_loss = 10000,patience = 15,epochs_without_min=0):
    
    logging.basicConfig(filename=f'{log_name}.log', filemode='a', level=logging.INFO)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val],generator=torch.Generator().manual_seed(42) )
    n_subtrain = int(n_train*dataset_fraction)
    subtrain, unused = random_split(train,[n_subtrain,n_train-n_subtrain],generator=torch.Generator().manual_seed(42) )
    train_loader = DataLoader(subtrain,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True)

    criterion = nn.CrossEntropyLoss()

    min_loss = False

    for epoch in range(epoch_start,epoch_end):

        model.train()

        for batch_no, batch in enumerate(train_loader):
            imgs = batch['image']
            labels = batch['label']

            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            out = model(imgs)
            loss = criterion(out, labels)

            correct = (out.argmax(1) == labels)
            acc = [correct[labels == i].double().mean().item() for i in range(model.num_classes)]

            logging.info((
                f"{epoch}:{batch_no} loss={loss.item():.3f} p(pred=i | true=i)={acc}"
            ))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.1)
            optimizer.step()

        ce_loss , total_acc, total_softdiceloss = eval_net(model, val_loader, device,epoch)
        writer.add_scalar('Cross_Entropy_Loss',ce_loss,epoch)
        for c in range(model.num_classes):
            writer.add_scalar(f'Dice_Loss/{c}',total_softdiceloss[c],epoch)
        
        #scheduler.step(val_score)
        if ce_loss < min_ce_loss:
            min_loss = True
            min_ce_loss = ce_loss
            epochs_without_min = 0
        
        else: 
            epochs_without_min += 1

        if epochs_without_min > patience:
            with open(checkpoint_path+'/training_progress.json','w') as f:
                d = {'epoch': epoch, 'epochs_without_min': epochs_without_min, 'done': True, 'ce_loss':ce_loss.item()}
                f.write(json.dumps(d)) 
            break
            

        if not save_only_min:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ce_loss,
                        }, f'{checkpoint_path}/model_{epoch:03}.pt')
        
        elif save_only_min and min_loss:
            print('min loss:', epoch)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ce_loss,
                        }, f'{checkpoint_path}/model_min.pt')
        
        min_loss = False

        with open(checkpoint_path+'/training_progress.json','w') as f:
            d = {'epoch': epoch, 'epochs_without_min': epochs_without_min, 'done': False, 'ce_loss':ce_loss.item()}
            f.write(json.dumps(d)) 


def eval_net(net, loader, device,epoch,three_d=True):
    """Validation"""

    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    total_acc = np.zeros(net.num_classes)
    total_softdiceloss = np.zeros(net.num_classes)
    criterion = nn.CrossEntropyLoss()

    for batch_no, batch in enumerate(loader):
        logging.info(( f"{batch_no}"))


        imgs, true_masks = batch['image'], batch['label']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            output = net(imgs)
            if torch.isnan(output).any():
                logging.info(("NaN"))
                n_val -= 1
                continue
            
        pred = output.softmax(1)
        correct = (output.argmax(1) == true_masks)
        acc = [correct[true_masks == i].double().mean().item() for i in range(net.num_classes)]
        total_acc+=acc

        logging.info(( f" p(pred=i | true=i)={acc}"))
        #logging.info(( f" p(pred=i | true=i)={100*acc[0]:.0f}%, {100*acc[1]:.0f}% and {100*acc[2]:.0f}%"))

        one_hot_true_masks = F.one_hot(true_masks,net.num_classes)

        for label in range(net.num_classes):
            try:
                total_softdiceloss[label] += SoftDiceLoss()(pred[:,label,:,:,:].unsqueeze(0),one_hot_true_masks[...,label].unsqueeze(0)).item()
            except:
                total_softdiceloss[label] += SoftDiceLoss()(pred[:,label,:,:].unsqueeze(0),one_hot_true_masks[...,label].unsqueeze(0)).item()
        tot += criterion(output, true_masks)

    net.train()

    total_acc /= n_val
    total_softdiceloss /= n_val
    

    logging.info((
        f"validation: {epoch}: ce loss={tot/n_val:.3f} p(pred=i | true=i)={total_acc}"
    ))
    logging.info((
        f"validation: {epoch}: dice loss={total_softdiceloss[0]:.3f}, {total_softdiceloss}"
    ))

    return tot/n_val , total_acc, total_softdiceloss
