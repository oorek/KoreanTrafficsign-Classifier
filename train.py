import argparse
import torch
import os
from glob import glob
from utils import *
import pdb
from dataset import TrafficSign
from torch.utils.data import DataLoader
from models import ImageModel
from tqdm import tqdm
import wandb
from datetime import datetime

def train(args, model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_train_loss = 0
    total_train_score = 0
    batch_iter = tqdm(enumerate(train_loader), desc='Training', total=len(train_loader), ncols=120)
    for batch_idx, batch_item in batch_iter:
        #pdb.set_trace()
        optimizer.zero_grad()
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)
        #print(img.shape)
        pred = model(img.float())
        train_loss = criterion(pred, label)

        train_loss.backward()
        optimizer.step()

        train_score = score_function(label,pred)
        total_train_loss += train_loss
        total_train_score += train_score

        log = f'[EPOCH {epoch}] Train Loss : {train_loss.item():.4f}({total_train_loss / (batch_idx + 1):.4f}), '
        log += f'Train Acc : {train_score.item():.4f}({total_train_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Train Loss : {total_train_loss / (batch_idx + 1):.4f}, '
            log += f'Train Acc : {total_train_score / (batch_idx + 1):.4f}, '
            log += f"LR : {optimizer.param_groups[0]['lr']:.2e}"

        batch_iter.set_description(log)
        batch_iter.update()

    _lr = optimizer.param_groups[0]['lr']
    train_mean_loss = total_train_loss / len(batch_iter)
    train_mean_acc = total_train_score / len(batch_iter)
    batch_iter.close()
    
    if args.wandb:
        wandb.log({'train_mean_loss': train_mean_loss, 'lr': _lr, 'train_mean_acc': train_mean_acc}, step=epoch)

@torch.no_grad()
def valid(args, model, val_loader, criterion, epoch):
    model.eval()
    total_val_score = 0
    total_val_loss = 0
    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=120)
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        pred = model(img.float())
        val_loss = criterion(pred, label)

        val_score = score_function(label, pred)
        total_val_loss += val_loss
        total_val_score += val_score

        log = f'[EPOCH {epoch}] val Loss : {val_loss.item():.4f}({total_val_loss / (batch_idx + 1):.4f}), '
        log += f'val Acc : {val_score.item():.4f}({total_val_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] val Loss : {total_val_loss / (batch_idx + 1):.4f}, '
            log += f'val Acc : {total_val_score / (batch_idx + 1):.4f}, '
            log += f"LR : {optimizer.param_groups[0]['lr']:.2e}"
        batch_iter.set_description(log)
        batch_iter.update()

    val_mean_loss = total_val_loss / len(batch_iter)
    val_mean_acc = total_val_score / len(batch_iter)

    batch_iter.close()

    return val_mean_loss, val_mean_acc

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default='/Users/user/ml/trafficsign/data/')
    parser.add_argument('-sd', '--save_dir', type=str, default='/Users/user/ml/trafficsign/savedir/')
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b0')
    #parser.add_argument('-is', '--img_size', type=int, default=32)
    #parser.add_argument('-av', '--aug_ver', type=int, default=0)
    
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)

    parser.add_argument('-l', '--loss', type=str, default='ce', choices=['ce', 'focal', 'smoothing_ce'])
    #parser.add_argument('-ls', '--label_smoothing', type=float, default=0.5)
    parser.add_argument('-ot', '--optimizer', type=str, default='adam',
                        choices=['adam', 'radam', 'adamw', 'adamp', 'ranger', 'lamb', 'adabound'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-sc', '--scheduler', type=str, default='cos_base', choices=['cos_base', 'cos', 'cycle'])
    # wandb config:
    parser.add_argument('--wandb', type=bool, default=False)

    # 입력받은 인자값을 args에 저장
    args = parser.parse_args()
    #args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
    #args.device = 'cpu'
    #print(f"device: {args.device}")
    #scaler = torch.cuda.amp.GradScaler() if args.amp else None
    #### SEED EVERYTHING ####
    #seed_everything(args.seed)
    #########################

    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    label_description = [l_d for l_d in label_description if l_d[0] != '.']
    label_encoder = {key:idx for idx, key in enumerate(label_description)}
    #{speedlimit20,0}, {speedlimit30,1}
    label_decoder = {val:key for key, val in label_encoder.items()}
    #[0~29]
    train_data = sorted(glob(f'{os.path.join(args.data_dir, "train")}/*/*.png'))
    #1000 * 30 = 30000
    val_data = sorted(glob(f'{os.path.join(args.data_dir, "val")}/*/*.png'))
    train_label = [data.split('/')[-2] for data in train_data]
    train_labels = [label_encoder[k] for k in train_label]
    val_label = [data.split('/')[-2] for data in val_data]
    val_labels = [label_encoder[k] for k in val_label]
    #pdb.set_trace()
    #####################

    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
    save_dir = os.path.join(args.save_dir, f'{args.model}_{c_date}_{c_time}')
    os.makedirs(save_dir)
    #### SET WANDB ####
    run = None
    ###################

    #### LOAD DATASET ####
    train_dataset = TrafficSign(args, train_data, train_labels, mode='train')
    val_dataset = TrafficSign(args, val_data, val_labels, mode = 'val')
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)
    iter_per_epoch = len(train_loader)
    # 30000 / batch_size
    print('> DATAMODULE BUILT')
    ######################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='train')
    #pdb.set_trace()
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    print('> MODEL BUILT')
    ####################

    #### SET TRAINER ####
    optimizer = set_optimizer(args, model)
    criterion = set_loss(args)
    scheduler = set_scheduler(args, optimizer, iter_per_epoch)
    print('> TRAINER SET')
    #####################

    best_val_acc = .0
    best_val_loss = 9999.
    best_epoch = 0

    print('> START TRAINING')
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        val_loss, val_acc = valid(args, model, val_loader, criterion, epoch)
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = val_acc

            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])
            
            #torch.save({'model_state_dict': model.state_dict(),
            #            'optimizer_state_dict': optimizer.state_dict(),
            #            'scheduler' : scheduler.state_dict(),
            #            'epoch' : epoch,},
            #            f'{save_dir}/ckpt_best.pt')
            #print(f'> SAVED model ({epoch:02d}) at {save_dir}/ckpt_best.pt')

        if args.scheduler in ['cos_base', 'cos']:
            scheduler.step()
    
    del model
    del optimizer, scheduler
    del train_dataset

    if args.wandb:
        run.finish()