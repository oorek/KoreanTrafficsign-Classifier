import argparse
import torch
import os
from glob import glob
from torch.utils.data import DataLoader
from models import ImageModel
from dataset import TrafficSign
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb

@torch.no_grad()
def test_no_label(model, test_loader):
    model.eval()

    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)
    preds, img_names = [], []
  #  pdb.set_trace()
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        img_name = batch_item['img_name']
       # pdb.set_trace()
        pred = model(img.float())
        preds.extend(torch.softmax(pred, dim=1).clone().detach().cpu().numpy())  # probabillity, not label
        img_names.extend(img_name)

    return preds, img_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dd', '--data_dir', type=str, default='./test/final/')
    parser.add_argument('-sd', '--save_dir', type=str, default='./test/result/')
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b0')

    parser.add_argument('-ckpt', '--checkpoint', type=str, default='./ckpt_best_try1.pt')
    parser.add_argument('-bs', '--batch_size', type=int, default=3)
    parser.add_argument('-is', '--img_size', type=int, default=100)

    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-cv', '--csv_name', type=str, default='test')
    args = parser.parse_args()

    args.device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'

    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join('./data/','train')))
    label_description = [l_d for l_d in label_description if l_d[0] != '.']
    label_encoder = {key:idx for idx, key in enumerate(label_description)}

    test_data = sorted(glob(f'{os.path.dirname(args.data_dir)}/*.jpg'))
    #####################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='test')
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################

    #### LOAD DATASET ####
    test_dataset = TrafficSign(args, test_data, labels=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size= args.batch_size, num_workers=args.num_workers, shuffle=False)
    print('> DATAMODULE BUILT')
    ######################

    #### INFERENCE START ####
    print('> START INFERENCE ')
    preds, img_names = test_no_label(model, test_loader)
    preds = np.array(preds)
    val = np.max(preds, axis=1)
    preds = np.argmax(preds, axis=1)
    submission = pd.DataFrame()
    submission['image_name'] = img_names
    submission['label'] = preds
    submission['val'] = val

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    submission.to_csv(f'{args.save_dir}/{args.csv_name}.csv', index=False)