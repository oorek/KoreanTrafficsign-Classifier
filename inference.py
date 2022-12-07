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
from utils import score_function

def make_list(i, preds):
   # print(preds[i,rank1[i]])
    if preds[i,rank1[i]] >= 0.8:
        return True
    return False

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

@torch.no_grad()
def test_with_label(args, model, test_loader):
    model.eval()
    total_test_score = 0
    preds = []
    answer = []
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        pred = model(img.float())

        test_score = score_function(label, pred)
        total_test_score += test_score
        preds.extend(torch.argmax(pred, dim=1).clone().cpu().numpy())
        answer.extend(label.cpu().numpy())

        log = f'[TEST] Test Acc : {test_score.item():.4f}({total_test_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[TEST] Test Acc : {total_test_score / (batch_idx + 1):.4f}'

        batch_iter.set_description(log)
        batch_iter.update()

    test_mean_acc = total_test_score / len(batch_iter)

    batch_iter.close()

    return test_mean_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-cd', '--class_dir', type=str, default='D:\\result\\result\\data\\')
   # parser.add_argument('-dd', '--data_dir', type=str, default='D:\\images\\')
    parser.add_argument('-dd', '--data_dir', type=str, default='D:\\result\\result\\data')
    parser.add_argument('-sd', '--save_dir', type=str, default='C:/Users\Moon/trafficsign/KoreanTrafficsign-Classifier/result')
    parser.add_argument('-m', '--model', type=str, default='convnext_tiny')

    parser.add_argument('-ckpt', '--checkpoint', type=str, default='./savedir/convnext_tiny_1207_032152/ckpt_best.pt')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-is', '--img_size', type=int, default=224)

    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-cv', '--csv_name', type=str, default='test')
    parser.add_argument('-l', '--label', type=int, default=0)
    args = parser.parse_args()

    #args.device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join(args.class_dir,'train')))
    label_description = [l_d for l_d in label_description if l_d[0] != '.']
    label_encoder = {key:idx for idx, key in enumerate(label_description)}

   # test_data = sorted(glob(f'{os.path.dirname(args.data_dir)}/*/*.jpg'))
    test_data = sorted(glob(f'{os.path.join(args.data_dir, "test")}/*/*.jpg'))
    #pdb.set_trace()
    #####################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='test')
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################

    if args.label == 1:
        #### LOAD DATASET ####
        test_dataset = TrafficSign(args, test_data, labels=None, mode='test')
        test_loader = DataLoader(test_dataset, batch_size= args.batch_size, num_workers=args.num_workers, shuffle=False)
        print('> DATAMODULE BUILT')
        ######################

        #### INFERENCE START ####
        print('> START INFERENCE ')
        preds, img_names = test_no_label(model, test_loader)
        preds = np.array(preds)
       # pdb.set_trace()
        P = np.argsort(preds, axis=1)
        rank1 = P[:,-1]
        rank2 = P[:,-2]
        rank3 = P[:,-3]

        val = np.max(preds, axis=1)
    # preds = np.argmax(preds, axis=1)

        #submission = pd.DataFrame()
        #submission['image_name'] = img_names
        #submission['val'] = val
        #submission['rank1'] = rank1
        #submission['rank2'] = rank2
        #submission['rank3'] = rank3  


        #if not os.path.exists(args.save_dir):
        #    os.makedirs(args.save_dir)
        
        #submission.to_csv(f'{args.save_dir}/{args.csv_name}.csv', index=False)
      #  submission = [idx for idx in rank1 if preds[i,rank1] >= 0.8 for i in range(len(preds))]
        submission = [idx for i, idx in enumerate(rank1) if make_list(i, preds) == True] 
        print(submission)
       # return submission
    
    else:
        test_label = [data.split('\\')[-2] for data in test_data]  # '가자미전'
      #  pdb.set_trace()
        test_labels = [label_encoder[k] for k in test_label]  # 0

        #### LOAD DATASET ####
        test_dataset = TrafficSign(args, test_data, labels=test_labels, mode='valid')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        print('> DATAMODULE BUILT')
        ######################

        #### INFERENCE START ####
        print('> START INFERENCE ')
        test_acc = test_with_label(args, model, test_loader)

        print("=" * 50 + "\n\n")
        print(f"final accuracy: {test_acc * 100:.2f}% \n\n")
        print("=" * 50 + "\n\n")
        #########################

    