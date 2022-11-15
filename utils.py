import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def set_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),
                                    lr=args.learning_rate)
    return optimizer
def set_loss(args):
    criterion = None
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    return criterion

def set_scheduler(args, optimizer, iter_per_epoch):
    scheduler = None
    if args.scheduler == 'cos_base':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler

def score_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = accuracy_score(real, pred)
    return score