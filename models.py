import timm
import torch
from torch import nn
import pdb

class ImageModel(nn.Module):
    def __init__(self, model_name, class_n, mode):
        super().__init__()
        self.model_name = model_name
        self.class_n = class_n
        self.mode = mode
        self.encoder = timm.create_model(self.model_name, pretrained=False)
        names = []
        modules = []
        for name, module in self.encoder.named_modules():
            names.append(name)
            modules.append(module)
        #pdb.set_trace()
        self.fc_in_features = self.encoder.num_features
        print(f'The layer was modified...')

        fc_name = names[-1].split('.')
        print(
            f'{getattr(self.encoder, fc_name[0])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
        setattr(self.encoder, fc_name[0], nn.Linear(self.fc_in_features, class_n))
        #pdb.set_trace()

        
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.encoder(x)
        return x