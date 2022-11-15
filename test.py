import timm
from pprint import pprint

model_names = timm.list_models(pretrained=False)
pprint(model_names)