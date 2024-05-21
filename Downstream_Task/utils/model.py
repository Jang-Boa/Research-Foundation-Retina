import torch
from torchvision import models
import torch.nn as nn

def ResNetModel(model_name, num_classes=2):
    model_ft = None
    input_size = 0
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        #         model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes),
                                    )
#     elif model_name == 'resnet101':
#         model_ft = models.resnet101(pretrained=False)
#         num_ftrs = model_ft.fc.in_features
# #         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1024),
#                                     nn.BatchNorm1d(1024),
#                                     nn.ReLU(),
#                                     nn.Linear(1024, 512),
#                                     nn.BatchNorm1d(512),
#                                     nn.ReLU(),
#                                     nn.Linear(512, num_classes),
#                                    )
    
    else:
        print("Invalid model name, exiting...")
        exit()

    model_ft = nn.DataParallel(model_ft, device_ids=[0], output_device=0)
    return model_ft