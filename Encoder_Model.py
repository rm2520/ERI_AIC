import torch
import torchvision
from torch import nn
import torchvision.models as models
class InceptionModel(nn.Module):
    def __init__(self, **kwargs):
        super(ResnetModel, self).__init__()
        middle_dim=256
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in  self.inception.parameters():
            param.requires_grad_(False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, middle_dim)
        self.bn = nn.BatchNorm1d(middle_dim, momentum=0.01)
    def forward(self, image):
        img_fet=self.inception(image)
        image_embedding= self.bn(img_fet)
        return image_embedding


class ResnetModel(nn.Module):
    def __init__(self, **kwargs):
        super(ResnetModel, self).__init__()

        middle_dim=256
        resnet = torchvision.models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        module = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*module)
        self.embed = nn.Linear(resnet.fc.in_features, middle_dim)

        self.bn = nn.BatchNorm1d(middle_dim, momentum=0.01)
    def forward(self, image):
        img_fet=self.resnet(image)
        img_fet = img_fet.view(img_fet.size(0), -1)
        img_fet = self.embed(img_fet)
        image_embedding= self.bn(img_fet)
        return image_embedding








__factory = {
    'inception': InceptionModel,
    'resnet': ResnetModel,

}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):

    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
