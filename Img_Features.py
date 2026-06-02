
import torch
import torch.nn as nn
import torchvision.models as models
import os.path as osp
import pickle
from PIL import Image
from torch.nn import functional as F
import os


class Vgg16Model(object):
    def __init__(self,images,images_root,transform,save_features):
        save_features=save_features+  '.pkl'
        #get vgg features
        if not(osp.exists(os.path.join('logs/', save_features))):
            vgg_model = models.vgg16(pretrained=True).cuda()
            vgg_model.features = vgg_model.features[:]
            vgg_model.classifier = vgg_model.classifier[:4]

            vgg_model.eval()

            features = dict()
            with torch.no_grad():
                for img_name in images:
                    img_path=os.path.join(images_root, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except IOError:
                        print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    if transform is not None:
                        image = transform(image)
                    image = torch.unsqueeze(image, 0).cuda()
                    feat = vgg_model(image)
                    img_name = img_name.split('.')[0]
                    features[img_name]=feat
            pickle.dump(features, open(os.path.join('logs/',save_features), 'wb')) #save vgg features

        else:
            features = pickle.load(open(os.path.join('logs/', save_features), 'rb')) #retrieve vgg features
        self.features = features
        temp=images[0].split('.')[0]
        _,imgf_dim=features[temp].shape
        self.imgf_dim=imgf_dim


class MobilentModel(object):
    def __init__(self,images,images_root,transform,save_features):
        from itertools import islice
        save_features=save_features+  '.pkl'
        #get MobileNet features
        if not(osp.exists(os.path.join('logs/', save_features))):
            mobilenet_model = models.mobilenet_v2(pretrained=True).cuda()

            removed = list(mobilenet_model.children())[:-1]
            mobilenet_model= nn.Sequential(*removed)
            mobilenet_model.eval()

            features = dict()
            with torch.no_grad():
                for img_name in images:
                    img_path=os.path.join(images_root, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except IOError:
                        print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    if transform is not None:
                        image = transform(image)
                    image = torch.unsqueeze(image, 0).cuda()
                    feat = mobilenet_model(image)
                    feat = feat.cpu().detach()
                    img_name = img_name.split('.')[0]
                    features[img_name]=feat
            save_path = os.path.join('logs', save_features)

            chunk_size = 1000
            with open(save_path, 'wb') as f:
                it = iter(features.items())
                while True:
                    chunk = dict(islice(it, chunk_size))
                    if not chunk:
                        break
                    pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_path = os.path.join('logs', save_features)
            features = dict()
            with open(save_path, 'rb') as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        features.update(chunk)
                    except EOFError:
                        break
        self.features = features
        temp=images[0].split('.')[0]
       # _,imgf_dim=features[temp].shape
       # self.imgf_dim=imgf_dim

class InceptionModel(object):
    def __init__(self,images,images_root,transform,save_features):
        save_features = save_features +  '.pkl'
        # get inception features
        if not(osp.exists(os.path.join('logs/', save_features))):
            inception_model=models.inception_v3(pretrained=True, aux_logits=False)

            removed = list(inception_model.children())[:-1]
            inception_model = nn.Sequential(*removed)



            inception_model.eval()


            features = dict()
            with torch.no_grad():
                for img_name in images:
                    img_path=os.path.join(images_root, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except IOError:
                        print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    if transform is not None:
                        image = transform(image)
                    image = torch.unsqueeze(image, 0)
                    feat = inception_model(image)
                    ##feat = F.avg_pool2d(feat,feat.size()[2:])
                    pool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
                    feat=pool(feat)
                    feat=torch.unsqueeze(torch.squeeze(feat),0)
                    img_name = img_name.split('.')[0]

                    features[img_name]=feat


            print('hi')
            pickle.dump(features, open(os.path.join('logs/',save_features), 'wb'))

        else:
            features = pickle.load(open(os.path.join('logs/', save_features), 'rb'))
        self.features = features
        temp=images[0].split('.')[0]
        _,imgf_dim=features[temp].shape
        self.imgf_dim=imgf_dim


class ResnetModel(object):
    def __init__(self, images, images_root, transform, save_features):
        save_features = save_features + '.pkl'
        # get resnet features
        if not (osp.exists(os.path.join('logs/', save_features))):
            resnet = models.resnet101(pretrained=True).cuda() # pretrained ImageNet ResNet-101

            # Remove linear and pool layers (since we're not doing classification)
            modules = list(resnet.children())[:-1]
            resnet = nn.Sequential(*modules)


            resnet.eval()

            features = dict()
            with torch.no_grad():
                for img_name in images:
                    img_path = os.path.join(images_root, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except IOError:
                        print(
                            "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    if transform is not None:
                        image = transform(image)
                    image = torch.unsqueeze(image, 0).cuda()
                    feat = resnet(image)
                    feat = torch.unsqueeze(torch.squeeze(feat), 0)
                    img_name = img_name.split('.')[0]
                    features[img_name] = feat

            pickle.dump(features, open(os.path.join('logs/', save_features), 'wb'))

        else:
            features = pickle.load(open(os.path.join('logs/', save_features), 'rb'))
        self.features = features
        temp = images[0].split('.')[0]
        _, imgf_dim = features[temp].shape
        self.imgf_dim = imgf_dim




__factory = {
     #image features
    'vgg': Vgg16Model,
    'inception':InceptionModel,
    'resnet':ResnetModel,
    'mobile_v2':MobilentModel,

}

def get_names():
    return __factory.keys()

def init_imgmodel(name_feaure,save_features,images,images_directory,transform=None, *args, **kwargs):

    if name_feaure not in __factory.keys():
        raise KeyError("Unknown img Model: {}".format(name_feaure))


    return __factory[name_feaure](images,images_directory,transform,save_features,*args, **kwargs)







if __name__ == '__main__':
    # test

    Model = Vgg16Model()
