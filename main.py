import os
import argparse
import numpy as np
import pickle
import torch

from torchvision import transforms as T
from utils import  AverageMeter,accuracy
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from rouge_score import rouge_scorer



from torch.autograd import Variable
from torch.optim import lr_scheduler
from itertools import zip_longest,filterfalse


import Data_Manager
import Img_Features
import Text_Features
import Data_Preparing
from data_loader import Load_Data
import models
from PIL import Image
import torchvision.models as models_v
import encoder_model

parser = argparse.ArgumentParser(description='Train image with captian')
parser.add_argument('-d', '--dataset', type=str, default='coco',help='Flickr,ERI,FERI,coco,scol,strt,stsc',choices=Data_Manager.get_names())
parser.add_argument('--split_mode', type=str, default='file',help='File,split ')
parser.add_argument('--model_path', type=str, default='logs/', help='path for saving trained models')
parser.add_argument('--img_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")

parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--use-encoder',type=bool, default=True, help="use encoder")
parser.add_argument('--encoder-dim', default=2046, help="use encoder")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-batch', default=100, type=int, help="train batch size encoder 64")
parser.add_argument('--test-size', default=100, type=int, help="test size")

parser.add_argument('--img-fet', default='mobile_v2', type=str, help='vgg,mobile_v2',choices=Img_Features.get_names())
parser.add_argument('--threshold', type=int, default=3, help='minimum word count threshold')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


    print("==========\nArgs:{}\n==========".format(args))
    #split dataset get train , test
    dataset = Data_Manager.init_dataset(name=args.dataset,split=args.split_mode)
    #resize image to convert to tensor
    transform = T.Compose([
        T.Resize((args.img_size,args.img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    text_feature_pkl = args.dataset
    img_features_pkl = args.dataset + '_' + args.img_fet
    #get image features
    img_fet=Img_Features.init_imgmodel(args.img_fet,img_features_pkl,dataset.images,dataset.images_root,transform)
    features=img_fet.features
    dataset2 = Data_Manager.init_dataset(name='ERI', split=args.split_mode)
    img_features_pkl2 = "ERI" + '_' + args.img_fet
    img_fet2 = Img_Features.init_imgmodel(args.img_fet, img_features_pkl2, dataset2.images, dataset2.images_root, transform)
    features2 = img_fet2.features
    #get vocab in caption and text features
    vocab,text_fet=Text_Features.init_textmodel(dataset.captions,args.threshold,text_feature_pkl)

    pad_idx=vocab.word2idx['<pad>']


    train_Data=DataLoader(
        Load_Data(Data_Preparing.prepare_data(name=args.dataset, all_text=text_fet,split='train',imgs=dataset.trainimg, features=features)),
        batch_size=args.train_batch, shuffle=True,num_workers=args.num_workers,
        drop_last=True,collate_fn=lambda batch: pad_text(batch, split="train"),)

    test_Data=DataLoader(
        Load_Data(Data_Preparing.prepare_data(name=args.dataset, all_text=text_fet,split='test',imgs=dataset.testimg, features=features)),
        batch_size=args.test_size, shuffle=True,num_workers=args.num_workers,
        drop_last=True,collate_fn=lambda batch: pad_text(batch, split="test"),)



    model_name='sabri'

    #model_name = 'top'
    model=models.init_model(name=model_name,num_classes=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)



    total_step = len(train_Data)
    old_loss=None
    old_bleu=None
    for epoch in range(args.num_epochs):

        if (epoch+1)==100:
            torch.save(model.state_dict(),
                   os.path.join(args.model_path, 'ERI_epoch100.pth'))



        old_loss=train(train_Data, model, criterion, optimizer,epoch,total_step,old_loss,model_name)
        if (epoch + 1) % 2== 0 or (epoch + 1) == args.num_epochs:

            caption=print_sample(features2, model, vocab,transform)
            old_bleu= test(test_Data, model, criterion,  total_step,vocab, old_bleu)
            #evaluate(test_Data, model, criterion,  total_step,vocab, old_bleu)







def train(train_Data,model,criterion,optimizer,epoch,total_step,old_loss,model_name):
    model.train()

    for batch_idx, (img_feat, text_features,length,  img_names) in enumerate(train_Data):


        img_feat, text_features = img_feat.cuda(), text_features.cuda()


        # Forward, backward and optimize
        outputs = model(img_feat, text_features,length)
        targets = pack_padded_sequence(text_features, length, batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, length, batch_first=True)[0]




        loss = criterion(outputs, targets)
        ##loss = criterion(outputs.reshape(-1, outputs.shape[2]), text_features.reshape(-1))
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()


        if batch_idx % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, args.num_epochs, batch_idx, total_step, loss.item()))
        if old_loss is None:
            old_loss=loss.item()
        is_best=loss.item() < old_loss
        if is_best:
            old_loss=loss.item()
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, '{}_best.pth'.format(args.dataset)))


        # Save the model checkpoints
        if (batch_idx + 1) % args.save_step == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, 'seq-{}-{}-{}.pth'.format(epoch + 1, batch_idx + 1,args.dataset)))


def test(test_Data,model,criterion,total_step,vocab,old_bleu):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()


    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    with torch.no_grad():
        for batch_idx, (img_feat, text_features,length,  img_names,all_caps) in enumerate(test_Data):
            img_feat, text_features = img_feat.cuda(), text_features.cuda()

            # Forward, backward and optimize
            outputs = model(img_feat, text_features,length)
            outputs_copy = outputs.clone()
            targets = pack_padded_sequence(text_features, length, batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, length, batch_first=True)[0]
            loss = criterion(outputs, targets)
            # Keep track of metrics
            losses.update(loss.item(),sum(length))
            top5 = accuracy(outputs, targets, 5)
            batch_time.update(time.time() - start)
            start = time.time()


            for j in range(len(all_caps)):
                img_caps = all_caps[j]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {vocab.word2idx['<start>'], vocab.word2idx['<pad>'],vocab.word2idx['<end>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)
            # Hypotheses
            _, preds = torch.max(outputs_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:length[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
        all_refs_decoded = [
            [[vocab.idx2word[idx] for idx in sentence] for sentence in document]
            for document in references
        ]
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        all_hyps_decoded = [[vocab.idx2word[idx] for idx in sentence] for sentence in hypotheses]

        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0))
        bleu4 = corpus_bleu(references, hypotheses,  weights=(0.25, 0.25, 0.25, 0.25))
        meteor = sum(
            meteor_score(
                [" ".join(ref) for ref in refs],  # refs → list of strings
                " ".join(hyp)  # hyp  → string
            )
            for refs, hyp in zip(all_refs_decoded, all_hyps_decoded)
        ) / len(hypotheses)

        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        gts = {
            i: [" ".join(ref) for ref in ref_list]
            for i, ref_list in enumerate(all_refs_decoded)
        }

        res = {
            i: [" ".join(hyp)]
            for i, hyp in enumerate(all_hyps_decoded)
        }

        cider = Cider()
        cider_score, _ = cider.compute_score(gts, res)
        #spice_score, _ = Spice().compute_score(gts, res)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        total_rougeL = 0.0
        count = 0

        for key in res.keys():
            gt_text = gts[key]
            res_text = res[key][0]
            max_rouge = 0
            for gt in gt_text:
                scores = scorer.score(gt, res_text)  # ref is string now
                if (max_rouge < scores['rougeL'].fmeasure):
                    max_rouge = scores['rougeL'].fmeasure



            total_rougeL += max_rouge
            count += 1
        avg_rougeL = total_rougeL / count

        if old_bleu is None:
            old_bleu = bleu4
        is_best= bleu4 > old_bleu
        if is_best:
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, '{}_bestbleu4.pth'.format(args.dataset)))

        print(
            '\n * LOSS - {loss.avg:.3f}, BLEU-1 - {bleu1}-BLEU-2 - {bleu2}-BLEU-3 - {bleu3}-BLEU-4 - {bleu4}- meteor - {meteor} - cider: {cider_score} -avg_rougeL : {avg_rougeL}\n'.format(
                loss=losses,
                bleu1=bleu1,
                bleu2=bleu2,
                bleu3=bleu3,
                bleu4=bleu4,
                meteor=meteor,
                cider_score=cider_score,
                avg_rougeL=avg_rougeL))
    return old_bleu


def evaluate(test_Data,model,criterion,total_step,vocab,old_bleu):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()


    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    with torch.no_grad():
        for batch_idx, (img_feat, text_features,length,  img_names,all_caps) in enumerate(test_Data):
            img_feat, text_features = img_feat.cuda(), text_features.cuda()

            # Forward, backward and optimize
            outputs = model.evaluate(img_feat, text_features,length)
            outputs_copy = outputs.clone()
            targets = pack_padded_sequence(text_features, length, batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, length, batch_first=True)[0]
            loss = criterion(outputs, targets)
            # Keep track of metrics
            losses.update(loss.item(),sum(length))
            top5 = accuracy(outputs, targets, 5)
            batch_time.update(time.time() - start)
            start = time.time()


            for j in range(len(all_caps)):
                img_caps = all_caps[j]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {vocab.word2idx['<start>'], vocab.word2idx['<pad>'],vocab.word2idx['<end>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)
            # Hypotheses
            _, preds = torch.max(outputs_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:length[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0))
        bleu4 = corpus_bleu(references, hypotheses,  weights=(0.25, 0.25, 0.25, 0.25))


        if old_bleu is None:
            old_bleu = bleu4
        is_best= bleu4 > old_bleu
        if is_best:
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, '{}_bestbleu4.pth'.format(args.dataset)))


        print(
            '\n * LOSS - {loss.avg:.3f}, BLEU-1 - {bleu1}-BLEU-2 - {bleu2}-BLEU-3 - {bleu3}-BLEU-4 - {bleu4}\n'.format(
                loss=losses,
                bleu1=bleu1,
                bleu2=bleu2,
                bleu3 = bleu3,
                bleu4=bleu4))
    return old_bleu



def print_sample(img_fet, model, vocab, transform):
    import re
    #test_img1 = "COCO_train2014_000000000368"
    #test_img2 = "COCO_train2014_000000000722"
    #test_img3 = "COCO_train2014_000000000927"
    #test_img4 = "COCO_train2014_000000000984"
    #test_img5 =  "COCO_train2014_000000001014"
    #test_img1 = "762947607_2001ee4c72"
    #test_img2 = "861795382_5145ad433d"
    #test_img3 = "35506150_cbdb630f4f"
    test_img1 = "dngr_02290"
    test_img2 = "strt_01769"
    test_img3 = "scol_01253"
    test_img4 = "reef_03092"



    feature1 = img_fet[test_img1]
    feature2 = img_fet[test_img2]
    feature3 = img_fet[test_img3]
    feature4 = img_fet[test_img4]
    #feature5 = img_fet[test_img5]
    model.eval()

    caption1 = model.caption_image(feature1.cuda(), vocab)
    caption2 = model.caption_image(feature2.cuda(), vocab)
    caption3 = model.caption_image(feature3.cuda(), vocab)
    caption4 = model.caption_image(feature4.cuda(), vocab)
    #caption5 = model.caption_image(feature5.cuda(), vocab)
    print(desegment(" ".join(caption1)))
    print(desegment(" ".join(caption2)))
    print(desegment(" ".join(caption3)))
    print(desegment(" ".join(caption4)))
    #print(desegment(" ".join(caption5)))


"""def print_sample(img_fet, model, vocab, transform):
    import re
    #test_img1 = "2501968935_02f2cd8079.jpg"
    #test_img2 = "2380765956_6313d8cae3.jpg"
    #test_img3 = "2744330402_824240184c.jpg"
    #test_img1="strt_01567.jpeg"
    #test_img2 ="reef_03127.jpeg"
    #test_img3 ="scol_00928.jpeg"
    #test_img4 ="strt_00538.jpeg"
    #test_img1 = "dngr_02290.jpeg"
    #test_img2 ="strt_01680.jpeg"
    #test_img3 ="strt_01769.jpeg"
    #test_img4 = "scol_01253.jpeg"
    #test_img5 ="reef_03092.jpeg"
    #test_img6 ="1149179852_acad4d7300.jpg"
    #test_img1 = "strt_00769.jpeg"
    #test_img1 = "dngr_02913.jpeg"
    #test_img2 = "strt_00590.jpeg"
    #test_img3 = "strt_00741.jpeg"
    #test_img4 = "strt_02095.jpeg"
    #test_img5 = "scol_01430.jpeg"
    #test_img6 = "scol_01206"
    #test_img7 = "reef_03161"

    #temp1 = test_img1.split('.')[0]
    #temp2 = test_img2.split('.')[0]
    #temp3 = test_img3.split('.')[0]
    #temp4= test_img4.split('.')[0]
    #temp5 = test_img5.split('.')[0]
    #temp6 = test_img6.split('.')[0]
    #temp7 = test_img7.split('.')[0]

    feature1 = img_fet[temp1]
    feature2 = img_fet[temp2]
    feature3 = img_fet[temp3]
    feature4 = img_fet[temp4]
    feature5 = img_fet[temp5]
    feature6 = img_fet[temp6]
    feature7 = img_fet[temp7]

    model.eval()

    caption1 = model.caption_image(feature1.cuda(), vocab)
    caption2 = model.caption_image(feature2.cuda(), vocab)
    caption3 = model.caption_image(feature3.cuda(), vocab)
    caption4 = model.caption_image(feature4.cuda(), vocab)
    caption5 = model.caption_image(feature5.cuda(), vocab)
    caption6 = model.caption_image(feature6.cuda(), vocab)
    caption7 = model.caption_image(feature7.cuda(), vocab)



    print(desegment(" ".join(caption1)))
    print(desegment(" ".join(caption2)))
    print(desegment(" ".join(caption3)))
    print(desegment(" ".join(caption4)))
    print(desegment(" ".join(caption5)))
    print(desegment(" ".join(caption6)))
    print(desegment(" ".join(caption7)))"""

def desegment(line):
    line = line.replace("+ ", "")
    line = line.replace(" +", "")
    line=line.replace("+", "")
    return line

def pad_text(data,split):


    data.sort(key=lambda x: len(x[1]), reverse=True)
    ##img_f, text_f, img_names=zip(*data)
    img_f = [item[0] for item in data]
    img_f = torch.cat(img_f, dim=0)
    text_f = [item[1] for item in data]
    lengths=[len(cpt) for cpt in text_f]
    max_len=max(lengths)
    img_names = [item[2] for item in data]



    padeed_text=[torch.cat([torch.tensor(text),torch.zeros((max_len - len(text)),dtype=torch.int64)]) for text in text_f]
    #img_f = torch.stack(img_f, 0)
    #img_f = torch.squeeze(img_f)
    padeed_text=torch.stack(padeed_text,0)
    if split=="test":
        allcaps = [item[3] for item in data]
        return img_f, padeed_text, lengths, img_names,allcaps

    return img_f, padeed_text,lengths,img_names







if __name__ == '__main__':
    main()

