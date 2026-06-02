import torch
import torchvision
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
import os
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SeqtoSeq(nn.Module):

    def __init__(self, num_classes,img_dim, **kwargs):
        super(SeqtoSeq, self).__init__()
        middle_dim = 256
        hidden_size =512
        num_layers=1
        self.bn = nn.BatchNorm1d(middle_dim, momentum=0.01)
        self.img_embedding = nn.Linear(img_dim, middle_dim)
        self.tanh=nn.Tanh()
        self.relu = nn.ReLU()
        self.text_embedding= nn.Embedding (num_classes,middle_dim)
        self.lstm = nn.LSTM(middle_dim, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)




    def forward(self, img_fet,text_fet,length):

        image_embedding= self.bn(self.img_embedding(img_fet))
        image_embedding=self.tanh(image_embedding)
        text_embedding=self.text_embedding(text_fet)
        embeddings = torch.cat((image_embedding.unsqueeze(1), text_embedding), dim=1)

        packed = pack_padded_sequence(embeddings, length, batch_first=True)



        hiddens, _ = self.lstm(packed)
        if not self.training:
            hiddens_unpack, _= nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
            outputs = self.fc1(hiddens_unpack)
            return outputs





        outputs=self.fc1(hiddens[0])






        return outputs

    def caption_image(self,img_fet,vocabulary,max_length=20):
        state=None
        result_caption = []

        image_embedding = self.img_embedding(img_fet)
        image_embedding = self.tanh(image_embedding).unsqueeze(0)
        inputs=image_embedding


        for _ in range(max_length):
            hiddens, state = self.lstm(inputs, state)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc1(hiddens.squeeze(0))  # outputs:  (batch_size, vocab_size)
            predicted = outputs.argmax(1)  # predicted: (batch_size)
            result_caption.append(predicted.item())
            inputs = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(0)  # inputs: (batch_size, 1, embed_size)
            if vocabulary.idx2word[predicted.item()] == "<end>":
                break


        return [vocabulary.idx2word[idx] for idx in result_caption]



class Jundi(nn.Module):
    #jundi
    def __init__(self, num_classes, **kwargs):
        super(Jundi, self).__init__()
        self.middle_dim = 256
        self.hidden_size = 512
        self.num_classes=num_classes
        feature_dim=1280
        num_layers=1
        dropout=0.5
        self.img_embedding = nn.Linear(feature_dim, self.middle_dim)


        self.relu = nn.ReLU()
        self.tanh =nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.text_embedding= nn.Embedding (num_classes,self.middle_dim)
        self.gru=nn.GRU(self.hidden_size,self.hidden_size,num_layers, batch_first=True)
        self.u_att = nn.Linear(self.middle_dim, self.hidden_size)
        self.w_att = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_att = nn.Linear(self.hidden_size, 1)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.text_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, img_fet,text_fet,length):
        batch_size = img_fet.size(0)
        img_fet = img_fet.permute(0, 2, 3, 1)




        state = None
        image_embedding= self.relu(self.img_embedding(img_fet))
        img_dim = image_embedding.size(-1)
        image_embedding = image_embedding.view(batch_size, -1, img_dim)
        text_embedding=self.text_embedding(text_fet)

        h1,c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        # Create tensors to hold word predicion scores

        predictions = torch.zeros(batch_size, max(length), self.num_classes).to(device)
        for t in range(max(length)):
            input = text_embedding[:, t, :]
            score = self.v_att(self.tanh(self.u_att(image_embedding) + self.w_att(h1.unsqueeze(1))))
            context_vector = score * image_embedding
            context_vector=torch.mean(context_vector,dim=1)
            embeddings = torch.cat((context_vector.unsqueeze(1), input.unsqueeze(1)), dim=-1)
            out, state = self.gru(embeddings, state)
            h1=state.squeeze(0)
            out =self.relu(self.fc1(out))
            out= torch.reshape(out, (-1, out.shape[2]))
            out=self.fc2(self.dropout(out))
            predictions[:, t, :] = out

        return predictions


    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.hidden_size).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c


    def caption_image(self,img_fet,vocabulary,max_length=20):

        result_caption = []
        img_fet = img_fet.view(img_fet.size(0), img_fet.size(2) * img_fet.size(3), img_fet.size(1))
        state = None
        image_embedding = self.relu(self.img_embedding(img_fet))
        text=  torch.tensor([vocabulary.word2idx['<start>']]).to(device)
        text_embedding = self.text_embedding(text)
        h1 = self.init_hidden_state(1)

        inputs=text_embedding


        for _ in range(max_length):
            score = self.v_att(self.tanh(self.u_att(image_embedding) + self.w_att(h1.unsqueeze(1))))
            context_vector = score * image_embedding
            context_vector = torch.mean(context_vector, dim=1)
            embeddings = torch.cat((context_vector.unsqueeze(1), inputs.unsqueeze(1)), dim=-1)
            out, state = self.gru(embeddings, state)  # hiddens: (batch_size, 1, hidden_size)
            h1 = state.squeeze(0)
            out = self.relu(self.fc1(out))
            out = torch.reshape(out, (-1, out.shape[2]))
            outputs = self.fc2(self.dropout(out))
            predicted = outputs.argmax(1)  # predicted: (batch_size)
            result_caption.append(predicted.item())
            inputs = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)

            if vocabulary.idx2word[predicted.item()] == "<end>":
                break


        return [vocabulary.idx2word[idx] for idx in result_caption]



class Sabri(nn.Module):
    #sabri
    def __init__(self, num_classes, **kwargs):
        super(Sabri, self).__init__()
        self.middle_dim = 256
        self.hidden_size = 512

        self.num_classes=num_classes
        feature_dim=1280
        dropout=0.5

        self.img_embedding = nn.Linear(feature_dim, self.middle_dim)
        self.relu = nn.ReLU()
        self.tanh =nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.dropout = nn.Dropout(p=dropout)
        self.text_embedding= nn.Embedding (num_classes,self.middle_dim)
        self.lstm=nn.LSTMCell(self.hidden_size , self.hidden_size, bias=True)
        #self.lstm = nn.GRUCell(self.hidden_size, self.hidden_size, bias=True)
        self.init_h = nn.Linear(self.middle_dim, self.hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.middle_dim, self.hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.u_att = nn.Linear(self.middle_dim, self.hidden_size)
        self.w_att = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_att = nn.Linear(self.hidden_size, 1)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.text_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, img_fet,text_fet,length):

        batch_size = img_fet.size(0)
        img_fet = img_fet.permute(0, 2, 3, 1)

        #encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = image_embedding.size(-1)
        image_embedding = image_embedding.view(batch_size, -1, img_dim)

        #decoder

        h1, c1 = self.init_hidden_state(image_embedding)   # (batch_size, decoder_dim)
        # Create tensors to hold word predicion scores

        predictions = torch.zeros(batch_size, max(length), self.num_classes).to(device)
        text_embedding=image_embedding.mean(dim=1)
        for t in range(max(length)):

            score = self.v_att(self.tanh(self.u_att(image_embedding) + self.w_att(h1.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (image_embedding * score).sum(dim=1)


            embeddings = torch.cat((text_embedding,context_vector ), dim=-1)
            h1, c1 = self.lstm(embeddings, (h1,c1))
            #h1 = self.lstm(embeddings, h1)
            out =self.fc1(h1)

            out=self.fc2(self.dropout(out))
            predictions[:, t, :] = out
            input = text_fet[:, t]
            text_embedding = self.text_embedding(input)

        return predictions


    def init_hidden_state(self, image_feature):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_image_out = image_feature.mean(dim=1)
        h = self.init_h(mean_image_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_image_out)
        return h, c


    def caption_image(self,img_fet,vocabulary,max_length=20):


        img_fet = img_fet.permute(0, 2, 3, 1)

        # encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = image_embedding.size(-1)
        image_embedding = image_embedding.view(1, -1, img_dim)

        # decoder

        h1, c1 = self.init_hidden_state(image_embedding)  # (batch_size, decoder_dim)
        # Create tensors to hold word predicion scores



        result_caption = []

        #text= torch.tensor([vocabulary.word2idx['<start>']]).to(device)

        #text_embedding = self.text_embedding(text)
        text_embedding=image_embedding.mean(dim=1)



        for _ in range(max_length):


            score = self.v_att(self.tanh(self.u_att(image_embedding) + self.w_att(h1.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (image_embedding * score).sum(dim=1)

            embeddings = torch.cat((text_embedding, context_vector), dim=-1)
            h1, c1 = self.lstm(embeddings, (h1, c1))
            #h1 = self.lstm(embeddings, h1)
            out = self.fc1(h1)

            out = self.fc2(self.dropout(out))
            predicted = out.argmax(1)  # predicted: (batch_size)
            result_caption.append(predicted.item())
            text_embedding = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)

            if vocabulary.idx2word[predicted.item()] == "<end>":
                break
        return [vocabulary.idx2word[idx] for idx in result_caption]


    def evaluate(self,img_fet,vocabulary,length):
        batch_size = img_fet.size(0)
        img_fet = img_fet.permute(0, 2, 3, 1)

        # encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = image_embedding.size(-1)
        image_embedding = image_embedding.view(1, -1, img_dim)

        # decoder

        h1, c1 = self.init_hidden_state(image_embedding)  # (batch_size, decoder_dim)
        # Create tensors to hold word predicion scores

        result_caption = []

        predictions = torch.zeros(batch_size, max(length), self.num_classes).to(device)
        text_embedding = image_embedding.mean(dim=1)

        for t in range(max(length)):

            score = self.v_att(self.tanh(self.u_att(image_embedding) + self.w_att(h1.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (image_embedding * score).sum(dim=1)

            embeddings = torch.cat((text_embedding, context_vector), dim=-1)
            # h1, c1 = self.gru(embeddings, (h1, c1))
            h1 = self.lstm(embeddings, h1)
            out = self.fc1(h1)

            out = self.fc2(self.dropout(out))
            predictions[:, t, :] = out
            predicted = out.argmax(1)  # predicted: (batch_size)

            text_embedding = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)



        return predictions

class TopDown(nn.Module):
    #TopDown
    def __init__(self, num_classes, **kwargs):
        super(TopDown, self).__init__()
        self.middle_dim = 256
        self.hidden_size = 512

        self.num_classes=num_classes
        feature_dim=1280
        dropout=0.5

        self.img_embedding = nn.Linear(feature_dim, self.middle_dim)
        self.relu = nn.ReLU()
        self.tanh =nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.dropout = nn.Dropout(p=dropout)
        self.text_embedding= nn.Embedding (num_classes,self.middle_dim)
        #self.lstm=nn.LSTMCell(self.hidden_size , self.hidden_size, bias=True)
        #self.lstm = nn.GRUCell(self.hidden_size, self.hidden_size, bias=True)
        self.lstm = nn.LSTMCell(self.hidden_size+feature_dim  , self.hidden_size, bias=True)
        self.top_down_attention = nn.LSTMCell(feature_dim + self.middle_dim + self.hidden_size, self.hidden_size, bias=True)
        self.init_h = nn.Linear(feature_dim, self.hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(feature_dim, self.hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.u_att = nn.Linear(feature_dim, self.hidden_size)
        self.w_att = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_att = nn.Linear(self.hidden_size, 1)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.text_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, img_fet,text_fet,length):

        batch_size = img_fet.size(0)
        img_fet = img_fet.permute(0, 2, 3, 1)

        #encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = img_fet.size(-1)
        img_fet = img_fet.view(batch_size, -1, img_dim)


        #decoder

        h1, c1 = self.init_hidden_state(img_fet)   # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(img_fet)
        # Create tensors to hold word predicion scores

        predictions = torch.zeros(batch_size, max(length), self.num_classes).to(device)

        img_mean=img_fet.mean(dim=1)


        image_embedding = image_embedding.view(batch_size, -1, self.middle_dim)
        text_embedding=image_embedding.mean(dim=1)
        for t in range(max(length)):

            h2, c2 = self.top_down_attention(torch.cat([h1, img_mean, text_embedding], dim=-1),(h2,c2))
            score = self.v_att(self.tanh(self.u_att(img_mean) + self.w_att(h2.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (img_mean * score).sum(dim=1)


            embeddings = torch.cat((h2,context_vector ), dim=-1)
            h1, c1 = self.lstm(embeddings, (h1,c1))
            #h1 = self.lstm(embeddings, h1)
            out =self.fc1(h1)

            out=self.fc2(self.dropout(out))
            predictions[:, t, :] = out
            input = text_fet[:, t]
            text_embedding = self.text_embedding(input)

        return predictions


    def init_hidden_state(self, image_feature):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_image_out = image_feature.mean(dim=1)
        h = self.init_h(mean_image_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_image_out)
        return h, c


    def caption_image(self,img_fet,vocabulary,max_length=20):


        img_fet = img_fet.permute(0, 2, 3, 1)

        # encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = img_fet.size(-1)
        img_fet = img_fet.view(1, -1, img_dim)

        # decoder

        h1, c1 = self.init_hidden_state(img_fet)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(img_fet)
        # Create tensors to hold word predicion scores



        result_caption = []

        #text= torch.tensor([vocabulary.word2idx['<start>']]).to(device)

        #text_embedding = self.text_embedding(text)
        img_mean=img_fet.mean(dim=1)
        image_embedding = image_embedding.view(1, -1, self.middle_dim)
        text_embedding=image_embedding.mean(dim=1)



        for _ in range(max_length):

            h2, c2 = self.top_down_attention(torch.cat([h1, img_mean, text_embedding], dim=1), (h2, c2))
            score = self.v_att(self.tanh(self.u_att(img_mean) + self.w_att(h2.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (img_mean * score).sum(dim=1)

            embeddings = torch.cat((h2, context_vector), dim=-1)
            h1, c1 = self.lstm(embeddings, (h1, c1))
            #h1 = self.lstm(embeddings, h1)
            out = self.fc1(h1)

            out = self.fc2(self.dropout(out))
            predicted = out.argmax(1)  # predicted: (batch_size)
            result_caption.append(predicted.item())
            text_embedding = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)

            if vocabulary.idx2word[predicted.item()] == "<end>":
                break
        return [vocabulary.idx2word[idx] for idx in result_caption]


    def evaluate(self,img_fet,vocabulary,length):
        batch_size = img_fet.size(0)
        img_fet = img_fet.permute(0, 2, 3, 1)

        # encoder
        image_embedding = self.relu(self.img_embedding(img_fet))
        img_dim = img_fet.size(-1)
        img_fet = img_fet.view(batch_size, -1, img_dim)

        # decoder

        h1, c1 = self.init_hidden_state(img_fet)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(img_fet)
        # Create tensors to hold word predicion scores

        result_caption = []

        predictions = torch.zeros(batch_size, max(length), self.num_classes).to(device)
        img_mean = img_fet.mean(dim=1)
        image_embedding = image_embedding.view(batch_size, -1, self.middle_dim)
        text_embedding = image_embedding.mean(dim=1)

        for t in range(max(length)):
            h2, c2 = self.top_down_attention(torch.cat([h1, img_mean, text_embedding], dim=1), (h2, c2))
            score = self.v_att(self.tanh(self.u_att(img_mean) + self.w_att(h2.unsqueeze(1))))
            score = self.softmax(score)
            context_vector = (img_mean * score).sum(dim=1)

            embeddings = torch.cat((h2, context_vector), dim=-1)
            h1, c1 = self.lstm(embeddings, (h1, c1))
            # h1 = self.lstm(embeddings, h1)
            out = self.fc1(h1)

            out = self.fc2(self.dropout(out))

            predictions[:, t, :] = out
            predicted = out.argmax(1)  # predicted: (batch_size)

            text_embedding = self.text_embedding(predicted)  # inputs: (batch_size, embed_size)



        return predictions








__factory = {
    'seq': SeqtoSeq,

    'top':TopDown,

    'Jundi':Jundi,
    'sabri': Sabri ,

}

def get_names():
    return __factory.keys()

def init_model(name, num_classes, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes,*args, **kwargs)