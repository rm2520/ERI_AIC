from collections import Counter
import pickle
import os
import os.path as osp
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('corpus')
from arabic_preprocessing import Arabic_preprocessing

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class Text_Model(object):
    def __init__(self,caption,threshold):
        counter = Counter()
        arabic_process = Arabic_preprocessing()
        #prprocess each caption for each image and save word counts its frequency
        for img, cpt in caption.items():
            processed_captions = [arabic_process.preprocess_arabic_text(c) for c in cpt]
            caption[img] = processed_captions

            tokens =[nltk.tokenize.word_tokenize(processed_cpt)  for processed_cpt in processed_captions]
            [counter.update(token) for token in tokens ]
        #get word that frequecy >thresgold
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        x = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1]) if v>=10}
        file = open('frequency.txt', 'w', encoding='utf-8')
        for k, v in x.items():
            file.write(k)
            file.write(':')
            file.write(str(v))

            file.write('\n')
        file.close()

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word)
        self.vocab=vocab





def init_textmodel(cpts,threshold,save_features, *args, **kwargs):
    vocab_feat_pkl=save_features+'_ vocab_'+str(threshold)+' .pkl'
    tokenized_feat_pkl=save_features+'_ tokenized .pkl'
    # get vocab in captions
    if not (osp.exists(os.path.join('logs/', vocab_feat_pkl))):
        vocab_feat = Text_Model(cpts, threshold)
        pickle.dump(vocab_feat, open(os.path.join('logs/', vocab_feat_pkl), 'wb'))
    else:
        vocab_feat = pickle.load(open(os.path.join('logs/', vocab_feat_pkl), 'rb'))



    if not (osp.exists(os.path.join('logs/', tokenized_feat_pkl))):
        tokenized_cpts = {}
        for img, cpt in cpts.items():
            for c in cpt:
                tokenized_captions=( [vocab_feat.vocab(token) for token in c.split()])
                tokenized_captions.insert(0,vocab_feat.vocab('<start>'))
                tokenized_captions.insert(len(tokenized_captions),vocab_feat.vocab('<end>'))

                if img not in tokenized_cpts:
                    tokenized_cpts[img] = [tokenized_captions]
                else:
                    tokenized_cpts[img].append(tokenized_captions)
        pickle.dump(tokenized_cpts, open(os.path.join('logs/', tokenized_feat_pkl), 'wb'))

    else:
        tokenized_cpts = pickle.load(open(os.path.join('logs/', tokenized_feat_pkl), 'rb'))





    return vocab_feat.vocab,tokenized_cpts























if __name__ == '__main__':
    # test

    Model = Text_Model()
