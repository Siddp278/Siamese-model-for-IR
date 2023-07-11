import re
import nltk 
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class Language:
    def __init__(self):
        """ 
        Language class keeps track of the datasets vocabulary and creates 
        a words to index dictionary that will be required in the pytroch dataset
        """
        self.word2index = {'unk':0}  # sets index accodringly to unique ness - most common lower index e.g.1 
        self.word2count = {'unk':0}  # counts each unique word 
        self.index2word = {0: 'unk'}  # reverse of word3index
        self.n_words = 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.word2count[word] = 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def text_to_wordlist(text, remove_stopwords, stem_words):

    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    text = text.strip()
    return text, len(text)


def convert_data_to_tuples(df, remove_stopwords, stem_words):
    questions_pair = []
    labels = []
    max_length = 0
    for _, row in df.iterrows():

        q1, l1 = text_to_wordlist(str(row['question1']), remove_stopwords, stem_words)
        q2, l2 = text_to_wordlist(str(row['question2']), remove_stopwords, stem_words)
        if max_length < l1 and max_length < l2:
            if l1 < l2:
                max_length = l2
            else:
                max_length = l1 
        if max_length > l1 and max_length < l2: 
            max_length = l2 
        if max_length < l1 and max_length > l2:
            max_length = l1             
        label = int(row['is_duplicate'])
        if q1 and q2:
            questions_pair.append((q1, q2))
            labels.append(label)

    print ('Question Pairs: ', len(questions_pair))
    return questions_pair, labels, max_length


def word_to_embed(q_pair, labels, language, threshold, max_length):
    assert len(q_pair) == len(labels)
    q1_pairs = []
    q2_pairs = []
    for q1, q2 in q_pair:
        q1_indices = []
        q2_indices = []
        for word in q1.split():
            if language.word2count.get(word, -1) > threshold: 
                q1_indices.append(language.word2index[word])
            else:
                q1_indices.append(language.word2index['unk'])    
        for word in q2.split():
            if language.word2count.get(word, -1) > threshold: 
                q2_indices.append(language.word2index[word])
            else:
                q2_indices.append(language.word2index['unk']) 
        q1_indices.extend([0]*(max_length - len(q1_indices)))
        q2_indices.extend([0]*(max_length - len(q2_indices)))
        q1_pairs.append(q1_indices)
        q2_pairs.append(q2_indices)  
        # print(q1_pairs)         
    return q1_pairs, q2_pairs, labels    


class data_(Dataset):
    def __init__(self, xq1, xq2, y):
        self.xq1 = xq1
        self.xq2 = xq2
        self.y = y
        
    def __len__(self):
        return len(self.xq1)
    
    def __getitem__(self, index):
        # assert len(self.xq1[index]) == 145 and len(self.xq2[index]) == 145, f"length is wrong: {len(self.xq1[index]), len(self.xq2[index])}"
        # assert len(self.y[index]) == 1, f"y is wrong: {len(self.y[index])}"
        return self.xq1[index], self.xq2[index], self.y[index]


def dataset_generator(X, y, batch_size, split=0.1, max_length=None):
    xq1, xq2 = X
    l = 1-split
    xq1 = torch.Tensor(xq1)
    xq2 = torch.Tensor(xq2)
    y = torch.Tensor(y)
    train_loader = DataLoader(data_(xq1[:int(l*len(xq1))], xq2[:int(l*len(xq1))], y[:int(l*len(xq1))]), 
                              shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(data_(xq1[int(l*len(xq1)):], xq2[int(l*len(xq1)):], y[int(l*len(xq1)):]), 
                             shuffle=False, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader