
''' toy example of sentence clustering from:
https://pythonprogramminglanguage.com/kmeans-text-clustering/
'''
import os, sys, codecs, re, string, argparse, nltk
import numpy as np
from sklearn.cluster import KMeans
from nltk import PerceptronTagger
from nltk import tokenize
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# function that tokenizes text same as Stanford CoreNLP
def core_tokenize(text):
    ''' Takes a text string and returns tokenized string using NLTK word_tokenize 
    same as in Stanford CoreNLP. space, \n \t are lost. "" are replace by ``''
    '''
    # tokenize | _ ^ / ~ + = * that are not tokenized by word_tokenize
    text = text.replace("|", " | ") ; text = text.replace("_", " _ ")
    text = text.replace("^", " ^ ") ; text = text.replace("/", " / ")
    text = text.replace("+", " + ") ; text = text.replace("=", " = ")
    text = text.replace("~", " ~ ") ; text = text.replace("*", " * ") 
   
    # tokenize with word_tokenize preserving lines similar to Stanford CoreNLP
    tokens = word_tokenize(text, preserve_line=True)

    # fix the unsplit . problem 
    for i, tok in enumerate(tokens):
        if tok == '...':
            continue
        # double match
        if re.match(r'[^.\s]{2,}\.[^.\s]{2,}', tok):
            tokens[i] = tok.replace('.', ' . ')
        # left match
        if re.match(r'[^.\s]{2,}\.', tok):
            tokens[i] = tok.replace('.', ' . ')
        # right match
        if re.match(r'\.[^.\s]{2,}', tok):
            tokens[i] = tok.replace('.', ' . ')

    # put all tokens together
    text = ' '.join(tokens)
    # remove double+ spaces
    text = re.sub(r'\s{2,}', " ", text)

    # # optional lowercase
    # text = text.lower()
    
    # # remove special characters by performing encode-decode in ascii
    # text = text.encode('ascii', 'ignore').decode('ascii')

    return text

# removes dialogue speakers of the form A:, B: etc
def remove_speakers(text):
    '''
    :param text: long string of input text
    :return clean_text: text after removing speakers
    '''
    clean_text = re.sub(r'[A-Z]:', '', text)
    return clean_text

# run with keep_speakers=False, lower=True to replicate TSD-28-
def clean_text(text, keep_speakers=True, lower=False, eos=False):
    '''
    :param text: long string of input text
    :param lower: lowercase or not the input text - boolean
    :param speakers: keep or remove speakers from dialogues - boolean
    :param eos: add or not <EOS> after each sentence
    :return clean_text: long string of lowercased and tokenized input string
    '''
    # remove speakers from dialogues if required
    if not keep_speakers:
        text = remove_speakers(text)

    # strip leading and trailing white space
    text = text.strip()

    # convert to lowercase if required
    if lower:
        text = text.lower()

    # split in sentences 
    sent_list = tokenize.sent_tokenize(text)

    # tokenize each sentence with core_tokenize
    sent_list = list(map(core_tokenize, sent_list))

    # correct speakers A : to A:
    sent_list = [re.sub(r'([a-zA-Z])\s+(:)', r'\1\2', sent) for sent in sent_list]

    # correct description < other > to <other>
    sent_list = [re.sub(r'<\s(([\w]+))\s>', r'<\1>', sent) for sent in sent_list]

    # add <EOS> at the end of each sentence
    if eos:
        sent_list = [s + ' <EOS>' for s in sent_list]

    # join together all sentences in one text
    clean_text = ' '.join(sent_list)

    return clean_text

# split transcript into list of sentences 
def get_sents_from_trans(trans, keep_speakers=True):
    '''
    :param trans: the text of the transcript
    :return sent_lst: list of transcript sentences
    '''
    # remove speakers from dialogues if required - keep or remove...?
    if not keep_speakers:
        text = remove_speakers(text)
    # strip leading and trailing white space
    text = text.strip()

    # split in sentences 
    sent_lst = tokenize.sent_tokenize(trans)

    # tokenize each sentence with core_tokenize
    sent_lst = list(map(core_tokenize, sent_lst))

    # correct speakers A : to A:
    sent_lst = [re.sub(r'([a-zA-Z])\s+(:)', r'\1\2', sent) for sent in sent_lst]

    return sent_lst


documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)

## methods for finding optimal number of clusters
# distortions, distances = [], [] 
# K = range(3, 8)
# print(X)
# print(type(X))
# # iterate to find optimal number of clusters (3 - 7)
# for n_c in K:
#   model = KMeans(n_clusters=n_c, init='k-means++', max_iter=100, n_init=1)
#   model.fit(X)
#   # distortions.append(model.inertia_)
#   print(model.cluster_centers_)
#   print(type(model.cluster_centers_))
#   distortions.append(sum(np.min(cdist(np.array(X), model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')
# optimum_nc = kn.knee

# # elbow method for finding the optimal number of clusters
# for i in range(0, 5):
#   p1 = Point(initx=1,inity=distortions[0])
#   p2 = Point(initx=5,inity=distortions[4])
#   p = Point(initx=i+1,inity=distortions[i])
#   distances.append(p.distance_to_line(p1,p2))
# optimum_nc = distances.index(max(distances)) + 3


