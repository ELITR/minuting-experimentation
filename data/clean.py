import os, sys, codecs, re, string
import nltk
import word_attraction
import numpy as np
from sklearn.cluster import KMeans
from nltk import PerceptronTagger
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import *

# function that tokenizes text same as Stanford CoreNLP
def core_tokenize(text, alb=False):
    ''' Takes a text string and returns tokenized string using NLTK word_tokenize 
    same as in Stanford CoreNLP. space, \n \t are lost. "" are replace by ``''
    '''
    # tokenize | _ ^ / ~ + = * that are not tokenized by word_tokenize
    text = text.replace("|", " | ")
    text = text.replace("_", " _ ")
    text = text.replace("^", " ^ ")
    text = text.replace("/", " / ")
    text = text.replace("+", " + ")
    text = text.replace("=", " = ")
    text = text.replace("~", " ~ ")
    text = text.replace("*", " * ") 
   
    # tokenize with word_tokenize preserving lines similar to Stanford CoreNLP
    tokens = word_tokenize(text, preserve_line=True)

    # fix the unsplit . problem and Albanian language short forms
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

        # corrections for albanian texts -- may add n' | t'
        if alb:
            p = re.match(r"(s' | c' | ç')([\w]+)", tok, re.VERBOSE) 
            if p:
                tokens[i] = ' '.join([p.group(1), p.group(2)])

    # put all tokens together
    text = ' '.join(tokens)
    # remove double+ spaces
    text = re.sub(r'\s{2,}', " ", text)

    # # optional lowercase
    # text = text.lower()
    
    # # remove special characters by performing encode-decode in ascii
    # text = text.encode('ascii', 'ignore').decode('ascii')

    return text

# lowercase and tokenize
def clean_text(text):
	'''
	:param text: long string of input text
	:return clean_text: long string of lowercased and tokenized input string
	'''
    # convert to lower case
    text = text.lower()
    # strip extra white space
    text = re.sub(' +', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize with core_tokenize
    clean_text = core_tokenize(text)
    return clean_text

