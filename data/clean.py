
import os, sys, codecs, re, string, argparse
import nltk
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

# removes dialogue speakers of the form A:, B: etc
def remove_speakers(text):
	text = re.sub(r'[A-Z]:', '', text)
	return text

# lowercase and tokenize
def clean_text(text, speakers=True, lower=False):
	'''
	:param text: long string of input text
	:param lower: lowercase or not the input text - boolean
	:param speakers: keep or remove speakers from dialogues - boolean
	:return clean_text: long string of lowercased and tokenized input string
	'''
	# remove speakers from dialogues if required
	if not speakers:
		text = remove_speakers(text)

	# strip extra white space
	text = re.sub(r' +', ' ', text)
	# strip leading and trailing white space
	text = text.strip()

	# convert to lowercase if required
	if lower:
	    text = text.lower()

	# tokenize with core_tokenize
	clean_text = core_tokenize(text)
	return clean_text

# main 
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--src_file', default='./src.txt', help='src_file')
	args = parser.parse_args()

	# create name of output file
	path, in_file = os.path.split(args.src_file)
	out_file= os.path.join(path, 'clean_' + in_file)

	# read text from input file
	with open(args.src_file, 'rt') as in_f:
		in_text = in_f.read()

	# apply cleaning operations
	out_text = clean_text(in_text, speakers=False, lower=True)

	# write to output file
	with open(out_file, 'wt') as out_f:
		out_f.write(out_text)




