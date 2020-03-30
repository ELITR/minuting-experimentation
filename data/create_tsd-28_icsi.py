
''' converting recovered ICSI data in the format they use in TSD-28 '''

import os, sys, glob, re
import string, argparse
import xml.etree.ElementTree as ET
from collections import OrderedDict
import nltk
from sklearn.cluster import KMeans
from nltk import PerceptronTagger
from nltk import tokenize
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

# find paths of certain files under a source directory
def list_paths(src_dir, filt_string):
	'''
	:param src_dir: the source directory where to search for files
	:param filt_string: the pattern of file names to look for
	:return file_list: list of file paths
	'''
	read_paths = "{}/*/{}".format(src_dir, filt_string)
	file_list = glob.glob(read_paths)
	return file_list

# function that reads the dialogues das and abstractive summaries from all files
def read_records(file_list):
	'''
	:param file_list: list of file paths to read
	:return rec_list: list of records to return
	'''
	rec_list = []
	for f_name in file_list:
		f = open(f_name, 'rt')
		text = f.read()
		text = text.strip()
		rec_list.extend(text.split('\n\n'))
		f.close()
	return rec_list

# write the string of the list each in one line of the file
def write_list(in_list, out_file):
	'''
	:param in_list: the list with the strings 
	:param out_file: the path of the file where to write
	'''
	out_f = open(out_file, 'wt')
	out_string = '\n'.join(in_list)
	out_f.write(out_string)
	return

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./src', help='source folder')
parser.add_argument('--dest', default='./dest', help='destination folder')
args = parser.parse_args()

da_list, src_list, abst_list = [], [], []
file_list = list_paths(args.src, 'abst_summs.txt')
rec_list = read_records(file_list)

# spliting the 3 parts of each record in the corresponding lists
for rec in rec_list:
	try:
		# split in 3 parts
		temp_list = rec.split('abst_sum - ')
		a = temp_list[0]
		b = temp_list[1]
	except:
		continue
	# clean and add the pieces
	src = clean_text(a, keep_speakers=False, lower=True, eos=True)
	abst = clean_text(b, keep_speakers=False, lower=True)
	src_list.append(src)
	abst_list.append(abst)

if not os.path.exists(args.dest):
	os.makedirs(args.dest)

# form the paths of destination files
write_src = "{}/{}".format(args.dest, "in")
write_abst = "{}/{}".format(args.dest, "sum")

# write lists in each file
write_list(src_list, write_src)
write_list(abst_list, write_abst)
