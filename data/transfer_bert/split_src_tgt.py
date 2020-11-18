
'''
Author: Erion Çano
Descri:	Script that prepares the data for transfer learning summarization using BERT
        https://github.com/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning
        The source samples are the ones of ami-icsi_train and ami-icsi_test. The source
        and target texts are separated and stored in different files.  
Langu: 	Python 3.6.9
Usage:	python split_src_tgt.py -inpath INDIR -outpath OUTDIR
'''

import os, sys, fnmatch, re, json, argparse, pickle
from shutil import *
from nltk import tokenize
import multiprocessing as mp
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import *

# read file json lines from given file path and return them in a list
def file_lines_to_list(file_path):
    '''read json lines and store them in a list that is returned'''
    with open(file_path, "r") as inf:
        # strips \n at the end of each line
        line_list = [json.loads(line) for line in inf]
    return line_list

# function that tokenizes text same as Stanford CoreNLP
def core_tokenize(text, alb=False):
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
            p = re.match(r"(s' | c' | รง')([\w]+)", tok, re.VERBOSE) 
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

# write list of dicts (json lines) as lines in a file path
def dictlst_to_file_lines(file_path, line_lst):
    '''write list of dicts in a file path - change to "a" to append'''
    outf = open(file_path, "w+", encoding='utf-8')
    for itm in line_lst:
        json.dump(itm, outf)
        outf.write('\n')
    outf.close()

# write list of string records as lines in a given file path
def strlst_to_file_lines(file_path, line_list):
    '''write list strings in a file path that is opened'''
    outf = open(file_path, "a+", encoding='utf-8')
    outf.write("\n".join(line_list))
    outf.close()

# removes dialogue speakers of the form A:, B: etc
def remove_speakers(text):
	'''
	:param text: long string of input text
	:return clean_text: text after removing speakers
	'''
	clean_text = re.sub(r'[A-Z]:', '', text)
	# strip leading and trailing white space
	clean_text = clean_text.strip()
	# filter out more than 2 consequetive spaces
	clean_text = re.sub(r' {2,}', " ", clean_text)
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

	# filter out more than 2 consequetive spaces
	clean_text = re.sub(r' {2,}', " ", clean_text)

	return clean_text

# break one transcript in several discussions 
def break_trans(trans):
	'''splits on '\n\n' and returns the list of discussions'''
	# strip leading and trailing white space
	trans = trans.strip()
	disc_lst = trans.split('\n\n')
	return disc_lst

# filter out the descriptions and get the discussion and summary as dict
def get_src_tgt(trans):
	'''returns a dict of the form {dialogue: <d>, summary: <s>}'''
	# filter out all between descriptions: and \n\n
	trans = re.sub(r'descriptions: (.*?)\n', '\n', trans)
	try:
		trans_pieces = trans.split("abst_sum - ")
		src = trans_pieces[0] ; tgt = trans_pieces[1]
	except: 
		src = "" ; tgt = ""
	src = clean_text(src, keep_speakers=False, lower=True)
	tgt = clean_text(tgt, keep_speakers=False, lower=True)
	disc = dict()
	disc["dialogue"] = src ; disc["summary"] = tgt
	return disc

# spliting text of src and tgt in tokens
def split_src_tgt(dict_item):
	new_item = dict()
	new_src_lst, new_tgt_lst = [], []
	# split the two texts
	src_lst = dict_item["abstract"].split(' ')
	tgt_lst = dict_item["title"].split(' ')
	# append the lists in new lists to comply with the required format
	new_src_lst.append(src_lst)
	new_tgt_lst.append(tgt_lst)
	# store in a new dict
	new_item["src"] = new_src_lst
	new_item["tgt"] = new_tgt_lst
	return new_item

# spliting text of src and tgt in sentences and then in tokens
def split_src_tgt2(dict_item):
	new_item = dict()
	new_src_lst, new_tgt_lst, tmp_lst = [], [], []
	# split the two texts in sentences on . and ,

	src_lst = re.split("[.,]", dict_item["src"])
	tgt_lst = re.split("[.,]", dict_item["tgt"])

	# # splitting on . and preserving .
	# src_lst = [s + '.' for s in  dict_item["abstract"].split('.')]
	# # remove trailing separator
	# src_lst[-1] = src_lst[-1].strip('.')

	# tgt_lst = [s + '.' for s in  dict_item["title"].split('.')]
	# # remove trailing separator
	# tgt_lst[-1] = tgt_lst[-1].strip('.')

	src_lst = list(filter(None, src_lst)) # filter []
	tgt_lst = list(filter(None, tgt_lst)) # filter []

	# split each sentence in tokens
	for sent in src_lst:
		tmp_lst = sent.split(' ')
		tmp_lst = list(filter(None, tmp_lst)) # remove ""
		new_src_lst.append(tmp_lst)
	for sent in tgt_lst:
		tmp_lst = sent.split(' ')
		tmp_lst = list(filter(None, tmp_lst)) # remove ""
		new_tgt_lst.append(tmp_lst)

	new_src_lst = list(filter(None, new_src_lst)) # filter []
	new_tgt_lst = list(filter(None, new_tgt_lst)) # filter []
	# store in a new dict
	new_item["src"] = new_src_lst
	new_item["tgt"] = new_tgt_lst
	return new_item

# split a list of src: src, tgt: tgt dicts in two lists of strings
def split_src_tgt(dict_lst):
	src_lst = [d["src"] for d in dict_lst if (len(d["src"]) > 20 and len(d["src"]) > 10)]
	tgt_lst = [d["tgt"] for d in dict_lst if (len(d["src"]) > 20 and len(d["src"]) > 10)]
	return src_lst, tgt_lst

parser = argparse.ArgumentParser()
parser.add_argument('--inpath', required=True, help='input folder to read')
parser.add_argument('--outpath', required=True, help='output folder to write')
args = parser.parse_args()

if __name__ == "__main__":
	# loop through files in source path (there should be train and test only)
	for file in os.listdir(args.inpath):
		read_file = os.path.join(args.inpath, file)

		# reading train file
		if file.endswith("train.txt"):
			read_lst = file_lines_to_list(read_file)
			# prepare src and tgt write files 
			train_src = args.outpath + "/" + "train_story.txt"
			train_tgt = args.outpath + "/" + "train_summ.txt"
			# split src and tgt in two lists
			src_lst, tgt_lst = split_src_tgt(read_lst)
			# write the two files 
			strlst_to_file_lines(train_src, src_lst)
			strlst_to_file_lines(train_tgt, tgt_lst)

		# reading test file
		elif file.endswith("test.txt"):
			read_lst = file_lines_to_list(read_file)
			# prepare src and tgt write files 
			test_src = args.outpath + "/" + "eval_story.txt"
			test_tgt = args.outpath + "/" + "eval_summ.txt"
			# split src and tgt in two lists
			src_lst, tgt_lst = split_src_tgt(read_lst)
			# write the two files 
			strlst_to_file_lines(test_src, src_lst)
			strlst_to_file_lines(test_tgt, tgt_lst)

		# skip other files
		else:
			print("Wrong file! Skipping...")
			continue