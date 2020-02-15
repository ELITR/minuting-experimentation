
'''
Script that walks the folder structure of rec_ami/rec_icsi, reads certain 
files (e.g. abst_summs.txt or compl_extr_summ.txt) of each meeting and 
stores the dialogues-summaries in rec_ami-icsi_src-tgt/ folder.
'''

import os, sys, fnmatch, re, json
from shutil import *
from nltk import tokenize
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

# write list of dicts (json lines) as lines in a file path
def dictlst_to_file_lines(file_path, line_lst):
    '''write list of dicts in a file path - change to "a" to append'''
    outf = open(file_path, "w+", encoding='utf-8')
    for itm in line_lst:
        json.dump(itm, outf)
        outf.write('\n')
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

# parser = argparse.ArgumentParser()
# parser.add_argument('--inpath', required=True, help='input folder for reading')
# parser.add_argument('--outpath', required=True, help='output folder for writing')
# args = parser.parse_args()

read_path = "./rec_ami/"
# read_path = "./rec_icsi/"
write_path = "./rec_ami-icsi_src-tgt"
samp_lst = []

# i = 0
if __name__=="__main__":
	# read all lines from file
	file_lines_to_list(file_path)

