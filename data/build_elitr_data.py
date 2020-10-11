
'''
Author:		Erion Çano
Descript:	Script that walks the folder structure of ELITR minuting data, 
			reads files named basic_sample.txt of each meeting and builds 
			the entire ELITR minuting dataset
Language: 	Python 3.6
Usage:		python build_elitr_data.py -i INPATH -o OUTPATH
'''

import os, sys, fnmatch, re , argparse, json
from shutil import *
from tqdm import *
from nltk import tokenize, word_tokenize, sent_tokenize 

# write list of dicts (json lines) as lines in a file path
def dictlst_to_file_lines(file_path, line_lst):
    '''write list of dicts in a file path - change to "a" to append'''
    outf = open(file_path, "w+", encoding='utf-8')
    for itm in line_lst:
        json.dump(itm, outf)
        outf.write('\n')
    outf.close()
	
## function that tokenizes text same as Stanford CoreNLP
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

    # put all tokens together
    text = ' '.join(tokens)
    # remove double+ spaces
    text = re.sub(r'\s{2,}', " ", text)

    # # optional lowercase
    # text = text.lower()
    
    # # remove special characters by performing encode-decode in ascii
    # text = text.encode('ascii', 'ignore').decode('ascii')

    return text

## write list of dicts (json lines) as lines in a file path
def dictlst_to_file_lines(file_path, line_lst):
    '''write list of dicts in a file path - change to "a" to append'''
    outf = open(file_path, "w+", encoding='utf-8')
    for itm in line_lst:
        json.dump(itm, outf)
        outf.write('\n')
    outf.close()

## removes dialogue speakers of the form A:, B: etc
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

## run with keep_speakers=False, lower=True to replicate TSD-28-
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

	# # correct description < other > to <other>
	# sent_list = [re.sub(r'<\s(([\w]+))\s>', r'<\1>', sent) for sent in sent_list]

	# remove all descriptions "<desc>" to ""
	sent_list = [re.sub('<[^>]+>', '', s) for s in sent_list]

	# add <EOS> at the end of each sentence
	if eos:
		sent_list = [s + ' <EOS>' for s in sent_list]

	# join together all sentences in one text
	clean_text = ' '.join(sent_list)

	# filter out more than 2 consequetive spaces
	clean_text = re.sub(r' {2,}', " ", clean_text)

	return clean_text

## walk the entire data path and get the content file paths
def get_file_paths(read_path):
	file_paths = []
	for root, dir, files in os.walk(read_path):
		for file in files:
			if file.endswith("basic_sample.txt"):
				infile = os.path.join(root, file)
				file_paths.append(infile)
	return file_paths

## filter out the descriptions and get the discussion and summary as dict
def get_src_tgt(content):
	'''returns a dict of the form {dialogue: <d>, summary: <s>}'''
	content = content.strip()	# remove leading and trailing whitespaces
	content = content.split("<transcript>")[1]		# remove <transcript>
	## get the important pieces
	try:
		src = content.split("<summary>")[0]			# get the transcript
		tgt = content.split("<summary>")[1]			# get the summary
	except:
		src = "" ; tgt = ""	
	src = clean_text(src, keep_speakers=False, lower=True)	# clean the src
	tgt = clean_text(tgt, keep_speakers=False, lower=True)	# clean the tgt
	disc = {}
	disc["src"] = src ; disc["tgt"] = tgt
	return disc
	
## for the fancy command line invocation
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inpath', required=True, help='input folder for reading')
parser.add_argument('-o', '--outpath', default='./', help='output path for writing')
args = parser.parse_args()

## write path of the built dataset
write_path = os.path.join(args.outpath + "elitr_minuting_dataset.txt")

## main 
if __name__=="__main__":
	files = get_file_paths(args.inpath)			# get all file paths
	contents = []								# to hold the content strings
	
	## read content of each basic_sample.txt file
	for f in files:
		with open(f, 'rt', encoding='utf-8') as inf:
			content = inf.read()
		contents.append(content)				# add this content in the list
	
	## preprocess the entire file list
	src_tgt_lst = map(get_src_tgt, contents)	# create list of dictionaries
	dictlst_to_file_lines(write_path, src_tgt_lst)	# write data to file
