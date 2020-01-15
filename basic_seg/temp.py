
## Reporting the length mean and std and plotting the length distribution
## for each sample field

import json, gzip, re, string
import glob, os, sys
from json import JSONDecoder
from functools import partial
import functools
import ijson, gc, itertools, argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textdistance
from tqdm import tqdm

# read file json lines from given file path and return them in a list
def file_lines_to_list(file_path):
    '''read json lines and store them in a list that is returned'''
    with open(file_path, "r") as inf:
        # strip \n at the end of each line
        line_list = [json.loads(line) for line in inf]
    return line_list

# write list of dict (json) records as lines in a given file path
def list_to_file_lines(file_path, line_list):
    '''write list of dict lines in a file path that is opened'''
    outf = open(file_path, "a+", encoding='utf-8')
    outf.write("\n".join(line_list))
    outf.close()

# count and return number of lines in a given file path
def count_file_lines(file_path):
	'''counts number of lines in the file path given as argument'''
	fp = open(file_path, "r")
	num_lines = sum(1 for line in fp)
	return num_lines

# remove , from keyword string
def keywords_comma_fix(keys):
	key_lst = keys.split(' , ')
	fixed_keys = ' '.join(key_lst)
	return fixed_keys

# compute jaccard index between two strings
def jindex(source, target):
	# split and rejoin the target 
	targ = keywords_comma_fix(target)
	# remove punctuation from both strings
	s = source.translate(translator)
	t = targ.translate(translator)
	# compute and return jaccard index
	return textdistance.jaccard(s.split(), t.split())

# compute jaccard indexes between two string lists and return them as a list
def jindex_of_lists(src_lst, tgt_lst):
	# first check inequalith of lengths
	if len(src_lst) != len(tgt_lst):
		print("Different list lengths! Exiting...")
		return
	else:
		return list(map(jindex, src_lst, tgt_lst))

# join three strings with a space between them
def join_two_strings(a, b):
	return a + " " + b

# remove punctuation and extra spaces
def clean_punct(dirty_str):
	clean_str = dirty_str.translate(translator)
	# remove double+ spaces
	clean_str = re.sub(r'\s{2,}', " ", clean_str)
	return clean_str

parser = argparse.ArgumentParser()
parser.add_argument('--inpath', required=True, help='input folder for reading')
parser.add_argument('--outpath', required=True, help='output folder for writing')
args = parser.parse_args()

# for removing punctuation - used in jindex function
translator = str.maketrans('', '', string.punctuation)

# path of files to read
read_path = "./oagsx/length_clear"
write_path = "./oagsx/merged"

if __name__ == "__main__":

    for filename in tqdm(os.listdir(read_path)):
        
        # reset list of current file
        file_rec_lst, merged_lst = [], []

        # full path of file being read and writen
        fn = os.path.join(read_path, filename)
        fout = os.path.join(write_path, filename)
        print("\nAdding records from %s ..." % fn)
        file_rec_lst = file_lines_to_list(fn)
        file_tit_lst = [d["title"] for d in file_rec_lst if "title" in d]
        file_abs_lst = [d["abstract"] for d in file_rec_lst if "abstract" in d]

        # merge the strings of each list into one list
        merged_lst = list(map(join_two_strings, file_tit_lst, file_abs_lst))
        # perform cleaning with clean_punct function
        merged_lst = list(map(clean_punct, merged_lst))

        # append strings to file, each in one line
        list_to_file_lines(fout, merged_lst)

        # clean memory
        del file_rec_lst ; del merged_lst ; gc.collect()



