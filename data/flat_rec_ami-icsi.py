
'''
Script that walks the folder structure of rec_ami/rec_icsi, reads a certain 
files (e.g. abst_summs.txt or compl_extr_summ.txt) of each meeting and 
stores them in the respective rec_ami_*/rec_icsi_* folders.
'''

import os, sys, fnmatch, re 
from shutil import *
from tqdm import *

read_path = "./rec_icsi/"
write_path = "./rec_icsi_extr_sums/"

# filter out the descriptions and abstractive summaries from transcripts
def filter_trans(trans):
	# filter out all between descriptions: and \n\n
	trans = re.sub(r'descriptions: (.*?)\n', '\n', trans)
	trans = re.sub(r'abst_sum - (.*?)\n', '\n', trans)
	# filter out more than 2 consequetive newlines 
	trans= re.sub(r'\n{2,}', "\n\n", trans)
	return trans

# parser = argparse.ArgumentParser()
# parser.add_argument('--inpath', required=True, help='input folder for reading')
# parser.add_argument('--outpath', required=True, help='output folder for writing')
# args = parser.parse_args()

i = 0
if __name__=="__main__":
	# walk in the directory to find all files named abst_summs.txt
	for root, dir, files in os.walk(read_path):
		for file in files:
			if file.endswith("extr_summ.txt"):
				infile = os.path.join(root, file)
				inf = open(infile, 'rt', encoding='utf-8')
				trans = inf.read() ; inf.close()
				trans = filter_trans(trans)
				outfile = os.path.join(write_path, str(i) + ".txt")
				outf = open(outfile, 'wt', encoding='utf-8')
				outf.write(trans) ; outf.close()
				# copyfile(infile, os.path.join(write_path, str(i) + ".txt"))
				i += 1