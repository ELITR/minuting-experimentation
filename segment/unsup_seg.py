
'''
Author:	Erion Çano
Descri: Unsupervised transcript segmentation script that reads all transcript 
        files of the input path and writes the corresponding files to the 
        output path. Each transcript is split into sentences which are then 
        thematically clustered. The optimal number of clusters (from 2 to 7) 
        is found comparing the intracluster similarities between cluster 
		sentences. 
Langu: 	Python 3.6.9
Usage:	python unsup_seg.py --inpath INDIR --outpath OUTDIR
'''

import os, sys, codecs, re, string, argparse
import nltk, collections, math
from statistics import mean
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, \
AffinityPropagation, SpectralClustering
from nltk import PerceptronTagger, tokenize
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords

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
		trans = remove_speakers(trans)
	# strip leading and trailing white space
	trans = trans.strip()

	# split in sentences 
	sent_lst = tokenize.sent_tokenize(trans)

	# tokenize each sentence with core_tokenize
	sent_lst = list(map(core_tokenize, sent_lst))

	# correct speakers A : to A:
	sent_lst = [re.sub(r'([a-zA-Z])\s+(:)', r'\1\2', sent) for sent in sent_lst]

	return sent_lst

# print the clusters and the corresponding sentences
def print_cluster_sents(clusters, sent_lst):
	'''print the sentences of each cluster in a grouped form'''
	print("Number of clusters: ", len(clusters))
	for cl in range(len(clusters)):
		print("cluster " + str(cl) + ":")
		for i, s in enumerate(clusters[cl]):
			print("\tsentence " + str(i) + ": " + sent_lst[s])

# transform the cluster dict into a list of lists
def get_cluster_sents(clusters, sents):
	'''form the list with sentence lists according to clusters'''
	clustered_sents = []
	clustered_indexes = list(clusters.values())
	for cl in range(len(clusters)):
		this_cl_sents = []
		this_indexes = clustered_indexes[cl]
		for s in this_indexes:
			this_cl_sents.append(sents[s])
		clustered_sents.append(this_cl_sents)
	return clustered_sents

# vectorize the words of a sentence
def text_to_vec(t):
	'''get vector representation of text to compute cosine similarities'''
	WORD = re.compile(r'\w+')
	words = WORD.findall(t)
	return collections.Counter(words)

# compute the cosine similarity between two word vectors
def get_cosine_sim(vec1, vec2):
	'''compute cosine similarity between vec1 and vec2'''
	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	# mind the divizion by 0
	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

# compute cosine similarity between each pair of sents in sent_lst
def get_avg_cosine_list(s_lst):
	'''get cosine similarity between each pair of sents in sent_lst'''
	# return 0 if only 1 sentence in list
	if len(s_lst) <= 1:
		return 0
	sent_pairs = [(s_lst[p1], s_lst[p2]) for p1 in range(len(s_lst)) for p2 in range(p1+1,len(s_lst))]
	vec_pairs = [(text_to_vec(s_lst[p1]), text_to_vec(s_lst[p2])) for p1 in range(len(s_lst)) for p2 in range(p1+1,len(s_lst))]
	sent_pair_sims = list(map(get_cosine_sim, *zip(*vec_pairs)))
	this_c_avg = mean(sent_pair_sims)
	return this_c_avg

# compute the average similarity of sentences in a cluster
def get_avg_cluster_sim(clusters, sents):
	'''
	:param clusters: clusters dict like {0: [2, 3, 6], 1: [0, 1, 4, 5], ...}
	:param sent_lst: list of all sentences
	:return avg_cl_sim: average intracluster similarity
	'''
	cl_sents = get_cluster_sents(clusters, sents)

	avg_cl_sim, sum_cl_sim = 0, 0
	for c in range(len(cl_sents)):
		sum_cl_sim += get_avg_cosine_list(cl_sents[c])
	avg_cl_sim = sum_cl_sim / len(cl_sents)
	return avg_cl_sim

# create clusters of sentences - 5 clustering algorithms available
def cluster_sentences(n_c, sent_lst):
	'''create the clusters of sentences'''
	vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
		max_df=0.9, min_df=0.1, lowercase=True)
	#builds a tf-idf matrix for the sentences
	X = vectorizer.fit_transform(sent_lst)

	# # k-means
	# model = KMeans(n_clusters=n_c)
	# model.fit(X)
	# labels = model.labels_

	# Agglomerative - the best according to the conducted experiments
	model = AgglomerativeClustering(n_clusters=n_c, linkage="ward")
	model.fit(X.toarray())
	labels = model.labels_

	# # MeanShift - finds n_clusters itself
	# model = MeanShift()
	# model.fit(X.toarray())
	# labels = model.labels_

	# # Spectral
	# model = SpectralClustering(n_clusters=n_c)
	# model.fit(X)
	# labels = model.labels_

	# # Affinity Propagation - finds n_clusters itself
	# model = AffinityPropagation()
	# model.fit(X)
	# labels = model.labels_

	clusters = collections.defaultdict(list)
	for i, label in enumerate(labels):
	        clusters[label].append(i)
	return dict(clusters)

# find optimal number of clusters for the list of sentences
def optimal_cluster_num(sent_lst):
	'''find optimal number of clusters for the list of sentences'''
	avg_sims = []
	for n_c in range(2, 8):
		clusters = cluster_sentences(n_c, sent_lst)
		avg_sims.append(get_avg_cluster_sim(clusters, sent_lst))
	max_ind = avg_sims.index(max(avg_sims))
	max_val = max(avg_sims)
	# correcting to find optimal number of clusters
	opt_cl_n = max_ind + 2 
	return opt_cl_n

# format cluster texts for output files
def format_clusters(clusters, sent_lst):
	'''format cluster texts for output files'''
	cluster_lst, out_text = [], ""
	for cl in range(len(clusters)):
		cluster_str = []
		for s in clusters[cl]:
			cluster_str.append(sent_lst[s])
		cluster_lst.append('\n'.join(cluster_str))
	out_text = '\n\n<cluster_separator>\n\n'.join(cluster_lst)
	return out_text

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--inpath', required=True, help='input folder for reading')
parser.add_argument('--outpath', required=True, help='output folder for writing')
args = parser.parse_args()

if __name__=="__main__":
	# loop through all files of input path
	for filename in tqdm(os.listdir(args.inpath)):

		# full path of file being read and writen
		fin = os.path.join(args.inpath, filename)
		fout = os.path.join(args.outpath, filename)

		# read content of fin and split in sentences
		with open(fin, 'rt') as i:
			trans_txt = i.read()
		# get the sentences
		sent_lst = get_sents_from_trans(trans_txt, keep_speakers=False)

		# find optimal clusters
		nclusters = optimal_cluster_num(sent_lst)
		clusters = cluster_sentences(nclusters, sent_lst)
		out_text = format_clusters(clusters, sent_lst)
		# print("Avg sim: " + str(get_avg_cluster_sim(clusters, sent_lst)))

		# write content of fout 
		with open(fout, 'wt') as o:
			o.write(out_text)
			