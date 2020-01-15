
''' write come comments here
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
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

sent_lst = ["Nature is beautiful","I like green apples",
"We should protect the trees","Fruit trees provide fruits",
"Green apples are tasty", "Sun is a natural source of light", 
"Staying in naure is beautiful", "Apples and pears are good fruits",
"We need to plant more trees", "Fires burn and destroy trees"]

# parser = argparse.ArgumentParser()
# parser.add_argument('--inpath', required=True, help='input folder for reading')
# parser.add_argument('--outpath', required=True, help='output folder for writing')
# args = parser.parse_args()

# vectorize the sentences
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sent_lst)
# model = Doc2Vec(sent_lst, vector_size=5, window=2, min_count=1, workers=4)

distortions, distances = [], [] 
K = range(3, 8)
print(X)
print(type(X))
# iterate to find optimal number of clusters (3 - 7)
for n_c in K:
	model = KMeans(n_clusters=n_c, init='k-means++', max_iter=100, n_init=1)
	model.fit(X)
	# distortions.append(model.inertia_)
	print(model.cluster_centers_)
	print(type(model.cluster_centers_))
	distortions.append(sum(np.min(cdist(np.array(X), model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')
optimum_nc = kn.knee

# # elbow method for finding the optimal number of clusters
# for i in range(0, 5):
# 	p1 = Point(initx=1,inity=distortions[0])
# 	p2 = Point(initx=5,inity=distortions[4])
# 	p = Point(initx=i+1,inity=distortions[i])
# 	distances.append(p.distance_to_line(p1,p2))
# optimum_nc = distances.index(max(distances)) + 3

# building the final clustering model
final_model = KMeans(n_clusters=optimum_nc, init='k-means++', max_iter=100, n_init=1)
final_model.fit(X)
clusters = collections.defaultdict(list)
for i, label in enumerate(final_model.labels_):
	clusters[label].append(i)
final_clusers = dict(clusters)

for cl in range(optimum_nc):
	print("cluster " + cl + ":")
	for i, s in enumerate(final_clusers[cl]):
		print("\tsentence " + i + ": " + sentences[sent_lst])

# main 
# if __name__=="__main__":
# 	# if len(sys.argv) != 5:
# 	# 	print("usage: python script --inpath <source_dir> --outpath <dest_dir>")
# 	# 	sys.exit()

# 	for filename in tqdm(os.listdir(args.inpath)):

# 		# full path of file being read and writen
# 		fin = os.path.join(args.inpath, filename)
# 		fout = os.path.join(args.outpath, filename)

# 		# read content of fin and split in sentences
# 		with open(fin, 'rt') as i:
# 			trans_txt = i.read()

# 		# get the sentences
# 		sent_lst = get_sents_from_trans(trans_txt, keep_speakers=False)

# 		# vectorize the sentences
# 		vectorizer = TfidfVectorizer(stop_words='english')
# 		X = vectorizer.fit_transform(sent_lst)

# 		distortions, distances = [], [] 
# 		# iterate to find optimal number of clusters (3 - 7)
# 		for n_c in range(5) + 3:
# 			model = KMeans(n_clusters=n_c, init='k-means++', max_iter=100, n_init=1)
# 			model.fit(X)
# 			distortions.append(kmeans.inertia_)

# 		# elbow method for finding the optimal number of clusters
# 		for i in range(0, 5):
# 			p1 = Point(initx=1,inity=distortions[0])
# 			p2 = Point(initx=5,inity=distortions[4])
# 			p = Point(initx=i+1,inity=distortions[i])
# 			distances.append(p.distance_to_line(p1,p2))
# 		optimum_nc = distances.index(max(distances)) + 3

# 		# building the final clustering model
# 		final_model = KMeans(n_clusters=optimum_nc, init='k-means++', max_iter=100, n_init=1)
# 		final_model.fit(X)
# 		clusters = collections.defaultdict(list)
# 		for i, label in enumerate(final_model.labels_):
# 			clusters[label].append(i)
# 		final_clusers = dict(clusters)










# 		# write content of fout 
# 		with open(fout, 'wt') as o:
# 			o.write()



		
# 	# create name of output file
# 	in_file = args.src
# 	out_file= args.dest

# 	# read text from input file
# 	with open(in_file, 'rt') as in_f:
# 		in_text = in_f.read()

# 	# apply cleaning operations
# 	out_text = clean_text(in_text, keep_speakers=False, lower=True, eos=True)

# 	# write to output file
# 	with open(out_file, 'wt') as out_f:
# 		out_f.write(out_text)




