import pandas as pd
import numpy as np
import os, sys, json 
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import *

# funciton that counts word frequencies in a given text and returns a dictionary
def string_word_freqs(record):
    '''takes a text string and returns dictionary of word counts'''
    # create dictionary of current text
    wdict = {}
    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    for word in record.split():
        # update dictionary of current text 
        try:
        	wdict[word] += 1
        except KeyError:
        	wdict[word] = 1
    return wdict 

# function that counts word frequencies in a list of texts and returns a dictionary
def list_str_word_freqs(str_list):
    '''takes a list of strings and returns a dictionary of word counts'''
    wc_list = {}
    # itearate over all strings of the list to get dictionaries of appearing words
    for i in str_list:
        text_dict = string_word_freqs(i)
        wc_list = merge_sum_dicts(wc_list, text_dict)
    return wc_list

# returns min, max, avg token lengths of strings in the list argument
def return_strlist_stats(str_list):
    '''gets a list of strings and returns min, max, and avg token lenghts'''
    tot_len, clen = 0, 0
    avg_len, min_len, max_len = 0, 10000000, 0
    # loop through all list strings
    for i in trange(len(str_list)):
        clen = len(str_list[i].split())
        min_len, max_len = min(clen, min_len), max(clen, max_len)
        tot_len += clen
    avg_len = tot_len / len(str_list)
    return tot_len, min_len, max_len, avg_len

# prints min, max, avg token lengths of strings in the list argument
def print_strlist_stats(str_list):
    '''gets a list of strings and prints min, max, and avg lenght in tokens'''
    total, minL, maxL, avgL = return_strlist_stats(str_list)
    print("Number of records: ", str(len(str_list)))
    print("Total tokens in list: " + str(total))
    print("Min length: " + str(minL) + " tokens")
    print("Max length: " + str(maxL) + " tokens")
    print("Avg length: " + str(avgL) + " tokens")

# length distribution of text entries in a list
def strlist_len_dist(str_list, nbins):
    '''returns nbins histogram of token lengths in string list artument'''
    text_lengths = [len(x.split()) for x in str_list]
    # print num_strings, minL, maxL, avgL
    print_strlist_stats(str_list) 
    # counting and printing distributions
    counts, bins = np.histogram(text_lengths, bins=nbins, range=(MIN_LENGTH, MAX_LENGTH // 10))
    print(counts) ; print([int(x) for x in bins])
    return bins 

# returns total nr elements, min, max, avg lengths of lists in the list argument
def return_listlist_stats(list_of_lists):
    '''returns total nr elements, min, max, avg lengths of lists in the list argument'''
    len_abs, clen_abs = 0, 0 
    avg_len_abs, min_len_abs, max_len_abs = 0, 10000000, 0
    # loop through all list strings
    for i in list_of_lists:
        clen_abs = len(i)
        min_len_abs, max_len_abs = min(clen_abs, min_len_abs), max(clen_abs, max_len_abs)
        len_abs += clen_abs
    avg_len_abs = len_abs / len(list_of_lists)
    return len_abs, min_len_abs, max_len_abs, avg_len_abs

# read file json lines from given file path and return them in a list
def file_lines_to_list(file_path):
    '''read json lines and store them in a list that is returned'''
    with open(file_path, "r", encoding='utf-8') as inf:
        # strips \n at the end of each line
        line_list = [json.loads(line) for line in inf]
    return line_list

def print_strlist_stats(str_list):
    '''gets a list of strings and prints min, max, and avg lenght in tokens'''
    total, minL, maxL, avgL = return_strlist_stats(str_list)
    print("Number of records: ", str(len(str_list)))
    print("Total tokens in list: " + str(total))
    print("Min length: " + str(minL) + " tokens")
    print("Max length: " + str(maxL) + " tokens")
    print("Avg length: " + str(avgL) + " tokens")
	
## main
if __name__ == '__main__':
	## read the file 
	path = './elitr_minuting_dataset.txt'
	samp_lst = file_lines_to_list(path)
	
	trans_lst = [d['src'] for d in samp_lst]
	summ_lst = [d['tgt'] for d in samp_lst]
	
	print_strlist_stats(trans_lst)
	print('\n')
	print_strlist_stats(summ_lst)
	
	
	
	
	
	
	