#!/usr/bin/env python3

############################################################
# This script tries to match CTM and TXT files exactly.
# Usually they differ in 5% and so it doesn't always work.
# For that reason merge_xml.py is used by default.
############################################################


import sys
import re
import nltk

from argparse import ArgumentParser
from collections import namedtuple


CtmItem = namedtuple('CtmItem', 'fname channel start duration token conf')


def consume_tokens(lines: list):
    for line in lines:
        for token in line.split(): #nltk.word_tokenize(line):
            yield token
        yield '\n'


def parallel_read(ctm_items, lines):

    text_tokens = consume_tokens(lines)
    ctm_items = iter(ctm_items)
    had_apostrophe = False
    should_skip_token = False
    should_output_ctms = False
    ctm = last_ctm = None

    while True:
        try:
            if not should_skip_token:
                last_ctm = ctm
                ctm = next(ctm_items)
            if not should_output_ctms:
                token = next(text_tokens)
            should_skip_token = False
        except StopIteration:
            print("Done!")
            break
        
        if token == '\n':
            # add line
            yield "\n", ctm
            token = next(text_tokens)
        
        #print(ctm.token, token)
        ctm_stripped = re.sub(r'[^a-zA-Z0-9_\']+', '', ctm.token.lower())
        txt_stripped = re.sub(r'[^a-zA-Z0-9_\']+', '', token.lower())
        cmp_len = min(len(ctm_stripped), len(txt_stripped))     
        if not txt_stripped:
            should_output_ctms = False
            # just punctuation
            yield token, ctm
        elif ctm_stripped[:cmp_len] != txt_stripped[:cmp_len]:
            # invalid token
            print("!!! WARNING !!!", ctm_stripped, '!=', txt_stripped, file=sys.stderr)
            if should_output_ctms:  # when text tokens are not desired (numbers)
                yield ctm.token, ctm
            elif had_apostrophe:
                print("! Skipping due to apostrophe\n", file=sys.stderr)
                should_skip_token = True
                yield token, last_ctm
                continue
            elif any(c.isdigit() for c in txt_stripped): #.isnumeric():
                should_output_ctms = True
                yield ctm.token, ctm
                token = next(text_tokens)
            else:
                raise RuntimeError()
        else:
            should_output_ctms = False
            while ctm_stripped[:cmp_len] == txt_stripped[:cmp_len]:
                yield token, ctm
                ctm_stripped = ctm_stripped[cmp_len:]
                if not ctm_stripped: break
                token = next(text_tokens)
                if token == '\n':
                    yield '\n', ctm
                    break
                print("-", list(token), "+", ctm_stripped)
                cmp_len = min(len(ctm_stripped), len(txt_stripped))        
        
        had_apostrophe = "'" in ctm.token

        continue
        if not token[-1].isalnum():
            # text and punctuation
            yield token[:-1], ctm
            yield token[-1], ctm
        else:  
            # just text
            yield token, ctm


def write_xml(filename, it):
    i = 1
    with open(filename, "w") as fp:
        fp.write("""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>""" + '\n')
        fp.write("""<nite:root nite:id="EN2002d.A.words" xmlns:nite="http://nite.sourceforge.net/">""" + '\n')
        last = 0.0
        for token, ctm in it:
            # print("#", token, ctm.token)
            start, end = float(ctm.start), float(ctm.start) + float(ctm.duration)
            if token == '\n':
                start = end = last
                continue
            elif not token.isalnum() and len(token) == 1:
                start = end
                line = '<w nite:id="{form_fname}{i}" starttime="{start:2.2f}" endtime="{end:2.2f}" punc="true">'.format(form_fname=ctm.fname, i=i, start=start, end=end)
            else:
                line = '<w nite:id="{form_fname}{i}" starttime="{start:2.2f}" endtime="{end:2.2f}">'.format(form_fname=ctm.fname, i=i, start=start, end=end)
            last = end
            fp.write("\t" + line + token + "</w>\n")
            i += 1
        fp.write("</nite:root>\n")


if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("ctm")
    argp.add_argument("txt")
    argp.add_argument("xml")
    args = argp.parse_args()

    with open(args.ctm) as fp:
        lines = fp.read().split('\n')[:-2]
        ctm_items = [CtmItem._make(line.split()) for line in lines]
    
    with open(args.txt) as fp:
        txt_items = fp.read().split('\n')

    it = parallel_read(ctm_items, txt_items)
    #for i in it:
    #    print(i)
    write_xml(args.xml, it)
