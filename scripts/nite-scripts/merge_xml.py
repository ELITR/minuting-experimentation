#!/usr/bin/env python3

#######################################################################
# This script can be used to merge CTM and TXT files.
# It defaults to CTM outputs where they don't match (about 5% tokens).
#######################################################################

import os, re, sys
from collections import namedtuple
from argparse import ArgumentParser

CtmItem = namedtuple('CtmItem', 'fname channel start duration token conf')

def get_tokens(filename):
    with open(filename) as f:
        return f.read().split("\n")

def get_ctms(filename):
    ctms = []
    with open(filename) as f:
        for line in f.readlines()[:-1]:
            if line.endswith("\n"):
                line = line[:-1]
            ctms.append(CtmItem._make(line.split()))
    return ctms

def is_match(x, y):
    x_ = re.sub(r'[^a-zA-Z0-9_\']+', '', x.lower())
    y_ = re.sub(r'[^a-zA-Z0-9_\']+', '', y.lower())
    return x_ == y_


def find_next_overlap(i, j):
    # TODO: select best, not first overlap
    matches = []
    for k in range(i, min(i+50, len(ctms))):
        for l in range(j, min(j+50, len(tokens))):
            ctm, token = ctms[k], tokens[l]
            if is_match(ctm.token, token):
                matches.append((k, l))
    if matches:
        return min(matches, key=lambda x: x[0]+x[1])
    return -1, -1


def write_xml(filename, nite_entries):
    i = 1
    with open(filename, "w") as fp:
        fp.write("""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>""" + '\n')
        fp.write("""<nite:root nite:id="EN2002d.A.words" xmlns:nite="http://nite.sourceforge.net/">""" + '\n')
#         last = 0.0
        for ctm in nite_entries:
            # print("#", token, ctm.token)
            start, end = float(ctm.start), float(ctm.start) + float(ctm.duration)
            token = ctm.token
            
            if not token[-1].isalnum() and token[:-1].isalnum():  # punctuation
                punc, token = token[-1], token[:-1]
                line = '<w nite:id="{form_fname}{i}" starttime="{start:2.2f}" endtime="{end:2.2f}" punc="true">'\
                        .format(form_fname=ctm.fname, i=i, start=end, end=end)
                fp.write("\t" + line + punc + "</w>" + "\n")
                i += 1
            
            if token:
                line = '<w nite:id="{form_fname}{i}" starttime="{start:2.2f}" endtime="{end:2.2f}">'\
                        .format(form_fname=ctm.fname, i=i, start=start, end=end)
                fp.write("\t" + line + token + "</w>\n")
                i += 1
                
#             last = end
        fp.write("</nite:root>\n")


if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument('text')
    argp.add_argument('ctms')
    argp.add_argument('nite_xml')
    args = argp.parse_args()

    tokens = get_tokens(args.text)
    ctms = get_ctms(args.ctms)

    nite_entries = []
    i = j = 0
    last = -1
    matching = 0

    while i < len(ctms) and j < len(tokens):
        token, ctm = tokens[j], ctms[i]
        k, l = find_next_overlap(i, j)
        print(ctms[k].token, tokens[l])
        if k > i:
            # ignore text tokens until overlap
            nite_entries += [c for c in ctms[i:k]]
        elif k == -1:
            # disrepancy between files, only use ctms?
            print("Ended @ idx %d" % i)
            break
        nite_entries.append(ctms[k]._replace(token=tokens[l]))
        matching += 1
        i = k + 1
        j = l + 1
        #print(i, j)

    #print(nite_entries)
    print("Ended @:", i, j)
    print("Lengths:", len(ctms), len(tokens))
    print("Matching tokens: %d (%f%%)" % (matching, matching*100/i))

    write_xml(args.nite_xml, nite_entries)
    print("Done!")
