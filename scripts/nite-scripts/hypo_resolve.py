#!/usr/bin/python

# Thai-Son's script for stripping partial hypotheses from ASR .ctm output.

import sys

conv = "conv"

def generate_ctm(hypo):
    preword = ""
    for i in range(len(hypo)):
        word, start, end = hypo[i]
        if word.startswith("<") and word.endswith(">"): continue # Skiping noise model tokens
        #if word == preword: continue
        #preword = word
        print("%s 1 %7.2f %7.2f %-20s -1.00" % (conv, start, (end-start), word))
    return

TKNA_file = "H_TKNA.ctm"

if len(sys.argv) == 2:
    TKNA_file = sys.argv[1]

    f = open(TKNA_file, 'r')
else:
    f = sys.stdin
hypo = []
hypo_start = 0.0
new_hypo = True
idx = 0
for line in f:
    if line.startswith('#'):
        tokens = line.split()
        new_hypo = True
        continue
    if not line.startswith('#'):
        tokens = line.split()
        conv, channel, start, end, word = tokens[:5]

        start = float(start)
        end = start + float(end) + 0.001 # Adding a bias to fix float comparision

        if new_hypo:
            hypo_start = start + 0.01
            idx = len(hypo) 
            while idx > 0 and hypo[idx-1][2] > hypo_start: idx -= 1
            new_hypo = False
        if idx >= len(hypo):
            hypo.append([word, start, end])
            idx = len(hypo)
        else:
            if hypo[idx][0].lower() != word.lower():
                hypo = hypo[:idx+1]
            hypo[idx] = [word, start, end]
            idx += 1

generate_ctm(hypo)

print("# Done.")

