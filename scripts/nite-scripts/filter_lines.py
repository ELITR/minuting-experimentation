import sys
import re

#######################################################
# This script can be used to extract final hypotheses 
# from ASR+segmenter outputs (TXT files).
#######################################################


###  It looks like the segmenter output (text file) contains 2 kinds of lines:
# 1) intermediate outputs that should be displayed in the real-time translation GUI (typically overlapping)
# 2) shorter lines with final outputs (which are typically substrings of the intermediate outputs)

###  This script iterates over all triples of consecutive lines and extracts the final outputs using ~ these 3 rules:
# 1) this_line[concat]next_line == previous_line => this_line is extracted
#    - Example:
#       Lorem ipsum dolor sit ...
#       Lorem           <= final output
#       ipsum dolor sit amet ...
#       ipsum dolor     <= final output
#       sit amet ...
#       ...
# 2) this_line ends with <br> => this_line is extracted
#    - Example:
#       Hello world.
#       Hello           <= final output (rule 1)
#       world. <br><br> <= final output (rule 2)
#       Lorem ipsum ...
# 3) this_line is the beginning of sentence and occurs only once
#    - Example:
#       world. <br><br> <= final output (rule 2)
#       Yeah.           <= final output (rule 3)
#       Yeah and ...    <= someone else said that
#
###  And that's it! The rules were selected using the trial-and-error method after a very productive working day.

def has_common_prefix(a, b):
    a = re.sub(r'[^A-Za-z0-9]', '', a)
    b = re.sub(r'[^A-Za-z0-9]', '', b)
    try:
        l = min(len(a), len(b))
        diff_idx = min([i for i in range(l) if a[i] != b[i]])
        return diff_idx != 0
    except ValueError:
        # totally the same
        return True

def cleanmatch(a, b, c):
    b_ = re.sub(r'[^A-Za-z0-9]', '', b).lower().strip()
    c_ = re.sub(r'[^A-Za-z0-9]', '', c).lower().strip()
    if b_ != c_[:len(b_)]:
        return False  # B must be a prefix of C
    c = c_[len(b_):]
    a = re.sub(r'[^A-Za-z0-9]', '', a).lower().strip()
    #c = re.sub(r'[^A-Za-z0-9]', '', c).lower().strip()
    l = min(len(a), len(c), 10)
    if l <= 0: return False
    return a[:l] == c[:l]

history = []

# find every 3 lines such that prefix(A) == B.prefix_of(C)

for line in sys.stdin:
    if line.endswith("\n"):
        line = line[:-1]
    history.insert(0, line)
    if len(history) < 3: continue

    A = history[0]
    B = history[1]
    C = history[2]
#    print(A, '||', B)
#    print("_")
    is_ending  = history[1].rstrip().endswith("<br>")
#    is_unique  = not has_common_prefix(history[0], history[1]) and \
#                 not has_common_prefix(history[1], history[2]) and \
#                 (not history[1].rstrip().endswith('.') or history[2].rstrip().endswith('<br>'))
    is_unique = history[2].rstrip().endswith('<br>') \
                and not history[1].rstrip().endswith('...') \
                and history[0].rstrip().endswith('.')
    if cleanmatch(A, B, C) or is_ending or is_unique:
        for word in history[1].replace("<br>", "").split():
            print(word)
#        print(history[1])

