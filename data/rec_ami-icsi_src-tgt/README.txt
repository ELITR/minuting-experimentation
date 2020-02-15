
These files were produced running flat_rec_src-tgt.py on rec_ami and rec_icsi folders. 
They contain the transcripts split in dialogues and the abstractive summary of each dialogue. 
The texts were processed with the clean script (tokenized, lowercased, removed speakers). 
Lines shorter than 50 characters were removed with sed -ri '/.{50}/!d' file.txt 