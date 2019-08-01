In this wiki, we are going to demonstrate the structure of our recovered data from the ICSI corpus.

# Output
The output folder will contain one folder for each meeting with its name.
## Meeting
Each meeting folder includes a variable number of files, depends on the types of data that are available for that meeting and the speakers participated.
### Topic Segmentation
First and the one necessary thing for a meeting is the topic segmentation. if a meeting doesn't have any, no folder will be created for it. So every folder in the output, at least, contain one main files **conv_by_topic**.
#### conv_by_topic
This file simply contains the meeting description, segmented by topic and at the start of each line, its speaker is identified. different segments are separated by a blank line.
For every segment in the **conv_by_topic**, is a line of description. Description annotation belongs to the corpus itself. For AMI, there are 27 default topic description available in file **default-topics.xml** in folder **ontologies**, which are used for describing the topic of every segment. Besides, for some segments, there is a manually written description. Both of these descriptions are written in **conv_by_topic**, with the format of *"default desc_manual desc"*.
### Dialogue acts
Another data type that is recovered from datasets is **dialogue acts**. All of the utterances of speakers are separated under the name of something called dialogue acts. Dialogue acts have variable lengths, from half of a sentence to one or more. There is two dialogue act file for each speaker in the meeting.
#### transcrpits_by_da_{speaker}
This file contains all of the dialogue acts of one speaker in the whole meeting. Each line represents one dialogue act.
### Extractive Summary
There is one extractive summary file for every meeting that has been annotated for this type of summary. So if the meeting has an extractive summary, then there is one file that includes those dialogue acts of the meeting that constitute the extractive summary of that meeting, with their speaker specified in file **compl_extr_summ***.
### Abstractive Summary
There is also an abstractive summary available for some meetings, which is manually written by annotators. Each abstractive summary may (and may not) be related to some dialogue acts (relations are written in {meeting_name}.summlink.xml in the **extractive** folder. In the **abst_summs** file, we've written all of the dialogue acts related to one summary. The abstractive summary is written in format *"abst_sum - {category}: {summary}"*. There are 4 categories for each summary which are: **abstract**, **actions**, **decisions**, and **problems**. If the abstractive summary has no dialogue act directly related to it, "None" is written instead of dialogue acts.