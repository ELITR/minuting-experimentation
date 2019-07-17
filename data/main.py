import os
import string, argparse
import xml.etree.ElementTree as ET
from collections import OrderedDict

output_root = './output'
if not os.path.exists(output_root):
    os.makedirs(output_root)

data_root = './input'
words_folder = 'words'
topics_folder = 'topics'
segments_folder = 'segments'
dialogue_acts_folder = 'dialogueActs'
extractive_sum_folder = 'extractive'
abstractive_sum_folder = 'abstractive'
ontologies_folder = 'ontologies'

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='./src', help='src_dir')
parser.add_argument('--dest_dir', default='./dest', help='dest_dir')
args = parser.parse_args()


# check if a meeting has a topic file or not
def has_topic(data_root, topics_folder, meeting_name):
    '''
    :param data_root: root folder of all data
    :param topics_folder: folder of all topics
    :param meeting_name: name of the meeting being analyzed
    :return boolean - topic folder exists or not
    '''
    if not os.path.exists(os.path.join(data_root, topics_folder, meeting + '.topic.xml')):
        return False
    return True


# recover the words of speaker_name in meating_name
def parse_words_file(data_root, words_folder, meeting_name, speaker_name):
    '''
    :param data_root: root folder of all data
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :param speaker_name: name or letter of the speaker
    :return words: list of words of the speaker - list of strings
    '''
    # list with speaker words
    words = []
    tree = ET.parse(os.path.join(data_root, words_folder, meeting_name + '.' + speaker_name + '.words.xml'))
    root = tree.getroot()
    # recovering the words from xml tree
    for child in root:
        tag = child.tag
        if tag == 'w':
            words.append(child.text)
        elif tag == 'vocalsound':
            words.append('<' + child.get('type') + '>')
        else:
            words.append('')
    # returning the list of recovered words
    return words


def get_speakers_words(data_root, topics_folder, words_folder, meeting_name):
    '''
    :param data_root: root folder of all data
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :param topics_folder: folder of all topics
    :return speakers_words: list of speakers containing lists of their words - list of lists of strings
    '''
    # if there is no topic for this meeting, just ignore it
    if not has_topic(data_root, topics_folder, meeting):
        return

    # list of speakers each containing their list of words
    speakers_words = []

    # get the words of meeting
    for c in string.ascii_uppercase:
        try:
            speaker_words = parse_words_file(data_root, words_folder, meeting, c)
            speakers_words.append(speaker_words)
        except FileNotFoundError:
            continue
    return speakers_words


# recover the default topics and their ids
def parse_default_topics(data_root, ontologies_folder):
    '''
    :param data_root: root folder of all data
    :param ontologies_folder: folder of ontologies
    :return default_topic_dict - dictionary of ids and topic descriptions
    '''
    default_topics_dict = dict()
    default_topics = ET.parse(os.path.join(data_root, ontologies_folder, 'default-topics.xml')).getroot()
    for topic_names in default_topics:
        default_topics_dict['id(' + topic_names.get('{http://nite.sourceforge.net/}id') + ')'] = topic_names.get('name')
        for topic_name in topic_names:
            default_topics_dict['id(' + topic_name.get('{http://nite.sourceforge.net/}id') + ')'] = topic_name.get(
                'name')
    return default_topics_dict


# returns words in topics and their description for each topic
def get_words_in_topics(data_root, topics_folder, meeting_name, default_topics_dict, speakers_words):
    '''
    :param data_root: root folder of all data
    :param topics_folder: folder of all topics
    :param meeting_name: name of the meeting being analyzed
    :param default_topic_dict: dictionary of default topics
    :param speakers_words: list of speakers each containing their list of words
    :return: words_in_topics, topic_description
    '''
    # parsing a topic file
    meeting_topics = ET.parse(os.path.join(data_root, topics_folder, meeting_name + '.topic.xml'))
    root = meeting_topics.getroot()

    topic_description = []
    words_in_topics = []
    for topic in root:
        words_in_topic = []
        for child in topic:
            if str(child.tag).find('pointer') != -1:
                href = child.get('href').split('#')[1]
                topic_description.append(default_topics_dict[href] + ':\t' + str(topic.get('other_description')))
            elif str(child.tag).find('child') != -1:
                href = child.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                start_word = int(href[1][href[1].find('words') + 5: href[1].find(')')])
                end_word = int(href[1][href[1].rfind('words') + 5: href[1].rfind(')')])
                words_in_subtopic = [speaker + ': '] + speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]
                words_in_topic.append(words_in_subtopic)
        words_in_topics.append(words_in_topic)
    return words_in_topics, topic_description


def save_meeting_converses_by_topic(output_root, meeting_name, words_in_topics):
    '''
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting being analyzed
    :param words_in_topics: words in the topics
    '''
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the transcripts by topic
    transcrpits_by_topic = open(output_dir_for_meeting + "/converses_by_topic.txt", "w+")
    for topics_it in words_in_topics:
        for words_of_each_speaker in topics_it:
            for word_it in words_of_each_speaker:
                transcrpits_by_topic.write(word_it + " ")
            transcrpits_by_topic.write('\n')
        transcrpits_by_topic.write('\n\n')
    return


def save_meeting_topics_descriptions(output_root, meeting_name, descriptions):
    '''
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting being analyzed
    :param descriptions: words in the topics
    '''
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the topic file
    topic_file = open(output_dir_for_meeting + "/topic_descriptions.txt", "w+")
    for desc in descriptions:
        topic_file.write(str(desc) + "\n")
    topic_file.close()
    return


# recover the default dialogue acts and their IDs
def parse_default_dialogue_acts(data_root, ontologies_folder):
    '''
    :param data_root: root folder of all data
    :param ontologies_folder: folder of ontologies
    :return default_das_dict - dictionary of IDs and dialogue acts
    '''
    # get the default dialogue acts
    default_das_dict = dict()
    default_das = ET.parse(os.path.join(data_root, ontologies_folder, 'da-types.xml')).getroot()
    for da_names in default_das:
        default_das_dict['id(' + da_names.get('{http://nite.sourceforge.net/}id') + ')'] = da_names.get('gloss')
        for da_name in da_names:
            default_das_dict['id(' + da_name.get('{http://nite.sourceforge.net/}id') + ')'] = da_name.get('gloss')
    return default_das_dict


def get_speaker_dialogue_acts(data_root, dialogue_acts_folder, meeting_name, speakers_words, default_das_dict, c):
    '''
    :param c: speaker name
    :param data_root: root folder of all data
    :param dialogue_acts_folder: folder of all dialogue acts
    :param meeting_name: name of the meeting being analyzed
    :param default_das_dict: dictionary of default dialogue acts
    :param speakers_words: list of speakers each containing their list of words
    :return: speaker_das, speaker_das_descriptions
    '''
    speaker_das = OrderedDict()
    speaker_das_descriptions = OrderedDict()
    tree = ET.parse(os.path.join(data_root, dialogue_acts_folder, meeting_name + '.' + c + '.dialog-act.xml'))
    root = tree.getroot()
    for da in root:
        words_in_da = []
        index = int(da.get('{http://nite.sourceforge.net/}id')[da.get('{http://nite.sourceforge.net/}id').rfind('.') + 1:])
        for child in da:
            if str(child.tag).find('pointer') != -1:
                href = child.get('href').split('#')[1]
                speaker_das_descriptions[index] = default_das_dict[href]
                # speaker_das_descriptions.append(default_das_dict[href])
                # speaker_das_descriptions.insert(index, default_das_dict[href])
            elif str(child.tag).find('child') != -1:
                href = child.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                start_word = int(href[1][href[1].find('words') + 5: href[1].find(')')])
                end_word = int(href[1][href[1].rfind('words') + 5: href[1].rfind(')')])
                words_in_da = speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]
                # selected_words = speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]
                # words_in_da.append(selected_words)

        speaker_das[index] = words_in_da
        # speaker_das.insert(index, words_in_da)
        # speaker_das.append(words_in_da)
    return speaker_das, speaker_das_descriptions


def save_meeting_dialogue_acts(output_root, meeting_name, speaker_das, c):
    '''
    :param c: speaker name
    :param data_root: root folder of all data
    :param meeting_name: name of the meeting being analyzed
    :param speaker_das: dialogue acts of the speaker
    '''
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the transcripts by da
    transcrpits_by_da = open(output_dir_for_meeting + "/transcrpits_by_da_" + str(c) + ".txt", "w+")
    for das_it in speaker_das:
        for word in speaker_das[das_it]:
            transcrpits_by_da.write(word + " ")
        transcrpits_by_da.write('\n')
    return


def save_meeting_dialogue_acts_descriptions(output_root, meeting_name, speaker_das_descriptions, c):
    '''
    :param c: speaker name
    :param data_root: root folder of all data
    :param meeting_name: name of the meeting being analyzed
    :param speaker_das_descriptions: descriptions of speaker das
    '''
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the dialogue_acts file
    da_file = open(output_dir_for_meeting + "/dialogue_acts_" + str(c) + ".txt", "w+")
    for desc in speaker_das_descriptions:
        da_file.write(str(speaker_das_descriptions[desc]) + "\n")
    da_file.close()
    return


# get set of meetings' names
name_of_meetings = {x.split('.')[0] for x in os.listdir(os.path.join(data_root, words_folder))}


def get_extractive_summary(data_root, extractive_sum_folder, meeting_name, speakers_das):
    """

    :param data_root: root folder of all data
    :param extractive_sum_folder:
    :param meeting_name: name of the meeting being analyzed
    :param speakers_das: dialogue acts of all of speakers
    """

    summ = ET.parse(os.path.join(data_root, extractive_sum_folder, meeting_name + '.extsumm.xml'))
    root = summ.getroot()
    das_in_summary = []
    for ext_summ in root:
        for child in ext_summ:
            if str(child.tag).find('child') != -1:
                href = child.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                start_da = int(href[1][href[1].find('dharshi') + 8: href[1].find(')')])
                end_da = int(href[1][href[1].rfind('dharshi') + 8: href[1].rfind(')')])
                selected_das = [speaker + ': ']
                for i in range(start_da, end_da + 1):
                    selected_das.append(speakers_das[ord(speaker) - ord('A')][i])
                das_in_summary.append(selected_das)
    return das_in_summary


def save_meeting_extractive_summary(output_root, meeting, das_in_summary):
    """

    :param output_root:
    :param meeting:
    :param das_in_summary:
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the summary file
    extractiv_summary = open(output_dir_for_meeting + "/extractiv_summary.txt", "w+")
    for das in das_in_summary:
        speaker = das[0]
        extractiv_summary.write(speaker + " ")
        for word in das[1]:
            extractiv_summary.write(word + " ")
        extractiv_summary.write('\n\n')


for meeting in name_of_meetings:
    # list of speakers each containing their list of words
    speakers_words = get_speakers_words(data_root, topics_folder, words_folder, meeting)
    if speakers_words is None:
        continue

    # getting default topics
    default_topics_dict = parse_default_topics(data_root, ontologies_folder)
    words_in_topics, topic_description = get_words_in_topics(data_root, topics_folder, meeting, default_topics_dict,
                                                             speakers_words)
    save_meeting_converses_by_topic(output_root, meeting, words_in_topics)
    save_meeting_topics_descriptions(output_root, meeting, topic_description)

    default_das_dict = parse_default_dialogue_acts(data_root, ontologies_folder)
    # get the dialogue_acts
    speakers_das = []
    speakers_das_descriptions = []
    for c in string.ascii_uppercase:
        try:
            speaker_das, speaker_das_descriptions = get_speaker_dialogue_acts(data_root, dialogue_acts_folder,
                                                                              meeting, speakers_words, default_das_dict,
                                                                              c)
            speakers_das.append(speaker_das)
            speakers_das_descriptions.append(speaker_das_descriptions)
        except FileNotFoundError:
            continue
        save_meeting_dialogue_acts(output_root, meeting, speaker_das, c)
        save_meeting_dialogue_acts_descriptions(output_root, meeting, speaker_das_descriptions, c)

    # extractive_summary
    try:
        das_in_summary = get_extractive_summary(data_root, extractive_sum_folder, meeting, speakers_das)
        save_meeting_extractive_summary(output_root, meeting, das_in_summary)
    except FileNotFoundError:
        pass
print("Done!")


# tree1 = ET.parse('data/words/ES2002a.A.words.xml')
# root1 = tree1.getroot()
# tree2 = ET.parse('data/words/ES2002a.B.words.xml')
# root2 = tree2.getroot()
#
# words = []
# for child in root1:
#     words.append(child)
#     # print(child.tag, child.attrib, child.text)
# for child in root2:
#     words.append(child)
# words.sort(key=lambda x: float(x.get('starttime')))
# for element in words:
#     print(element.tag, element.attrib, element.text)
# print("hi")
# # print(root)
