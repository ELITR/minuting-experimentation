import os, sys
import string, argparse
import xml.etree.ElementTree as ET
from collections import OrderedDict


# check if a meeting has a topic file or not
def has_topic(data_root, topics_folder, meeting_name):
    """
    :param data_root: root folder of all data
    :param topics_folder: folder of all topics
    :param meeting_name: name of the meeting being analyzed
    :return boolean - topic folder exists or not
    """
    if not os.path.exists(os.path.join(data_root, topics_folder, meeting_name + '.topic.xml')):
        return False
    return True


# recover the words of speaker_name in meating_name
def parse_words_file(data_root, words_folder, meeting_name, speaker_name):
    """
    :param data_root: root folder of all data
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :param speaker_name: name or letter of the speaker
    :return words: list of words of the speaker - list of strings
    """

    # list with speaker words
    words = OrderedDict()
    tree = ET.parse(os.path.join(data_root, words_folder, meeting_name + '.' + speaker_name + '.words.xml'))
    root = tree.getroot()
    # recovering the words from xml tree
    for child in root:
        tag = child.tag
        id = child.get('{http://nite.sourceforge.net/}id')
        if tag == 'w':
            words[id] = child.text
        elif tag == 'vocalsound':
            words[id] = '<' + child.get('description') + '>'
        elif tag == 'nonvocalsound':
            # Maybe it's better to ignore these guys.
            words[id] = '<' + child.get('description') + '>'
        else:
            words[id] = ''
    # returning the list of recovered words
    return words


def get_speakers_words(data_root, topics_folder, words_folder, meeting_name):
    """
    :param data_root: root folder of all data
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :param topics_folder: folder of all topics
    :return speakers_words: list of speakers containing lists of their words - list of lists of strings
    """
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


def parse_segments_file(data_root, segments_folder, meeting_name, speaker_name, speakers_words):
    tree = ET.parse(os.path.join(data_root, segments_folder, meeting_name + '.' + speaker_name + '.segs.xml'))
    root = tree.getroot()

    speaker_segments = OrderedDict()
    for segment in root:
        id = segment.get('{http://nite.sourceforge.net/}id')
        words_in_segment = []
        for child in segment:
            href = child.get('href').split('#')
            speaker = href[0][href[0].find('.') + 1]
            start_word = href[1][href[1].find('(') + 1: href[1].find(')')]
            end_word = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
            keys_list = list(speakers_words[ord(speaker) - ord('A')].keys())
            start_index = keys_list.index(start_word)
            end_index = keys_list.index(end_word)
            words_in_segment = list(speakers_words[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
            words_in_segment.insert(0, speaker + ': ')
            speaker_segments[id] = words_in_segment
            # words_in_segment = [speaker + ': '] + speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]

    return speaker_segments


def get_segments(data_root, segments_folder, meeting_name, speakers_words):
    speakers_segments = []

    # get the words of meeting
    for c in string.ascii_uppercase:
        try:
            speaker_segments = parse_segments_file(data_root, segments_folder, meeting_name, c, speakers_words)
            speakers_segments.append(speaker_segments)
        except FileNotFoundError:
            break
    return speakers_segments


# output_root = './output'
# data_root = './input'

words_folder = 'words'
topics_folder = 'topics'
segments_folder = 'segments'
dialogue_acts_folder = 'dialogueActs'
extractive_sum_folder = 'extractive'
abstractive_sum_folder = 'abstractive'
ontologies_folder = 'ontologies'

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./src', help='source directory')
parser.add_argument('--dest', default='./dest', help='destination directory')
args = parser.parse_args()

data_root = parser.parse_args().src
output_root = parser.parse_args().dest

if not os.path.exists(data_root):
    sys.exit('source directory does not exist.')
if not os.path.exists(output_root):
    os.makedirs(output_root)

# get set of meetings' names
name_of_meetings = {x.split('.')[0] for x in os.listdir(os.path.join(data_root, words_folder))}
name_of_meetings = sorted(name_of_meetings)


def get_words_in_topic(topic, rec_segments, output_words_in_topics_dict, output_descriptions_in_topics_dict):
    topic_id = topic.get('{http://nite.sourceforge.net/}id')
    topic_description = topic.get('description')
    segments_in_topic = []
    for child in topic.findall('{http://nite.sourceforge.net/}child'):
        href = child.get('href').split('#')
        speaker = href[0][href[0].find('.') + 1]
        start_seg = href[1][href[1].find('(') + 1: href[1].find(')')]
        end_seg = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
        keys_list = list(rec_segments[ord(speaker) - ord('A')].keys())
        start_index = keys_list.index(start_seg)
        end_index = keys_list.index(end_seg)
        words_in_segment = list(rec_segments[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
        segments_in_topic.append(words_in_segment)
    output_words_in_topics_dict[topic_id] = segments_in_topic
    output_descriptions_in_topics_dict[topic_id] = topic_description
    for topic in topic.findall('topic'):
        get_words_in_topic(topic, segments, output_words_in_topics_dict, output_descriptions_in_topics_dict)
    return


def get_words_in_topics(data_root, topics_folder, meeting_name, segments):
    # parsing a topic file
    meeting_topics = ET.parse(os.path.join(data_root, topics_folder, meeting_name + '.topic.xml'))
    root = meeting_topics.getroot()

    topic_description = OrderedDict()
    words_in_topics = OrderedDict()
    for each_topic in root:
        get_words_in_topic(each_topic, segments, words_in_topics, topic_description)

    return words_in_topics, topic_description


for meeting in name_of_meetings:
    speakers_words = get_speakers_words(data_root, topics_folder, words_folder, meeting)
    if speakers_words is None:
        continue

    segments = get_segments(data_root, segments_folder, meeting, speakers_words)
    words_in_topics, topic_description = get_words_in_topics(data_root, topics_folder, meeting, segments)

    # save_meeting_converses_by_topic(output_root, meeting, words_in_topics)
    # save_meeting_topics_descriptions(output_root, meeting, topic_description)
