
''' 
Recover ELITR meeting transcripts of raw_own_meetings from xml 
to plaintext format stored in rec_own_meetings
'''

import argparse, os, string, sys
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

# recover the words of speaker_name in meeting_name
def parse_words_file(data_root, words_folder, meeting_name, speaker_name):
    """
    :param data_root: root folder of all data
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :param speaker_name: name or letter of the speaker
    :return words: dictionary of words of the speaker
    """

    # list with speaker words
    words = OrderedDict()
    tree = ET.parse(os.path.join(data_root, words_folder, meeting_name + '.' + speaker_name + '.words.xml'))
    root = tree.getroot()
    # recovering the words from xml tree
    for child in root:
        tag = child.tag
        word_id = child.get('{http://nite.sourceforge.net/}id')
        if tag == 'w':
            words[word_id] = child.text
        elif tag == 'vocalsound':
            words[word_id] = '<' + child.get('description') + '>'
        elif tag == 'nonvocalsound':
            # Maybe it's better to ignore these guys.
            words[word_id] = '<' + child.get('description') + '>'
        else:
            words[word_id] = ''
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
    # if not has_topic(data_root, topics_folder, meeting_name):
    #     return

    # list of speakers each containing their list of words
    speakers_words = []

    # get the words of meeting
    for c in string.ascii_uppercase:
        try:
            speaker_words = parse_words_file(data_root, words_folder, meeting_name, c)
            speakers_words.append(speaker_words)
        except FileNotFoundError:
            continue
    return speakers_words


def get_words_in_segment(segment, speaker_segments, speakers_words):
    """

    :param segment: one segment
    :param speaker_segments: all of the segments of this speaker
    :param speakers_words: all of the words of speakers
    """
    segment_id = segment.get('{http://nite.sourceforge.net/}id')
    for child in segment.iter(tag='{http://nite.sourceforge.net/}child'):
        href = child.get('href').split('#')
        speaker = href[0][href[0].find('.') + 1]
        start_word = href[1][href[1].find('(') + 1: href[1].find(')')]
        end_word = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
        keys_list = list(speakers_words[ord(speaker) - ord('A')].keys())
        start_index = keys_list.index(start_word)
        end_index = keys_list.index(end_word)
        words_in_segment = list(speakers_words[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
        words_in_segment.insert(0, speaker + ': ')
        speaker_segments[segment_id] = words_in_segment
    for segment in segment.findall('segment'):
        get_words_in_segment(segment, speaker_segments, speakers_words)


def parse_segments_file(data_root, segments_folder, meeting_name, speaker_name, speakers_words):
    """

    :param data_root: root folder of all data
    :param segments_folder: folder of all segments
    :param meeting_name: name of the meeting being analyzed
    :param speaker_name: name or letter of the speaker
    :param speakers_words: all of the words of speakers
    :return: segments of one speaker
    """
    tree = ET.parse(os.path.join(data_root, segments_folder, meeting_name + '.' + speaker_name + '.segs.xml'))
    root = tree.getroot()

    speaker_segments = OrderedDict()
    for segment in root.findall('segment'):
        get_words_in_segment(segment, speaker_segments, speakers_words)

    return speaker_segments


def get_segments(data_root, segments_folder, meeting_name, speakers_words):
    """

    :param data_root: root folder of all data
    :param segments_folder: folder of all segments
    :param meeting_name: name of the meeting being analyzed
    :param speakers_words: all of the words of speakers
    :return: segments of all of the speakers
    """
    speakers_segments = []

    # get the words of meeting
    for c in string.ascii_uppercase:
        try:
            speaker_segments = parse_segments_file(data_root, segments_folder, meeting_name, c, speakers_words)
            speakers_segments.append(speaker_segments)
        except FileNotFoundError:
            break
    return speakers_segments


def get_words_and_descriptions_in_topic(topic, segments, words_in_topics_dict, descriptions_in_topics_dict):
    """

    :param topic: one topic
    :param segments: segments of all of the speakers
    :param words_in_topics_dict: words in each topic which is here to be filled
    :param descriptions_in_topics_dict: descriptions in each topic which is here to be filled
    """
    topic_id = topic.get('{http://nite.sourceforge.net/}id')
    topic_description = topic.get('description')
    segments_in_topic = []
    for child in topic.findall('{http://nite.sourceforge.net/}child'):
        href = child.get('href').split('#')
        speaker = href[0][href[0].find('.') + 1]
        start_seg = href[1][href[1].find('(') + 1: href[1].find(')')]
        end_seg = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
        keys_list = list(segments[ord(speaker) - ord('A')].keys())
        start_index = keys_list.index(start_seg)
        end_index = keys_list.index(end_seg)
        words_in_segment = list(segments[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
        segments_in_topic.append(words_in_segment)
    words_in_topics_dict[topic_id] = segments_in_topic
    descriptions_in_topics_dict[topic_id] = topic_description
    for topic in topic.findall('topic'):
        get_words_and_descriptions_in_topic(topic, segments, words_in_topics_dict, descriptions_in_topics_dict)


def get_words_in_topics(data_root, topics_folder, meeting_name, segments):
    """
    :param data_root: root folder of all data
    :param meeting_name: name of the meeting being analyzed
    :param topics_folder: folder of all topics
    :param segments:segments of all of the speakers
    :return: dictionaries of words and descriptions of each topic
    """
    # parsing a topic file
    meeting_topics = ET.parse(os.path.join(data_root, topics_folder, meeting_name + '.topic.xml'))
    root = meeting_topics.getroot()

    topic_description = OrderedDict()
    words_in_topics = OrderedDict()
    for each_topic in root:
        get_words_and_descriptions_in_topic(each_topic, segments, words_in_topics, topic_description)

    return words_in_topics, topic_description


def save_meeting_converses_by_topic(output_root, meeting_name, words_in_topics, descriptions):
    """
    :param descriptions: descriptions of topics
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting being analyzed
    :param words_in_topics: words in the topics
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the transcripts by topic
    transcrpits_by_topic = open(output_dir_for_meeting + "/conv_by_topic.txt", "w+")
    for topics_it in words_in_topics:
        for words_of_each_speaker in words_in_topics[topics_it]:
            for one_segment in words_of_each_speaker:
                for word_it in one_segment:
                    transcrpits_by_topic.write(word_it + " ")
                transcrpits_by_topic.write('\n')
        transcrpits_by_topic.write('topic_description:\t' + str(descriptions[topics_it]) + "\n")
        transcrpits_by_topic.write('\n\n')

    transcrpits_by_topic.close()
    return


def save_meeting_topics_descriptions(output_root, meeting_name, descriptions):
    """
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting being analyzed
    :param descriptions: words in the topics
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the topic file
    topic_file = open(output_dir_for_meeting + "/topic_descriptions.txt", "w+")
    for desc in descriptions:
        topic_file.write(str(descriptions[desc]) + "\n")
    topic_file.close()


def get_speaker_dialogue_acts(data_root, dialogue_acts_folder, meeting_name, speakers_words, c):
    """
    :param data_root: root folder of all data
    :param dialogue_acts_folder: folder of all dialogue acts
    :param meeting_name: name of the meeting being analyzed
    :param speakers_words: list of speakers each containing their list of words
    :param c: speaker name
    :return: speaker_das, speaker_das_descriptions
    """
    speaker_das = OrderedDict()
    tree = ET.parse(os.path.join(data_root, dialogue_acts_folder, meeting_name + '.' + c + '.dialog-act.xml'))
    root = tree.getroot()
    for da in root:
        words_in_da = []
        index = da.get('{http://nite.sourceforge.net/}id')
        for child in da.findall('{http://nite.sourceforge.net/}child'):
            href = child.get('href').split('#')
            speaker = href[0][href[0].find('.') + 1]
            start_word = href[1][href[1].find('(') + 1: href[1].find(')')]
            end_word = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
            keys_list = list(speakers_words[ord(speaker) - ord('A')].keys())
            start_index = keys_list.index(start_word)
            end_index = keys_list.index(end_word)
            words_in_da = list(speakers_words[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
            words_in_da.insert(0, speaker + ': ')
            speaker_das[id] = words_in_da

        speaker_das[index] = words_in_da
    return speaker_das


def save_meeting_dialogue_acts(output_root, meeting_name, speaker_das, c):
    """
    :param output_root: main folder of output data
    :param c: speaker name
    :param meeting_name: name of the meeting being analyzed
    :param speaker_das: dialogue acts of the speaker
    """
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


def get_extractive_summary(data_root, extractive_sum_folder, meeting_name, speakers_das):
    """
    :param data_root: root folder of all data
    :param extractive_sum_folder: folder of extractive summaries
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
                start_seg = href[1][href[1].find('(') + 1: href[1].find(')')]
                end_seg = href[1][href[1].rfind('(') + 1: href[1].rfind(')')]
                try:
                    keys_list = list(speakers_das[ord(speaker) - ord('A')].keys())
                except IndexError:
                    print("problem with meeting %s. speaker %s has no dialogue act" % (meeting_name, speaker))
                    return None
                    # because the meeting BMR012 has no speaker speaker J dialogue acts file
                start_index = keys_list.index(start_seg)
                end_index = keys_list.index(end_seg)
                words_in_da = list(speakers_das[ord(speaker) - ord('A')].values())[start_index: end_index + 1]
                das_in_summary.append(words_in_da)

    return das_in_summary


def save_meeting_extractive_summary(output_root, meeting_name, das_in_summary):
    """
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting_name being analyzed
    :param das_in_summary: dialogue acts which are included in summary
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the summary file
    extractive_summary = open(output_dir_for_meeting + "/compl_extr_summ.txt", "w+")
    for das in das_in_summary:
        for da in das:
            for word in da:
                extractive_summary.write(word + " ")
            extractive_summary.write('\n')
        extractive_summary.write('\n')


def get_dialogue_acts(data_root, dialogue_acts_folder, meeting_name, speakers_words, output_root):
    """

    :param data_root: main folder of output data
    :param dialogue_acts_folder: folder of all dialogue acts
    :param meeting_name: name of the meeting_name being analyzed
    :param speakers_words: list of speakers each containing their list of words
    :param output_root: main folder of output data
    :return: dialogue acts of all of speakers
    """
    # get the dialogue_acts
    speakers_das = []
    for c in string.ascii_uppercase:
        try:
            speaker_das = get_speaker_dialogue_acts(data_root, dialogue_acts_folder, meeting_name, speakers_words, c)
            speakers_das.append(speaker_das)
        except FileNotFoundError:
            continue
        save_meeting_dialogue_acts(output_root, meeting_name, speaker_das, c)
    return speakers_das


def get_the_abstractive_summary(data_root, abstractive_sum_folder, meeting_name):
    """
    :param data_root: main folder of output data
    :param abstractive_sum_folder: folder of abstractive summaries
    :param meeting_name: name of the meeting_name being analyzed
    """
    summ = ET.parse(os.path.join(data_root, abstractive_sum_folder, meeting_name + '.abssumm.xml'))
    root = summ.getroot()
    summaries = OrderedDict()
    for abstracts in root.findall('abstract'):
        for sentence in abstracts.findall('sentence'):
            summaries[sentence.get('{http://nite.sourceforge.net/}id')] = 'abstract: ' + sentence.text

    for actions in root.findall('actions'):
        for sentence in actions.findall('sentence'):
            summaries[sentence.get('{http://nite.sourceforge.net/}id')] = 'action: ' + sentence.text

    for decisions in root.findall('decisions'):
        for sentence in decisions.findall('sentence'):
            summaries[sentence.get('{http://nite.sourceforge.net/}id')] = 'decisions: ' + sentence.text

    for problems in root.findall('problems'):
        for sentence in problems.findall('sentence'):
            summaries[sentence.get('{http://nite.sourceforge.net/}id')] = 'problems: ' + sentence.text

    for issues in root.findall('issues'):
        for sentence in issues.findall('sentence'):
            summaries[sentence.get('{http://nite.sourceforge.net/}id')] = 'problems: ' + sentence.text
    return summaries


def get_the_dialogue_acts_related_to_abssumms(data_root, extractive_sum_folder, meeting_name, speakers_das):
    """

    :param data_root: root folder of all data
    :param extractive_sum_folder: folder of extractive summaries
    :param meeting_name: name of the meeting being analyzed
    :param speakers_das: dialogue acts of all of speakers
    :return: dictionary from summaries to their corresponding dialogue acts
    """
    summ = ET.parse(os.path.join(data_root, extractive_sum_folder, meeting_name + '.summlink.xml'))
    root = summ.getroot()
    summ_to_da_dict = OrderedDict()
    for summlink in root.findall('summlink'):
        selected_das = []
        for pointer in summlink:
            if pointer.get('role') == 'extractive':
                href = pointer.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                da_id = href[1][href[1].rfind('(') + 1: href[1].find(')')]
                try:
                    selected_das = speakers_das[ord(speaker) - ord('A')][da_id].copy()
                except IndexError:
                    print("problem with meeting %s. speaker %s has no dialogue act" % (meeting_name, speaker))
                    return None
                    # because the meeting BMR012 has no speaker speaker J dialogue acts file
            elif pointer.get('role') == 'abstractive':
                href = pointer.get('href').split('#')
                summary_id = href[1][href[1].find('id') + 3: -1]
                if summary_id in summ_to_da_dict:
                    summ_to_da_dict[summary_id].append(selected_das)
                else:
                    summ_to_da_dict[summary_id] = [selected_das]
    return summ_to_da_dict


def save_meeting_abs_summaries_and_related_das(output_root, meeting_name, abs_summeries, summ_to_da_dict):
    """

    :param meeting_name: name of the meeting_name being analyzed
    :param output_root: main folder of output data
    :param meeting_name: name of the meeting being analyzed
    :param abs_summeries: abstract summaries
    :param summ_to_da_dict: dictionary from summaries to their corresponding dialogue acts
    """
    # create a directory for meeting_name
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the summary file
    abstractiv_summary = open(output_dir_for_meeting + "/abst_summs.txt", "w+", encoding="utf-8")
    for abs_summ in abs_summeries:
        if summ_to_da_dict.get(abs_summ) is None:
            # some abstract summaries are not mapped to any dialogue acts
            abstractiv_summary.write('None\n')
        else:
            for da in summ_to_da_dict.get(abs_summ):
                for word in da:
                    abstractiv_summary.write(word + ' ')
                abstractiv_summary.write('\n')
        try:
            abstractiv_summary.write('abst_sum - ' + abs_summeries.get(abs_summ) + '\n\n')
        except UnicodeEncodeError as e:
            raise e

def initialize_arguemnents():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source directory')
    parser.add_argument('--dest', default='./dest', help='destination directory')
    args = parser.parse_args()
    return args


def recover_icsi_from_source():
    args = initialize_arguemnents()

    data_root = args.src
    output_root = args.dest

    if not os.path.exists(data_root):
        sys.exit('source directory does not exist.')
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    words_folder = 'words'
    topics_folder = 'topics'
    segments_folder = 'segments'
    dialogue_acts_folder = 'dialogueActs'
    extractive_sum_folder = 'extractive'
    abstractive_sum_folder = 'abstractive'

    # get set of meetings' names
    name_of_meetings = {x.split('.')[0] for x in os.listdir(os.path.join(data_root, words_folder))}
    name_of_meetings = sorted(name_of_meetings)

    for meeting in name_of_meetings:
        speakers_words = get_speakers_words(data_root, topics_folder, words_folder, meeting)
        if speakers_words is None:
            continue

        # segments = get_segments(data_root, segments_folder, meeting, speakers_words)

        # words_in_topics, topic_description = get_words_in_topics(data_root, topics_folder, meeting, segments)
        # save_meeting_converses_by_topic(output_root, meeting, words_in_topics, topic_description)

        speakers_das = get_dialogue_acts(data_root, dialogue_acts_folder, meeting, speakers_words, output_root)

        # extractive_summary
        try:
            das_in_summary = get_extractive_summary(data_root, extractive_sum_folder, meeting, speakers_das)
            if das_in_summary is not None:
                save_meeting_extractive_summary(output_root, meeting, das_in_summary)
        except FileNotFoundError:
            print('Extractive Summary is not available for meeting ' + meeting)

        # abstractive_summary
        try:
            abs_summeries = get_the_abstractive_summary(data_root, abstractive_sum_folder, meeting)
            summ_to_da_dict = get_the_dialogue_acts_related_to_abssumms(data_root, extractive_sum_folder, meeting,
                                                                        speakers_das)
            if summ_to_da_dict is not None:
                save_meeting_abs_summaries_and_related_das(output_root, meeting, abs_summeries, summ_to_da_dict)
        except FileNotFoundError:
            print('abstractive Summary is not available for meeting ' + meeting)

        print(meeting)

if __name__=="__main__":
    recover_icsi_from_source()
