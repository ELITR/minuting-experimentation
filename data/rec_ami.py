
''' Recovering AMI corpus transcripts from the XML format '''

import argparse
import os
import string
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict


# check if a meeting has a topic file or not
def has_topic(data_root, topics_folder, meeting_name):
    """
    :param data_root: root folder of all data
    :param topics_folder:  folder of all topics
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
    """
    :param data_root: root folder of all data
    :param topics_folder: folder of all topics
    :param words_folder: folder of all words
    :param meeting_name: name of the meeting being analyzed
    :return speakers_words: list of speakers containing lists of their words - list of lists of strings
    """
    # if there is no topic for this meeting, just ignore it
    if not has_topic(data_root, topics_folder, meeting_name):
        return

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


# recover the default topics and their ids
def parse_default_topics(data_root, ontologies_folder):
    """
    :param data_root: root folder of all data
    :param ontologies_folder: folder of ontologies
    :return default_topic_dict - dictionary of ids and topic descriptions
    """
    default_topics_dict = dict()
    default_topics = ET.parse(os.path.join(data_root, ontologies_folder, 'default-topics.xml')).getroot()
    for topic_names in default_topics:
        default_topics_dict['id(' + topic_names.get('{http://nite.sourceforge.net/}id') + ')'] = topic_names.get('name')
        for topic_name in topic_names:
            default_topics_dict['id(' + topic_name.get('{http://nite.sourceforge.net/}id') + ')'] = topic_name.get(
                'name')
    return default_topics_dict


def get_words_in_topic(topic, speakers_words, output_words_in_topics_dict, output_descriptions_in_topics_dict,
                       default_topics_dict):
    topic_id = topic.get('{http://nite.sourceforge.net/}id')
    topic_description = topic.get('other_description')
    words_in_topic = []
    for child in topic.findall('{http://nite.sourceforge.net/}pointer'):
        href = child.get('href').split('#')[1]
        other_description = str(topic.get('other_description'))
        if other_description is None:
            topic_description = default_topics_dict[href]
        else:
            topic_description = default_topics_dict[href] + '_' + other_description
    for child in topic.findall('{http://nite.sourceforge.net/}child'):
        href = child.get('href').split('#')
        speaker = href[0][href[0].find('.') + 1]
        start_word = int(href[1][href[1].find('words') + 5: href[1].find(')')])
        end_word = int(href[1][href[1].rfind('words') + 5: href[1].rfind(')')])
        words_in_subtopic = [speaker + ': '] + speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]
        words_in_topic.append(words_in_subtopic)
    output_words_in_topics_dict[topic_id] = words_in_topic
    output_descriptions_in_topics_dict[topic_id] = topic_description
    for topic in topic.findall('topic'):
        get_words_in_topic(topic, speakers_words, output_words_in_topics_dict, output_descriptions_in_topics_dict,
                           default_topics_dict)
    return


# returns words in topics and their description for each topic
def get_words_in_topics(data_root, topics_folder, meeting_name, default_topics_dict, speakers_words):
    """
    :param data_root: root folder of all data
    :param topics_folder: folder of all topics
    :param meeting_name: name of the meeting being analyzed
    :param default_topics_dict: dictionary of default topics
    :param speakers_words: list of speakers each containing their list of words
    :return: words_in_topics, topic_description
    """
    # parsing a topic file
    meeting_topics = ET.parse(os.path.join(data_root, topics_folder, meeting_name + '.topic.xml'))
    root = meeting_topics.getroot()

    topic_description = OrderedDict()
    words_in_topics = OrderedDict()

    for each_topic in root:
        get_words_in_topic(each_topic, speakers_words, words_in_topics, topic_description, default_topics_dict)
    return words_in_topics, topic_description


def save_meeting_converses_by_topic(output_root, meeting_name, words_in_topics, topic_description):
    """
    :param topic_description:
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
            for word_it in words_of_each_speaker:
                transcrpits_by_topic.write(word_it + " ")
            transcrpits_by_topic.write('\n')
        transcrpits_by_topic.write('topic_description:\t' + str(topic_description[topics_it]) + "\n")
        transcrpits_by_topic.write('\n\n')
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
        topic_file.write(str(desc) + "\n")
    topic_file.close()
    return


# recover the default dialogue acts and their IDs
def parse_default_dialogue_acts(data_root, ontologies_folder):
    """
    :param data_root: root folder of all data
    :param ontologies_folder: folder of ontologies
    :return default_das_dict - dictionary of IDs and dialogue acts
    """
    # get the default dialogue acts
    default_das_dict = dict()
    default_das = ET.parse(os.path.join(data_root, ontologies_folder, 'da-types.xml')).getroot()
    for da_names in default_das:
        default_das_dict['id(' + da_names.get('{http://nite.sourceforge.net/}id') + ')'] = da_names.get('gloss')
        for da_name in da_names:
            default_das_dict['id(' + da_name.get('{http://nite.sourceforge.net/}id') + ')'] = da_name.get('gloss')
    return default_das_dict


def get_speaker_dialogue_acts(data_root, dialogue_acts_folder, meeting_name, speakers_words, default_das_dict, c):
    """
    :param c: speaker name
    :param data_root: root folder of all data
    :param dialogue_acts_folder: folder of all dialogue acts
    :param meeting_name: name of the meeting being analyzed
    :param default_das_dict: dictionary of default dialogue acts
    :param speakers_words: list of speakers each containing their list of words
    :return: speaker_das, speaker_das_descriptions
    """
    speaker_das = OrderedDict()
    speaker_das_descriptions = OrderedDict()
    tree = ET.parse(os.path.join(data_root, dialogue_acts_folder, meeting_name + '.' + c + '.dialog-act.xml'))
    root = tree.getroot()
    for da in root:
        words_in_da = []
        index = int(
            da.get('{http://nite.sourceforge.net/}id')[da.get('{http://nite.sourceforge.net/}id').rfind('.') + 1:])
        has_pointer = 0
        for child in da:
            if str(child.tag).find('pointer') != -1:
                href = child.get('href').split('#')[1]
                speaker_das_descriptions[index] = default_das_dict[href]
                has_pointer = 1
            elif str(child.tag).find('child') != -1:
                href = child.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                if has_pointer == 0:
                    print('Unusual case! in meeting %s, speaker %c, dialogue act number %s: has no description' % (
                        meeting_name, speaker, index))
                    speaker_das_descriptions[index] = 'None'
                    # Some unusual dialogue acts have no pointer and also no description!
                else:
                    has_pointer = 0
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
    """
    :param output_root: destination folder for the outputs.
    :param meeting_name: name of the meeting being analyzed
    :param speaker_das: dialogue acts of the speaker
    :param c: speaker name
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


def save_meeting_dialogue_acts_descriptions(output_root, meeting_name, speaker_das_descriptions, c):
    """
    :param output_root: destination folder for the outputs.
    :param meeting_name: name of the meeting being analyzed
    :param speaker_das_descriptions: descriptions of speaker das
    :param c: speaker name
    """
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


def get_extractive_summary(data_root, extractive_sum_folder, meeting_name, speakers_das):
    """
    :rtype: list of dialogue acts which are included in summary
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
                ids = href[1].split('..')

                selected_das = [speaker + ': ']
                start_da = int(ids[0][ids[0].rfind('.') + 1:ids[0].rfind(')')])
                if len(ids) == 1:
                    end_da = start_da
                else:
                    end_da = int(ids[1][ids[1].rfind('.') + 1:ids[1].rfind(')')])
                for i in range(start_da, end_da + 1):
                    try:
                        selected_das.append(speakers_das[ord(speaker) - ord('A')][i])
                    except KeyError:
                        # some dialogue acts are missing, its natural.
                        pass
                das_in_summary.append(selected_das)
    return das_in_summary


def save_meeting_extractive_summary(output_root, meeting_name, das_in_summary):
    """
    :param output_root: destination folder for the outputs.
    :param meeting_name: name of the meeting_name being analyzed
    :param das_in_summary: list of dialogue acts which are included in summary
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the summary file
    extractiv_summary = open(output_dir_for_meeting + "/compl_extr_summ.txt", "w+")
    for das in das_in_summary:
        speaker = das[0]
        extractiv_summary.write(speaker + " ")
        for word in das[1]:
            extractiv_summary.write(word + " ")
        extractiv_summary.write('\n\n')


def get_the_abstractive_summary(data_root, abstractive_sum_folder, meeting_name):
    """
    :param data_root: root folder of all data
    :param abstractive_sum_folder: folder of abstractive summaries
    :param meeting_name: name of the meeting being analyzed
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


def get_the_dialogue_acts_related_to_abssumms(data_root, extractive_sum_folder, meeting_name, speakers_das,
                                              speakers_das_descriptions):
    """
    :param data_root: root folder of all data
    :param extractive_sum_folder: folder of extractive summaries
    :param meeting_name: name of the meeting being analyzed
    :param speakers_das: dialogue acts of all of speakers
    :param speakers_das_descriptions: descriptions of dialogue acts of all of speakers
    :return: dictionaries from summaries to their corresponding dialogue acts and descriptions
    """
    summ = ET.parse(os.path.join(data_root, extractive_sum_folder, meeting_name + '.summlink.xml'))
    root = summ.getroot()
    summ_to_da_dict = OrderedDict()
    summ_to_da_desc_dict = OrderedDict()
    for summlink in root.findall('summlink'):
        selected_das = []
        selected_das_desc = []
        for pointer in summlink:
            if pointer.get('role') == 'extractive':
                href = pointer.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                da_id = int(href[1][href[1].rfind('.') + 1: href[1].find(')')])
                selected_das = speakers_das[ord(speaker) - ord('A')][da_id].copy()
                selected_das.insert(0, speaker + ': ')
                try:
                    selected_das_desc = speakers_das_descriptions[ord(speaker) - ord('A')][da_id]
                except KeyError as e:
                    raise e
            elif pointer.get('role') == 'abstractive':
                href = pointer.get('href').split('#')
                summary_id = href[1][href[1].find('id') + 3: -1]
                if summary_id in summ_to_da_dict:
                    summ_to_da_dict[summary_id].append(selected_das)
                else:
                    summ_to_da_dict[summary_id] = [selected_das]
                if summary_id in summ_to_da_desc_dict:
                    summ_to_da_desc_dict[summary_id].append(selected_das_desc)
                else:
                    summ_to_da_desc_dict[summary_id] = [selected_das_desc]
    return summ_to_da_dict, summ_to_da_desc_dict


def save_meeting_abs_summaries_and_related_das(output_root, meeting_name, abs_summaries, summ_to_da_dict,
                                               summ_to_da_desc_dict):
    """

    :param summ_to_da_desc_dict: dictionary from summaries to their corresponding dialogue acts descriptions
    :param output_root: destination folder for the outputs
    :param meeting_name: name of the meeting being analyzed
    :param abs_summaries: abstractive summaries
    :param summ_to_da_dict: dictionary from summaries to their corresponding dialogue acts
    """
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output_root, meeting_name)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)

    # writing the summary file
    abstractiv_summary = open(output_dir_for_meeting + "/abst_summs.txt", "w+")
    for abs_summ in abs_summaries:
        if summ_to_da_dict.get(abs_summ) is None:
            # some abstract summaries are not mapped to any dialogue acts
            abstractiv_summary.write('None\n')
        else:
            for da in summ_to_da_dict.get(abs_summ):
                for word in da:
                    abstractiv_summary.write(word + ' ')
                abstractiv_summary.write('\n')
            abstractiv_summary.write('descriptions: ')
            for da_desc in summ_to_da_desc_dict.get(abs_summ):
                abstractiv_summary.write(da_desc)
                abstractiv_summary.write(' ')
        abstractiv_summary.write('\nabst_sum - ' + abs_summaries.get(abs_summ) + '\n\n')


def initialize_arguemnents():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source directory')
    parser.add_argument('--dest', default='./dest', help='destination directory')
    args = parser.parse_args()
    return args


def recover_ami_from_source():
    args = initialize_arguemnents()

    data_root = args.src
    output_root = args.dest

    if not os.path.exists(data_root):
        sys.exit('source directory does not exist.')
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    words_folder = 'words'
    topics_folder = 'topics'
    dialogue_acts_folder = 'dialogueActs'
    extractive_sum_folder = 'extractive'
    abstractive_sum_folder = 'abstractive'
    ontologies_folder = 'ontologies'

    # get set of meetings' names from source directory
    name_of_meetings = {x.split('.')[0] for x in os.listdir(os.path.join(data_root, words_folder))}
    name_of_meetings = sorted(name_of_meetings)

    for meeting in name_of_meetings:
        # list of speakers each containing their list of words
        speakers_words = get_speakers_words(data_root, topics_folder, words_folder, meeting)
        if speakers_words is None:
            continue
        print(meeting)
        # getting default topics
        default_topics_dict = parse_default_topics(data_root, ontologies_folder)
        words_in_topics, topic_description = get_words_in_topics(data_root, topics_folder, meeting, default_topics_dict,
                                                                 speakers_words)
        save_meeting_converses_by_topic(output_root, meeting, words_in_topics, topic_description)
        # save_meeting_topics_descriptions(output_root, meeting, topic_description)

        default_das_dict = parse_default_dialogue_acts(data_root, ontologies_folder)
        # get the dialogue_acts
        speakers_das = []
        speakers_das_descriptions = []
        for c in string.ascii_uppercase:
            try:
                speaker_das, speaker_das_descriptions = get_speaker_dialogue_acts(data_root, dialogue_acts_folder,
                                                                                  meeting, speakers_words,
                                                                                  default_das_dict,
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
            print('Extractive Summary is not available for meeting ' + meeting)

        # abstractive_summary
        try:
            abs_summaries = get_the_abstractive_summary(data_root, abstractive_sum_folder, meeting)
            summ_to_da_dict, summ_to_da_desc_dict = get_the_dialogue_acts_related_to_abssumms(data_root,
                                                                                              extractive_sum_folder,
                                                                                              meeting,
                                                                                              speakers_das,
                                                                                              speakers_das_descriptions)
            save_meeting_abs_summaries_and_related_das(output_root, meeting, abs_summaries, summ_to_da_dict,
                                                       summ_to_da_desc_dict)
        except FileNotFoundError:
            # raise e
            print('abstractive Summary is not available for meeting ' + meeting)

    print("Done!")

if __name__=="__main__":
    recover_ami_from_source()
