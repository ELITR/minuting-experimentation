import os
import string
import xml.etree.ElementTree as ET

output = './output'
if not os.path.exists(output):
    os.makedirs(output)

data_directory = './data'
words_folder = 'words'
topics_folder = 'topics'
segments_folder = 'segments'
dialogue_acts_folder = 'dialogueActs'
extractive_sum_folder = 'extractive'
abstractive_sum_folder = 'abstractive'
ontologies_folder = 'ontologies'


# for name in
name_of_meetings = {x.split('.')[0] for x in os.listdir(os.path.join(data_directory, words_folder))}
print(name_of_meetings)


for meeting in name_of_meetings:
    # if there is no topic for this meeting, just ignore it
    if not os.path.exists(os.path.join(data_directory, topics_folder, meeting + '.topic.xml')):
        continue

    speakers_words = []

    # get the words of meeting
    for c in string.ascii_uppercase:
        try:
            tree = ET.parse(os.path.join(data_directory, words_folder, meeting + '.' + c + '.words.xml'))
            # speakers_words.append(tree)
            root = tree.getroot()
            words = []
            for child in root:
                tag = child.tag
                if tag == 'w':
                    words.append(child.text)
                elif tag == 'vocalsound':
                    words.append('<' + child.get('type') + '>')
                else:
                    words.append('')
                # print(child.tag, child.attrib, child.text)
            speakers_words.append(words)
        except FileNotFoundError:
            pass

    # get the topic file of meeting
    default_topics_dict = dict()
    default_topics = ET.parse(os.path.join(data_directory, ontologies_folder, 'default-topics.xml')).getroot()
    for topic_names in default_topics:
        default_topics_dict['id(' + topic_names.get('{http://nite.sourceforge.net/}id') + ')'] = topic_names.get('name')
        for topic_name in topic_names:
            default_topics_dict['id(' + topic_name.get('{http://nite.sourceforge.net/}id') + ')'] = topic_name.get('name')

    topics = ET.parse(os.path.join(data_directory, topics_folder, meeting + '.topic.xml'))
    root = topics.getroot()
    descriptions = []
    topics_words = []
    for topic in root:
        # descriptions.append(topic.get('other_description'))
        words_in_topic = []
        for child in topic:
            if str(child.tag).find('pointer') != -1:
                href = child.get('href').split('#')[1]
                descriptions.append(default_topics_dict[href] + ':\t' + str(topic.get('other_description')))
            elif str(child.tag).find('child') != -1:
                href = child.get('href').split('#')
                speaker = href[0][href[0].find('.') + 1]
                start_word = int(href[1][href[1].find('words') + 5: href[1].find(')')])
                end_word = int(href[1][href[1].rfind('words') + 5: href[1].rfind(')')])
                selected_words = [speaker + ': '] + speakers_words[ord(speaker) - ord('A')][start_word:end_word + 1]
                words_in_topic.append(selected_words)
                # print(child.attrib)
                # print(child.tag, child.attrib, child.text)
        topics_words.append(words_in_topic)
    # create a directory for meeting
    output_dir_for_meeting = os.path.join(output, meeting)
    if not os.path.exists(output_dir_for_meeting):
        os.makedirs(output_dir_for_meeting)
    # writing the topic file
    topic_file = open(output_dir_for_meeting + "/topic.txt", "w+")
    for desc in descriptions:
        topic_file.write(str(desc) + "\n")
    topic_file.close()
    # writing the transcripts by topic
    transcrpits_by_topic = open(output_dir_for_meeting + "/transcrpits_by_topic.txt", "w+")
    for topics_it in topics_words:
        for words_of_each_speaker in topics_it:
            for word_it in words_of_each_speaker:
                transcrpits_by_topic.write(word_it + " ")
            transcrpits_by_topic.write('\n')
        transcrpits_by_topic.write('\n\n')


    # get the default dialogue acts
    default_das_dict = dict()
    default_das = ET.parse(os.path.join(data_directory, ontologies_folder, 'da-types.xml')).getroot()
    for da_names in default_das:
        default_das_dict['id(' + da_names.get('{http://nite.sourceforge.net/}id') + ')'] = da_names.get('gloss')
        for da_name in da_names:
            default_das_dict['id(' + da_name.get('{http://nite.sourceforge.net/}id') + ')'] = da_name.get('gloss')

    # get the dialogue_acts
    speakers_das = []
    speakers_das_descriptions = []
    for c in string.ascii_uppercase:
        try:
            tree = ET.parse(os.path.join(data_directory, dialogue_acts_folder, meeting + '.' + c + '.dialog-act.xml'))
            root = tree.getroot()
            for da in root:
                words_in_da = []
                for child in da:
                    if str(child.tag).find('pointer') != -1:
                        href = child.get('href').split('#')[1]
                        speakers_das_descriptions.append(default_das_dict[href])
                    elif str(child.tag).find('child') != -1:
                        href = child.get('href').split('#')
                        speaker = href[0][href[0].find('.') + 1]
                        start_word = int(href[1][href[1].find('words') + 5: href[1].find(')')])
                        end_word = int(href[1][href[1].rfind('words') + 5: href[1].rfind(')')])
                        selected_words = speakers_words[ord(speaker) - ord('A')][
                                                            start_word:end_word + 1]
                        words_in_da.append(selected_words)
                speakers_das.append(words_in_da)
                # TODO
                # das_words.append(words_in_da)
            # writing the dialogue_acts file
            da_file = open(output_dir_for_meeting + "/dialogue_acts_" + str(c) + ".txt", "w+")
            for desc in speakers_das_descriptions:
                da_file.write(str(desc) + "\n")
            da_file.close()
            # writing the transcripts by da
            transcrpits_by_da = open(output_dir_for_meeting + "/transcrpits_by_da_" + str(c) + ".txt", "w+")
            for das_it in speakers_das:
                for words_of_each_speaker in das_it:
                    for word_it in words_of_each_speaker:
                        transcrpits_by_da.write(word_it + " ")
                    transcrpits_by_da.write('\n')
                transcrpits_by_da.write('\n')
        except FileNotFoundError:
            pass

    # extractive_summary
    try:
        summ = ET.parse(os.path.join(data_directory, extractive_sum_folder, meeting + '.extsumm.xml'))
        root = summ.getroot()
        for summ in root:
            # descriptions.append(topic.get('other_description'))
            words_in_summary = []
            for child in summ:
                if str(child.tag).find('pointer') != -1:
                    pass
                elif str(child.tag).find('child') != -1:
                    href = child.get('href').split('#')
                    speaker = href[0][href[0].find('.') + 1]
                    start_word = int(href[1][href[1].find('dharshi') + 8: href[1].find(')')])
                    end_word = int(href[1][href[1].rfind('dharshi') + 8: href[1].rfind(')')])
                    selected_words = [speaker + ': '] + speakers_das[ord(speaker) - ord('A')][start_word:end_word + 1]
                    words_in_topic.append(selected_words)
                    # print(child.attrib)
                    # print(child.tag, child.attrib, child.text)
            topics_words.append(words_in_topic)
        # create a directory for meeting
        output_dir_for_meeting = os.path.join(output, meeting)
        if not os.path.exists(output_dir_for_meeting):
            os.makedirs(output_dir_for_meeting)
        # writing the topic file
        topic_file = open(output_dir_for_meeting + "/topic.txt", "w+")
        for desc in descriptions:
            topic_file.write(str(desc) + "\n")
        topic_file.close()
        # writing the transcripts by topic
        transcrpits_by_topic = open(output_dir_for_meeting + "/transcrpits_by_topic.txt", "w+")
        for topics_it in topics_words:
            for words_of_each_speaker in topics_it:
                for word_it in words_of_each_speaker:
                    transcrpits_by_topic.write(word_it + " ")
                transcrpits_by_topic.write('\n')
            transcrpits_by_topic.write('\n\n')
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
