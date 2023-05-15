import argparse
import json
import os
import re
import sys

import spacy
from tqdm import tqdm

MODELS_DIR = 'data/models'


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_path', type=str, default='data/VUA18',
                        help='Directory of where vau18 data held.')
    return parser.parse_args()


sentiment_map = {1: 'metaphor', -1: 'non_meta'}

def read_file(file_name, parser):
    '''
    Read twitter data and extract text and store.
    return sentences of [sentence, aspect_sentiment, from_to]
    '''
    print('loading data...')
    with open(file_name, 'r') as f:
        data = f.readlines()
        data = [d.strip('\n') for d in data]
    # list of dict {text, aspect, sentiment}
    sentences = []
    i = 0
    while i<len(data):
        ori_sentence = data[i]
        ori_index = data[i+1]
        ori_pos = data[i+2]
        text = data[i+3]
        aspect = data[i+4]
        sentiment = data[i+5]
        i+=6
        try:
            sentence = get_sentence(text, aspect, sentiment, ori_sentence, ori_index, ori_pos, parser)
        except:
            print(i)
            continue
        else:
            sentences.append(sentence)
    print(file_name, len(sentences))
    with open(file_name.replace('.raw', '.txt'), 'w') as f:
        for sentence in sentences:
            f.write(sentence['sentence'] + '\n')

    return sentences

def get_sentence(text, aspect, sentiment, ori_sentence, ori_index, ori_pos, parser):
    sentence = dict()
    sentence['sentence'] = text.replace('$T$', aspect)
    sentence['aspect_sentiment'] = [[aspect, sentiment_map[int(sentiment)]]]
    frm = text.split().index('$T$')
    sentence['ori_sentence'] = ori_sentence
    sentence['ori_aspect'] = aspect
    sentence['ori_index'] = ori_index
    sentence['ori_pos'] = ori_pos
    to = frm + len(aspect.split())
    index = 0
    t_to = 0
    t_frm = 0
    for i, token in enumerate(sentence['sentence'].split()):
        if i == frm:
            t_frm = index
        elif i == to:
            t_to = index - 1
            break
        index += len(parser(token))
    if t_to < t_frm:
        t_to = len(parser(sentence['sentence']))-1
    #print(frm, to,'$$$$$$$', t_frm, t_to)
    sentence['from_to'] = [[t_frm, t_to]]
    return sentence

def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences

def text2docs(file_path, parser):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    for st in sentences:
        print(st.split())
        break
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(parser(str(sentences[i])))
        
    return docs

def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    

    sentence = {}
    sentence['tokens'] = []
    sentence['predicted_dependencies'] = []
    sentence['predicted_heads'] = []
    sentence['dependencies'] = []
    sentence['tags'] = []
    for token in doc:
        sentence['tokens'].append(token.text)
        if token.dep_ == 'ROOT':
            head = 0
        else:
            head = token.head.i+1
        dependence = [token.dep_, head, (token.i+1)]
        sentence['dependencies'].append(dependence)
        sentence['predicted_heads'].append(head)
        sentence['predicted_dependencies'].append(token.dep_)
        sentence['tags'].append(token.pos_)

    return sentence


def syntaxInfo2json(sentences, sentences_with_dep, file_name):
    json_data = []
    #tk = TreebankWordTokenizer()
    # mismatch_counter = 0
    for idx, sentence in enumerate(sentences):
        sentence['tokens'] = sentences_with_dep[idx]['tokens']
        sentence['tags'] = sentences_with_dep[idx]['tags']
        sentence['predicted_dependencies'] = sentences_with_dep[idx]['predicted_dependencies']
        sentence['dependencies'] = sentences_with_dep[idx]['dependencies']
        sentence['predicted_heads'] = sentences_with_dep[idx]['predicted_heads']
        # sentence['energy'] = sentences_with_dep[idx]['energy']
        json_data.append(sentence)
    for sentence in sentences:
        print(sentence['tokens'])
        print(sentence['ori_sentence'])
        break
    
    with open(file_name.replace('.txt', '_rospacy.json'), 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))


def main():
    args = parse_args()

    parser = spacy.load("en_core_web_trf")
    print('predictor loaded!')

    train_file = os.path.join(args.data_path, 'train.raw')
    test_file = os.path.join(args.data_path, 'test.raw')
    val_file = os.path.join(args.data_path, 'val.raw')

    # raw -> txt -> json
    
    train_sentences = read_file(train_file, parser)
    train_sentences_with_dep = get_dependencies(os.path.join(args.data_path, 'train.txt'), parser)    
    syntaxInfo2json(train_sentences, train_sentences_with_dep, os.path.join(args.data_path, 'train.txt'))
    
    test_sentences = read_file(test_file, parser)
    test_sentences_with_dep = get_dependencies(os.path.join(args.data_path, 'test.txt'), parser)
    syntaxInfo2json(test_sentences, test_sentences_with_dep, os.path.join(args.data_path, 'test.txt'))
    
    val_sentences = read_file(val_file, parser)
    val_sentences_with_dep = get_dependencies(os.path.join(args.data_path, 'val.txt'), parser)
    syntaxInfo2json(val_sentences, val_sentences_with_dep, os.path.join(args.data_path, 'val.txt'))
    

if __name__ == "__main__":
    main()



























    
