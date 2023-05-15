from torch.nn.utils.rnn import pad_sequence
import argparse
import codecs
import json
import linecache
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import numpy as np
import simplejson as json
import torch
from allennlp.modules.elmo import batch_to_ids
from lxml import etree
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset

from senten_pair import sentence_pair

logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args):
    train, test, val = get_dataset(args.dataset_name)

    # Our model takes unrolled data, currently we don't consider the MAMS cases(future experiments)
    _, train_all_unrolled, _, _ = get_rolled_and_unrolled_data(train, args)
    _, test_all_unrolled, _, _ = get_rolled_and_unrolled_data(test, args)
    _, val_all_unrolled, _, _ = get_rolled_and_unrolled_data(val, args)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_all_unrolled))
    logger.info('Test set size: %s,', len(test_all_unrolled))
    logger.info('val set size: %s,', len(val_all_unrolled))
        
    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_all_unrolled+test_all_unrolled+val_all_unrolled, args)
    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding

    train_dataset = ASBA_Depparsed_Dataset(
        train_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    val_dataset = ASBA_Depparsed_Dataset(
        val_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)

    return train_dataset, test_dataset, val_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def get_dataset(dataset_name):
    '''
    Already preprocess the data and now they are in json format.(only for semeval14)
    Retrieve train and test set
    With a list of dict:
    e.g. {"sentence": "Boot time is super fast, around anywhere from 35 seconds to 1 minute.",
    "tokens": ["Boot", "time", "is", "super", "fast", ",", "around", "anywhere", "from", "35", "seconds", "to", "1", "minute", "."],
    "tags": ["NNP", "NN", "VBZ", "RB", "RB", ",", "RB", "RB", "IN", "CD", "NNS", "IN", "CD", "NN", "."],
    "predicted_dependencies": ["nn", "nsubj", "root", "advmod", "advmod", "punct", "advmod", "advmod", "prep", "num", "pobj", "prep", "num", "pobj", "punct"],
    "predicted_heads": [2, 3, 0, 5, 3, 5, 8, 5, 8, 11, 9, 9, 14, 12, 3],
    "dependencies": [["nn", 2, 1], ["nsubj", 3, 2], ["root", 0, 3], ["advmod", 5, 4], ["advmod", 3, 5], ["punct", 5, 6], ["advmod", 8, 7], ["advmod", 5, 8],
                    ["prep", 8, 9], ["num", 11, 10], ["pobj", 9, 11], ["prep", 9, 12], ["num", 14, 13], ["pobj", 12, 14], ["punct", 3, 15]],
    "aspect_sentiment": [["Boot time", "positive"]], "from_to": [[0, 2]]}
    '''
    rest_train = 'data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
    rest_test = 'data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'

    laptop_train = 'data/semeval14/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = 'data/semeval14/Laptops_Test_Gold_biaffine_depparsed.json'

    twitter_train = 'data/twitter/train_biaffine.json'
    twitter_test = 'data/twitter/test_biaffine.json'

    vua18_train = 'data/VUA18/train_rospacy.json'
    vua18_test = 'data/VUA18/test_rospacy.json'
    vua18_val = 'data/VUA18/val_rospacy.json'

    verb18_train = 'data/VUA/train_spacy.json'
    verb18_test = 'data/VUA/test_spacy.json'
    verb18_val = 'data/VUA/val_spacy.json'

    vua20_train = 'data/VUA20/train_rospacy.json'
    vua20_test = 'data/VUA20/test_rospacy.json'
    vua20_val = 'data/VUA20/val_rospacy.json'

    trofi_test = 'data/TroFi/test_rospacy.json'

    mohx_test = 'data/MOH-X/CLS/test_rospacy.json'
    
    verb20_train = 'data/verb20/train_biaffine.json'
    verb20_test = 'data/verb20/test_biaffine.json'
    verb20_val = 'data/verb20/val_biaffine.json'
    
    sample_train = 'data_sample/VUA/train_rospacy.json'
    sample_test = 'data_sample/VUA/test_rospacy.json'
    sample_val = 'data_sample/VUA/val_rospacy.json'

    ds_train = {'rest': rest_train, 'vua18': vua18_train, 'vua20':vua20_train, 'trofi': vua20_train, 'mohx':vua20_train,
            'verb18': verb18_train, 'verb20': verb20_train, 'sample':sample_train, 'laptop': laptop_train, 'twitter': twitter_train}
    ds_test = {'rest': rest_test, 'vua18': vua18_test, 'vua20':vua20_test, 'trofi': trofi_test, 'mohx': mohx_test,
            'verb18': verb18_test, 'verb20': verb20_test, 'sample': sample_test,'laptop': laptop_test, 'twitter': twitter_test}
    ds_val = {'rest': rest_test, 'vua18': vua18_val, 'vua20':vua20_val, 'trofi': trofi_test, 'mohx': mohx_test,
            'verb18': verb18_val,'verb20': verb20_val, 'sample':sample_val, 'laptop': laptop_test, 'twitter': twitter_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))

    val = list(read_sentence_depparsed(ds_val[dataset_name]))
    logger.info("# Read %s Val set: %d", dataset_name, len(val))
    return train, test, val


def reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def get_rolled_and_unrolled_data(input_data, args):
    '''
    In input_data, each sentence could have multiple aspects with different sentiments.
    Our method treats each sentence with one aspect at a time, so even for
    multi-aspect-multi-sentiment sentences, we will unroll them to single aspect sentence.

    Perform reshape_dependency_tree to each sentence with aspect

    return:
        all_rolled:
                a list of dict
                    {sentence, tokens, pos_tags, pos_class, aspects(list of aspects), sentiments(list of sentiments)
                    froms, tos, dep_tags, dep_index, dependencies}
        all_unrolled:
                unrolled, with aspect(single), sentiment(single) and so on...
        mixed_rolled:
                Multiple aspects and multiple sentiments, ROLLED.
        mixed_unrolled:
                Multiple aspects and multiple sentiments, UNROLLED.
    '''
    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
                        'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    all_rolled = []
    all_unrolled = []
    mixed_rolled = []
    mixed_unrolled = []

    unrolled = []
    mixed = []
    unrolled_ours = []
    mixed_ours = []

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'metaphor': 1, 'non_meta': 0}

    logger.info('*** Start processing data(unrolling and reshaping) ***')
    tree_samples = []
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []
        dep_tags = []
        dep_index = []
        dep_dirs = []

        # Classify based on POS-tags

        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # We would tokenize the aspect while at it.
            aspect = word_tokenize(aspect)
            sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            # Center on the aspect.
            dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(frm, to+1, e['dependencies'],
                                                       multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)

            # Because of tokenizer differences, aspect opsitions are off, so we find the index and try again.
            if len(dep_tag) == 0:
                zero_dep_counter += 1
                as_sent = e['aspect_sentiment'][i][0].split()
                as_start = e['tokens'].index(as_sent[0])
                # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
                as_end = e['tokens'].index(
                    as_sent[-1]) if len(as_sent) > 1 else as_start + 1
                print("Debugging: as_start as_end ", as_start, as_end)
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(as_start, as_end, e['dependencies'],
                                                           multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
                if len(dep_tag) == 0:  # for debugging
                    print("Debugging: zero_dep",
                          e['aspect_sentiment'][i][0], e['tokens'])
                    print("Debugging: ". e['dependencies'])
                else:
                    zero_dep_counter -= 1

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies'],
                 'ori_sentence': e['ori_sentence'], 'ori_index': e['ori_index'], 'ori_pos': e['ori_pos'], 'ori_aspect': e['ori_aspect']})


        # All sentences with multiple aspects and sentiments rolled.
        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
             'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

        # Ignore sentences with single aspect or no aspect
        if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
            mixed_rolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
                 'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

            # Unrolling
            for i, as_sent in enumerate(e['aspect_sentiment']):
                mixed_counter[as_sent[1]] += 1
                mixed_unrolled.append(
                    {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspects[i], 'sentiment': sentiments[i],
                     'from': froms[i], 'to': tos[i], 'dep_tag': dep_tags[i], 'dep_idx': dep_index[i], 'dependencies': e['dependencies']})


    logger.info('Total sentiment counter: %s', total_counter)
    logger.info('Multi-Aspect-Multi-Sentiment counter: %s', mixed_counter)

    return all_rolled, all_unrolled, mixed_rolled, mixed_unrolled


def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and glove embeddings.
    # Elmo and bert have it's own vocab and embeddings.
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir, 'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def _default_unk_index():
    return 1


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']
        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


# def build_dep_tag_vocab_energy():  # 47 in total, all tags plus pad and non-connect
#     '''
#     biaffine dep_tag Vocab : {0: 'punct', 1: 'prep', 2: 'pobj', 3: 'det', 4: 'nn',
#         5: 'nsubj', 6: 'amod', 7: 'root', 8: 'dobj', 9: 'aux', 10: 'advmod', 11: 'conj',
#         12: 'cc', 13: 'num', 14: 'poss', 15: 'ccomp', 16: 'dep', 17: 'xcomp', 18: 'mark',
#         19: 'cop', 20: 'number', 21: 'possessive', 22: 'rcmod', 23: 'auxpass', 24: 'appos',
#         25: 'nsubjpass', 26: 'advcl', 27: 'partmod', 28: 'pcomp', 29: 'neg', 30: 'tmod',
#         31: 'quantmod', 32: 'npadvmod', 33: 'prt', 34: 'infmod', 35: 'parataxis',
#         36: 'mwe', 37: 'expl', 38: 'acomp', 39: 'iobj', 40: 'csubj', 41: 'predet',
#         42: 'preconj', 43: 'discourse', 44: 'csubjpass'}
#     This is used in energy case.
#     '''
#     head_tags = {0: 'punct', 1: 'prep', 2: 'pobj', 3: 'det', 4: 'nn', 5: 'nsubj', 6: 'amod', 7: 'root', 8: 'dobj', 9: 'aux', 10: 'advmod', 11: 'conj', 12: 'cc', 13: 'num', 14: 'poss', 15: 'ccomp', 16: 'dep', 17: 'xcomp', 18: 'mark', 19: 'cop', 20: 'number', 21: 'possessive', 22: 'rcmod', 23: 'auxpass', 24: 'appos',
#                  25: 'nsubjpass', 26: 'advcl', 27: 'partmod', 28: 'pcomp', 29: 'neg', 30: 'tmod', 31: 'quantmod', 32: 'npadvmod', 33: 'prt', 34: 'infmod', 35: 'parataxis', 36: 'mwe', 37: 'expl', 38: 'acomp', 39: 'iobj', 40: 'csubj', 41: 'predet', 42: 'preconj', 43: 'discourse', 44: 'csubjpass', 45: '<pad>', 46: 'non-connect'}
#     itos = [head_tags[i] for i in range(len(head_tags))]
#     stoi = defaultdict()
#     stoi.update({tok: i for i, tok in enumerate(itos)})
#     return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


class ASBA_Depparsed_Dataset(Dataset):
    '''
    Convert examples to features, numericalize text to ids.
    data:
        -list of dict:
            keys: sentence, tags, pos_class, aspect, sentiment,
                predicted_dependencies, predicted_heads,
                from, to, dep_tag, dep_idx, dependencies, dep_dir

    After processing,
    data:
        sentence
        tags
        pos_class
        aspect
        sentiment
        from
        to
        dep_tag
        dep_idx
        dep_dir
        predicted_dependencies_ids
        predicted_heads
        dependencies
        sentence_ids
        aspect_ids
        tag_ids
        dep_tag_ids
        text_len
        aspect_len
        if bert:
            input_ids
            word_indexer

    Return from getitem:
        sentence_ids
        aspect_ids
        dep_tag_ids
        dep_dir_ids
        pos_class
        text_len
        aspect_len
        sentiment
        deprel
        dephead
        aspect_position
        if bert:
            input_ids
            word_indexer
            input_aspect_ids
            aspect_indexer
        or:
            input_cat_ids
            segment_ids
    '''

    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], \
            e['pos_class'], e['text_len'], e['aspect_len'], e['sentiment'],\
            e['dep_rel_ids'], e['predicted_heads'], e['aspect_position'], e['dep_dir_ids']
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        elif self.args.embedding_type == 'elmo':
            items_tensor = e['sentence_ids'], e['aspect_ids']
            items_tensor += tuple(torch.tensor(t) for t in items)
        else:  # bert and roberta
            if self.args.melbert:
                bert_items = e['input_ids'], e['word_indexer'], e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids'], e['aspect_segment'], e['attention_mask'], e['aspect_attention'], e['neighbor_mask']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            
        return items_tensor

    def first_mask(self, i):
        first_neighb = np.zeros(len(self.data[i]['predicted_heads']))
        num_neighb = 0
        for idx in range(len(self.data[i]['predicted_heads'])):
            if str(self.data[i]['predicted_heads'][idx]) == str(self.data[i]['from']):
                first_neighb[idx] = 1
                num_neighb += 1
        return first_neighb, num_neighb
    
    def dep_mask(self, i):
        dep_neighb = np.zeros(len(self.data[i]['predicted_heads']))
        num_neighb = 0
        for idx in range(len(self.data[i]['predicted_heads'])):
            if str(self.data[i]['dep_dir'][idx]) != 0:
                dep_neighb[idx] = 1
                num_neighb += 1
        return dep_neighb, num_neighb
    
            
    def convert_features_melbert(self, i):
        
        cls_token = self.args.tokenizer.cls_token
        sep_token = self.args.tokenizer.sep_token
        pad_token = 0
        # tokenizer = self.args.tokenizer

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []
        self.data[i]['1st_neighbor'], self.data[i]['num_neighbor'] = self.dep_mask(i)
        neighbor_mask = []
        dep_tag_ids = []
        target_index = self.data[i]['from']

        #tokens = self.args.tokenizer.tokenize(self.data[i]['ori_sentence'])
        '''
        if len(self.args.tokenizer.tokenize(self.data[i]['ori_sentence']))+1 != len(self.data[i]['sentence']):
            print(self.args.tokenizer.tokenize(self.data[i]['ori_sentence']))
            print(self.data[i]['sentence'])
            print("%%%%%%%%%%%%%%%%%%%%%%")
        '''
        
        sentence = sentence_pair(self.data[i]['ori_sentence'],self.data[i]['sentence'])
            
        for idx, word in enumerate(sentence):
            '''
            if idx<len(self.data[i]['sentence'])-1:
                word_tokens = self.args.tokenizer.tokenize(' '+word) if ' '+word+self.data[i]['sentence'][idx+1] or ' '+word+' ' in self.data[i]['ori_sentence'] else self.args.tokenizer.tokenize(word)
            else:
                word_tokens = self.args.tokenizer.tokenize(word)
            '''
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            if idx == int(self.data[i]['from']):
                target_position = token_idx
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)
            if self.data[i]['dep_dir'][idx] != 0:
                dep_tag_ids.extend([self.data[i]['dep_tag_ids'][idx]]*len(word_tokens))
                neighbor_mask.extend(np.ones(len(word_tokens)))
            else:
                dep_tag_ids.extend(np.zeros(len(word_tokens)))
                neighbor_mask.extend(np.zeros(len(word_tokens)))

            '''
            if self.data[i]['1st_neighbor'][len(word_indexer)-1]:
                neighbor_mask.extend((np.ones(len(word_tokens))/len(word_tokens))/self.data[i]['num_neighbor'])
            else:
                neighbor_mask.extend(np.zeros(len(word_tokens)))
            '''
        '''
        #tokens = self.args.tokenizer.tokenize(self.data[i]['ori_sentence'])
        # aspect
        #for word in self.data[i]['ori_aspect']:
        w_tok = None
        target_position = 0
        for idx,w in enumerate(self.data[i]['ori_sentence'].split()):
            if idx == int(self.data[i]['ori_index']):
                aspect_tokens = self.args.tokenizer.tokenize(w) if i == 0 else self.args.tokenizer.tokenize(" " + w)
                log_aspect = w
                break
            w_tok = self.args.tokenizer.tokenize(w) if i == 0 else self.args.tokenizer.tokenize(" " + w)
            if w_tok:
                target_position += len(w_tok) - 1
        '''
        for idx, aspect_token in enumerate(self.data[i]['aspect']):
            word_aspect_tokens = self.args.tokenizer.tokenize(' '+aspect_token) if idx==0 and int(self.data[i]['ori_index']) != 0 else self.args.tokenizer.tokenize(aspect_token)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        '''
        print('****8*****', self.data[i]['ori_sentence'])
        print('$$$$$$$$$$',tokens)
        print('$$$$$$$$$$', self.data[i]['ori_index'])
        print('$$$$$$$$$$',target_position)
        print('$$$$$$$$$$',self.data[i]['ori_aspect'])
        #print('$$$$$$$$$$',log_aspect)
        print('$$$$$$$$$$', aspect_tokens)
        '''
        

        # pos
        pos_token = self.data[i]['ori_pos']
        pos_token = self.args.tokenizer.tokenize(pos_token)
        
        if len(tokens) > self.args.max_seq_length - 2:
            tokens = tokens[:(self.args.max_seq_length - 2)]
            dep_tag_ids = dep_tag_ids[:(self.args.max_seq_length - 2)]
            neighbor_mask = neighbor_mask[:(self.args.max_seq_length - 2)]
            
        
        ## First Features
        tokens = [cls_token] + tokens + [sep_token]
        neighbor_mask = [0] + neighbor_mask + [0]
        dep_tag_ids = [0] + dep_tag_ids + [0]
        target_position += 1
        
        if self.args.use_local_context:
            local_start = 1
            local_end = local_start + len(aspect_tokens)
            comma1 = self.args.tokenizer.tokenize(",")[0]
            comma2 = self.args.tokenizer.tokenize(" ,")[0]
            for j, w in enumerate(tokens):
                if j < target_position and (w in [comma1, comma2]):
                    local_start = j
                if j > target_position and (w in [comma1, comma2]):
                    local_end = j
                    break
            segment_ids = [
                2 if i >= local_start and i <= local_end else 0 for i in range(len(tokens))
            ]
        else:
            segment_ids = [0] * len(tokens)
           
        if target_position + len(aspect_tokens) < self.args.max_seq_length - 2:
            for idx in range(len(aspect_tokens)):
                segment_ids[target_position + idx] = 1
        
        tokens += aspect_tokens + [sep_token] + pos_token + [sep_token]
        segment_ids += [3] * (len(aspect_tokens)+len(pos_token)+2)
        neighbor_mask += [0] * (len(aspect_tokens)+len(pos_token)+2)
        dep_tag_ids += [0] * (len(aspect_tokens)+len(pos_token)+2)
        if len(tokens) != len(neighbor_mask):
            print(len(tokens), len(neighbor_mask))

        target_mask = [0]*len(tokens)
        if target_position + len(aspect_tokens) < self.args.max_seq_length - 2:
            for idx in range(len(aspect_tokens)):
                target_mask[target_position + idx] = 1

                
        ## Second Features
        aspect_segment = [0]+ [1]* len(aspect_tokens) + [0]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]



        word_indexer = [i+1 for i in word_indexer]
        aspect_indexer = [i+1 for i in aspect_indexer]

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        #assert len(word_indexer) == len(self.data[i]['sentence'])
        #assert len(aspect_indexer) == len(self.data[i]['aspect'])
        assert len(input_ids) == len(segment_ids)
        #assert len(neighbor_mask) == len(input_ids)

        # THE STEP:Zero-pad up to the sequence length, save to collate_fn.

        input_cat_ids = input_ids
        attention_mask = [1] * len(input_cat_ids)
        aspect_attention = [1] * len(input_aspect_ids)

        self.data[i]['input_cat_ids'] = input_cat_ids
        self.data[i]['segment_ids'] = segment_ids
        self.data[i]['input_ids'] = target_mask
        self.data[i]['word_indexer'] = word_indexer
        self.data[i]['input_aspect_ids'] = input_aspect_ids
        self.data[i]['aspect_segment'] = aspect_segment
        self.data[i]['aspect_indexer'] = aspect_indexer
        self.data[i]['attention_mask'] = attention_mask
        self.data[i]['aspect_attention'] = aspect_attention
        self.data[i]['neighbor_mask'] = neighbor_mask
        self.data[i]['dep_tag_ids'] = dep_tag_ids

        self.data[i]['mel_mask'] = None
        self.data[i]['mel_seg'] = None
        self.data[i]['mel_ids'] = None



    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):

            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['dep_tag']]
            self.data[i]['dep_dir_ids'] = [idx
                                           for idx in self.data[i]['dep_dir']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['tags']]
            self.data[i]['aspect_len'] = len(self.data[i]['aspect'])

            self.data[i]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
                                           for r in self.data[i]['predicted_dependencies']]

            
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspect']]
            elif self.args.embedding_type == 'elmo':
                self.data[i]['sentence_ids'] = self.data[i]['sentence']
                self.data[i]['aspect_ids'] = self.data[i]['aspect']
            elif self.args.embedding_type == 'bert':  
                self.convert_features_bert(i)
            elif self.args.melbert:
                self.convert_features_melbert(i)
            else:# self.args.embedding_type == 'roberta'
                self.convert_features_roberta(i)

            self.data[i]['text_len'] = len(self.data[i]['sentence'])
            self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
            try:  # find the index of aspect in sentence
                for j in range(self.data[i]['from'], self.data[i]['to']):
                    self.data[i]['aspect_position'][j] = 1
            except:
                for term in self.data[i]['aspect']:
                    self.data[i]['aspect_position'][self.data[i]
                                                    ['sentence'].index(term)] = 1



def my_collate_melbert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, aspect_segment, attention_mask, aspect_attention, neighbor_mask, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids = zip(        
            *batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value =0)
    aspect_segment = pad_sequence(aspect_segment, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    aspect_attention = pad_sequence(aspect_attention, batch_first=True, padding_value=0)
    neighbor_mask = pad_sequence(neighbor_mask, batch_first=True, padding_value=0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer, batch_first=True, padding_value=1)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)
    #sentence = pad_sequence(sentence, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_segment = aspect_segment[sorted_idx]
    attention_mask = attention_mask[sorted_idx]
    aspect_attention = aspect_attention[sorted_idx]
    neighbor_mask = neighbor_mask[sorted_idx]
    
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]
    #sentence = sentence[sorted_idx]

    return input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids, attention_mask, aspect_attention, neighbor_mask, aspect_segment #, sentence


