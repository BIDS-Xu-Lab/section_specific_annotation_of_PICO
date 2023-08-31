# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def get_labels(labels_path, data_dir=None, file_list=None):
    """get labels from labels_path at first, else enumerate labels from train/dev/test files from data_dir/file_list"""
    labels_set = set()
    labels = list()
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r") as rf:
            labels = rf.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
    elif file_list:
        for file in file_list:
            if data_dir:
                pfn = os.path.join(data_dir, file)
            else:
                pfn = file
            with open(pfn, 'r', encoding='utf-8') as rf:
                for line in rf.readlines():
                    line = line.rstrip()
                    if not line or (line.startswith('###') and line.endswith('$$$')):
                        continue
                    tokens = line.split("\t")
                    if len(tokens) >= 2:
                        if tokens[-1] != "O":
                            labels_set.add(tokens[-1][2:])
        labels = ["O"]
        for label in labels_set:
            labels.extend(["B-"+label, "I-"+label])
    else:
        labels = ['O',]
    return labels

def update_data_to_max_len(max_len, in_data_file, out_data_file, model_name_or_path, sep_tag_type='tab'):
    sep_tag = ' ' if sep_tag_type=='space' else '\t'
    print(f'update data to max len {max_len} for {in_data_file} to {out_data_file}...')
    sent_word_tokened_len = 0
    sent_word_tokened_since_previous_B_len = 0
    label_previous_B_num = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # no num_special_tokens_to_add() in tokenizer under transformer v3
    # just ignore it, take 4 by default (2 generally for ner, 4 for paired tasks like translation task)
    #max_len -= tokenizer.num_special_tokens_to_add(), 
    max_len -= 4    
    out_data = []
    #if out_data_file:
    #    out_fp = open(out_data_file, 'w', encoding='utf-8')
    #else:
    #    out_fp = None
    show_count = 0
    with open(in_data_file, "rt") as in_fp:
        for line in in_fp:
            line = line.rstrip()
            if (not line) or (line.startswith('###') and line.endswith('$$$')):
                #print(line) #, file=out_fp)
                out_data.append(line)
                sent_word_tokened_len = 0
                sent_word_tokened_since_previous_B_len
                label_previous_B_num = 0
                continue
            line_sep = line.split(sep_tag)
            word = line_sep[0]
            label = line_sep[-1]
            word_tokened = tokenizer.tokenize(word)
            current_word_tokened_len = len(word_tokened)
            if (current_word_tokened_len > 1) and (show_count < 3):
                show_count += 1
                print("{}->{}".format(word, ','.join(word_tokened)))            
            # Token contains strange control characters like \x96 or \x95,
            # Filter out the complete line
            if current_word_tokened_len == 0:
                print(f'Warning: {word} tokenized to empty str in {line}')                
                continue
            if (sent_word_tokened_len + current_word_tokened_len) > max_len:
                #print("", file=out_fp)
                #print(line, file=out_fp)
                sent_idx = len(out_data) - label_previous_B_num
                out_data.insert(sent_idx, '')
                sent_word_tokened_len = sent_word_tokened_since_previous_B_len
                
            #line = '{}{}{}'.format(word, sep_tag, label)
            #print(line, file=out_fp)
            #print(line)
            out_data.append(line)
            sent_word_tokened_len += current_word_tokened_len
            if label == 'O':
                label_previous_B_num = 0
                sent_word_tokened_since_previous_B_len = 0
            else:
                label_previous_B_num += 1
                sent_word_tokened_since_previous_B_len += current_word_tokened_len
    if out_data_file:
        with open(out_data_file, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(out_data))

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


# mode will be like [train|test|dev].[txt|bio]
def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, mode)
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n" or \
                    (line.startswith('###') and line.endswith('$$$\n')):
                if words:
                    assert len(words) == len(labels)
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                 words=words,
                                                 labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            assert len(words) == len(labels)
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) <= 0:
                print(f'{word} tokenized to empty word!')
                tokens.append(word)
                label_ids.append(label_map[label])
            else:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        if tokens.count('[UNK]') > 3:
            print(f"TOO MANY UNK:{tokens.count('[UNK')} in tokenized {example.words}")
        labels_len = len(label_ids)
        tokens_len = len(tokens)
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            print(f'Tokens length {len(tokens)}+{special_tokens_count} exceed max_seq_length {max_seq_length}')
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        if len(label_ids) != max_seq_length:
            logger.info("*** Error ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("labels_len: %s", str(labels_len))
            logger.info("tokens_len: %s", str(tokens_len))
            logger.info("input_len: %s", str(input_len))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features

