from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def list_tokenizer(tokenizer, text_list):
    """Split train dataset corpus to list """
    line_list = list()
    for text in text_list:
        line_list.append(tokenizer.tokenize(text))

    return line_list


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_tokens_to_features(tokens_list, max_seq_length, tokenizer):
    """Convert tokens list to ids features list."""
    features, total_token_num = [], 0

    tokens_a = tokens_list[0]

    if tokens_list[1]:
        tokens_b = tokens_list[1]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]


    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    features.append(
        InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids))
    return features

class KorELECTRADataset():
    def __init__(self, train_examples, tokenizer, args):
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.args = args

    def __getitem__(self, index):
        train_dataset = convert_tokens_to_features(list_tokenizer(self.tokenizer, self.train_examples[index]), self.args.max_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_dataset], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_dataset], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_dataset], dtype=torch.long)

        del train_dataset

        return tuple([all_input_ids, all_input_mask, all_segment_ids])

    def __len__(self):
        return len(self.train_examples)






