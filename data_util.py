import pickle
from collections import OrderedDict

from nltk.tokenize import word_tokenize

_buckets = [(15, 15), (25, 25), (35, 35), (50, 35), (100, 35), (150, 35)]
special_tokens = ["<eos>", "<go>", "<pad>", "<unknow>"]
eos_id = 0
go_id = 1
pad_id = 2
unknow_id = 3


def save_utter_dict(file_name="utter_dict.p"):
    """save utter to a dict has the form "L212": "i have a dream" """
    utter_dict = {}
    with open("data/movie_lines.txt", encoding="latin-1") as f:
        for line in f:
            items = line.rstrip().split(" +++$+++ ")
            utter_dict[items[0]] = items[-1].lower()

    pickle.dump(utter_dict, open(file_name, "wb"))
    return utter_dict


def load_utter_dict(file_name="utter_dict.p"):
    """load utter dict from saved pickle"""
    return pickle.load(open(file_name, "rb"))


def save_conv(file_name="conv_list.p"):
    """save conversation list in a form of [['L1','L2','L3'],['L5','L6']]"""
    conv_list = []
    with open("data/movie_conversations.txt", encoding="latin-1") as f:
        for line in f:
            conv = line.rstrip().split(" +++$+++ ")[-1]
            # example['L864', 'L123','L234']
            conv = conv[1:-1].replace(" ", "").replace("\'", "").split(",")
            conv_list.append(conv)
        pickle.dump(conv_list, open(file_name, "wb"))
    return conv_list


def load_conv(file_name="conv_list.p"):
    """ get conv_list which has a form of [['L1','L2'],['L5','L6']]"""
    return pickle.load(open(file_name, "rb"))


def utter_matrix(word_dict, utter_dict, add_go=True,
                 add_eos=False, file_name="utter_matrix.p"):
    """ convert utter_dict values to correspond index lists
    utter_matrix has the form: {"L1234": [1, 2, 3, 4, 5]}"""
    utter_matrix_data = {}
    for key in utter_dict:
        utter_matrix_data[key] = word_dict.sentence_to_indexs(utter_dict[key],
                                                              add_go=add_go,
                                                              add_eos=add_eos)

    pickle.dump(utter_matrix_data, open(file_name, "wb"))
    return utter_matrix_data


def load_utter_matrix(file_name="utter_matrix.p"):
    return pickle.load(open(file_name, "rb"))


def conv_index_list(utter_matrix, conv_list, file_name="conv_index.p"):
    """save index_list in a form [[[1,2,3,4],[2,3,4,5]],
    [[11,22,3,4],[2,3,4,5]]]"""

    conv_index = []
    for con in conv_list:
        current_index = []
        for key in con:
            current_index.append(utter_matrix[key])
        conv_index.append(current_index)
    pickle.dump(conv_index, open(file_name, "wb"))
    return conv_index


def load_conv_index(file_name="conv_index.p"):
    return pickle.load(open(file_name, "rb"))


def conv_split(conv_index, max_conv=2, file_name="conv_split.p"):
    """split conv in to a conv which only has a two sentences,
    will modify this function to have more sentences in a split"""
    conv_split_data = []
    for convs in conv_index:
        for i in range(len(convs) - 1):
            conv_split_data.append([convs[i], convs[i + 1]])
    pickle.dump(conv_split_data, open(file_name, "wb"))
    return conv_split_data


def load_conv_split(file_name="conv_split.p"):
    return pickle.load(open(file_name, "rb"))


def split_to_dataset(conv_split, buckets=_buckets, file_name="dataset.p"):
    data_set = [[] for _ in buckets]
    for x in conv_split:
        for buck_id, (question_size, answer_size) in enumerate(buckets):
            if len(x[0]) <= question_size and len(x[1]) <= answer_size:
                data_set[buck_id].append([x[0], x[1]])
                break
    pickle.dump(data_set, open(file_name, "wb"))
    return data_set


def load_dataset(file_name="dataset.p"):
    return pickle.load(open(file_name, "rb"))


def add_pad(data_set, _buckets, file_name="padded_data.p"):
    new_data = [[] for _ in _buckets]
    for i in range(len(_buckets)):
        input_len = _buckets[i][0]
        output_len = _buckets[i][1]
        for data in data_set[i]:
            current_data = [0, 0]
            current_data[0] = [pad_id] * input_len
            current_data[1] = [pad_id] * output_len
            for j in range(len(data[0])):
                current_data[0][-1 * j - 1] = data[0][j]
            for j in range(len(data[1])):
                current_data[1][j] = data[1][j]
            new_data[i].append(current_data)
    pickle.dump(new_data, open(file_name, "wb"))
    return new_data


def process_data(new_utter_dict=True, new_save_conv=True,
                 new_utter_matrix=True, max_words=20000,
                 new_conv_index_list=True, new_conv_split=True,
                 new_split_to_dataset=True, new_worddict=True,
                 buckets=_buckets, add_go=True,
                 add_eos=True):
    if not new_split_to_dataset and not new_worddict:
        data_set = load_dataset()
        word_dict = pickle.load(open("word_dict.p", "rb"))
        return word_dict, data_set

    if new_utter_dict:
        utter_dict = save_utter_dict()
    else:
        utter_dict = load_utter_dict()
    print("utter dictionary processed")

    if new_save_conv:
        conv_list = save_conv()
    else:
        conv_list = load_conv()

    print("conversation list processed")

    if new_worddict:
        word_dict = WORDDICT(max_words)
        word_dict.update_dict(utter_dict)
    else:
        word_dict = pickle.load(open("word_dict.p", "rb"))

    print("word dictionary class processed")

    if new_utter_matrix:

        utter_matrix_data = utter_matrix(word_dict, utter_dict,
                                         add_go=add_go,
                                         add_eos=add_eos)
    else:
        utter_matrix_data = load_utter_matrix()

    print("utter matrix  processed")

    if new_conv_index_list:
        conv_index = conv_index_list(utter_matrix_data, conv_list)
    else:
        conv_index = load_conv_index()
    print("conversation index list processed")

    if new_conv_split:
        conv_split_data = conv_split(conv_index)
    else:
        conv_split_data = load_conv_split()

    print("conversation index list splitted into desired length of conv")

    if new_split_to_dataset:
        data_set = split_to_dataset(conv_split_data, buckets=buckets)
    else:
        data_set = load_dataset()

    print("final dataset processed ")

    return word_dict, data_set


class WORDDICT():

    def __init__(self, max_words=20000):
        self.max_words = max_words

        self.count_dict = {}
        self.word_to_index = {}
        self.index_to_word = {}

    def add_words(self, utter_dict):
        "update worddict class with utter_dict"
        lines = utter_dict.values()
        words = []
        for line in lines:
            tokens = []
            tokens.append(special_tokens[go_id])
            tokens += word_tokenize(line)
            tokens.append(special_tokens[eos_id])
            words += tokens
        for word in words:
            self.count_dict[word] = self.count_dict.get(word, 0) + 1

    def get_word_index(self):
        tmp = OrderedDict(sorted(self.count_dict.items(),
                          key=lambda x: -1 * x[1]))
        max_count = len(tmp)
        index = 0

        """ assume go and eos have large enough count, so it will be added in
        the iteration later"""
        index = 0
        for i in range(len(special_tokens)):
            self.word_to_index[special_tokens[i]] = index
            index += 1
        max_count = self.max_words
        for word, _ in tmp.items():
            if not (word in special_tokens):
                self.word_to_index[word] = index
                index += 1
                if index == max_count:
                    break

    def get_index_word(self):
        self.index_to_word = dict((v, k) for k, v in
                                  self.word_to_index.items())

    def sentence_to_indexs(self, sentence, add_go=True, add_eos=False):
        sentence = sentence.lower()
        index_list = []  # word list
        if add_go:
            index_list.append(special_tokens[go_id])
        index_list += word_tokenize(sentence)
        sentence = sentence.lower()
        if add_eos:
            index_list.append(special_tokens[eos_id])
        result = []
        for x in index_list:
            result.append(self.word_to_index.get(x,
                          self.word_to_index[special_tokens[unknow_id]]))
        return result

    def indexs_to_sentence(self, indexs):
        return [self.index_to_word[x] for x in indexs]

    def update_dict(self, utter_dict):
        """ update utter_dict to Word_dict"""
        self.add_words(utter_dict)
        self.get_word_index()
        self.get_index_word()
        self.save()

    def save(self, file_name="word_dict.p"):
        """save word dict to pickle file"""
        pickle.dump(self, open(file_name, "wb"))


if __name__ == "__main__":
    create_new = False
    if create_new:
        word_dict, data_set = process_data()
    else:
        word_dict, data_set = process_data(new_split_to_dataset=False,
                                           new_worddict=False)
    for x in data_set:
        print(len(x))
    # x = add_pad(data_set, _buckets)
