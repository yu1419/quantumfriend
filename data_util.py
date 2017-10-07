import pickle
from nltk.tokenize import word_tokenize
from collections import OrderedDict

def save_utter_dict(file_name="utter_dict.p"):
    """ save utter to a dict has the form "L212": "I have a dream" """
    utter_dict = {}
    with open("data/movie_lines.txt", encoding="latin-1") as f:
        for line in f:
            items = line.rstrip().split(" +++$+++ ")
            utter_dict[items[0]] = items[-1].lower()

    pickle.dump(utter_dict, open(file_name, "wb"))


def load_utter_dict(file_name="utter_dict.p"):
    """load utter dict from saved pickle"""
    return pickle.load(open(file_name, "rb"))

 
def save_conv(file_name="conv_list.p"):
    """save conversation list in a form of [['L1','L2'],['L5','L6']]"""
    conv_list = []
    with open("data/movie_conversations.txt", encoding="latin-1") as f:
        for line in f:
            conv = line.rstrip().split(" +++$+++ ")[-1]  #example['L864', 'L123']
            conv = conv[1:-1].replace(" ", "").replace("\'","").split(",")
            conv_list.append(conv)
        pickle.dump(conv_list, open(file_name, "wb"))

def load_conv(file_name="conv_list.p"):
    """ get conv_list which has a form of [['L1','L2'],['L5','L6']]"""
    return pickle.load(open(file_name, "rb"))

def utter_matrix(word_dict, utter_dict, file_name="utter_matrix.p"):
    """ convert utter_dict values to correspond index lists"""
    utter_matrix = {}
    for key in utter_dict:
        utter_matrix[key] = word_dict.sentence_to_indexs(utter_dict[key])
     
    pickle.dump(utter_matrix, open(file_name, "wb")) 
    return utter_matrix

def conv_index_list(utter_matrix, conv_list, file_name="conv_index.p"):
    conv_index = []
    for con in conv_list:
        current_index = []
        for key in con:
            current_index += utter_matrix[key]
        conv_index.append(current_index)      
    pickle.dump(conv_index, open(file_name, "wb"))
    return conv_index 

def load_conv_index(file_name="conv_index.p"):
    return pickle.load(open(file_name, "rb"))

class Word_dict():

    def __init__(self, max_words=None):
        self.max_words = max_words

        self.count_dict = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.go_token = '<go>'
        self.end_token = '<eos>'
        self.unknow_token = '<unkwn>'
        self.count_dict[self.go_token] = 0
        self.count_dict[self.end_token] = 0

    def add_words(self, conv):
        lines = utter_dict.values()
        words = []
        for line in lines:
            tokens = []
            tokens.append(self.go_token)
            tokens += word_tokenize(line)
            tokens.append(self.end_token)
            words += tokens
        for word in words:
            self.count_dict[word] = self.count_dict.get(word, 0) + 1

    def get_word_index(self):
        tmp = OrderedDict(sorted(self.count_dict.items(), key=lambda x:-1*x[1]))
        max_count = len(tmp)
        if not self.max_words is None:
            max_count = self.max_words
        index = 0
        for word,_ in tmp.items():
            self.word_to_index[word] = index
            index += 1
            if index == max_count:
                break
        self.word_to_index[self.unknow_token] = index

    def get_index_word(self):
        self.index_to_word = dict((v,k) for k,v in self.word_to_index.items())

    def sentence_to_indexs(self, sentence):
        sentence = sentence.lower()
        index_list = []  #word list
        index_list.append(self.go_token)
        index_list += word_tokenize(sentence)
        sentence = sentence.lower()
        index_list.append(self.end_token)
        result = []
        for x in index_list:
            result.append(self.word_to_index.get(x, self.word_to_index[self.unknow_token]))

        return result
    def indexs_to_sentence(self, indexs):
        return [self.index_to_word[x] for x in indexs]

    def update_dict(self, utter_dict):
        """ update utter_dict to Word_dict"""
        self.add_words(utter_dict)
        self.get_word_index()
        self.get_index_word()

    def save(self, file_name="word_dict.p"):
        pickle.dump(self, open(file_name, "wb"))


if __name__ == "__main__":
    #save_utter_dict()
    utter_dict = load_utter_dict()
    #save_conv()"""
    conv_list = load_conv()
    """
    d = Word_dict(15000)
    d.add_words(utter_dict)
    d.get_word_index()
    d.get_index_word()
    """
    word_count = 25000
    """d = Word_dict(word_count)
    d.update_dict(utter_dict)
    d.save()"""

    c = pickle.load(open("word_dict.p", "rb"))
    #utter_matrix = utter_matrix(c, utter_dict)
    utter_matrix = pickle.load(open("utter_matrix.p", "rb"))
    unknown_lines = 0
    """for key in utter_matrix:
        values = utter_matrix[key]
        if word_count in values:
            unknown_lines = unknown_lines + 1
    print(unknown_lines)
    print(len(utter_matrix))
    """
    #print(conv_index_list(utter_matrix, conv_list))
    print(load_conv_index())
