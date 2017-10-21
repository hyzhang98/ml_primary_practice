import re
import jieba


laplace_constants = 1 #Laplace
spam_set = []
normal_set = []
pattern = re.compile('^{data_flag:(.*?)}$')
normal_dictionary = {}
spam_dictionary = {}
total_count = 2
normal_count = 1
spam_count = 1


def main(path_set):
    print('start')
    prepare(path_set)
    training_set_to_dictionary()


def predict(email_content):
    words = get_words(email_content)
    p = px_given_y(words, True) * py(True) / (px_given_y(words, True) + px_given_y(words, False))
    print(p)
    return True if p >= 0.5 else False


def px_given_y(email_content, is_normal):
    normal_count = len(normal_set) + 1
    spam_count = len(spam_set) + 1
    total_count = normal_count + spam_count
    dic = normal_dictionary if is_normal else spam_dictionary
    p = 1.0
    y_count = (normal_count if is_normal else spam_count) + 1
    flag = True
    for d in dic:
        has_word = d in email_content
        count = dic.get(d, 0) + 1 if has_word else y_count - dic.get(d, 0) + 1
        p = p * (count / y_count)
        if flag:
            flag = False
    return p


def py(is_normal):
    normal_count = len(normal_set) + 1
    spam_count = len(spam_set) + 1
    total_count = normal_count + spam_count
    y_count = (normal_count if is_normal else spam_count) + 1;
    return y_count / total_count


def prepare(path_set):
    for path in path_set:
        raw_data = open(path)
        print('path: ' + path)
        sample = ''
        handle_raw_data_set_from_file(raw_data)


def handle_raw_data_set_from_file(raw_data):
    sample = ''
    data_flag = -1
    while True:
        line = raw_data.readline()
        if not line:
            break;
        line = line.strip('\n')
        if is_invalid_line(line):
            continue
        result = pattern.match(line)
        if result is None:
            sample += line
        else:
            if data_flag == -1:
                data_flag = int(result.group(1))
                continue
            if data_flag == 0:
                add_to_traing_set(spam_set, sample)
            elif data_flag == 1:
                add_to_traing_set(normal_set, sample)
            data_flag = int(result.group(1))
            sample = ''


def is_invalid_line(line):
    return line is None or line is '' or line.isspace()


def add_to_traing_set(pre_set, sample):
    pre_set.append(sample)


def training_set_to_dictionary():
    for sample in normal_set:
        text_set = get_words(sample)
        add_word_count_to_dictionary(normal_dictionary, text_set)
        add_word_to_dictionary(spam_dictionary, text_set)
    for sample in spam_set:
        text_set = get_words(sample)
        add_word_count_to_dictionary(spam_dictionary, text_set)
        add_word_to_dictionary(normal_dictionary, text_set)


def get_words(sample):
    jieba_set = jieba.cut(sample)
    result = []
    for word in jieba_set:
        if word.isspace():
            continue
        result.append(word)
    return set(result)


def add_word_count_to_dictionary(dictionary, text_set):
    for word in text_set:
        count = dictionary.get(word, 0)
        dictionary[word] = count + 1;


def add_word_to_dictionary(dictionary, text_set):
    for word in text_set:
        if dictionary.get(word, 0) == 0:
            dictionary[word] = 0
            

if __name__ == '__main__':
    path_base = '/Users/hyzhang/MachineLearning/python/primary_practice/data/'
    path_set = [path_base + 'emails1.data']
    main(path_set)
    file = open('/Users/hyzhang/Desktop/test.data')
    content = ''
    while True:
        line = file.readline()
        if not line:
            break;
        line = line.strip('\n')
        if is_invalid_line(line):
            continue
        content += line
    print(content)
    flag = predict(content)
    print_content = "This is spam email!" if not flag else "This is a normal email"
    print(print_content)

