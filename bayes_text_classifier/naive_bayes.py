import re


training_set_size = 1 #Laplace
spam_set = {}
normal_set = {}


def main(path):
    print('start')
    prepare(path_set)


def predict(email_content):
    p = 1.0
    for word in email_content:
        key_count = spam_set.get(word)
        key_count = key_count if key_count is not None else 1 #Laplace
        p *= key_count / training_set_size
    return 1 if p >= 0.5 else 0


def prepare(path_set):
    pattern = re.compile('^{data_flag:(.*?)$}')
    for path in path_set:
        raw_data = open(path)
        sample = ''
        data_flag = 0
        handle_one_data_set(raw_data, data_flag)


def handle_one_data_set(raw_data, data_flag):
    sample = ''
    while True:
        line = raw_data.readline()
        if not line:
            break;
        result = pattern.match(line)
        if result is None:
            sample += line
        else:
            training_set_size += 1
            if data_flag == 0:
                add_feature(spam_set, sample)
            else data_flag == 1:
                add_feature(normal_set, sample)
            data_flag = int(result.group(1))
            sample = ''


def add_feature(pre_set, sample):
    if sample is None or sample is '':
        return;
    text_set = []
    for char in sample:
        text_set.append(char)
    text_set = set(text_set)
    for word in text_set:
        if pre_set.get(word) == None:
            pre_set[word] = 2 #Laplace
        else:
            pre_set[word] += 1

