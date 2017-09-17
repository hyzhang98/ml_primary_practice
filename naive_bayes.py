import re

def main(path):
    print('start')
    handle_email(path)


def prepare(path_set):
    pattern = re.compile('^{data_flag:(.*?)$}')
    spam_set = {}
    normal_set = {}
    for path in path_set:
        raw_data = open(path)
        sample = ''
        data_flag = 0
        handle_one_data_set(raw_data, spam_set, normal_set)


def handle_one_data_set(raw_data, spam_set, normal_set, data_flag):
    sample = ''
    while True:
        line = raw_data.readline()
        if not line:
            break;
        result = pattern.match(line)
        if result is None:
            sample += line
        else:
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
            pre_set[word] = 1
        else:
            pre_set[word] += 1

