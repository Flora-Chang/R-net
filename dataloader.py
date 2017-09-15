import json
from nltk.tokenize import word_tokenize
import random

class DataLoader:
    def __init__(self, filename, batch_size=128, is_flat=True, max_len=300):
        self.pointer = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.max_len = max_len
        if is_flat:
            f = open(filename, 'r')
            self.flat_data = json.loads(f.readline())
        else:
            self.flat_data = self.read_data(filename)
        random.shuffle(self.flat_data)


    def read_data(self, filename):
        f = open(filename, 'r')
        root = json.loads(f.readline())
        flat_data = []
        non_count = 0
        for theme in root['data']:
            for paragraph in theme['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    max_length = 0
                    best_answer = ''
                    answer_list = qa['answers']
                    for answer in answer_list:
                        if max_length < len(answer['text'].split()):
                            max_length = len(answer['text'].split())
                            best_answer = answer
                    if max_length > 0:
                        d = dict()
                        d['context'] = word_tokenize(context.lower())
                        d['question'] = word_tokenize(question.lower())
                        d['id'] = qa['id']
                        answer_text = word_tokenize(best_answer['text'].lower())
                        l = len(answer_text)
                        for i in range(len(d['context'])):
                            if answer_text == d['context'][i:i + l]:
                                d['start'] = i
                                d['end'] = i + l - 1
                        if not d.has_key('start'):
                            for ans in answer_list:
                                answer_text = word_tokenize(ans['text'].lower())
                                l = len(answer_text)
                                for i in range(len(d['context'])):
                                    if answer_text == d['context'][i:i + l]:
                                        d['start'] = i
                                        d['end'] = i + l - 1
                        if not d.has_key('start'):
                            non_count += 1
                        else:
                            flat_data.append(d)
        print non_count
        return flat_data



    def get_next_batch(self):
        context_list = []
        start_label_list = []
        end_label_list = []
        question_list = []
        id_list = []
        while (True):
            if self.pointer >= len(self.flat_data):
                self.pointer = 0
                self.epoch += 1
                random.shuffle(self.flat_data)
            d = self.flat_data[self.pointer]
            self.pointer += 1
            if d['end'] >= self.max_len:
                continue
            context_list.append(d['context'])
            start_label_list.append(d['start'])
            end_label_list.append(d['end'])
            question_list.append(d['question'])
            id_list.append(d['id'])

            if len(context_list) == self.batch_size:
                break
        return context_list, question_list, start_label_list, end_label_list, id_list, self.epoch
