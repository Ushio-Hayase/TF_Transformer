import json
import math
import os

import numpy as np
import tensorflow as tf
from kobert_tokenizer import KoBERTTokenizer

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1',
                                            sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})


class Preprocess(tf.keras.utils.Sequence):
    def __init__(self, batch_size: tf.int32, shuffle=False, name=None):
        super(Preprocess, self).__init__()
        self.data = []
        self.max_length = tf.cast(128, tf.int32)
        self.pad_idx = 3
        self.batch_size: tf.int32 = batch_size
        self.shuffle = shuffle

        if not os.path.exists(f'train.json'):
            self.__load__("")
            f = open(f'{name}.json', 'wt')
            json.dump(self.data, f)
            f.close()
        else:
            f = open(f'train_tf.json', 'rt', encoding='utf-8')
            self.data = json.load(f)


        self.on_epoch_end()

    def __load__(self, path):
        for (p, dirs, files) in os.walk(path):
            print(f"{p} \n {dirs} ")
            self.__open_file__(p, files)

    # 파일 읽기
    def __open_file__(self, path, files):
        for file in files:
            if os.path.splitext(file)[1] != '.json':
                continue
            f = open(os.path.join(path, file), 'r', encoding='utf-8')
            if (not f.readable()):
                continue
            line = f.read()

            j = json.loads(line)  # json 데이터

            persona_cl = j['personaInfo']['clInfo']
            persona_cp = j['personaInfo']['cpInfo']

            cl_id: str = str(persona_cl['personaID'])
            cp_id: str = str(persona_cp['personaID'])

            self.data[0][cl_id] = [str(i) for i in persona_cl['personaFeatures']]
            self.data[0][cp_id] = [str(i) for i in persona_cp['personaFeatures']]

            for sessions in j['sessionInfo']:
                for i, session in enumerate(sessions['dialog']):
                    if not i < len(sessions['dialog']) - 1: continue
                    self.data[1].append([str(sessions['dialog'][i]['personaID']), str(session['utterance'])])

            f.close()
        print('complete')

    def __sort__(self):
        data = [[],[]]
        for i in range(len(self.data[1])):
            data[1].append([tokenizer.encode(self.data[0][self.data[1][i][0]][0], padding='max_length', truncation='only_first',
                               max_length=self.max_length)])
            data[0].append([tokenizer.encode(self.data[1][i][1], padding='max_length', truncation='only_first',
                               max_length=self.max_length)])
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data[1]) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        a = [self.data[0][i] for i in indices]
        b = [self.data[1][i]for i in indices]

        x = {"input": tf.squeeze((np.array(b))),
             "dec_input": tf.squeeze((np.array(a)))}
        y = np.array(a)

        return x, tf.squeeze(y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[1]))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
