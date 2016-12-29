# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import io
from collections import OrderedDict


org_train_file = 'training.1600000.processed.noemoticon.csv'
org_test_file = 'testdata.manual.2009.06.14.csv'

# 提取文件中有用的字段
def usefull_field(org_file, output_file):
    output = io.open(output_file, 'w', encoding = 'utf-8')
    with io.open(org_file, buffering = 10000, encoding = 'utf-8') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                clf = line.split(',')[0]
                if clf == '0':
                    clf = [0,0,1] # 消极评论
                elif clf == '2':
                    clf = [0,1,0] # 中性评论
                elif clf == '4':
                    clf = [0,0,1] # 积极评论

                tweet = line.split(',')[-1]
                outputline = str(clf) + ':%:%:%:' + tweet
                output.write(outputline)
        except Exception as e:
            print(e)
    output.close()

# usefull_field(org_train_file, 'training.csv')
# usefull_field(org_test_file, 'testing.csv')

# 创建词汇表
def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with io.open(train_file, buffering = 10000, encoding = 'utf-8') as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(line.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    # print '---' + word
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1
 
            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                print '+++' + word
                if count_word[word] < 100000 and count_word[word] > 100:  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex

lex = create_lexicon('training.csv')
 
# with io.open('lexcion.pickle', 'wb') as f:
#     pickle.dump(lex, f)
