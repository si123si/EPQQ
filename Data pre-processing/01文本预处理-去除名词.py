# -*- codeing = utf-8 -*-
# @Time : 2023/11/21 10:35
# @Author : 陈倩倩
# @File : 01文本预处理-去除名词.py
# @Software : PyCharm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import jieba.posseg as psg
import itertools
import jieba
import os
import numpy as np

os.chdir("D:/研究生论文/小论文2/代码/01数据预处理")

def Data_Ychuli():
    data = pd.read_excel("./00data-初始-未合并集数/桃花马上请长缨/combined_data.xlsx")
    data = data[['内容']].drop_duplicates()  # 去重
    # data.to_excel('data_tmp.xlsx')
    content = data['内容']

    # 数据清洗

    # 去除去除英文、数字等
    # 由于评论主要为商品的评论，因此去除这些词语
    str_tmp = re.compile('[0-9a-zA-Z]|评论|啊啊啊|垃圾|真的|动画片|发|真|这部')  # re模块 正则表达式
    content = content.apply(lambda x: str_tmp.sub('', str(x)))  # 空值替换匹配内容

    # 词典和合并词替换  表情删除

    # 分词、词性标注、去除停用词代码

    # 分词
    word_tmp = lambda s: [(x.word, x.flag) for x in psg.cut(s)]  # 自定义简单分词函数
    seg_word = content.apply(word_tmp)

    # 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
    n_word = seg_word.apply(lambda x: len(x))  # 每一评论中词的个数

    n_content = [[x + 1] * y for x, y in zip(list(seg_word.index), list(n_word))]
    index_content = sum(n_content, [])  # 将嵌套的列表展开，作为词所在评论的id

    seg_word = sum(seg_word, [])
    word = [x[0] for x in seg_word]  # 词

    nature = [x[1] for x in seg_word]  # 词性

    # content_type = [[x] * y for x, y in zip(list(data['评论类型']), list(n_word))]
    # content_type = sum(content_type, [])  # 评论类型

    result = pd.DataFrame({"index_content": index_content,
                           "word": word,
                           "nature": nature,
                           # "content_type": content_type}
                           })

    # 删除标点符号
    result = result[result['nature'] != 'x']  # x表示标点符号

    # 删除停用词
    stop_path = open("./stop_dic/stopwords.txt", 'r', encoding='UTF-8')
    stop = stop_path.readlines()
    stop = [x.replace('\n', '') for x in stop]
    word = list(set(word) - set(stop))
    result = result[result['word'].isin(word)]

    # 构造各词在对应评论的位置列
    n_word = list(result.groupby(by=['index_content'])['index_content'].count())
    index_word = [list(np.arange(0, y)) for y in n_word]
    index_word = sum(index_word, [])  # 表示词语在该评论的位置

    # 合并评论id，评论中词的id，词，词性，评论类型
    result['index_word'] = index_word

    # 提取含有名词类的评论
    ind = result[['n' in x for x in result['nature']]]['index_content'].unique()
    result = result[[x in ind for x in result['index_content']]]

    # 绘制词云

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    frequencies = result.groupby(by=['word'])['word'].count()
    frequencies = frequencies.sort_values(ascending=False)
    backgroud_Image = plt.imread('./stop_dic/pl.jpg')
    wordcloud = WordCloud(font_path="./stop_dic/simsun.ttf",
                          max_words=200,
                          background_color='white',
                          mask=backgroud_Image)
    my_wordcloud = wordcloud.fit_words(frequencies)
    plt.imshow(my_wordcloud)
    plt.axis('off')
    plt.show()

    # 将结果写出
    result.to_excel("./02输出结果-预处理/桃花马上请长缨/1.(去除无效评论).xlsx",index=False)

# 每个评论中只留名词

if __name__ == '__main__':
    Data_Ychuli()


