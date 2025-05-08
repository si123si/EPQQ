# -*- codeing = utf-8 -*-
# @Time : 2023/11/21 10:38
# @Author : 陈倩倩
# @File : 02文本预处理.py
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

data = pd.read_excel("./02输出结果-预处理/桃花马上请长缨/1.(去除无效评论).xlsx")
data = data[['index_content']].drop_duplicates()  # 去重 , '评论类型'
data.head()
data.to_excel("./02输出结果-预处理/桃花马上请长缨/2.(去除无效评论)-标签项.xlsx", index=False)

#读取原始评论文本
data = pd.read_excel("./00data-初始-未合并集数/桃花马上请长缨/combined_data.xlsx")
data.head()
# 在DataFrame中添加一列作为序号
data.insert(0, 'index_content', range(1, len(data) + 1))
data.head()
data.to_excel("./02输出结果-预处理/桃花马上请长缨/3.序号评论项.xlsx", index=False)
# 读取第一个Excel文件
file1_path = "./02输出结果-预处理/桃花马上请长缨/2.(去除无效评论)-标签项.xlsx"
df1 = pd.read_excel(file1_path)

# 读取第二个Excel文件
file2_path = "./02输出结果-预处理/桃花马上请长缨/3.序号评论项.xlsx"
df2 = pd.read_excel(file2_path)

# 找到两个DataFrame中的重复项
duplicates = pd.merge(df1, df2, how='inner')

# 打印重复项
print("重复项：")
print(duplicates)
duplicates.to_excel("./02输出结果-预处理/桃花马上请长缨/4.有效评论项.xlsx", index=False)
