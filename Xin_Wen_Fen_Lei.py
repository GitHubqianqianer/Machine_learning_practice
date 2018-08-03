#coding=utf-8

'''
新闻分类

'''

import os
import jieba
from sklearn.datasets.base import Bunch
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer#feature_extraction‘特征提取’,Convert a collection of raw documents to a matrix of TF-IDF features.
'''
The sklearn.feature_extraction module deals with feature extraction from raw data. 
It currently includes methods to extract features from text and images.
The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯分类算法，Naive Bayes classifier for multinomial models
'''
The multinomial Naive Bayes classifier is suitable for classification with discrete features
 (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.
  However, in practice, fractional counts such as tf-idf may also work.
'''
from sklearn.metrics import classification_report#metrics度量，分类报告Build a text report showing the main classification metrics


text_train_cnews='data/char-level/cnews.train.txt'
text_val_cnews='data/char-level/cnews.val.txt'
text_test_cnews='data/char-level/cnews.test.txt'
'''
以上三个文本的内容大概为：
体育	黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季。。。。。。。
教育	09年12月英语六级听力参考答案(沪江)最新、最完整的。。。。。。
'''


text_category_cnews='data/char-level/cnews.category.txt'
#category的内容为：
'''
体育	0
财经	1
房产	2
家居	3
教育	4
科技	5
时尚	6
时政	7
游戏	8
娱乐	9
'''

word_level='data/word_level'
text_jieba_cnews='data/word_level/cnews.jieba.txt'
text_cnews='data/word_level/cnews.word.jieba.dat'

#创建文件夹，用来存放处理后的数据
if not os.path.exists(word_level):
    os.mkdir(word_level)

class Categories:
    def __init__(self,filename):#初始化
        self.category={}#将类别名和类别数值存储在字典里面
        for line in open(filename,'r',encoding='utf-8'):
            c,label=line.strip('\r\n').split('\t')#体育 0；财经 1；房产 2；
            self.category[c]=label
    #根据类别名获取类别的数值
    def get_category_label(self,name):
        return self.category[name]


categories=Categories(text_category_cnews)
# h=categories.get_category_label('体育')#
# print(h)

#
def get_word(filenamelist,save_filename):#把所有的数据全部读取，并且输出存到同一个文本文件内
    labels=[]
    count=0
    for filename in filenamelist:
        count+=1
        print('*****正在拆分第%s个输入文本的信息，请稍等*******'%count)
        with open(filename,'r',encoding='utf-8') as f:
            lines=f.readlines()#一开始写成了readline，然后找了半天都没有找出来哪里报错了，fuck！！！！！
        with open(save_filename,'a',encoding='utf-8') as f1:#用w的方式，每读取一个文件，输出的文本会将之前的数据覆盖了，用a的话就不会，追加
            for line in lines:
                label,content=line.strip('\r\n').split('\t')
                labels.append(label)
                content_list=list(jieba.cut(content))#jieba分词
                content_word=''
                #将list里面的元素（词），用‘’拼接成字符串
                for word in content_list:
                    word=word.strip()
                    if word !='':#不等于空的数值，进行存储
                        content_word +=word+' '#使用空格进行区分
                wordli='%s\t%s\n'%(label,content_word.strip())#存储分词后的新闻内容
                f1.write(wordli)
        print(set(labels))#打印出labels中不重复的键值

filenamelist=[text_val_cnews,text_test_cnews,text_train_cnews]
get_word(filenamelist,text_jieba_cnews)


def save_bunch(input_file_name,out_file_name,category_file):#将输入的文本文件进行分词拆分、然后分类
    categories=Categories(category_file)
    #实例化Bunch对象,包含targets、filenames、labels
    #contents
    bunch=Bunch(targets=[],filenames=[],labels=[],contents=[])
    filename=0
    lab=[]
    print('***********正在调用save_bunch函数,请稍等**********')
    with open(input_file_name,'r',encoding='utf-8') as f:
        lines=f.readlines()
    for line in lines:
        filename +=1
        category,content=line.strip('\r\n').split('\t')#由于get_word 函数中最后的wordli又通过\t和\n将分类和对应的新闻又组合在了一起
        bunch.contents.append(content)
        label=categories.get_category_label(category)#label的取值为0到9，代表体育、财经、房产。。。。。
        bunch.labels.append(label)
        bunch.filenames.append(str(filename))
        lab.append(label)
    bunch.targets=list(set(lab))#targets存的是大的分类，而labels是下属的每条新闻都对应一个分类，而这个label是和大的分类一致的。

    #print(bunch) #{'targets': ['0'], 'filenames': ['1', '2', '3', '4', '5', '6'], 'labels': ['0', '0', '0', '0', '0', '0'], 'contents': ['黄蜂 vs 湖人 首发 ： 科比 带伤 战

    with open(out_file_name,'wb') as f:#将bunch存储到文件
        pickle.dump(bunch,f)
#
save_bunch(text_jieba_cnews,text_cnews,text_category_cnews)
text_tfdif_cnews='data/word_level/cnews.word.tfdif.jieba.dat'
stop_word_file='data/中文停用词库.txt'
#
def _read_bunch(filename):
    with open(filename,'rb') as f:
        bunch=pickle.load(f)
    return bunch

def _write_bunch(bunch,filename):
    with open(filename,'wb') as f:
        pickle.dump(bunch,f)

def get_stop_words(filename=stop_word_file):
    stop_word=[]
    for line in open(filename,'r',encoding='gb18030'):
        stop_word.append(line.strip())
    return stop_word
#
# #权重策略TF——IDF
# '''
# TF-IDF(Term frequency * Inverse Doc Frequency)词权重
# 在较低的文本语料库中，一些词非常常见（例如，英文中的“the”，“a”，“is”），因此很少带有文档实际内容的有用信息。
# 如果我们将单纯的计数数据直接喂给分类器，那些频繁出现的词会掩盖那些很少出现但是更有意义的词的频率。
# 为了重新计算特征的计数权重，以便转化为适合分类器使用的浮点值，通常都会进行tf-idf转换。
# 词重要性度量一般使用文本挖掘的启发式方法：TF-IDF。
# 这是一个最初为信息检索（作为搜索引擎结果的排序功能）开发的词加权机制，在文档分类和聚类中也是非常有用的
# 由于tf-idf经常用于文本特征，因此有另一个类称为TfidfVectorizer，
# 将CountVectorizer和TfidfTransformer的所有选项合并在一个模型中
# '''

def tfidf_deal_cnews(input_file_name,out_file_name):
    bunch=_read_bunch(input_file_name)#读取数据
    stop_words=get_stop_words()#得到停用词
    #实例化bunch对象
    #tmd（权重列表）
    #vocabulary（词典索引）
    space_bunch=Bunch(targets=bunch.targets,filename=bunch.filenames,labels=bunch.labels,tmd=[],vocabulary={})
    #使用特征提取函数TfidfVectorizer初始化向量空间模型
    vector=TfidfVectorizer(stop_words=stop_words,sublinear_tf=True,max_df=0.5)#提取函数的初始化，啥数据都没有处理。选择能代表新闻特征、独一无二的词汇，词频大于50%的就被过滤掉？？？如果过大、过小会如何？
    space_bunch.tmd=vector.fit_transform(bunch.contents)#contents只有新闻内容，没有分类。用df-idf训练转化，获得if-idf权值矩阵：fit_transform(raw_documents[, y])	Learn vocabulary and idf, return term-document matrix.
    '''
    print(space_bunch.tmd)输出格式为以下:
    (0, 834)	0.2608786231499419
    (0, 38)	0.2104752305319886
    (0, 557)	0.29664039933480035
    (0, 820)	0.2104752305319886
    '''
    space_bunch.vocabulary=vector.vocabulary_#词典索引，统计词频
    '''
    print(space_bunch.vocabulary)输出格式如下：
    {'黄蜂': 834, 'vs': 38, '湖人': 557, '首发': 820, '科比': 609, '带伤': 352, '保罗': 156,
    '''
    _write_bunch(space_bunch,out_file_name)#写入文件

tfidf_deal_cnews(text_cnews,text_tfdif_cnews)
bunch=_read_bunch(text_tfdif_cnews)
# print(bunch.tmd.shape)#结果为(65000, 379716)

#构建分类器
x_train,x_test,y_train,y_test=train_test_split(bunch.tmd,bunch.labels,test_size=0.2,random_state=100)
'''
X_train格式：
0, 592)	0.05975232195788132
  (0, 59)	0.07286741411594184
  (0, 697)	0.07286741411594184
  (0, 296)	0.07286741411594184
  (0, 224)	0.07286741411594184
  (0, 469)	0.07286741411594184
  (0, 513)	0.07286741411594184
  (0, 514)	0.07286741411594184
  (0, 414)	0.07286741411594184
  (0, 517)	0.07286741411594184
  (0, 84)	0.07286741411594184
  
  Y_train格式：
  ['0', '0', '0', '0']

'''

#以上，提取新闻文档中的词频和对应的新闻分类代号
'''
random_state : int, RandomState instance or None, optional (default=None)If int, 
random_state is the seed used by the random number generator; If RandomState instance, 
random_state is the random number generator; 
If None, the random number generator is the RandomState instance used by np.random.
将总样本划分为100份，从每份中取20%作为训练集，这样的话，可以使得训练集和测试集的误差最小最小。
'''

#先实例化模型，然后调用methods，如调用fit、predict
nb=MultinomialNB(alpha=0.01)#实例化模型 alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
nb.fit(x_train,y_train)#训练模型	Fit Naive Bayes classifier according to X, y
y_pred=nb.predict(x_test)#预测测试集X; Perform classification(分类) on an array of test vectors X.
print(classification_report(y_test,y_pred))#打印输出评分
'''
                precision  recall   f1-score   support

          0       1.00      0.99      1.00      1291
          1       0.94      0.92      0.93      1276
          2       0.91      0.92      0.91      1323
          3       0.95      0.91      0.93      1245
          4       0.92      0.90      0.91      1283
          5       0.94      0.96      0.95      1332
          6       0.96      0.96      0.96      1305
          7       0.95      0.95      0.95      1315
          8       0.98      0.97      0.98      1315
          9       0.92      0.99      0.95      1315

avg / total       0.95      0.95      0.95     13000
'''

































