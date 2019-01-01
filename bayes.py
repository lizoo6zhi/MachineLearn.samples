"""
1. 文章数据集：各种类型的文章数篇
2. 关键字数据集：有代表性的关键字
3. 算出各种类别的文章中关键字所占的比例
	以类型A文章系列(文章1，文章2)为例，计算A中各关键字所占比例
	a.统计文章1中包含的关键字总个数sum1
	b.统计文章2中包含的关键字总个数sum2
	c.统计文章1中各关键字分别出现的总次数count1
	d.统计文章2中各关键字分别出现的总次数count2
	e.计算A类型所有文章中包含的关键字总次数sum1+sum2
	f.计算A类型所有文章中各关键字分别出现的总次数count1+count2
	g.计算A类型中所有文章中各关键字出现在A类型文章中的总比例 (count1+count2)/(sum1+sum2)
	h.依次分别计算B类型、C类型等所有文章中各关键字出现在A类型文章中的总比例
"""
import numpy as np

def loadDataSet():
	"""
	创建数据集
	"""
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
	return postingList, classVec
	
def create_vocab_list(dataset):
	"""
	根据数据集创建词汇列表
	"""
	vocabset = set([])
	for doc in dataset:
		vocabset = vocabset | set(doc)
	return list(vocabset)
		
		
def trans_doc_to_vector(vocablist, inputset):
	"""
	将输入词汇通过词汇表转换为向量
	例：vocablist = ['hello','world','fate0729']
		inputset = ['a','hello','b','c']
		return [0,1,0,0]
	"""
	count = len(vocablist)
	return_vector = [0]*count
	for word in inputset:
		if word in vocablist:
			return_vector[vocablist.index(word)] = 1
	return return_vector	
	
def trainNB0(train_matrix,train_caltegory):
	"""
	train_matrix：转换后的文档向量
	train_caltegory：文档类型
	"""
	docs = len(train_matrix)  #文章数量
	wordnum_in_docs = len(train_matrix[0]) #文章的单词数
	p0num = np.zeros(wordnum_in_docs)
	p1num = np.zeros(wordnum_in_docs)
	p0Denom = 0.0
	p1Denom = 0.0
	for i in range(docs):
		if train_caltegory[i] == 1:
			p1num += train_matrix[i] #单词在类型1的所有文章中出现的次数
			p1Denom += sum(train_matrix[i]) #类型1的文章中在词汇表中出现的总次数
		else:
			p0num += train_matrix[i]
			p0Denom += sum(train_matrix[i])
	p1vect = p1num / p1Denom  #所有单词在类型1的文章中出现的概率
	p0vect = p0num / p0Denom  #所有单词在类型0的文章中出现的概率
	return p0vect,p1vect
	
if __name__ == '__main__':
	postingList, classVec = loadDataSet()
	vocablist = create_vocab_list(postingList)
	print('vocablist:',vocablist)
	return_vectors = []
	for item in postingList:
		return_vector = trans_doc_to_vector(vocablist,item)
		return_vectors.append(return_vector)
	p0vect,p1vect = trainNB0(return_vectors, classVec)
	print('p0vect:',p0vect)
	print('p0vect:',p1vect)
		
	