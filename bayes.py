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
4. 根据贝叶斯算法p(结果|特征) = p(特征|结果) * p(结果) / p(特征)
	注：p(特征,结果) = p(特征1,结果)+p(特征2,结果)+p(特征3,结果)+...
"""
import numpy as np
import math

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
			return_vector[vocablist.index(word)] += 1
	return return_vector	
	
def bayes_train(train_matrix,train_caltegory):
	"""
	训练集
	train_matrix：转换后的文档向量
	train_caltegory：文档类型
	"""
	docs = len(train_matrix)  #文章数量
	wordnum_in_docs = len(train_matrix[0]) #文章的单词数
	p0num = np.ones(wordnum_in_docs)
	p1num = np.ones(wordnum_in_docs)
	p0Denom = 2
	p1Denom = 2
	for i in range(docs):
		if train_caltegory[i] == 1:
			p1num += train_matrix[i] #单词在类型1的所有文章中出现的次数
			p1Denom += sum(train_matrix[i]) #类型1的文章中在词汇表中出现的总次数
		else:
			p0num += train_matrix[i]
			p0Denom += sum(train_matrix[i])
	
	p1vect = np.log(p1num / p1Denom)  #所有单词在类型1的文章中出现的概率
	p0vect = np.log(p0num / p0Denom)  #所有单词在类型0的文章中出现的概率
	pAb = sum(train_caltegory) / len(train_caltegory)#类型1在所有类型中的概率
	return p0vect,p1vect,pAb
	
def classify(pAb,p0vect,p1vect,test_vect):
	"""
	分类
	"""
	p1 = sum(test_vect*p1vect) + math.log(pAb)
	p0 = sum(test_vect*p0vect) + math.log((1-pAb))
	print("p0:",p0)
	print("p1:",p1)
	if p1 > p0:
		return 1
	else:
		return 0
		
if __name__ == '__main__':
	postingList, classVec = loadDataSet()
	vocablist = create_vocab_list(postingList)
	print('vocablist:',vocablist)
	return_vectors = []
	for item in postingList:
		return_vector = trans_doc_to_vector(vocablist,item)
		return_vectors.append(return_vector)
	p0vect,p1vect,pAb = bayes_train(return_vectors, classVec)
	print("pAb:",pAb)
	print('p0vect:',p0vect)
	print('p0vect:',p1vect)
	test_doc = ['love', 'my', 'dalmation']
	test_doc_vect = np.array(trans_doc_to_vector(vocablist, test_doc))
	label = classify(pAb,p0vect,p1vect,test_doc_vect)
	print("留言类型为:",label)
	test_doc1 = ['stupid', 'garbage']
	test_doc_vect1 = np.array(trans_doc_to_vector(vocablist, test_doc1))
	label1 = classify(pAb,p0vect,p1vect,test_doc_vect1)
	print("留言类型为:",label1)
		
	