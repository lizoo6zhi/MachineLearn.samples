import numpy as np
import math

def loadDataSet(fileName):      
    dataset = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) 
        dataset.append(fltLine)
	
    return dataset

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1
	
def regleaf(datamat):
	return np.mean(datamat[:,-1])
	
def regerro(datamat):
	return np.var(datamat[:,-1]) * np.shape(datamat)[0]  #通过均方差切分数据
	
def chooseBestSplit(datamat,leafType=regleaf,erroType=regerro,ops=(1,4)):
	tolS = ops[0]
	tolN = ops[1]
	classlist = [example[-1] for example in datamat.tolist()]

	if classlist.count(classlist[0]) == len(classlist): #所有分类结果相同
		return None,leafType(datamat)
	m,n = np.shape(datamat)
	svar = regerro(datamat)
	best_svar = np.inf
	best_index = 0
	best_value = 0
	for index in range(n-1):
		splitvalset = set(datamat[:,index].T.tolist()[0])
		for splitval in splitvalset:
			mat0,mat1 = binSplitDataSet(datamat,index,splitval)
			newsvar = regerro(mat0) + regerro(mat1)
			if newsvar < best_svar:
				best_svar = newsvar
				best_index = index
				best_value = splitval
	if svar - best_svar < tolS: #若误差减小不大则推出
		return None,leafType(datamat)
	mat0,mat1 = binSplitDataSet(datamat,best_index,best_value)
	if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:  #若切分出来的数据集很小则推出
		return None,leafType(datamat)
	return best_index,best_value
	
def createTree(datamat,leafType=regleaf,erroType=regerro,ops=(1,4)):
	index,val = chooseBestSplit(datamat)
	if index is None:
		return val
	tree = {}
	tree['spInd'] = index
	tree['spVal'] = val
	lset,rset = binSplitDataSet(datamat,index,val)
	tree['left'] = createTree(lset,leafType,erroType,ops)
	tree['right'] = createTree(rset,leafType,erroType,ops)
	return tree
	
def isTree(obj):
	return ((type(obj).__name__) == 'dict')
	
def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['right'] + tree['left']) / 2.0
	
def prune(tree,testData):
	"""
		回归树剪枝函数
	"""
	if np.shape(testData)[0] == 0:
		return getMean(tree)
	if isTree(tree['right']) or isTree(tree['left']):
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'],lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'],lSet)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
		errorNomerge = sum(np.power(lSet[:,-1] - tree['left'], 2)) + \
						 sum(np.power(rSet[:,-1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
		if errorMerge < errorNomerge:
			print("merger")
			return treeMean
		else:
			return tree
	else:
		return tree
		
if __name__ == "__main__":
	dataset = loadDataSet("dataSet\ex2.txt")
	datamat = np.mat(dataset)
	tree = createTree(datamat)
	print(tree)
	test_dataset = loadDataSet('dataSet\ex2test.txt')
	print(prune(tree,np.mat(test_dataset)))