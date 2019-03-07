#*--coding=utf-8--

import random
import numpy as np

'''爬山算法'''
def hillclimb(domain,costf):
	#创建一个随机解
	sol = [random.ranint(domain[i][0],domain[i][1]) for i in len(domain)]
	
	min_cost = 999999.9
	best_sol_index = -1
	#主循环
	while 1:
		neighbors = []
		co
		for i in rangge(len(domain)):
			if sol[i] > domain[i][0]:
				neighbors.append(sol[0:i]+[sol[i]-1]+sol[i+1:])
			if sol[i] < domain[i][1]:
				neighbors.append(sol[0:i] + [sol[i]+1],sol[i+1:])
		
		current_cost = costf(sol)
		min_cost = current_cost
		for j in range(len(neighbors)):
			cost = costf(neighbors[j])
			if cost < min_cost:
				min_cost = cost
				best_sol_index = j
		sol = neighbors[best_sol_index]
		
		if min_cost == current_cost:  #获取的是局部最优解并非全局最优解
			break
			
def annealing_optimaize(domain,costf,T=1000.0,cool=0.95,step=1):
	#创建一个随机解
	vec = [float(random.ranint(domain[i][0],domain[i][1]) for i in len(domain))]
	
	while T > 0.1:
		#选择一个索引值
		i = random.randint(0,len(domain)-1)
		#选择一个改变索引值的方向
		dir = random.randint(-step,step)
		
		#创建一个代表题解的新列表，改变其中的一个值
		vecb = vec[:]
		vecb[i] +=dir
		if vecb[i] < domain[i][0]:
			wecb[i] = domain[i][0]
		if vecb[i] > domain[i][1]:
			wecb[i] = domain[i][1]
			
		#计算当前的成本和新的成本
		ea = costf(vec)
		eb = costf(vecb)
		
		if (eb < ea) or (random.random() < math.pow(math.e,-(eb-ea)/T)):
			vec = vecb
		T = t *cool
	return vec

def selectJrand(i, m):
    """
    函数说明:随机选择alpha_j的索引值
 
    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
    """
    j = i                                 #选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j
	
def smo(dataset,classabels,C,toler,maxiter):
	'''序列最小最优化算法'''
	datamatrix = np.mat(dataset)
	label_matrix = np.mat(classabels)
	b = 0
	m,n = np.shape(datamatrix)
	alphas = np.zeros((m,1))
	iter = 0
	
	while(iter < maxiter):
		alphaPairsChanged = 0
		for i in range(m):
			fxi = float(np.mutiply(alphas,label_matrix).T*(datamatrix*(datamatrix[i,:].T)))+b
			Ei = fxi - flaot(label_matrix[i])
			if (alphas[i] > 0 and alphas[i] < C) and 
			(Ei*label_matrix[i] < -toler or Ei*label_matrix[i]>toler):
				j = selectJrand(i,m) #随机选择第二个alpha
				fxj = float(np.mutiply(alphas,label_matrix).T*(datamatrix*(datamatrix[j,:].T)))+b
				Ej = fxj - flaot(label_matrix[j])
				alphaiold = alphas[i]
				alphajold = alphas[j]
				
				if label_matrix[i] != label_matrix[j]: #y-x=k,L,H为对角线段端点的界
					L = max(0,fxj-fxi)
					H = min(C,C+fxj-fxi)
				else:
					L = max(0,fxj+fxi-C)
					H = min(C,fxj+fxi)
			if L == H:
				print("L==H")
				continue
			
		
				
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	