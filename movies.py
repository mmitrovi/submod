#Author: Marko Mitrovic (please email marko.mitrovic@yale.edu with any questions)
#Date Last Edited: Oct.20,2017
#Requires Keras with Theano backend!

#ratings.dat is a set of time-stamped movie reviews from various users.
#Given a sequence of movie reviews from a particular user, we want to predict which movies will be reviewed next.
#This code will run and compare the results of the following algorithms:
#Frequency: just return the most popular movies (i.e. movies that are reviewed most often)
#OMegA: algorithm from from Tschiatschek et al. (2017) (https://las.inf.ethz.ch/files/tschiatschek17ordered.pdf)
#sg: Sequence-Greedy from our paper
#hsg: Hyper Sequence-Greedy from our paper
#LSTM-RNN: Long Short-Term Memory - Recurrent Neural Network

#at the bottom of this code, you can play around with some parameters.


import time
import math
import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils


def importData(path):
	#import data from the ratings.dat file
	#data must be in the following format:
	#UserID::MovieID::Rating::Timestamp
	
	print 'Loading data...'
	arr = []
	with open(path, 'rU') as f:
		content = f.readlines()	#load the file into an array called content	
		for row in content:	
			row = row.split('::')
			row2 = []
			for entry in row:
				row2.append( int(entry) )
			arr.append(row2)
			
	return arr
	
def userDic(arr):
	#combine all user reviews into a single entry
	#i.e. dic[userID] is an array of entries
	#where each entry is [productID,rating,UnixTime]
	
	dic = {}
	for i in range(0,len(arr)):
		userID = arr[i][0]
		productID = arr[i][1]
		rating = arr[i][2]
		unixTime = arr[i][3]
		temp = [productID,rating,unixTime]
		
		if userID not in dic:
			dic[userID] = []
		dic[userID].append(temp)
	
	#sort each array by unixTime
	#and keep only the productIDs (using filterArr).
	for userID in dic:
		dic[userID] = filterArr(sorted(dic[userID], key=lambda x: x[2]),0)
	
	return dic
	
def removeUsers(users,m1,m2):
	#remove all users who have reviewed less than m1 movies or more than m2 movies.

	rmU = []
	for key in users:
		size = len(users[key])
		if size < m1 or size > m2:
			rmU.append(key)
	for key in rmU:
		users.pop(key)	

def productDic(users,bound):
	#dic[i][i] is number of times product i has been reviewed
	#dic[i][j] is probability of product j being reviewed given that product i was reviewed first
	#bound is the value we add to the denominator in the calculation of the probability above
	#i.e. Nij is number of times product j was reviewed after product i, Nii is number of times product was reviewed
	#then dic[i][j] = Nij/(Nii + bound)
	
	dic = {}
	for key in users: #first fill up dic[i][i]
		for pid in users[key]:
			if pid not in dic:
				dic[pid] = {}
				dic[pid][pid] = 0.0
			dic[pid][pid] += 1
	
	for key in users: #now fill up dic[i][j]
		temp = users[key]
		for j in range(1,len(temp)):
			for i in range(0,j):
				pi = temp[i]
				pj = temp[j]
				
				if pi not in dic:
					dic[pi] = {}
				if pj not in dic[pi]:
					dic[pi][pj] = 0.0
				dic[pi][pj] += 1
	
	for key1 in dic:
		for key2 in dic[key1]:
			if key1 != key2:				
				dic[key1][key2] = dic[key1][key2]/(dic[key1][key1]+bound)
	
	return dic

def hyperDic(users,prods,bound):
	#dic[i][j][k] is probability of reviewing movie k given that they reviewed movie i then movie j
	#bound again is number added to the denominator of the probability calculation.
	
	dic = {}
	for key in users:
		temp = users[key]
		for k in range(2,len(temp)):
			for j in range(1,k):
				for i in range(0,j):
					pi = temp[i]
					pj = temp[j]
					pk = temp[k]
				
					if pi not in dic:
						dic[pi] = {}
					if pj not in dic[pi]:
						dic[pi][pj] = {}
					if pk not in dic[pi][pj]:
						dic[pi][pj][pk] = 0.0
					dic[pi][pj][pk] += 1
		
	for key1 in dic:
		for key2 in dic[key1]:
			for key3 in dic[key1][key2]:
				
				num = prods[key1][key2] * (prods[key1][key1]+bound) #number of people that reviewed movie i and then movie j
				dic[key1][key2][key3] = dic[key1][key2][key3] / (num+bound)

	return dic

def freq(pseq,prods,k):
	#pseq is prior sequence (i.e. we observe these)
	#freq just selects the k most commonly reviewed movies (that are not already in pseq)
	
	fmov = [] #array of frequencies for movies
	for pid in prods:
		if pid not in pseq:
			score = prods[pid][pid]
			fmov.append([pid,score])
		
	fmov = sorted(fmov, key = lambda x: x[1], reverse = True)
	
	return fmov[:k]

def omega(pseq,prods,k,numUsers,bound):

	#algorithm from Tschiatschek et al. (2017)
	#https://las.inf.ethz.ch/files/tschiatschek17ordered.pdf
	
	dic = {}
	for pid in pseq:
		if pid in prods:
			for key in prods[pid]:
				if key not in pseq:
					score = prods[pid][key]
					if key in dic:
						dic[key] = dic[key] * (1-score)
					else:
						dic[key] = 1 - score
	
	arr = []				
	for key in dic:
		if bound > 0:
			dic[key] = dic[key] * (1 - (prods[key][key]/numUsers/bound))
		else:
			dic[key] = dic[key] * (1 - (prods[key][key]/numUsers))
		dic[key] = 1 - dic[key]
		arr.append([key,dic[key]])				
	
	arr = sorted(arr, key = lambda x: x[1], reverse = True)	
	
	return arr[:k]				

def sg(pseq,prods,k):
	#our algorithm: Sequence-Greedy
	
	arr = []
	for pid in pseq:
		if pid in prods:
			for key1 in prods[pid]:
				if key1 not in pseq:
					arr.append([key1,prods[pid][key1],0])
	
	#print 'Calculated Ours!'		
	arr = sorted(arr, key = lambda x: x[1], reverse = True)	
	
	if len(arr) < k:
		return freq(pseq,prods,k)
	return arr[:k]				

def hsg(pseq,prods,hyper,k):
	#our algorithm: Hyper Sequence-Greedy
	arr = []
	for i in range(0,len(pseq)):
		pid = pseq[i]
		if pid in hyper:
			temp = pseq[i:]
			for key1 in hyper[pid]:
				if key1 in temp:
					for key2 in hyper[pid][key1]:
						if key2 not in pseq:
							score = hyper[pid][key1][key2]
							arr.append([key2,score,1]) #the 1 at the end indicates that it's a hyper edge, 0 means regular edge.
													   #doesn't really do anything, just if we wanna compare which type of edges are being selected
													   #since we combine the hyper and regular edges
	
	arr2 = sg(pseq,prods,k) #the arr we calculated above considers only size 3 hyperedges
							  #it is possible that there are size 2 edges that are more valuable
							  #calculate this and add them to arr
	arr = arr + arr2
		
	#print 'Sorting...'
	arr = sorted(arr, key = lambda x: x[1], reverse = True)	
	
	return arr[:k]	
	
def filterArr(arr,ind):
	#basically make a new array, arr2, that contains only the ind index (i.e. arr[ind]) of each entry
	#just a helper function

	arr2 = []
	for entry in arr:
		arr2.append(entry[ind])
		
	return arr2

def results(guess,fseq):
	#Find what percentage of our predictions are actually eventually reviewed by the user
	
	count = 0.0
	for entry in guess:
		if entry in fseq:
			count += 1
			
	return count/len(guess)
	
def test(users,prods,hyper,kvalues,numUsers,bound,pseqLen):
	
	#runs all our algorithms on the given test set.

	#fseq is the entire sequence of items this person reviewed.
	#pseq is the sequence we show the algorithm.
	#pseqLen is the percentage of all movies we given the algorithm as input.
	#i.e. the algorithm sees pseqLen fraction of all the movies the user reviews.
	#we make k guesses for what they will review next and see how well it performs.
	
	print 'Comparing baseline algorithms...'
	freqRes = [] #results from freq
	omegaRes = [] #results from OMegA
	sgRes = [] #results from Sequence-Greedy
	hsgRes = [] #results from Hyper Sequence-Greedy
	for k in kvalues: #fill in the arrays with default starting values.
		freqRes.append(0.0)
		omegaRes.append(0.0)
		sgRes.append(0.0)
		hsgRes.append(0.0)
		
	k = kvalues[-1] #assuming the k values we went to test are sorted (ascending), take the largest k value
	for uid in users:
		uList = users[uid] #list of all movies the user has reviewed
		size = len(uList) #total number of movies the user has reviewed
		plen = int(size*pseqLen) #length of pseq
		pseq = uList[:plen]
		fseq = uList
		
		ans1 = filterArr(freq(pseq,prods,k),0) #use filterArr() to take only the pid and not the score.
		ans3 = filterArr(omega(pseq,prods,k,numUsers,bound),0)
		ans4 = filterArr(sg(pseq,prods,k),0)
		ans5 = filterArr(hsg(pseq,prods,hyper,k),0)
		
		for i in range(0,len(kvalues)):
			k = kvalues[i]
	
			freqRes[i] += results(ans1[:k],fseq) 
			omegaRes[i] +=  results(ans3[:k],fseq) 
			sgRes[i] += results(ans4[:k],fseq) 
			hsgRes[i] += results(ans5[:k],fseq) 
		
	size = float(len(users))
	for i in range(0,len(kvalues)):
		freqRes[i] = freqRes[i]/size
		omegaRes[i] = omegaRes[i]/size 
		sgRes[i] = sgRes[i]/size
		hsgRes[i] = hsgRes[i]/size
		
	return [freqRes,omegaRes,sgRes,hsgRes]

def kcv(uArr,users,i,numTrials): 
	#k-fold cross validation, where k = numTrials
	#splits the data into training data and testing data.
	
	trainSet = {}
	testSet = {}
	temp1 = int(i/float(numTrials)*len(uArr))
	temp2 = int((i+1)/float(numTrials)*len(uArr))
	
	for j in range(0,len(uArr)):
		uid = uArr[j]
		if j >= temp1 and j <= temp2:
			testSet[uid] = users[uid]
		else:
			trainSet[uid] = users[uid]
			
	return [trainSet,testSet]

def rnn(kv,users,train,test,sl,numNodes,dropout,ep,bs):
	#Train and test an LSTM-RNN (Long Short-Term Memory Recurrent Neural Network)
	#we will train on sequences of length sl and attempt to predict the (sl+1)th movie for each user
	#in the paper we use sl = 6, so the RNN will take a sequence of length 6 and attempt to predict the 7th movie in the sequence.
	
	
	print 'Training/Testing LSTM-RNN...'
	numMovies = 3706.0 #total number of movies, used for one hot encoding (this is a constant for this dataset).
	N2 = len(train)
	dataX = []
	dataY = []
	for key in train:
		user = users[key]
		size = len(user)
		temp = users[key][0:sl]
		while len(temp) < sl:
			temp.insert(0,0) #pad to length sl (prepend 0s to start).
		dataX.append(temp)
		dataY.append(users[key][sl])
			
	
	N = len(dataX)		
	# reshape X to be [samples, time steps, features]
	X = np.reshape(dataX, (N, sl, 1))
	# normalize
	X = X / numMovies
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model

	model = Sequential()  
	model.add(LSTM(numNodes, input_shape=(X.shape[1],X.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X, y, epochs = ep, batch_size = bs)

	avg = [0.0]*len(kv)
	for key in test:
		size = len(users[key])
		temp = users[key][0:sl]
		while len(temp) < sl:
			temp.insert(0,0) #pad to length sl
		temp = np.reshape(temp,(1,sl,1))
		answer = model.predict(temp)[0] #predict the next item given the length sl sequence temp.
										#the output will be a 3707x1 vector where each entry
										#is the relative probability of that entry being the next item.
		
		ans = []
		for i in range(0,len(answer)):
			ans.append([i,answer[i]])
		ans = sorted(ans, key = lambda x: x[1],reverse=True) #sort all the entries in the answer vector (descending)

		for i in range(0,len(kv)):
			k = kv[i]
			ans2 = []
			for j in range(0,k):
				ans2.append(ans[j])
			
			score = 0.0	
			for entry in ans2:
				if entry[0] in users[key][sl:]:
					score += 1	
			score = score/float(k)/float(len(test))	
			avg[i] += score

	return avg

def main(kval,minItems,maxItems,pseqLen,percTrain,bound,path,rnnSeqLen,numNodes,dropout,ep,bs):	

	print 'k values:',kval,'| pseqLen:',pseqLen,'| bound:',bound,'| min/maxItems:',minItems,maxItems,'| percTrain:',percTrain
	print 'rnnSeqLen: ',rnnSeqLen,'| numNodes:',numNodes,'| dropout:',dropout,'| epochs:',ep,'| batchsize:',bs
	bound1 = bound*percTrain #scale bound for training set
	bound2 = bound*(1-percTrain) #scale bound for testing set
	numTrials = int(1/(1-percTrain)) #find how many cross-validations we can do with the given split of testing/training.
	
	arr = importData(path) #import data
	users = userDic(arr) #arrange data by user 
					     #i.e. users is a dictionary where the key is the userID
					     #and the entry is an array of all the movies the user has reviewed.
	removeUsers(users,minItems,maxItems) #remove all users who have reviewed less than minItems or more than maxItems.
	
	uArr = [] #need array of users, not dic, so we can shuffle and do cross-validation.
	for key in users:
		uArr.append(key)
	random.shuffle(uArr) #randomly shuffle the users
	
	final = [[],[],[],[],[]] #initialize array to hold final results
	for i in range(0,5):
		for j in range(0,len(kval)):
			final[i].append(0.0)
			
	
	for n in range(0,numTrials):
		print 'Trial number:',n+1 #print which cross-validation number we are on.
		print 'Training edge values for hypergraph...'
		[trainSet,testSet] = kcv(uArr,users,n,numTrials) #split data into training and testing (k-fold cross validation).
		prods = productDic(trainSet,bound1) #find values of size 1 edges (loops) and size 2 edges (regular edges)
		hyper = hyperDic(trainSet,prods,bound1) #find values of size 3 edges (hyperedges)
		numUsers = float(len(trainSet)) #number of users in the training set.
		temp = test(testSet,prods,hyper,kval,numUsers,bound,pseqLen) #run all our algorithms on the test set.
		for i in range(0,len(final)-1): #for each algorithm
			for j in range(0,len(kval)): #for each k value we want to test.
				final[i][j] = final[i][j] + temp[i][j]
		
		#for each user in the train set, use the first sl movies to train an LSTM-RNN 
		#to predict the (sl+1)th movie that the user will review.
		#for each user in the test set, use the first sl movies and the trained LSTM-RNN
		#to predict the (sl+1)th movie that the user will review.
		rnnTemp = rnn(kval,users,trainSet,testSet,rnnSeqLen,numNodes,dropout,ep,bs) 
		for j in range(0,len(kval)):
			final[4][j] = final[4][j] + rnnTemp[j]
		
	for i in range(0,len(final)):
		for j in range(0,len(kval)):
			final[i][j] = round(final[i][j]/float(numTrials),4) #round to 4 decimal places for readability.
			

	#print results		
	print 'Freq:',final[0]
	print 'OMegA:',final[1]
	print 'Sequence-Greedy:',final[2]
	print 'Hyper Sequence-Greedy:',final[3]
	print 'LSTM-RNN:',final[4]
		
#========
#========

kval = [4,6,8,10,12] #set of k values we want to test, should be sorted (ascending)
bound = 20 #extra number we add to denominator (to give more value to edges that appear more often).
minItems = 20 #minimum number of items for a user to be considered
maxItems = 30 #maximum number of items for a user to considered
pseqLen = 0.5 #fraction of reviews that are given as input to algorithms
percTrain = 0.9 #percentage of points for training. 
path = 'ratings.dat' #file we are reading from.

rnnSeqLen = 6 #length of sequences we train for LSTM-RNN
numNodes = 256 #number of lstm nodes in our neural network
dropout = 0.2 #dropout layer
ep = 30 #number of epochs to train neural network
bs = 128 #batchsize when training neural network

main(kval,minItems,maxItems,pseqLen,percTrain,bound,path,rnnSeqLen,numNodes,dropout,ep,bs) 


	