#Author: Marko Mitrovic (please email marko.mitrovic@yale.edu with any questions)
#Date Last Edited: Mar.1,2018
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

def removeProducts(users,m):
	
	#Function to remove all movies with fewer than m reviews.
	#It removes them directly from the users dictionary.	
	
	countP = {}
	for uid in users:
		for entry in users[uid]:
			if entry not in countP:
				countP[entry] = 0.0
			countP[entry] += 1
	
	print 'Original Products:',len(countP)	
	
	rmP = {} #list of products to remove
	count = 0
	for key in countP:
		if countP[key] < m:
			rmP[key] = 0
		else:
			count += 1
	print 'Reduced Products: ',count
	
			
	for uid in users:
		tbr = [] #indices to be removed
		for i in range(0,len(users[uid])):
			if users[uid][i] in rmP:
				tbr.append(i)
		temp = []
		for i in range(0,len(users[uid])):
			if i not in tbr:
				temp.append(users[uid][i])
		users[uid] = temp	
		
	return count	
	
def removeUsers(users,m1,m2):
	#remove all users who have reviewed less than m1 movies or more than m2 movies.

	print 'Original users:',len(users)

	rmU = []
	for key in users:
		size = len(users[key])
		if size < m1 or size > m2:
			rmU.append(key)
	for key in rmU:
		users.pop(key)	
		
	print 'Reduced users:',len(users)	

def productDic(users,bound,thresh = 0.0):
	#dic[i][i] is number of times movie i has been reviewed
	#dic[i][j] is number of times movie j was reviewed after movie i was reviewed first
	#dic2[i][i] is the probability of movie i being reviewed.
	#dic2[i][j] is probability of movie i being reviewed given that movie j was reviewed first
	
	N = float(len(users))
	dic = {}
	for key in users: #now fill up dic[i][j]
		temp = users[key]
		for j in range(0,len(temp)):
			for i in range(0,j+1):
				pi = temp[i]
				pj = temp[j]
				
				if pi not in dic:
					dic[pi] = {}
				if pj not in dic[pi]:
					dic[pi][pj] = 0.0
				dic[pi][pj] += 1
	
	dic2 = {}
	for key1 in dic:
		for key2 in dic[key1]:
							
				if key1 == key2:
					score = dic[key1][key1]/N #probability of key1 being reviewed.	
				else:
					score = dic[key1][key2]/(dic[key1][key1]+bound) #conditional probability of key2 being reviewed (given key1 reviewed first).
				
				if score > thresh:
					if key1 not in dic2:
						dic2[key1] = {}
					dic2[key1][key2] = round(score,4)	
					
	return (dic,dic2)

def hyperDic(users,p1,bound,thresh = 0.0):
	#dic[i][j][k] is number of people that reviewed i and j and then k
	#dic2[i][j][k] is probability of reviewing movie k given that movie i was reviewed and then movie j was reviewed.
	
	dic = {}
	for key in users: #fill in dic[i][j][k]
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
	
	dic2 = {}	
	for key1 in dic:
		for key2 in dic[key1]:
			for key3 in dic[key1][key2]:
				
				score = dic[key1][key2][key3]/(p1[key1][key2]+bound)
				if score > thresh:
					
					if key1 not in dic2:
						dic2[key1] = {}
					if key2 not in dic2[key1]:
						dic2[key1][key2] = {}
					dic2[key1][key2][key3] = round(score,4)
					
	return (dic,dic2)

def argmax(arr,searchInd,ansInd):
	#find the index that has maximum value in array
	#each entry in arr might be a list so 
	#searchInd is the max value we are trying to find
	#ansInd is the argument we are trying to return.
	
	max = arr[0][searchInd]
	ans = arr[0][ansInd]
	for i in range(1,len(arr)):
		if arr[i][searchInd] > max:
			max = arr[i][searchInd]
			ans = arr[i][ansInd]
	return (ans,max)

def freq(pseq,prods,k):
	#pseq is prior sequence (i.e. we observe these)
	#freq just selects the k most commonly reviewed movies (that are not already in pseq)
	
	fmov = [] #array of frequencies for movies
	for pid in prods:
		if pid not in pseq:
			score = prods[pid][pid]
			fmov.append([pid,score])
		
	fmov = sorted(fmov, key = lambda x: x[1], reverse = True) #sort by decreasing frequency
	
	fmov2 = []
	for i in range(0,k): #take the top k movies (but only the movieID, not the score).
		fmov2.append(fmov[i][0])
	
	return fmov2

def omega(pseq,p2,k):
	#Implementation of OMegA from Tschiatschek et al. (2017).
	
	pseqProb = {} #dictionary of probabilities that this item was reviewed.
				  #everything that starts off in pseq has value 1
				  #then if an element is added to new, it is added to pseqProb
				  #with probability equal to the certainty that it should be added.
	for entry in pseq:
		pseqProb[entry] = 1.0
	
	dic = {}
	for j in range(0,len(pseq)):
		pid = pseq[j]
		if pid in p2:
			for key in p2[pid]:
				if key not in pseq:
					score = p2[pid][key]
					if key not in dic:
						dic[key] = 1.0
					dic[key] = dic[key] * (1-score)
	
	dic2 = {}
	arr = []
	for key in dic:
		dic2[key] = dic[key] * (1 - p2[key][key])
		dic2[key] = 1 - dic[key]
		arr.append([key,dic2[key]])
	best,v = argmax(arr,1,0)
	dic.pop(best)
	pseqProb[best] = v
		
	new = [best]
	for i in range(1,k):
		for key in p2[best]:
			if key not in pseq and key not in new:
				score = p2[best][key]*pseqProb[best]
				if key not in dic:
					dic[key] = 1.0
				dic[key] = dic[key] * (1-score)
		
		dic2 = {}		
		arr = []
		for key in dic:
			dic2[key] = dic[key] * (1 - p2[key][key])
			dic2[key] = 1 - dic2[key]
			arr.append([key,dic2[key]])
		best,v = argmax(arr,1,0)
		dic.pop(best)
		pseqProb[best] = dic2[best]
		new.append(best)
	
	return new

def sg(pseq,p2,k):
	#Implementation of Sequence-Greedy (Algorithm 1 in the paper).
	
	pseqProb = {} #dictionary of probabilities that this item was reviewed.
				  #everything that starts off in pseq has value 1
				  #then if an element is added to new, it is added to pseqProb
				  #with probability equal to the certainty that it should be added.
	for entry in pseq:
		pseqProb[entry] = 1.0
	
	arr = []
	for i in range(0,len(pseq)): #for each movie already seen
		pid = pseq[i]
		if pid in p2: #if the movie has an edge leaving from it (it should)
			for key1 in p2[pid]: #all movies key1 s.t. they have an edge from pid
				if key1 not in pseq: #if key1 hasn't already been reviewed/recommended
					arr.append([key1,p2[pid][key1]])
	
	new = []
	for i in range(0,k):
		best,v = argmax(arr,1,0) #find argmax where 1 is the index we want to maximize, and 0 is the arg index.
		arr2 = []
		for entry in arr:
			if entry[0] != best:
				arr2.append(entry)
		arr = arr2
		new.append(best)
		
		pseqProb[best] = v
		for key1 in p2[best]:
			if key1 not in pseq and key1 not in new:
				arr.append([key1,p2[best][key1]*pseqProb[best]])
		
	return new

def hsg(pseq,p2,h2,k):
	#Implementation of algorithm 3 in the paper.
	
	
	pseqProb = {} #dictionary of probabilities that this item was reviewed.
				  #everything that starts off in pseq has value 1
				  #then if an element is added to new, it is added to pseqProb
				  #with probability equal to the certainty that it should be added.
	for entry in pseq:
		pseqProb[entry] = 1.0
		
	arr = []
	for i in range(0,len(pseq)):
		pid = pseq[i]
		if pid in h2:
			temp = pseq[i+1:]
			for key1 in h2[pid]:
				if key1 in temp:
					for key2 in h2[pid][key1]:
						if key2 not in pseq:
							score = h2[pid][key1][key2]
							arr.append([key2,score])
		
		for i in range(0,len(pseq)): #for each movie already seen
			pid = pseq[i]
			if pid in p2: #if the movie has an edge leaving from it (it should)
				for key1 in p2[pid]: #all movies key1 s.t. they have an edge from pid
					if key1 not in pseq: #if key1 hasn't already been reviewed/recommended
						arr.append([key1,p2[pid][key1]])
	
	new = []
	for i in range(0,k):
		best,v = argmax(arr,1,0) #find argmax where 1 is the index we want to maximize, and 0 is the arg index.
		arr2 = []
		for entry in arr:
			if entry[0] != best:
				arr2.append(entry)
		arr = arr2
		new.append(best)
		
		pseqProb[best] = v
		for key1 in p2[best]: #size 2 edges
			if key1 not in pseq and key1 not in new:
				arr.append([key1,p2[best][key1]*pseqProb[best]])
		
		temp = pseq + new[:-1]
		for key1 in temp: #size 3 edges
			if best in h2[key1]:
				for key3 in h2[key1][best]:
					if key3 not in pseq and key3 not in new:
						score = h2[key1][best][key3]*pseqProb[key1]*pseqProb[best]
						arr.append([key3,score])	
				
		
	return new	
	
def filterArr(arr,ind):
	#basically make a new array, arr2, that contains only the ind index (i.e. arr[ind]) of each entry
	#just a helper function

	arr2 = []
	for entry in arr:
		arr2.append(entry[ind])
		
	return arr2

def genPairs(seq):
	#seq is an ordered list, we want to return list of ordered pairs
	#for example: if fseq = [1,2,3,4]
	#then we return [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
	
	pairs = []
	k = len(seq)
	for i in range(1,k):
		for j in range(0,i):
			pairs.append([seq[j],seq[i]])
			
	return pairs
		
def moranScore(ans,fseq):
	#This is basically a modified version of Kendall tau distance.

	#ans is our guess for the next k movies to be reviewed.
	#fseq are the actual next k movies that were reviewed.
	
	#Basically, for each ordered pair of movies in fseq,
	#we will check if that pair appears in the correct order in ans.
	
	#for example: if fseq = [1,2,3,4]
	#then we will check for (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
	#therefore if ans = [1,4,2,5], then it only contains the ordered pairs
	#(1,4) and (1,2).
	
	ansPairs = genPairs(ans)
	fseqPairs = genPairs(fseq)
	
	score = 0.0
	for entry in fseqPairs:
		if entry in ansPairs:
			score += 1/float(len(fseqPairs))
			
	return score
			
def test(users,prods,hyper,kvalues,numUsers,bound,pseqLen):
	
	#runs all our algorithms on the given test set.

	#fseq is the entire sequence of items this person reviewed.
	#pseq is the sequence we show the algorithm.
	#pseqLen is the number of movies we give the algorithm as input.
	
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
		pseq = uList[:pseqLen]
		fseq = uList[pseqLen:]
		
		ans1 = freq(pseq,prods,k) 
		ans3 = omega(pseq,prods,k)
		ans4 = sg(pseq,prods,k)
		ans5 = hsg(pseq,prods,hyper,k)
		
		for i in range(0,len(kvalues)):
			k2 = kvalues[i]
	
			freqRes[i] += moranScore(ans1[:k2],fseq[:k2]) 
			omegaRes[i] += moranScore(ans3[:k2],fseq[:k2]) 
			sgRes[i] += moranScore(ans4[:k2],fseq[:k2]) 
			hsgRes[i] += moranScore(ans5[:k2],fseq[:k2]) 
		
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

def reMap(users):

	#Originally we have 3,706 movies and each movie has an ID in this range.
	#However, when we remove all movies with fewer than 1000 reviews, we end up with only 207 movies.
	#Therefore, we re-map the movie IDs so they range between 0-207.
	#This is mainly useful for neural network implementation (so we have a size 207 vector instead of size 3706).
	
	dic = {}
	count = 0
	for key in users:
		for mov in users[key]:
			if mov not in dic:
				dic[mov] = count
				count += 1
				
	return dic

def rnn(train,test,cats,k,numNodes,dropout,ep,bs):
	#LSTM where we output a single vector and take the top k highest values.
	
	print 'Training/Testing LSTM-RNN for k =',k,'...'
	
	numMovies = len(cats)
	
	trainX = []
	trainY = []
	for key in train:		
		tempX = []
		for i in range(0,k):
			temp = [0]*numMovies
			ind = cats[train[key][i]]
			temp[ind] = 1
			tempX.append(temp)
			
		tempY = [0]*numMovies	
		for i in range(k,k+k):
			ind = cats[train[key][i]]
			tempY[ind] = 1
			
		trainX.append(tempX)
		trainY.append(tempY)
		
	testX = []
	testY = []
	for key in test:		
		tempX = []
		for i in range(0,k):
			temp = [0]*numMovies
			ind = cats[test[key][i]]
			temp[ind] = 1
			tempX.append(temp)
			
		tempY = [0]*numMovies	
		for i in range(k,k+k):
			ind = cats[test[key][i]]
			tempY[ind] = 1
			
		testX.append(tempX)
		testY.append(tempY)
	
	N = len(trainX)		
	# reshape X to be [samples, time steps, features]
	X = np.reshape(trainX, (N, k, numMovies))
	y = np.reshape(trainY, (N,numMovies))

	# define the LSTM model
	model = Sequential()  
	model.add(LSTM(numNodes, input_shape=(X.shape[1],X.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(y.shape[1],activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	model.fit(trainX, trainY, epochs = ep, batch_size = bs,verbose=0)

	overall = 0.0
	overall2 = 0.0
	for j in range(0,len(testX)):
	
		temp = testX[j]
		temp = np.reshape(temp, (1,k,numMovies))
		answer = model.predict(temp)[0]
		
		truth = []
		for i in range(0,numMovies):
			if testY[j][i] == 1:
				truth.append(i)	
		
		ans = []
		for i in range(0,len(answer)):
			ans.append([i,answer[i]])
		ans = sorted(ans, key = lambda x: x[1],reverse=True) #sort all the entries in the answer vector (descending)
		
		output = [item[0] for item in ans][:k]
		
		score = 0.0
		for i in range(0,k):
			if output[i] in truth:
				score += 1.0/k
				
		score2 = moranScore(output,truth)
				
		
		overall += score/len(testX)
		overall2 += score2/len(testX)
	
	return overall2
	
def main(kval,minReviews,minItems,maxItems,pseqLen,percTrain,bound,path,numNodes,dropout,ep,bs):	

	print 'k values:',kval,'| pseqLen:',pseqLen,'| bound:',bound,'| minReviews:',minReviews,'| min/maxItems:',minItems,maxItems,'| percTrain:',percTrain
	print 'numNodes:',numNodes,'| dropout:',dropout,'| epochs:',ep,'| batchsize:',bs
	bound1 = bound*percTrain #scale bound for training set
	bound2 = bound*(1-percTrain) #scale bound for testing set
	numTrials = int(1/(1-percTrain)) #find how many cross-validations we can do with the given split of testing/training.
	
	arr = importData(path) #import data
	users = userDic(arr) #arrange data by user 
					     #i.e. users is a dictionary where the key is the userID
					     #and the entry is an array of all the movies the user has reviewed.
	
	numMovies = removeProducts(users,minReviews) #remove all movies that have been reviewed fewer than minReviews times.
	removeUsers(users,minItems,maxItems) #remove all users who have reviewed less than minItems or more than maxItems.
	cats = reMap(users) #remap indices of movies 
	
	count = 0
	for uid in users:
		for entry in users[uid]:
			count += 1
	print 'Number of reviews:',count		
	
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
		p1,p2 = productDic(trainSet,bound1) #find values of size 1 edges (loops) and size 2 edges (regular edges)
		h1,h2 = hyperDic(trainSet,p1,bound1) #find values of size 3 edges (hyperedges)
		numUsers = float(len(trainSet)) #number of users in the training set.
		temp = test(testSet,p2,h2,kval,numUsers,bound,pseqLen) #run all our algorithms on the test set.
		for i in range(0,len(final)-1): #for each algorithm
			for j in range(0,len(kval)): #for each k value we want to test.
				final[i][j] = final[i][j] + temp[i][j]
		
		#This is the LSTM part. Basically we train an LSTM on the first k movies each user
		#in the training set has reviewed, where the output is a vector of probabilities
		#for which movie will be reviewed next. For testing, we input the first k movies
		#each user in the test set has reviewed, and the output is the top k most likely 
		#movies to be reviewed next.
		for j in range(0,len(kval)):
			k = kval[j]
			final[4][j] = final[4][j] + rnn(trainSet,testSet,cats,k,numNodes,dropout,ep,bs)
		
	for i in range(0,len(final)):
		for j in range(0,len(kval)):
			final[i][j] = round(final[i][j]/float(numTrials),4) #round to 4 decimal places for readability.
			

	#print results		
	print 'Freq:',final[0]
	print 'OMegA:',final[1]
	print 'Sequence-Greedy:',final[2]
	print 'Hyper Sequence-Greedy:',final[3]
	print 'LSTM-RNN:',final[4]
	
	print 'freqR =',final[0]
	print 'krauseR =',final[1]
	print 'ourR =',final[2]
	print 'ourHR =',final[3]
	print 'rnn =',final[4]



		
#========
#========




kval = [2,3,4,5,6,8,10] #set of k values we want to test, should be sorted (ascending)
bound = 20 #extra number we add to denominator (to give more value to edges that appear more often).
minReviews = 1000 #minimum number of reviews for a movie to be considered.
minItems = 20 #minimum number of items for a user to be considered.
maxItems = 50 #maximum number of items for a user to considered.
pseqLen = 8 #length of given sequence (that we use for future predictions).
percTrain = 0.9 #percentage of points for training. 
path = 'ratings.dat' #file we are reading from.

numNodes = 512 #number of lstm nodes in our neural network
dropout = 0.5 #dropout layer
ep = 30 #number of epochs to train neural network
bs = 128 #batchsize when training neural network

main(kval,minReviews,minItems,maxItems,pseqLen,percTrain,bound,path,numNodes,dropout,ep,bs) 