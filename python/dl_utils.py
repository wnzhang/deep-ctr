import theano
import theano.tensor as T
import linecache
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import numpy
import os
from time import gmtime, strftime
import pickle

rng = numpy.random
rng.seed(1234)

#parameters
log_path = '../log/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_file=log_path + 'log-' + strftime("%Y-%m-%d", gmtime()) 

def save_weights(file,tuple_weights):
	pickle.dump( tuple_weights, open(file, "wb" ))
	
def save_prediction(file,prediction):
	pickle.dump( prediction, open(file, "wb" ))
#write logs for analysis
def log(msg,file=""):
	with open(log_file+file+'.txt', "a+") as myfile:
		myfile.write(msg+"\n")
def logfile(msg,file):
        print msg
        with open(log_path+file+'.txt', "a+") as myfile:
                myfile.write(msg+"\n")
def log_p(msg,file=""):
	log(msg,file)
	print msg

def init_weight(hidden1,hidden2,acti_type):
	v=rng.uniform(  low=-numpy.sqrt(6. / (hidden1 + hidden2)),
                high=numpy.sqrt(6. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
	if acti_type=='sigmoid':
		ww2=numpy.asarray((v*4))
	elif acti_type=='tanh':
		ww2=numpy.asarray((v))
	else:
		ww2=numpy.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))

	bb2=numpy.zeros(hidden2)
	return ww2,bb2
#get all test set
def get_all_data(file):
	if (not os.path.isfile(file)):
		print 'The file:' + str(file) + ' does not exist'
	else:
		array=[]
		arrayY=[]
		i=0
		with open(file, "r") as ins:
			array = []
			for line in ins:
				if line.strip()!="":
					i+=1
					y=line[0:line.index(',')]
					x=line[line.index(',')+1:]
					arr=[float(xx) for xx in x.split(',')]
					array.append(arr)
					arrayY.append(int(y))
		xarray=numpy.array(array, dtype=theano.config.floatX)
		yarray=numpy.array(arrayY, dtype=numpy.int32)
		return [xarray,yarray]


#Get batch data from training set
def get_batch_data(file,index,size):#1,5->1,2,3,4,5
	array=[]
	arrayY=[]
	for i in range(index,index+size):
		line=linecache.getline(file, i)
		if line.strip()!="":
			y=line[0:line.index(',')]
			x=line[line.index(',')+1:]
			arr=[float(xx) for xx in x.split(',')]
			array.append(arr)
			arrayY.append(int(y))
	xarray=numpy.array(array, dtype=theano.config.floatX)
	yarray=numpy.array(arrayY, dtype=numpy.int32)
	shared_x= theano.shared(numpy.asarray(xarray,dtype=theano.config.floatX))
	shared_y = theano.shared(numpy.asarray(yarray,dtype=theano.config.floatX))
	#return [shared_x,T.cast(shared_y, 'int32')]
	return xarray,yarray
	
	
#get x array and y 
def get_xy(line):
	y=int(line[0:line.index(',')])
	x=line[line.index(',')+1:]
	arr=[float(xx) for xx in x.split(',')]
	return arr,y   
	
	
#get file length
def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1
	
	
#get feature number
def feats_len(fname):
	with open(fname) as f:
		l=len(f.readline().split(','))
	return (l-1)
