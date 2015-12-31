from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import sys
import numpy
import time
import theano
import theano.tensor as T
import linecache
import math
import dl_utils as ut
import data_fm as fm
import pickle
import sampling_based_denosing_autoencoder as da
from theano.tensor.shared_randomstreams import RandomStreams
import os.path
srng = RandomStreams(seed=234)
rng = numpy.random
rng.seed(1234)
batch_size=1000                                                          #batch size
lr=0.0006                                                               #learning rate
lambda1=0.0001# .01 
hidden0=300                                                       #regularisation rate
hidden1 = 300 #hidden layer 1
hidden2 = 100 #hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 100                                                               #epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../data/train.fm.txt'             #training file
test_file='../data/test.fm.txt'                   #test file
fm_model_file='../data/fm.model.txt'                   #fm model file                 #fm model file
#feats = ut.feats_len(train_file)                                           #feature size


print train_file
train_size=ut.file_len(train_file)                    #training size
test_size=ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches
x_dim=133465
dropout=1
if advertiser=='2997':
    hidden0=200
    hidden1=300
    hidden2=100
    lr=0.0005
    lambda1=0.0000001
    dropout=0.99
    lambda1=0
    
def log_p(msg,m=""):
    ut.log_p(msg,"drop_mlp4da"+str(advertiser)) 
log_p('drop_mlp4da.py|ad:'+advertiser+'|drop:'+str(dropout)+'|b_size:'+str(batch_size)+' | X:'+str(x_dim) + ' | Hidden 0:'+str(hidden0)+ ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L_r:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        
# initialise parameters
arr=[]
arr.append(x_dim)
arr.append(hidden0)
arr.append(hidden1)
arr.append(hidden2)

ww0,bb0=ut.init_weight(x_dim,hidden0,'sigmoid')
ww1,bb1=ut.init_weight(hidden0,hidden1,'sigmoid')
ww2,bb2=ut.init_weight(hidden1,hidden2,'sigmoid')

# ww0,bb0,ww1,bb1,ww2,bb2=da.get_da_weights(train_file,arr,ncases=train_size,batch_size=100000)
# pickle.dump( (ww0,bb0,ww1,bb1,ww2,bb2), open( "2997_da_4l_10.p", "wb" ))

# (ww0,bb0,ww1,bb1,ww2,bb2)=pickle.load( open( "2997_da_4l_10.p", "rb" ) )

def num_feats(file):
    fi = open(file, 'r')
    line=fi.readline()
    s = len(line.strip().split(':'))
    fi.close()
    return (s-1)
numf=num_feats(train_file)
wfile="dropda_"+str(advertiser)+"_.p"
if os.path.isfile(wfile):
    (ww0,bb0,ww1,bb1,ww2,bb2)=pickle.load(open(wfile, "rb" ) )
else:
    ww0,bb0,ww1,bb1,ww2,bb2=da.get_da_weights(train_file,arr,num_feats=numf,ncases=train_size,batch_size=100000)
    pickle.dump( (ww0,bb0,ww1,bb1,ww2,bb2), open(wfile, "wb" ))

# ww2,bb2=ut.init_weight(hidden1,hidden2,'sigmoid')
ww3=rng.uniform(-0.05,0.05,hidden2)
ww3=numpy.zeros(hidden2)
#
bb3=0.


arr=[]
arr.append(x_dim)
arr.append(hidden1)
arr.append(hidden2)


ww3=numpy.reshape(ww3,hidden2)
bb3=float(bb3)


# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1", borrow=True)
w2 = theano.shared(ww2, name="w2", borrow=True)
w3 = theano.shared(ww3, name="w3", borrow=True)
b1 = theano.shared(bb1, name="b1", borrow=True)
b2 = theano.shared(bb2, name="b2", borrow=True)
b3 = theano.shared(bb3 , name="b3", borrow=True)


# Construct Theano expression graph
z1=T.dot(x, w1) + b1
if acti_type=='sigmoid':
    h1 = 1 / (1 + T.exp(-z1))              # hidden layer 1
elif acti_type=='linear':
    h1 = z1
elif acti_type=='tanh':
    h1=T.tanh(z1)
r1=srng.binomial(size=(1,hidden1),n=1,p=dropout)
d1=h1*r1[0]

z2=T.dot(h1, w2) + b2
if acti_type=='sigmoid':
    h2 = 1 / (1 + T.exp(-z2))              # hidden layer 2
elif acti_type=='linear':
    h2 = z2
elif acti_type=='tanh':
    h2=T.tanh(z2)
    
d2=T.tanh(T.dot(d1, w2) + b2)
r2=srng.binomial(size=(1,hidden2),n=1,p=dropout)
d2=d2*r2[0]

p_drop=1 / (1 + T.exp(-T.dot(d2, w3) - b3))
p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_drop) - (1-y) * T.log(1-p_drop)             # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w1 ** 2).sum() +
       (w2 ** 2).sum() + (w3 ** 2).sum() +
       (b1 ** 2).sum() + (b2 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1,w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
    auc,rmse=auc_rmse(file)
    log_p( msg + '\t' + str(auc) + '\t' + str(rmse))  

def auc_rmse(file,err_batch=100000):
    yp = []
    fi = open(file, 'r')
    flag_start=0
    flag=False
    xarray = []
    yarray=[]
    start_t = time.clock()
    while True:
        line=fi.readline()
        if len(line.strip()) == 0:
            flag=True
        else:
            flag_start+=1
            if flag==False:
                x_dense=numpy.zeros(ww0.shape[1])
                s = line.strip().replace(':', ' ').split(' ')
                for f in range(1, len(s), 2):
                    if int(s[f+1])==1:
                        x_dense += ww0[int(s[f])]
                x_dense+=bb0
                x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
                xarray.append((x_dense.tolist()))
                yarray.append(int(s[0]))
        if ((flag_start==err_batch) or (flag==True)):
#             print 'one epoch',time.clock()-start_t
            pred=predict(xarray)
            for p in pred:
                yp.append(p)
            flag_start=0
            xarray=[]
        if flag==True:
            break
    fi.close()
    auc = roc_auc_score(yarray, yp)
    rmse = math.sqrt(mean_squared_error(yarray, yp))
    return auc,rmse
#get error via batch
def whole_auc_rmse(file):
    yp = []
    fi = open(file, 'r')
    xarray = []
    yarray=[]
    index=0
    while True:
        line=fi.readline()
        if len(line.strip()) == 0:
            break
        else:
            index+=1
            x_dense=numpy.zeros(ww0.shape[1])
            s = line.strip().replace(':', ' ').split(' ')
            for f in range(1, len(s), 2):
                if int(s[f+1])==1:
                    x_dense += ww0[int(s[f])]
            x_dense+=bb0
            x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
            xarray.append((x_dense.tolist()))
            yarray.append(int(s[0]))
    yp=predict(xarray)
    tempx=xarray
    flag_start=0
    xarray=[]
    fi.close()
    try:
        auc = roc_auc_score(yarray, yp)
    except:
        print "error"
    rmse = math.sqrt(mean_squared_error(yarray, yp))
    return auc,rmse
def get_fi_h1_y(file,index,size):
    farray=[]
    xarray = []
    yarray=[]
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        if line.strip() != '':
            x_dense=numpy.zeros(ww0.shape[1])
            s = line.strip().replace(':', ' ').split(' ')
#             print 's:',s
            fi=[]
            for f in range(1, len(s), 2):
                if int(s[f+1])==1:
                    fi.append(int(s[f]))
#                     print 'int(s[f])',int(s[f])
                    x_dense += ww0[int(s[f])]
            x_dense+=bb0
#             print 'ww0',ww0
#             print 'x_dense',x_dense
            x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
            farray.append(fi)
            xarray.append(x_dense)
            yarray.append(int(s[0]))
    farray = numpy.array(farray, dtype = numpy.int32)
    xarray = numpy.array(xarray, dtype = theano.config.floatX)
    yarray = numpy.array(yarray, dtype = numpy.int32)
    return farray,xarray,yarray

# print_err(test_file,'InitTestErr:')


# Train
def mytrain():
    global bb0
    global ww1
    print "Training model:"
    min_err = 0
    min_err_epoch = 0
    times_reduce = 0
    for i in range(epoch):
        start_time = time.time()
        index = 1
        for j in range(n_batch):
            fi,x,y = get_fi_h1_y(train_file,index,batch_size)
            index += batch_size
            gx,ww1,ww2, ww3,bb1,bb2,bb3 = train(x,y)
            b_size = len(fi)
        
            for t in range(b_size):
                ft = fi[t]
                gxt = gx[t]
                xt=x[t]
                bb0=bb0-lr*gxt*xt*(1-xt)
                for feat in ft:
                    ww0[feat]=ww0[feat]-lr *gxt*xt*(1-xt)
                

        train_time = time.time() - start_time
        mins = int(train_time / 60)
        secs = int(train_time % 60)
        print 'training: ' + str(mins) + 'm ' + str(secs) + 's'

        start_time = time.time()
        print_err(train_file,'\t\tTraining Err: \t' + str(i))# train error
        train_time = time.time() - start_time
        mins = int(train_time / 60)
        secs = int(train_time % 60)
        print 'training error: ' + str(mins) + 'm ' + str(secs) + 's'

        start_time = time.time()
        auc, rmse = auc_rmse(test_file)
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        log_p( 'Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse))
        print 'test error: ' + str(mins) + 'm ' + str(secs) + 's'

        #stop training when no improvement for a while 
        if auc>min_err:
            min_err=auc
            min_err_epoch=i
            if times_reduce<3:
                times_reduce+=1
        else:
            times_reduce-=1
        if times_reduce<-2:
            break
    log_p( 'Minimal test error is '+ str( min_err)+' , at EPOCH ' + str(min_err_epoch))

mytrain()

        
