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
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rng = numpy.random
rng.seed(1234)
batch_size=100                                                          #batch size
lr=0.002                                                                #learning rate
lambda1=0.1 # .01                                                        #regularisation rate
hidden1 = 300 															#hidden layer 1
hidden2 = 100 															#hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 100                                                               #epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../data/train.fm.txt'             #training file
test_file='../data/test.fm.txt'                   #test file
fm_model_file='../data/fm.model.txt'                   #fm model file
#feats = ut.feats_len(train_file)                                           #feature size
if len(sys.argv) > 2 and advertiser=='all':
    train_file=train_file+'.5.txt'
elif len(sys.argv) > 2:
    train_file=train_file+'.10.txt'
print train_file

train_size=ut.file_len(train_file)                    #training size
test_size=ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches
x_drop=1

if advertiser=='2997':#
    lr=0.001
    x_drop=dropout=0.5
    hidden1=300
    hidden2=100
    lambda1=0.0
    lambda_fm=0.1

    
    

name_field = {'weekday':0, 'hour':1, 'useragent':2, 'IP':3, 'region':4, 'city':5, 'adexchange':6, 'domain':7, 'slotid':8,
       'slotwidth':9, 'slotheight':10, 'slotvisibility':11, 'slotformat':12, 'creative':13, 'advertiser':14, 'slotprice':15}

def log_p(msg,m=""):
    ut.logfile(msg,"fm"+str(advertiser))


log_p('ad:'+str(advertiser))
log_p( 'batch_size:'+str(batch_size))
feat_field = {}
feat_weights = {}
w_0 = 0
feat_num = 0
k = 0
xdim = 0
fi = open(fm_model_file, 'r')
first = True
for line in fi:
    s = line.strip().split()
    if first:
        first = False
        w_0 = float(s[0])
        feat_num = int(s[1])
        k = int(s[2]) + 1 # w and v
        xdim = 1 + len(name_field) * k
    else:
        feat = int(s[0])
        weights = [float(s[1 + i]) for i in range(k)]
        feat_weights[feat] = weights
        name = s[1 + k][0:s[1 + k].index(':')]
        field = name_field[name]
        feat_field[feat] = field

def feat_layer_one_index(feat, l):
    return 1 + feat_field[feat] * k + l

def feats_to_layer_one_array(feats):
    x = numpy.zeros(xdim)
    x[0] = w_0
    for feat in feats:
        x[feat_layer_one_index(feat, 0):feat_layer_one_index(feat, k)] = feat_weights[feat]
    return x

log_p('drop_mlp3fm.py|ad:'+advertiser+'|drop:'+str(dropout)+'|b_size:'+str(batch_size)+' | X:'+str(xdim) + ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L_r:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        
# initialise parameters
w=rng.uniform(  low=-numpy.sqrt(6. / (xdim + hidden1)),
                high=numpy.sqrt(6. / (xdim + hidden1)),
                size=(xdim,hidden1))
if acti_type=='sigmoid':
    ww1=numpy.asarray((w))
elif acti_type=='tanh':
    ww1=numpy.asarray((w*4))
else:
    ww1=numpy.asarray(rng.uniform(-1,1,size=(xdim,hidden1)))

bb1=numpy.zeros(hidden1)


v=rng.uniform(  low=-numpy.sqrt(6. / (hidden1 + hidden2)),
                high=numpy.sqrt(6. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
if acti_type=='sigmoid':
    ww2=numpy.asarray((v))
elif acti_type=='tanh':
    ww2=numpy.asarray((v*4))
else:
    ww2=numpy.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))

bb2=numpy.zeros(hidden2)

ww3=numpy.zeros(hidden2)


# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(0. , name="b3")


# Construct Theano expression graph

r0=srng.binomial(size=(1,xdim),n=1,p=x_drop)
x=x*r0[0]

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


p_drop=(1 / (1 + T.exp(-T.dot(d2, w3) - b3)))
p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_drop) - (1-y) * T.log(1-p_drop)             # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w3 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1, w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
    auc,rmse=get_err_bat(file)
    log_p( msg + '\t' + str(auc) + '\t' + str(rmse))    


#get error via batch
def get_err_bat(file,err_batch=100000):
    y = []
    yp = []
    fi = open(file, 'r')
    flag_start=0
    xx_bat=[]
    flag=False
    while True:
        line=fi.readline()
        if len(line) == 0:
            flag=True
        flag_start+=1
        if flag==False:
            xx,yy = get_xy(line)
            xx_bat.append(numpy.asarray(xx))
        if ((flag_start==err_batch) or (flag==True)):
            pred=predict(xx_bat)
            for p in pred:
                yp.append(p)
            flag_start=0
            xx_bat=[]
        if flag==False:
            y.append(yy)
        if flag==True:
            break
    fi.close()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    return auc,rmse

def get_batch_data(file,index,size):#1,5->1,2,3,4,5
    xarray = []
    yarray = []
    farray = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        if line.strip() != '':
            f, x, y = get_fxy(line.strip())
            xarray.append(x)
            yarray.append(y)
            farray.append(f)
    xarray = numpy.array(xarray, dtype = theano.config.floatX)
    yarray = numpy.array(yarray, dtype = numpy.int32)
    return farray, xarray, yarray

def get_xy(line):
    s = line.replace(':', ' ').split()
    y = int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return x, y

def get_fxy(line):
    s = line.replace(':', ' ').split()
    y=int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return feats,x,y

#print_err(test_file,'InitTestErr:')
def get_pred(file,best_w1,best_w2,best_w3,best_b1,best_b2,best_b3):
	w1.set_value(best_w1)
	w2.set_value(best_w2)
	w3.set_value(best_w3)
	b1.set_value(best_b1)
	b2.set_value(best_b2)
	b3.set_value(best_b3)
	yp = []
	fi = open(file, 'r')
	xx_bat=[]
	while True:
		line=fi.readline()
		if len(line) != 0:
			xx,yy = get_xy(line)
			xx_bat.append(numpy.asarray(xx))
		else:
			break
	pred=predict(xx_bat)
	fi.close()
	return pred

# Train
print "Training model:"
best_w1=w1.get_value()
best_w2=w2.get_value()
best_w3=w1.get_value()
best_b1=b1.get_value()
best_b2=b2.get_value()
best_b3=b3.get_value()
min_err = 0
min_err_epoch = 0
times_reduce = 0
for i in range(epoch):
    start_time = time.time()
    index = 1
    for j in range(n_batch):
        if index>train_size:
            break
        f,x,y = get_batch_data(train_file,index,batch_size)
        index += batch_size
        gx, w1t, w2t, w3t,b1t,b2t,b3t = train(x,y)
        b_size = len(f)
        for t in range(b_size):
            ft = f[t]
            gxt = gx[t]
            for feat in ft:
                for l in range(k):
                    feat_weights[feat][l] = feat_weights[feat][l] * (1 - 2. * lambda_fm * lr / b_size) \
                                            - lr * gxt[feat_layer_one_index(feat, l)] * 1

    
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
    auc, rmse = get_err_bat(test_file)
    test_time = time.time() - start_time
    mins = int(test_time / 60)
    secs = int(test_time % 60)
    log_p( 'Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse))
    print 'test error: ' + str(mins) + 'm ' + str(secs) + 's'

    #stop training when no improvement for a while 
    if auc>min_err:
        best_w1=w1t
        best_w2=w2t
        best_w3=w3t
        best_b1=b1t
        best_b2=b2t
        best_b3=b3t
        min_err=auc
        min_err_epoch=i
        if times_reduce<3:
            times_reduce+=1
    else:
        times_reduce-=1
    if times_reduce<0:
        break
log_p( 'Minimal test error is '+ str( min_err)+' , at EPOCH ' + str(min_err_epoch))
ut.save_weights("mlp3fm_train_"+advertiser+".p",get_pred(train_file,best_w1,best_w2,best_w3,best_b1,best_b2,best_b3))
ut.save_weights("mlp3fm_test_"+advertiser+".p",get_pred(test_file,best_w1,best_w2,best_w3,best_b1,best_b2,best_b3))
