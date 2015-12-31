import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#from logistic_sgd import load_data
#from utils import tile_raster_images
import dl_utils as ut
import data_fm as fm
rng = numpy.random
rng.seed(1234)
 
class dA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        is_sparse=0,
        sparse_len=0,
        sparse_W=None,
        sparse_b=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if not W:
            self.initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=self.initial_W, name='W', borrow=True)
        if ((is_sparse==1) and (not sparse_W)):
            self.init_sparse_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (sparse_len + n_visible)),
                    high=4 * numpy.sqrt(6. / (sparse_len + n_visible)),
                    size=(sparse_len, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            sparse_W = theano.shared(value=self.init_sparse_W, name='sparse_W', borrow=True)
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.sparse_W=sparse_W
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        
            
                      
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        
            
        return (cost, updates,z,self.W,self.b)


def da(row,col,file,results,learning_rate=0.1, training_epochs=15,
            batch_size=20,corruption_level=0,is_sparse=0,sparse_len=0,k=1):
    x = T.matrix('x') 
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=row,
        n_hidden=col
    )
    cost, updates,z,w,b = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )
    train_da = theano.function(
        [x],
        [cost,z,w,b],
        updates=updates
    )
    start_time = timeit.default_timer()
    ############
    # TRAINING #
    ############
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        batch=[]
        indexes=[]
        with open(file, "r") as ins:
            array = []
            size=0
            for line in ins:
                flag=False
                if line.strip()!="":
                    size+=1
                    s = line.strip().replace(':', ' ').split(' ')
                    x = []
                    index=[]
                    for f in range(1, len(s), 2):
                        x.append(int(s[f+1]))
                        index.append(int(s[f]))
                    batch.append(x)
                    indexes.append(index)
                    if size==batch_size: 
                        batcharr=numpy.array(batch, dtype=numpy.float32)
                        i=0
                        for r in results:
                            i+=1
                            if i%2==1:
                                weights=r
                            else:
                                if i==2:
                                    newbatch=numpy.zeros((len(batcharr),weights.shape[1]))
                                    bias=r
                                    hnum = weights.shape[1]
                                    case_num = len(batcharr)
                                    j = 0
                                    for row in indexes:
                                        sum=0
                                        for k in range(hnum):
                                            for r in row:
                                                sum += weights[r][k]
                                            newbatch[j][k] = sum
                                        j += 1
                                    batcharr=newbatch+bias
    #                                 batch=np.dot(batch,weights)+bias
                                else:
                                    bias=r
                                    batcharr=numpy.dot(batcharr,weights)+bias
                                batcharr = 1.0 / (1.0 + numpy.exp(-batcharr))
                        
                        [cost,z,w,b]=train_da(batcharr)
                        batch=[]
                        indexes=[]
                        size=0
                        c.append(cost)
                        
            if len(batch)>0:
                batcharr=numpy.array(batch, dtype=numpy.float32) 
                i=0
                for r in results:
                    i+=1
                    if i%2==1:
                        weights=r
                    else:
                        if i==2:
                            newbatch=numpy.zeros((len(batcharr),weights.shape[1]))
                            bias=r
                            hnum = weights.shape[1]
                            case_num = len(batcharr)
                            j = 0
                            for row in indexes:
                                sum=0
                                for k in range(hnum):
                                    for r in row:
                                        sum += weights[r][k]
                                    newbatch[j][k] = sum
                                j += 1
                            batcharr=newbatch+bias
#                                 batch=np.dot(batch,weights)+bias
                        else:
                            bias=r
                            batcharr=np.dot(batcharr,weights)+bias
                        batcharr = 1.0 / (1.0 + numpy.exp(-batcharr))
                
                [cost,z,w,b]=train_da(batcharr)
                batch=[]
                indexes=[]
                size=0
                c.append(cost)
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    return w,b

def sparse_da(row,col,file,learning_rate=0.1, training_epochs=15,
            batch_size=20,corruption_level=0,is_sparse=0,sparse_len=0,k=1):
    x = T.matrix('x') 
    ww = T.matrix('ww')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=row,#o_fm.xdim,
        n_hidden=col,
        is_sparse=is_sparse,
        sparse_len=sparse_len
    )
    cost, updates,z,w,b = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )
    train_da = theano.function(
        [x,ww],
        [cost,z,w,b],
        updates=updates,
        givens=[(da.W,ww)]
    )
    start_time = timeit.default_timer()
    ############
    # TRAINING #
    ############
    # go through training epochs
    #sampled weight
    init_sparse_W = numpy.asarray(
                rng.uniform(
                    low=-4 * numpy.sqrt(6. / (sparse_len + col)),
                    high=4 * numpy.sqrt(6. / (sparse_len + col)),
                    size=(sparse_len, col)
                ),
                dtype=theano.config.floatX
            )
    # whole weight
    initial_W = numpy.asarray(
                rng.uniform(
                    low=-4 * numpy.sqrt(6. / (row + col)),
                    high=4 * numpy.sqrt(6. / (row + col)),
                    size=(row, col)
                ),
                dtype=theano.config.floatX
            )
    init_sparse_b=numpy.zeros(
                    sparse_len,
                    dtype=theano.config.floatX
                )
    init_b=numpy.zeros(
                    row,
                    dtype=theano.config.floatX
                )
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        batch=[]
        batch_index=[]
        
        with open(file, "r") as ins:
            array = []
            size=0
            for line in ins:
                flag=False
                if line.strip()!="":
                    size+=1
                    s = line.strip().replace(':', ' ').split(' ')
                    x = []
                    indexes=[]
                    a=0
                    for f in range(1, len(s), 2):
                        for kk in range(k-1):
                            new_sample=int(rng.uniform(a,int(s[f])))
                            if (not (new_sample in indexes)):
                                x.append(0)
                                indexes.append(new_sample)
                        x.append(int(s[f+1]))
                        a=int(s[f])+1
                        indexes.append(int(s[f]))
                    batch.append(x)
                    if size==batch_size: 
                        batcharr=numpy.array(batch, dtype=numpy.float32)
                        i=0
#                         print 'indexes.size',len(indexes)
                        for f in indexes:
#                             print 'f',f
                            initial_W[i]=init_sparse_W[f]
                            init_b[i]=init_sparse_b[f]
                            i+=1
                            
                        [cost,z,w,b]=train_da(batcharr,initial_W)
                        
                        i=0
                        for f in indexes:
                            init_sparse_W[f]=w[i]
                            i+=1
                        batch=[]
                        batch_index=[]
                        size=0
                        c.append(cost)
            if len(batch)>0:
#                 batch_indexes=numpy.array(batch_index, dtype=numpy.float32)
                batch=numpy.array(batch, dtype=numpy.float32)
                [cost,z,w,b]=train_da(batch)
                c.append(cost)
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    return init_sparse_W,b
    
def get_da_weights(file, arr, ncases,num_feats=16, batch_size=100000):
    epochs=3
    row=0
    col=0
    results=[]
    weights=[]
    bias=[]
    index=0
    k=2                                    # k is the number of sampling
    for line in arr:
        index+=1
        if index==1:
            col=int(line)
        else:
            row=col
            col=int(line)
        if index==2:
            w,b=sparse_da(num_feats*k,col,file,training_epochs=epochs,sparse_len=row,is_sparse=1,batch_size=1,k=k)
            results.append(w)
            results.append(b)
        elif index>2:
            w,b=da(row,col,file,results,training_epochs=epochs,batch_size=1)
            results.append(w)
            results.append(b)
    return results
