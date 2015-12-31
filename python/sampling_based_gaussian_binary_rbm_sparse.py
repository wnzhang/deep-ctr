import sys
import time
import numpy as np
import dl_utils as ut
import data_fm as fm
import linecache
rng = np.random
rng.seed(1234)

class RBM(object):
    def __init__(self, nvis, nhid, mfvis=True, mfhid=False, initvar=0.1):
        nweights = nvis * nhid
        vb_offset = nweights
        hb_offset = nweights + nvis
        
        # One parameter matrix, with views onto it specified below.
        self.params = np.empty((nweights + nvis + nhid))

        # Weights between the hiddens and visibles
        self.weights = self.params[:vb_offset].reshape(nvis, nhid)

        # Biases on the visible units
        self.visbias = self.params[vb_offset:hb_offset]

        # Biases on the hidden units
        self.hidbias = self.params[hb_offset:]

        # Attributes for scratch arrays used during sampling.
        self._hid_states = None
        self._vis_states = None

        # Instance-specific mean field settings.
        self._mfvis = mfvis
        self._mfhid = mfhid

    @property
    def numvis(self):
        return self.visbias.shape[0]

    @property
    def numhid(self):
        return self.hidbias.shape[0]

    def _prepare_buffer(self, ncases, kind):
        if kind not in ['hid', 'vis']:
            raise ValueError('kind argument must be hid or vis')
        name = '_%s_states' % kind
        num = getattr(self, 'num%s' % kind)
        buf = getattr(self, name)
        if buf is None or buf.shape[0] < ncases:
            if buf is not None:
                del buf
            buf = np.empty((ncases, num))
            setattr(self, name, buf)
        buf[...] = np.NaN
        return buf[:ncases]

    def hid_activate(self, input, mf=False):
        input = np.atleast_2d(input)
        ncases, ndim = input.shape
        hid = self._prepare_buffer(ncases, 'hid')
        self._update_hidden(input, hid, mf)
        return hid

    def _update_hidden(self, vis, hid, mf=False):
        hid[...] = np.dot(vis, self.weights)
        hid[...] += self.hidbias
        hid *= -1.
        np.exp(hid, hid)
        hid += 1.
        hid **= -1.
        if not mf:
            self.sample_hid(hid)
    
    def _update_visible(self, vis, hid, mf=False):
        # Implements 1/(1 + exp(-WX) with in-place operations
        vis[...] = np.dot(hid, self.weights.T)
        vis[...] += self.visbias
        vis *= -1.
        np.exp(vis, vis)
        vis += 1.
        vis **= -1.
        if not mf:
           self.sample_vis(vis)
    
    @classmethod
    def binary_threshold(cls, probs):
        samples = rng.uniform(size=probs.shape)
        
        probs[samples < probs] = 1.

        # Anything not set to 1 should be 0 once floored.
        np.floor(probs, probs)

    # Binary hidden units
    sample_hid = binary_threshold

    # Binary visible units
    sample_vis = binary_threshold

    def gibbs_walk(self, nsteps, hid):
        hid = np.atleast_2d(hid)
        ncases = hid.shape[0]

        # Allocate (or reuse) a buffer with which to store 
        # the states of the visible units
        vis = self._prepare_buffer(ncases, 'vis')

        for iter in xrange(nsteps):
            
            # Update the visible units conditioning on the hidden units.
            self._update_visible(vis, hid, self._mfvis)

            # Always do mean-field on the last hidden unit update to get a
            # less noisy estimate of the negative phase correlations.
            if iter < nsteps - 1:
                mfhid = self._mfhid
            else:
                mfhid = True
            
            # Update the hidden units conditioning on the visible units.
            self._update_hidden(vis, hid, mfhid)

        return self._vis_states[:ncases], self._hid_states[:ncases]

class GaussianBinaryRBM(RBM):
    def _update_visible(self, vis, hid, mf=False):
        vis[...] = np.dot(hid, self.weights.T)
        vis += self.visbias
        if not mf:
            self.sample_vis(vis)
    @classmethod
    def sample_vis(self, vis):
        vis += rng.normal(size=vis.shape)
def get_fake_line(line):
    newline=[]
    for l in line:
        a,b=l.split(":")
        newline.append(l)
        newline.append(str(int(a)-1)+":0")
    return newline
def get_batch_x(file,index,size):
    xarray = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        if line.strip() != '':
            s = line.strip().replace(':', ' ').split(' ')
            x = {}
            for f in range(1, len(s), 2):
                x[int(s[f])] = int(s[f+1])
                x[int(s[f])-1] = 0        #fake
            #line=line[line.index(' ')+1:]
            #x=get_fake_line(line.split())
            xarray.append(x)
#     print xarray
    return xarray
class CDTrainer(object):
    """An object that trains a model using vanilla contrastive divergence."""
    
    def __init__(self, model, weightcost=0.0002, rates=(1e-4, 1e-4, 1e-4),
                 cachebatchsums=True):
        self._model = model
        self._visbias_rate, self._hidbias_rate, self._weight_rate = rates
        self._weightcost = weightcost
        self._cachebatchsums = cachebatchsums
        self._weightstep = np.zeros(model.weights.shape)

    def train(self,file_path,epochs,ncases,fm_model_file,results=None,cdsteps=1, minibatch=100, momentum=0.9):
        """
        Train an RBM with contrastive divergence, using `nsteps`
        steps of alternating Gibbs sampling to draw the negative phase
        samples.
        """
        model = self._model
        if self._cachebatchsums:
            batchsums = {}

        for epoch in xrange(epochs):

            # An epoch is a single pass through the training data.
            
            epoch_start = time.clock()
           
            # Mean squared error isn't really the right thing to measure
            # for RBMs with binary visible units, but gives a good enough
            # indication of whether things are moving in the right way.

            mse = 0
            offset=0
            while True:
                batch=get_batch_x(file_path,(offset+1),minibatch)
                if results !=None:
                    i=0
                    for r in results:
                        i+=1
                        if i%2==1:
                            weights=r
                        else:
                            if i==2:
                                newbatch=np.zeros((len(batch),weights.shape[1]))
                                bias=r
                                hnum = weights.shape[1]
                                case_num = len(batch)
                                j = 0
                                for x in batch:
                                    sums = np.zeros(hnum)
                                    for f in x:
                                        if x[f] == 1:
                                            sums += weights[f]
                                    newbatch[j] = sums
                                    j += 1
                                batch=newbatch+bias
#                                 batch=np.dot(batch,weights)+bias
                            else:
                                bias=r
                                batch=np.dot(batch,weights)+bias
                batch = 1.0 / (1.0 + np.exp(-batch))
                batchsize = batch.shape[0]
                # Mean field pass on the hidden units f
                hid = model.hid_activate(batch, mf=True)
                
                # Correlations between the data and the hidden unit activations
                poscorr = np.dot(batch.T, hid)
                
                # Activities of the hidden units
                posact = hid.sum(axis=0)

                # Threshold the hidden units so that they can't convey 
                # more than 1 bit of information in the subsequent
                # sampling (assuming the hidden units are binary,
                # which they most often are).
                model.sample_hid(hid)

                # Simulate Gibbs sampling for a given number of steps.
                vis, hid = model.gibbs_walk(cdsteps, hid)

                # Update the weights with the difference in correlations
                # between the positive and negative phases.
                
                thisweightstep = poscorr
                thisweightstep -= np.dot(vis.T, hid)
                thisweightstep /= batchsize
                thisweightstep -= self._weightcost * model.weights
                thisweightstep *= self._weight_rate
               
                self._weightstep *= momentum
                self._weightstep += thisweightstep

                model.weights += self._weightstep
                
                # The gradient of the visible biases is the difference in
                # summed visible activities for the minibatch.
                if self._cachebatchsums:
                    if offset not in batchsums:
                        batchsum = batch.sum(axis=0)
                        batchsums[offset] = batchsum
                    else:
                        batchsum = batchsums[offset]
                else:
                    batchsum = batch.sum(axis=0)
                
                visbias_step = batchsum - vis.sum(axis=0)
                visbias_step *= self._visbias_rate / batchsize

                model.visbias += visbias_step

                # The gradient of the hidden biases is the difference in
                # summed hidden activities for the minibatch.

                hidbias_step = posact - hid.sum(axis=0)
                hidbias_step *= self._hidbias_rate / batchsize

                model.hidbias += hidbias_step
            #     print 'vis',vis
#                 print 'bat',batch
                # Compute the squared error in-place.
                vis -= batch
                vis **= 2.
                
                # Add to the total epoch estimate.
                mse += vis.sum() / ncases

                offset+=batch.shape[0]
            #     print 'minibatch',minibatch
#                 print 'batsize',batch.shape[0]
                if batch.shape[0]<minibatch:
                    break

            print "Done epoch %d: %f seconds, MSE=%f" % \
                    (epoch + 1, time.clock() - epoch_start, mse)
            sys.stdout.flush()


class sparse_RBM(object):
    def __init__(self, nvis, nhid, nsparsevis, mfvis=True, mfhid=False, initvar=0.1):
        nweights = nvis * nhid
        vb_offset = nweights
        hb_offset = nweights + nvis
        self.params = np.empty((nweights + nvis + nhid))
        self.weights = self.params[:vb_offset].reshape(nvis, nhid)
        self.visbias = self.params[vb_offset:hb_offset]
        self.hidbias = self.params[hb_offset:]
        self._hid_states = None
        self._vis_states = None
        self._sparsevis_states=None
        self._mfvis = mfvis
        self._mfhid = mfhid
        self.nsparsevis=nsparsevis
        self.line=None
        self.line_dic=None
    @property
    def numvis(self):
        return self.visbias.shape[0]
    @property
    def numhid(self):
        return self.hidbias.shape[0]
    @property
    def numsparsevis(self):
        return self.nsparsevis
    def _prepare_buffer(self, ncases, kind):
        if kind not in ['hid', 'vis','sparsevis']:
            raise ValueError('kind argument must be hid or vis')
        name = '_%s_states' % kind
        num = getattr(self, 'num%s' % kind)
        buf = getattr(self, name)
        if buf is None or buf.shape[0] < ncases:
            if buf is not None:
                del buf
            buf = np.empty((ncases, num))
            setattr(self, name, buf)
        buf[...] = np.NaN
        return buf[:ncases]

    def hid_activate(self, input, mf=False):
        hid = self._prepare_buffer(1, 'hid')
        hid[...]=0
        self._update_hidden(input, hid, mf)
        return hid

    def _update_hidden(self, vis, hid, mf=False):
        hid[...]=0
        for i in range(self.numhid):
            sum=0
            j=0
            for f in sorted(self.line_dic):
                sum+=(self.weights[f][i])*float(vis[0][j])
                j+=1
            hid[0][i]=sum    
        hid[...] += self.hidbias
        hid *= -1.
        np.exp(hid, hid)
        hid += 1.
        hid **= -1.
        if not mf:
            self.sample_hid(hid)
    def _update_visible(self, vis, hid, mf=False):
        vis[...]=0
        i=0
        for f in sorted(self.line_dic):
            vis[0][i]=np.dot(hid,self.weights[f,:].T)
            vis[0][i]+=self.visbias[f]
            i+=1
        vis *= -1.
        np.exp(vis, vis)
        vis += 1.
        vis **= -1.
        if not mf:
           self.sample_vis(vis)
    
    @classmethod
    def binary_threshold(cls, probs):
        samples = rng.uniform(size=probs.shape)
        probs[samples < probs] = 1.
        # Anything not set to 1 should be 0 once floored.
        np.floor(probs, probs)

    # Binary hidden units
    sample_hid = binary_threshold

    # Binary visible units
    sample_vis = binary_threshold

    def gibbs_walk(self, nsteps, hid):
        hid = np.atleast_2d(hid)
        ncases = hid.shape[0]
        # Allocate (or reuse) a buffer with which to store 
        # the states of the visible units
        vis = self._prepare_buffer(ncases, 'sparsevis')

        for iter in xrange(nsteps):
            
            # Update the visible units conditioning on the hidden units.
            self._update_visible(vis, hid, self._mfvis)
            if iter < nsteps - 1:
                mfhid = self._mfhid
            else:
                mfhid = True
            
            # Update the hidden units conditioning on the visible units.
            self._update_hidden(vis, hid, mfhid)
#             print 'gibbs_walk',self._hid_states[:ncases]
        return self._sparsevis_states[:ncases], self._hid_states[:ncases]

class sparse_CDTrainer(object):
    def __init__(self, model, weightcost=0.0002, rates=(1e-4, 1e-4, 1e-4),
                 cachebatchsums=True):
        self._model = model
        self._visbias_rate, self._hidbias_rate, self._weight_rate = rates
        self._weightcost = weightcost
        self._cachebatchsums = cachebatchsums
        self._weightstep = np.zeros((model.numsparsevis,model.numhid))

    def train(self,file_path,epochs,ncases,cdsteps=1, momentum=0.9, k=1):
        model = self._model
        if self._cachebatchsums:
            batchsums = {}
        for epoch in xrange(epochs):
            epoch_start = time.clock()
            mse = 0
            offset=0
            with open(file_path, "r") as ins:
                array = []
                for line in ins:
                    if line.strip()!="":
                        s = line.strip().replace(':', ' ').split(' ')
                        x = {}
                        for f in range(1, len(s), 2):
                            x[int(s[f])-1] = 0 
                            x[int(s[f])] =1# int(s[f+1]) 
                        model.line_dic=x
                        v=[]
                        i=0
                        for f in sorted(model.line_dic):
                            i+=1
                            v.append(int(model.line_dic[f]))
                        v=np.asarray(v).reshape(1,i)
                        batch=v
                        hid = model.hid_activate(batch, mf=True)
                        poscorr = np.dot(batch.T, hid)
                        posact = hid.sum(axis=0)
                        model.sample_hid(hid)
                        # Simulate Gibbs sampling for a given number of steps.
                        vis, hid = model.gibbs_walk(cdsteps, hid)

#                         # Update the weights with the difference in correlations
#                         # between the positive and negative phases.
                        thisweightstep = poscorr
                        thisweightstep -= np.dot(vis.T, hid)
                        i=-1
                        for f in sorted(model.line_dic):
                            i+=1
                            thisweightstep[i]-=self._weightcost*model.weights[int(f)]    
                        thisweightstep *= self._weight_rate
                        self._weightstep *= momentum
                        self._weightstep += thisweightstep
                        
                        i=-1

                        for f in sorted(model.line_dic):
                            i+=1
                            model.weights[int(f)]+=self._weightstep[i]
                            model.weights[int(f)] += self._weightstep[i]
                            
#                         # The gradient of the visible biases is the difference in
#                         # summed visible activities for the minibatch.
                        if self._cachebatchsums:
                            if offset not in batchsums:
                                batchsum = batch.sum(axis=0)
                                batchsums[offset] = batchsum
                            else:
                                batchsum = batchsums[offset]
                        else:
                            batchsum = batch.sum(axis=0)
                        visbias_step = batchsum - vis.sum(axis=0)
                        visbias_step *= self._visbias_rate
#                         model.visbias += visbias_step
                        i=-1
                        # for l in model.line:
#                             i+=1
#                             a,b=l.split(":")
#                             model.visbias[int(a)] += visbias_step[i]
                        for f in sorted(model.line_dic):
                            i+=1
                            model.visbias[int(f)] += visbias_step[i]    
                            
                            
                            
                            
                        # The gradient of the hidden biases is the difference in
                        # summed hidden activities for the minibatch.

                        hidbias_step = posact - hid.sum(axis=0)
                        hidbias_step *= self._hidbias_rate

                        model.hidbias += hidbias_step
                    #     print 'sparse:'
#                         print 'vis',vis
#                         print 'bat',batch
                        # Compute the squared error in-place.
                        vis -= batch
                        vis **= 2.
                
                        # Add to the total epoch estimate.
                        mse += vis.sum() / ncases
                        offset+=1
            print "Done epoch %d: %f seconds, MSE=%f" % \
                    (epoch + 1, time.clock() - epoch_start, mse)
            sys.stdout.flush()
            
def get_rbm_weights(file, arr, ncases, fm_model_file,batch_size=1):
    epochs=3
    row=0
    col=0
    results=[]
    weights=[]
    bias=[]
    index=0
    n_sparse_vis=32
    k=2
    print "training RBM"
    for line in arr:
        print "line:",line
        index+=1
        if index==1:
            col=int(line)
        else:
            row=col
            col=int(line)
        if index==2:
            rbm = sparse_RBM(row, col,n_sparse_vis)
            rbm.params[:] = rng.uniform(-1./10, 1./10, len(rbm.params))
            trainer = sparse_CDTrainer(rbm)
            trainer.train(file,epochs,ncases,cdsteps=1,k=k)
            results.append(rbm.weights)
            results.append(rbm.hidbias)
        elif index>2:
            rbm = RBM(row, col)
            rbm.params[:] = rng.uniform(-1./10, 1./10, len(rbm.params))
            trainer = CDTrainer(rbm)
            trainer.train(file, epochs,ncases,results=results,minibatch=batch_size,fm_model_file=fm_model_file)
            results.append(rbm.weights)
            results.append(rbm.hidbias)
    return results
