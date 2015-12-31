import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import linecache
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import numpy
import os
from time import gmtime, strftime



class DataFM(object):
    def __init__(self,fm_model_file):
        # advertiser = '2997'
        # fm_model_file='../../make-ipinyou-data/' + advertiser + '/fm.model.txt'
        self.name_field = {'weekday':0, 'hour':1, 'useragent':2, 'IP':3, 'region':4, 'city':5, 'adexchange':6, 'domain':7, 'slotid':8,
               'slotwidth':9, 'slotheight':10, 'slotvisibility':11, 'slotformat':12, 'creative':13, 'advertiser':14, 'slotprice':15}
        self.fm_model_file=fm_model_file
        self.feat_field = {}
        self.feat_weights = {}
        self.w_0 = 0
        feat_num = 0
        self.k = 0
        self.xdim = 0
        fi = open(fm_model_file, 'r')
        first = True

        for line in fi:
            s = line.strip().split()
            if first:
                first = False
                self.w_0 = float(s[0])
                feat_num = int(s[1])
                self.k = int(s[2]) + 1 # w and v
                self.xdim = 1 + len(self.name_field) * self.k
            else:
                feat = int(s[0])
                weights = [float(s[1 + i]) for i in range(self.k)]
                self.feat_weights[feat] = weights
                name = s[1 + self.k][0:s[1 + self.k].index(':')]
                field = self.name_field[name]
                self.feat_field[feat] = field

    def feat_layer_one_index(self,feat, l):
        return 1 + self.feat_field[feat] * self.k + l

    def feats_to_layer_one_array(self,feats):
        x = numpy.zeros(self.xdim)
        x[0] = self.w_0
        for feat in feats:
            x[self.feat_layer_one_index(feat, 0):self.feat_layer_one_index(feat, self.k)] = self.feat_weights[feat]
        return x


    def get_batch_data(self,file,index,size):#1,5->1,2,3,4,5
        xarray = []
        yarray = []
        farray = []
        for i in range(index, index + size):
            line = linecache.getline(file, i)
            if line.strip() != '':
                f, x, y = self.get_fxy_fm(line.strip())
                xarray.append(x)
                yarray.append(y)
                farray.append(f)
        xarray = numpy.array(xarray, dtype = theano.config.floatX)
        yarray = numpy.array(yarray, dtype = numpy.int32)
        return farray, xarray, yarray

    def get_xy_fm(self,line):
        s = line.replace(':', ' ').split()
        y = int(s[0])
        feats = [int(s[j]) for j in range(1, len(s), 2)]
        x = self.feats_to_layer_one_array(feats)
        return x, y

    def get_fxy_fm(self,line):
        s = line.replace(':', ' ').split()
        y=int(s[0])
        feats = [int(s[j]) for j in range(1, len(s), 2)]
        x = self.feats_to_layer_one_array(feats)
        return feats,x,y

