#coding=utf-8
'''
Created on 2014年6月21日

@author: JianMing Song
'''
import numpy as np
def sigmrnd(x):
    return np.array(1/(1+np.exp(-x))>np.random.random_sample(x.shape),dtype=np.float) 

class rbm(object):
    def __init__(self,n_in,n_out,momentum=0,alpha=1,show=True):
        self.w=np.zeros((n_out,n_in))
        self.vw=np.zeros((n_out,n_in))
        self.b=np.zeros((n_in,))
        self.vb=np.zeros((n_in,))
        self.c=np.zeros((n_out,))
        self.vc=np.zeros((n_out,))
        self.momentum=momentum
        self.alpha=alpha
        self.show=show
        return None
    def train(self,data,epoch,batch_size,eta=0.1):
        num,d2=data.shape
        assert num%batch_size==0
        numbatches=num/batch_size
        for loop in xrange(0,epoch):
            kk=np.random.permutation(num)
            err=0
            for index in xrange(0,numbatches):
                batch=np.reshape(data[kk[index*batch_size:(index+1)*batch_size],:],[batch_size,d2])
                v1=batch
                h1=sigmrnd(np.tile(self.c.T,[batch_size,1])+np.dot(v1,self.w.T))
                v2=sigmrnd(np.tile(self.b.T,[batch_size,1])+np.dot(h1,self.w))
                h2=sigmrnd(np.tile(self.c.T,[batch_size,1])+np.dot(v2,self.w.T))
                
                c1=np.dot(h1.T , v1)
                c2=np.dot(h2.T,v2)
                
                self.vw=self.momentum*self.vw+self.alpha*(c1-c2)/batch_size
                self.vb=self.momentum*self.vb+self.alpha*np.sum((v1-v2).T,axis=1)/batch_size
                self.vc=self.momentum*self.vc+self.alpha*np.sum((h1-h2).T,axis=1)/batch_size
                
                self.w=self.w+self.vw
                self.b=self.b+self.vb
                self.c=self.c+self.vc
                
                err+=np.sum((v1-v2)**2)/batch_size
            if self.show:
                print('epoch %d/%d Average reconstruction error = %f'%(loop+1,epoch,err/numbatches))
if __name__=='__main__':
    from scipy.io import loadmat
    data=loadmat(r'E:\code\matlab\DeepLearnToolbox-master\data\mnist_uint8.mat')
    train_x=np.asarray(data['train_x'],np.float)/255.0
    nn=rbm(n_in=784,n_out=100)
    nn.train(train_x,batch_size=100,epoch=100)
