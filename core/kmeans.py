#coding=utf-8
'''
Created on 2014年6月18日
K-means 算法，将数据聚类到K个中心点
@author: sjm
'''
import numpy as np
import random

def dist(x,y):
    '''
    计算两个数据间的距离，使用马氏距离
    '''
    return np.sqrt(np.sum(x-y)**2)
def distMat(X,Y):
    '''
    计算两个矩阵间的距里，即矩阵里的每一个数据与另一个矩阵中每一个数据的距离
    '''
    mat=[map(lambda y:dist(x,y),Y) for x in X]
def sum_dist(data,label,center):
    s=0
    for i in range(data.shape[0]):
        s+=dist(data[i],center[label[i]])
    return s
def kmeans(data,cluster,threshold=1.0e-19,maxIter=100):
    m=len(data)
    labels=np.zeros(m)
    center=np.array(random.sample(data,cluster))
    s=sum_dist(data,labels,center)
    n=0
    while 1:
        n=n+1
        tmp_mat=distMat(data,center)
        labels=tmp_mat.argmin(axis=1)
        for i in xrange(cluster):
            idx=(labels==i).nozero()
            center[i]=np.mean(data[idx[0]],axis=1)
            d_i=data[idx[0]]
            d_i=d_i[0]
        s1=sum_dist(data,labels,center)
        if s-s1<threshold:
            break;
        s=s1
        if n>maxIter:
            break;
    return center