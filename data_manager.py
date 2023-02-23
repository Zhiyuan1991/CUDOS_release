# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

class DataManager_3dshapes(object):
  def load(self,set_condition=True,train_test_split_mode=0,seed=0,task=0):
    import h5py
    dataset = h5py.File('../../Data/3DShapes/3dshapes.h5', 'r')
    self.imgs=np.array(dataset['images'])
    self.labels=np.array(dataset['labels'])
    # self.imgs = np.load('../../Data/3dshape_images.npy')
    # self.labels=np.load('../../Data/3dshape_labels.npy')
    if set_condition:
      if task==1:
        ind=np.where((self.labels[:,0]==0)&(self.labels[:,1]==0))
        self.factor_labels=self.labels[ind]
        self.task_label=np.zeros([len(self.imgs),1])
      elif task==2:
        ind=np.where((self.labels[:,1]==0)&(self.labels[:,0]>0))
        self.factor_labels=self.labels[ind]
        self.task_label=np.ones([len(self.imgs),1])
      elif task==3:
        ind=np.where((self.labels[:,1]>0))
        self.factor_labels=self.labels[ind]
        self.task_label=np.ones([len(self.imgs),1])*2
      else:
        print('unknow task')
        quit()

      self.imgs=self.imgs[ind]
      self.task_label=self.task_label[ind]

    #select train or test data
    if train_test_split_mode:
      X_train, X_test, y_train, y_test, fl_train, fl_test = train_test_split(self.imgs, self.task_label,self.factor_labels, test_size=0.2, random_state=seed)
      if train_test_split_mode==1:
        self.imgs=X_train[:100000]
        self.task_label=y_train[:100000]
        self.factor_labels=fl_train[:100000]
      else:
        self.imgs=X_test
        self.task_label=y_test
        self.factor_labels=fl_test

    self.n_samples = self.imgs.shape[0]
    self.original_n_samples = self.imgs.shape[0]
    print("dataset size:",self.n_samples)

  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, index=0):
    return self.imgs[index]

  def get_images(self, indices,with_label=False):
    images = np.zeros([len(indices),64,64,3])
    labels = []
    count=0
    for index in indices:
      img = self.imgs[index]
      img = img/255.
      images[count]=img
      if with_label:
        labels.append(self.task_label[index])
      count+=1
    if with_label:
      return images,labels
    else:
      return images
  
  def get_regression_labels(self,indices,label_ind=5):
    labels = []
    for index in indices:
      labels.append(self.factor_labels[index][label_ind])
    return np.expand_dims(np.array(labels),1)

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)
  
  def add_temporal_samples(self,samples,keep_adding=False):
    if not keep_adding:
      self.imgs=self.imgs[:self.original_n_samples]
    if samples is not None:
      self.imgs=np.vstack((self.imgs,np.array(samples)))
      self.n_samples = self.imgs.shape[0]