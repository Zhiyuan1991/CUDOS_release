# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import lib.dist as dist
import torch
from torch.autograd import Variable
from utils import *
import imageio


class VAE(object):
  """ SOM Mixture Sparse Variational Auto Encoder. """
  def __init__(self,
               flags=None,
               image_size=64,
               imgs_chn=3,
               som_sigma=4,
               name=''):
    # general
    self.flags=flags
    self.name=name
    self.imgs_chn=imgs_chn
    self.image_size=image_size
    self.activation=tf.nn.elu

    # vae
    self.z_dim=flags.z_dim
    self.gamma = flags.gamma #beta
    self.learning_rate = flags.learning_rate
    
    # som 
    self.som_dim=flags.som_dim
    self.SOM_learning_rate = flags.SOM_learning_rate
    self.som_sigma=som_sigma
    self.som_model_spike=self.flags.som_model_spike
    
    # sparse coding
    self.sparse_c_start=5
    self.nu=10

    # continual learning
    self.maximum_envs=1+4
    self.cur_env=0
    if not flags.known_task_boundary:
      self.already_used_env=tf.compat.v1.get_variable("already_used_env_tf", [3],initializer=tf.compat.v1.constant_initializer(0))

    #others
    if self.flags.sparse_coding_MIG:
      self.q_dist=dist.slab_and_spike()
    else:
      self.q_dist = dist.Normal()
    self.x_dist = dist.Bernoulli()
    self.prior_dist = dist.Normal()
    self.prior_params = torch.zeros(self.z_dim, 2)
    self.task_flag = tf.compat.v1.placeholder(tf.float32, shape=[None,1],name="PH_task_flag")
    self.env_onehot=tf.one_hot(tf.cast(self.task_flag,tf.int32),depth=3)
    self.env_onehot=self.env_onehot[:,0,:]
    global_log_alpha1=tf.compat.v1.get_variable("global_logalpha1", [1]+[self.z_dim],
                          initializer=tf.compat.v1.constant_initializer(np.log(self.flags.comp_alpha_init)))
    global_log_alpha2=tf.compat.v1.get_variable("global_logalpha2", [1]+[self.z_dim],
                          initializer=tf.compat.v1.constant_initializer(np.log(self.flags.comp_alpha_init)))
    global_log_alpha3=tf.compat.v1.get_variable("global_logalpha3", [1]+[self.z_dim],
                        initializer=tf.compat.v1.constant_initializer(np.log(self.flags.comp_alpha_init)))
    self.global_log_alpha1=-tf.nn.relu(-global_log_alpha1)
    self.global_log_alpha2=-tf.nn.relu(-global_log_alpha2)
    self.global_log_alpha3=-tf.nn.relu(-global_log_alpha3)
      
    self._create_network()
    self._create_loss_optimizer()

  def _get_prior_params(self, batch_size=1):
    expanded_size = (batch_size,) + self.prior_params.size()
    prior_params = Variable(self.prior_params.expand(expanded_size))
    return prior_params
  
  def _calc_anneal_som_sigma(self, step):
    sigma0=5
    sigmaInf=0.05
    t0=750
    tinf=2000
    if step < t0:
      s = sigma0
    elif step > tinf:
      s = sigmaInf
    else:
      s = sigma0-(sigma0-sigmaInf) * ((step-t0) / (tinf-t0))
    return s

  def _sample_z(self, z_mean, z_logvar):
    eps_shape = tf.shape(input=z_mean)
    eps = tf.random.normal(eps_shape, 0, 1, dtype=tf.float32)
    z = tf.add(z_mean,
              tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps))
    return z
  
  def _sample_z_sparse(self, mu, logvar, logspike, return_selection=False,simple_sigmoid=True):
    weight=self.increasing_weight
    std = tf.exp(0.5*logvar)
    eps = tf.random.normal(tf.shape(input=mu), 0, 1, dtype=tf.float32)
    gaussian = tf.add(mu,tf.multiply(std, eps))
    if simple_sigmoid:
      sparse_c=self.nu
    else:
      sparse_c=self.sparse_c_start*weight
    eta = tf.random.uniform(tf.shape(input=mu),dtype=tf.float32)
    selection = tf.nn.sigmoid(sparse_c*(eta + tf.exp(logspike) - 1))
    if return_selection:
      return selection*gaussian,selection
    else:
      return selection*gaussian

  def _create_recognition_network(self, x, reuse=False, name=""):
    with tf.compat.v1.variable_scope(name+"rec", reuse=reuse) as scope:
      conv1 = tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d(x,32,4,strides=(2,2),padding='same',activation=self.activation))#32
      conv2 = tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d(conv1,64,4,strides=(2,2),padding='same',activation=self.activation))#16
      conv3 = tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d(conv2,128,4,strides=(2,2),padding='same',activation=self.activation))#8
      conv4 = tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d(conv3,256,4,strides=(2,2),padding='same',activation=self.activation))#4
      fc1=tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.dense(tf.compat.v1.layers.flatten(conv4),1024,activation=self.activation))
      fc2=tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.dense(fc1,1024,activation=self.activation))
      z_mean=tf.compat.v1.layers.dense(fc2,self.z_dim)
      z_logvar=tf.compat.v1.layers.dense(fc2,self.z_dim)
      if self.flags.sparse_coding:
        z_log_spike=tf.math.log(tf.expand_dims(self.env_onehot[:,0],1)*tf.exp(self.global_log_alpha1)+
                              tf.expand_dims(self.env_onehot[:,1],1)*tf.exp(self.global_log_alpha2)+
                              tf.expand_dims(self.env_onehot[:,2],1)*tf.exp(self.global_log_alpha3))
        return (z_mean, z_logvar, z_log_spike)
      else:
        return (z_mean, z_logvar)

  def _create_generator_network(self, z, reuse=False,env_z=None,name=""):
    with tf.compat.v1.variable_scope(name+"gen", reuse=reuse) as scope:
      fc1=tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.dense(z,1024,activation=self.activation))
      fc2=tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.dense(fc1,4*4*256,activation=self.activation))
      fc2_reshaped = tf.reshape(fc2, [-1, 4, 4, 256])
      deconv1=tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d_transpose(fc2_reshaped,128,4,strides=(2,2),padding='same',activation=self.activation))#8
      deconv2=tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d_transpose(deconv1,64,4,strides=(2,2),padding='same',activation=self.activation))#16
      deconv3=tf.compat.v1.layers.batch_normalization(
        tf.compat.v1.layers.conv2d_transpose(deconv2,32,4,strides=(2,2),padding='same',activation=self.activation))#32
      deconv4=tf.compat.v1.layers.conv2d_transpose(deconv3,self.imgs_chn,4,strides=(2,2),padding='same',activation=None)#64
      x_out_logit=deconv4
      return x_out_logit

  def _get_som_embeddings(self,reuse=False,name=""):
    with tf.compat.v1.variable_scope(name+"som", reuse=reuse) as scope:
      som_env=tf.compat.v1.get_variable("som_env",self.som_dim,
                          initializer=tf.compat.v1.constant_initializer(0),dtype=tf.int32)
      env_spike=tf.compat.v1.get_variable("som_env_spike",[self.maximum_envs,self.z_dim],
                         initializer=tf.compat.v1.constant_initializer(0.5),dtype=tf.float32)
      component_mean=tf.compat.v1.get_variable("som_comp_mean", self.som_dim+[self.z_dim]+[1],
                            initializer=tf.compat.v1.initializers.random_uniform(-0.05,0.05))
      component_logvar=tf.compat.v1.get_variable("som_comp_logvar", self.som_dim+[self.z_dim]+[1],
                            initializer=tf.compat.v1.truncated_normal_initializer(mean=-1,stddev=0.05))
      component_logalpha=tf.compat.v1.get_variable("som_comp_logalpha", self.som_dim+[self.z_dim]+[1],
                            initializer=tf.compat.v1.constant_initializer(np.log(self.flags.comp_alpha_init)))
      component_logalpha=-tf.nn.relu(-component_logalpha)
      component_logalpha=tf.math.log(tf.nn.sigmoid(self.nu*(tf.exp(component_logalpha)-0.5)))
      embeddings=tf.concat([component_mean,component_logvar,component_logalpha],axis=-1)
      mixture_prior=tf.compat.v1.get_variable("som_mixture_prior", self.som_dim,
                            initializer=tf.compat.v1.constant_initializer(1./(self.som_dim[0]*self.som_dim[1])),)
      return embeddings, mixture_prior, som_env, env_spike

  def get_normalize_prior(self,mixture_prior):
    mixture_prior=tf.reshape(mixture_prior,[-1])
    real_mixture_prior=tf.nn.softmax(mixture_prior)
    real_mixture_prior=tf.reshape(real_mixture_prior,[self.som_dim[0],self.som_dim[1]])
    return real_mixture_prior
  
  def log_gaussian_pdf(self, x, mean, logvar, alpha, som_model_spike):
    eps=1e-6
    stddev2 = tf.exp(logvar)
    arg = tf.pow(x-mean, 2)
    arg2 = arg/(stddev2+eps)
    arg3_log_gaussian = -0.5*(arg2+logvar+tf.math.log(2*np.pi))
    if som_model_spike:
      arg3_ss = alpha*tf.exp(arg3_log_gaussian)+\
          (1-alpha)*self.flags.delta_value*tf.stop_gradient(tf.cast(tf.abs(x-mean)<self.flags.delta_threshold,dtype=tf.float32))
      arg4 = tf.reduce_sum(tf.math.log(arg3_ss+eps),3)
    else:
      arg4 = tf.reduce_sum(arg3_log_gaussian,3)
    return arg4
  
  def pz_likelihood_anneal(self, x, mean, logvar, alpha, mixture_prior, neighbor_mask):      
    arg_gaussian_log=self.log_gaussian_pdf(x, mean, logvar, alpha, som_model_spike=self.som_model_spike)      
    arg_log=tf.math.log(tf.expand_dims(self.get_normalize_prior(mixture_prior),0))+arg_gaussian_log
    arg_log_flatten=tf.reshape(arg_log,[-1,self.som_dim[0]*self.som_dim[1]])
    arg_log_flatten2=tf.expand_dims(arg_log_flatten,1)
    arg_mul=arg_log_flatten2 * neighbor_mask
    convLogProbs=tf.reduce_sum(arg_mul, axis=-1)
    singleConvLogSampleSums=tf.reduce_max(convLogProbs, axis=1)
    loglikelihood=tf.reduce_mean(singleConvLogSampleSums)
    return loglikelihood

  def w_posterior_flat(self,input_z,embeddings,mixture_prior):
    mixture_prior_nor=self.get_normalize_prior(mixture_prior)
    """Computes the posterior of each mixture component."""
    z_dist=self.log_gaussian_pdf(
                tf.expand_dims(tf.expand_dims(input_z,1),1),
                tf.expand_dims(tf.stop_gradient(embeddings[:,:,:,0]),0),
                tf.expand_dims(tf.stop_gradient(embeddings[:,:,:,1]),0),
                tf.expand_dims(tf.stop_gradient(tf.exp(embeddings[:,:,:,2])),0),self.som_model_spike
                )
    p_z_w=tf.exp(z_dist+tf.expand_dims(tf.stop_gradient(tf.math.log(mixture_prior_nor)),0))+1e-20
    z_posterior = tf.divide(p_z_w,
                            tf.expand_dims(tf.expand_dims(tf.stop_gradient(tf.reduce_sum(p_z_w, axis=[1,2])),1),1))
    z_dist_flat = tf.reshape(z_posterior, [-1, self.som_dim[0]*self.som_dim[1]])
    return z_dist_flat,z_posterior

  def bmu(self,input_z,embeddings,mixture_prior):
    z_dist_flat,z_posterior=self.w_posterior_flat(input_z,embeddings,mixture_prior)
    bmu_ind = tf.stop_gradient(tf.argmax(z_dist_flat, axis=-1))
    w_1 = bmu_ind // self.som_dim[1]
    w_2 = bmu_ind % self.som_dim[1]
    return w_1,w_2,z_dist_flat,z_posterior

  def som_inference(self,input_z,embeddings,mixture_prior,som_env):
    """ Compute BMU """
    w_1,w_2,_,posterior=self.bmu(input_z,embeddings,mixture_prior)
    w_index=[w_1,w_2]

    """ set up inference """
    w_stacked = tf.stack(w_index, axis=1)
    winner_params = tf.gather_nd(embeddings, w_stacked)
    w_envs = tf.gather_nd(som_env, w_stacked)
    w_samples=self._sample_z_sparse(tf.stop_gradient(winner_params[:,:,0]),
                                    tf.stop_gradient(winner_params[:,:,1]),
                                    tf.stop_gradient(winner_params[:,:,2]))
    return w_index,winner_params,w_samples,w_envs,posterior
  
  def neighborhood_mask_gaussian(self, sigma):
    shift                    = 0
    n=self.som_dim[0]
    oneRow                   = np.roll(np.arange(-n // 2 + shift, n // 2 + shift, dtype=np.float32), n // 2 + shift).reshape(n)
    npxGrid                  = np.stack(n * [oneRow], axis=0)
    npyGrid                  = np.stack(n * [oneRow], axis=1)
    npGrid                   = np.array([ np.roll(npxGrid, x_roll, axis=1) ** 2 + np.roll(npyGrid, y_roll, axis=0) ** 2 for y_roll in range(n) for x_roll in range(n) ])
    xyGrid                   = tf.constant(npGrid)
    cm                       = tf.reshape(tf.exp(-xyGrid / (2. * sigma ** 2.)), (self.som_dim[0]*self.som_dim[1], -1)) #notes: self.somSigma will change during "train_step"
    convMasks                = cm / tf.reduce_sum(cm, axis=1, keepdims=True)
    return convMasks

  def som_sampling(self,embeddings):
    """ set up sampling """
    self.component_flatten_ind=tf.compat.v1.placeholder(tf.int32, shape=[1],name="PH_mixture_sampling_ind")
    sc_ind1=self.component_flatten_ind //self.som_dim[1]
    sc_ind2=self.component_flatten_ind % self.som_dim[1]
    ind_stacked = tf.stack([sc_ind1,sc_ind2], axis=-1)
    self.z_mixture_sample_params=tf.gather_nd(embeddings, ind_stacked)
    self.z_mixture_sample_params_alpha=tf.exp(self.z_mixture_sample_params[:,:,2])
    z_mixture_sample=self._sample_z_sparse(self.z_mixture_sample_params[:,:,0],
                                        self.z_mixture_sample_params[:,:,1],
                                        self.z_mixture_sample_params[:,:,2])
    return z_mixture_sample
  
  def vae_sampling(self):
    """ set up sampling """
    z_sample=self._sample_z(tf.zeros([1,self.z_dim]),
                            tf.zeros([1,self.z_dim]))
    return z_sample

  def setup_updating_som(self): #for current som only
    """ get neighbor area """
    self.convMasks=self.neighborhood_mask_gaussian(self.som_sigma_anneal)
    
    self.pz_likelihood=self.pz_likelihood_anneal(
                    tf.expand_dims(tf.expand_dims(tf.stop_gradient(self.input_z),1),1),
                    tf.expand_dims(self.embeddings[:,:,:,0],0),
                    tf.expand_dims(self.embeddings[:,:,:,1],0),
                    tf.expand_dims(tf.exp(self.embeddings[:,:,:,2]),0),
                    self.mixture_prior,
                    self.convMasks,
                    )

    """ total loss """
    self.mixture_loss=-self.pz_likelihood #maximize likelihood

  def _create_network(self):
    self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.image_size,self.image_size,self.imgs_chn],name="PH_input_x")
    self.z_consist = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z_dim],name="PH_z_consist")
    self.increasing_weight = tf.compat.v1.placeholder(tf.float32, shape=[],name="PH_increasing_weight")
    self.this_env = tf.compat.v1.placeholder(tf.float32, shape=[None,1],name="PH_this_env")
    self.lam = tf.compat.v1.placeholder(tf.float32, shape=[],name="PH_lamda")
    self.som_sigma_anneal = tf.compat.v1.placeholder(tf.float32, shape=[],name="PH_som_anneal_sigma")

    with tf.compat.v1.variable_scope("vae"):
      if self.flags.sparse_coding:
        self.z_mean,self.z_logvar,self.z_log_spike = self._create_recognition_network(self.x)
        self.z_logvar=tf.clip_by_value(self.z_logvar,-1e6, 2)
        self.z,self.selection = self._sample_z_sparse(self.z_mean, self.z_logvar,self.z_log_spike,True)
      else:
        self.z_mean,self.z_logvar = self._create_recognition_network(self.x)
        self.z_logvar=tf.clip_by_value(self.z_logvar,-1e6, 2)
        self.z_log_spike=0*tf.stop_gradient(self.z_mean)
        self.z = self._sample_z(self.z_mean, self.z_logvar)
      self.x_out_logit = self._create_generator_network(self.z)
      self.x_out = tf.nn.sigmoid(self.x_out_logit)
        
    with tf.compat.v1.variable_scope("som"):
      self.input_z=self.z
      self.embeddings, self.mixture_prior, self.som_env, self.env_spike=self._get_som_embeddings()
      self.w_index,self.winner_params,self.w_samples,_,self.z_posterior=self.som_inference(self.input_z,self.embeddings,self.mixture_prior,self.som_env)
      self.setup_updating_som()
      
    with tf.compat.v1.variable_scope("vae"):
        #current data's protopyte
        self.x_prototype_logit_current=self._create_generator_network(self.w_samples,reuse=True)

        #recon replay data
        self.x_out_old_z_logit=self._create_generator_network(self.z_consist,reuse=True)
        
        #infer z from old encoder
        if self.flags.sparse_coding:
          self.z_mean_old,self.z_logvar_old,self.z_log_spike_old = self._create_recognition_network(self.x,name="oldcopy")
          self.qz_x_sample_old = self._sample_z_sparse(self.z_mean_old, self.z_logvar_old,self.z_log_spike_old)
        else:
          self.z_mean_old,self.z_logvar_old = self._create_recognition_network(self.x,name="oldcopy")
          self.z_log_spike_old=0*tf.stop_gradient(self.z_mean_old)
          self.qz_x_sample_old = self._sample_z(self.z_mean_old, self.z_logvar_old)

        #get generative replay data
        if self.flags.Bayesian_SOM:
          self.z_mixture_sample=self.som_sampling(self.embeddings)
        else:
          self.z_mixture_sample=self.vae_sampling()
          self.z_mixture_sample_params_alpha=tf.constant(0.)
        self.x_sample=tf.nn.sigmoid(self._create_generator_network(self.z_mixture_sample,name="oldcopy"))
    
    with tf.compat.v1.variable_scope("som"):
        self.embeddings_old, self.mixture_prior_old, self.som_env_old, self.env_spike_old=self._get_som_embeddings(name="oldcopy")
    
    with tf.compat.v1.variable_scope("vae"):
      #find closest prototype and make the z_combined
      self.p_index,self.p_params,self.p_samples,self.p_envs,self.p_w_posterior=self.som_inference(self.input_z,self.embeddings_old,self.mixture_prior_old,self.som_env_old)
      self.x_prototype_logit=self._create_generator_network(self.p_samples,reuse=True,name="oldcopy")
      self.z_p_spike=tf.stop_gradient(tf.exp(self.p_params[:,:,2]))
      self.shared_dims_mask_prod=self.z_p_spike*tf.exp(self.z_log_spike)
      self.shared_dims_mask = tf.nn.sigmoid(self.nu*(self.shared_dims_mask_prod - 0.5))
      new_z=self._sample_z_sparse(self.z_mean, self.z_logvar,self.z_log_spike,simple_sigmoid=True)
      self.combined_z=new_z*self.shared_dims_mask+tf.stop_gradient(self.p_params[:,:,0])*(1-self.shared_dims_mask)
      self.x_out_zcomb_logit = self._create_generator_network(self.combined_z,reuse=True)
      self.x_out_oldd_zcomb_logit = self._create_generator_network(self.combined_z,reuse=True,name="oldcopy")

  def _create_loss_optimizer(self):
    # -----------------------------------Reconstruction loss---------------------------------------------
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                            logits=self.x_out_logit)
    reconstr_loss = tf.reduce_sum(input_tensor=reconstr_loss, axis=[1,2,3])
    self.reconstr_loss = tf.reduce_mean(input_tensor=reconstr_loss)

    # ----------------------------------------KL loss------------------------------------------------    
    #self.z_som_params=tf.expand_dims(self.winner_params,1) #just compare with the winner node
    self.z_som_params=tf.expand_dims(tf.reshape(self.embeddings,[-1,self.z_dim,3]),0)
    self.spike=tf.exp(self.z_log_spike)
    if self.flags.sparse_coding and self.flags.Bayesian_SOM:
      self.spike=tf.expand_dims(self.spike,1)
      self.w_alpha=tf.exp(self.z_som_params[:,:,:,2])
      prior1=-0.5*tf.reduce_sum((self.spike)* 
        (1 + tf.expand_dims(self.z_logvar,1) - self.z_som_params[:,:,:,1]
        - (tf.square(tf.expand_dims(self.z_mean,1)-self.z_som_params[:,:,:,0])
        + tf.expand_dims(tf.exp(self.z_logvar),1))/(tf.exp(self.z_som_params[:,:,:,1])+1e-6)), axis=-1)
      pos_weight=tf.reshape(tf.stop_gradient(self.z_posterior),[-1,self.som_dim[0]*self.som_dim[1]])
      prior1=tf.reduce_sum(prior1*pos_weight,axis=1)
      prior21 = (1 - self.spike)*tf.math.log((1-self.spike+1e-6)/(1-self.w_alpha+1e-6))
      prior22 = (self.spike)*tf.math.log((self.spike+1e-6)/(self.w_alpha+1e-6))
      prior2 = tf.reduce_sum(input_tensor=prior21 + prior22,axis=-1)
      prior2=tf.reduce_sum(prior2*pos_weight,axis=1)
      self.prior21=prior21
      self.prior22=prior22          

      self.prior1=tf.reduce_mean(input_tensor=prior1)
      self.prior2=tf.reduce_mean(input_tensor=prior2)
      self.latent_loss = self.gamma*self.prior1 + self.prior2

      self.alpha_prior=self.flags.comp_alpha_init
      prior31 = (1 - self.w_alpha+1e-6)*tf.math.log((1-self.w_alpha+1e-6)/(1-self.alpha_prior+1e-6))
      prior32 = (self.w_alpha+1e-6)*tf.math.log((self.w_alpha+1e-6)/(self.alpha_prior+1e-6))
      prior3 = tf.reduce_sum(input_tensor=prior31 + prior32,axis=-1)
      self.prior3=tf.reduce_mean(input_tensor=prior3)
      self.mixture_loss+=self.prior3

    elif self.flags.sparse_coding:
      prior1 = -0.5 * tf.reduce_sum(self.spike*(
                1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar)),1)
      self.w_alpha=self.flags.comp_alpha_init
      prior21 = (1 - self.spike+1e-6)*tf.math.log((1-self.spike+1e-6)/(1-self.w_alpha+1e-6))
      prior22 = (self.spike+1e-6)*tf.math.log((self.spike+1e-6)/(self.w_alpha+1e-6))
      prior2 = tf.reduce_sum(input_tensor=prior21 + prior22,axis=[1])
      self.prior1=tf.reduce_mean(input_tensor=prior1)
      self.prior2=tf.reduce_mean(input_tensor=prior2)
      self.latent_loss = self.gamma*self.prior1 + self.prior2
    elif self.flags.Bayesian_SOM:
      prior1=-0.5*tf.reduce_sum(
          (1 + tf.expand_dims(self.z_logvar,1) - self.z_som_params[:,:,:,1]
          - (tf.square(tf.expand_dims(self.z_mean,1)-self.z_som_params[:,:,:,0])
          + tf.expand_dims(tf.exp(self.z_logvar),1))/tf.exp(self.z_som_params[:,:,:,1])), axis=-1)
      pos_weight=tf.reshape(tf.stop_gradient(self.z_posterior),[-1,self.som_dim[0]*self.som_dim[1]])
      prior1=tf.reduce_sum(prior1*pos_weight,axis=1)
      self.prior1=tf.reduce_mean(input_tensor=prior1)
      self.prior2=tf.constant(0.)
      self.latent_loss = self.gamma*self.prior1 + self.prior2
    else:
      latent_loss = -0.5 * tf.reduce_sum(1 + self.z_logvar
                                        - tf.square(self.z_mean)
                                        - tf.exp(self.z_logvar), axis=1)

      self.prior1 = tf.reduce_mean(input_tensor=latent_loss)
      self.prior2=tf.constant(0.)
      self.latent_loss = self.gamma*self.prior1 + self.prior2

    # ---------------------------------------- posterior loss ----------------------------------------
    if self.flags.Bayesian_SOM:
      self.uniform_prior=tf.constant(1./(self.som_dim[0]*self.som_dim[1]),
                            shape=[self.flags.batch_size]+self.som_dim)
      self.posterior_loss=tf.reduce_mean(input_tensor=tf.reduce_sum(
              tf.square(tf.subtract(self.uniform_prior,self.z_posterior)),axis=[1,2]))
    else:
      self.posterior_loss=tf.constant(0.)

    # ----------------------------------------old z and old z loss--------------------------------------------
    # for old data. z_old loss
    latent_loss_consist=tf.reduce_sum(
        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z,labels=tf.nn.sigmoid(self.z_consist)),axis=1)
    self.old_z_loss = tf.reduce_mean(input_tensor=latent_loss_consist)
    
    # Mask where the prototypes have data on it. Can make the training more stable if the new task is too different
    use_old_som_mask=True
    if use_old_som_mask:
      self.old_som_mask=tf.cast(tf.stop_gradient(self.p_envs)>0,tf.float32)
    else:
      self.old_som_mask=tf.ones_like(tf.stop_gradient(self.p_envs))

    # for new data, z_new loss
    z_mean_old=self.p_params[:,:,0]
    z_logvar_old=self.p_params[:,:,1]
    prior1=-0.5*tf.reduce_sum(
        (1 + self.z_logvar - z_logvar_old
        - (tf.square(self.z_mean-z_mean_old)
        + tf.exp(self.z_logvar))/tf.exp(z_logvar_old))*self.shared_dims_mask,
          axis=-1)
    z_new_loss=prior1*self.old_som_mask
    self.z_new_loss = tf.reduce_mean(z_new_loss)
    
    # ---------------------------------------old x and new x loss----------------------------------------------------------
    #for replay data
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                            logits=self.x_out_old_z_logit)
    reconstr_loss = tf.reduce_sum(reconstr_loss, axis=[1,2,3])
    self.x_de_loss_old = tf.reduce_mean(reconstr_loss)

    #for new data
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.stop_gradient(tf.nn.sigmoid(self.x_out_oldd_zcomb_logit)),
                                                          logits=self.x_out_zcomb_logit)
    reconstr_loss = tf.reduce_sum(reconstr_loss, axis=[1,2,3])*self.old_som_mask
    self.x_de_loss_new = tf.reduce_mean(reconstr_loss)
    
    self.latent_loss_edited = self.latent_loss
                                  
    self.common_loss=self.reconstr_loss+self.latent_loss_edited+1*self.posterior_loss
    
    self.old_data_loss=1.*self.old_z_loss+0.25*self.x_de_loss_old
    self.new_data_loss=1.*self.z_new_loss+0.35*self.x_de_loss_new

    reconstr_loss_summary_op = tf.compat.v1.summary.scalar('reconstr_loss', self.reconstr_loss)
    latent_loss_summary_op   = tf.compat.v1.summary.scalar('latent_loss',   self.latent_loss)
    self.summary_op = tf.compat.v1.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op])

    # list up variables
    all_vars=tf.compat.v1.trainable_variables()
    self.vae_params=[var for var in all_vars if 'som' not in var.name and 'oldcopy' not in var.name]

    # optimizer for SOM, variance has smaller learning rate
    var_list_not_var=[var for var in all_vars if 'som' in var.name and "som_comp_logvar" not in var.name and "oldcopy" not in var.name]
    var_list_var=[var for var in all_vars if "som_comp_logvar" in var.name and "oldcopy" not in var.name]
    opt_notvar=tf.compat.v1.train.AdamOptimizer(learning_rate=self.SOM_learning_rate)
    opt_var=tf.compat.v1.train.AdamOptimizer(learning_rate=self.SOM_learning_rate*0.01)
    grads_som=tf.gradients(self.mixture_loss,var_list_not_var+var_list_var)
    train_op1=opt_notvar.apply_gradients(zip(grads_som[:len(var_list_not_var)],var_list_not_var))
    train_op2=opt_var.apply_gradients(zip(grads_som[len(var_list_not_var):],var_list_var))
    self.optimizer_som_alt=tf.group(train_op1,train_op2)
    
    # optimizers for VAE
    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.common_loss,
            var_list=[var for var in all_vars if 'som' not in var.name and 'oldcopy' not in var.name])
    self.optimizer_old = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
          self.old_data_loss,
          var_list=[var for var in all_vars if 'som' not in var.name and 'oldcopy' not in var.name])
    self.optimizer_new = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
          self.common_loss+self.new_data_loss,
          var_list=[var for var in all_vars if 'som' not in var.name and 'oldcopy' not in var.name])
    

  def partial_fit(self, sess, xs, inc_c, step,global_step,epoch,batch_class=None,summary_writer=None):
    feed_dict={self.x : xs,
               self.increasing_weight:inc_c}
    feed_dict[self.lam]=calc_increasing_capacity(0,100,0,1)
    feed_dict[self.som_sigma_anneal]=self._calc_anneal_som_sigma(global_step)*1.0
    feed_dict[self.task_flag]=batch_class
    for i in range(2): # VAE is updated more frequently than SOM 
      sess.run(self.optimizer,feed_dict=feed_dict)
    if self.flags.update_som:
      for j in range(1):
        for i in range(len(xs)):
          feed_dict[self.x]=[xs[i]]
          sess.run(self.optimizer_som_alt,feed_dict=feed_dict)
          feed_dict[self.x]=xs
        
    if global_step%500==0 or step==0:
      if self.flags.Bayesian_SOM:
        save_folder=os.path.join(self.flags.checkpoint_dir,"SOM_debug")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        x_prototype=sess.run(tf.nn.sigmoid(self.x_prototype_logit_current),feed_dict=feed_dict)
        imageio.imwrite(save_folder+"/data_vs_proto_e{}_step{}.png".format(epoch,step), 
                          np.hstack((np.vstack((xss for xss in xs)),np.vstack((xss for xss in x_prototype)))))
        
    res = sess.run((self.env_spike,
                    self.env_spike,
                    self.reconstr_loss,
                    self.latent_loss,
                    self.prior1,
                    self.prior2,
                    self.mixture_loss,
                    self.posterior_loss,
                    self.som_sigma_anneal,
                    self.summary_op),
                    feed_dict=feed_dict)
    if np.isnan(res[2]) or np.isnan(res[3]):
      print("nan found")
      quit()
    if res[3]>1e5:
      print("Huge KL, KL is larger than 1e5")
    if summary_writer is not None:
      summary_writer.add_summary(res[-1], global_step)
    return ["re:",str(res[2]),"KL:",str(res[3]),
            "p1:",str(res[4]),"p2:",str(res[5]),
            "mix:",str(res[6]),
            "pos:",str(res[7]),"ssigma:",str(res[8])
            ]

  def partial_fit_CL_old(self,sess,xs,old_zs,old_env,step,glob_step):
    feed_dict={self.x : xs,
                self.z_consist:old_zs,
                self.increasing_weight:2}
    feed_dict[self.som_sigma_anneal]=self._calc_anneal_som_sigma(glob_step)
    feed_dict[self.task_flag]=old_env
    if glob_step%500==0 or step==0:
      save_folder=os.path.join(self.flags.checkpoint_dir,"CL_debug")
      if not os.path.exists(save_folder):
        os.mkdir(save_folder)
      x_out_old_z=sess.run(tf.nn.sigmoid(self.x_out_old_z_logit),feed_dict=feed_dict)
      imageio.imwrite(save_folder+"/generate_de_replay{}.png".format(step), 
                        np.hstack((np.vstack((xss for xss in xs)),np.vstack((xss for xss in x_out_old_z)))))
    for i in range(2):
      res = sess.run(self.optimizer_old,feed_dict=feed_dict)
    if self.flags.update_som:
      for j in range(1):
        for i in range(len(xs)):
          feed_dict[self.x]=[xs[i]]
          sess.run(self.optimizer_som_alt,feed_dict=feed_dict)
          feed_dict[self.x]=xs
          
    res = sess.run((self.env_spike,
                    self.reconstr_loss,
                    self.latent_loss,
                    self.prior1,
                    self.prior2,
                    self.old_z_loss,
                    self.x_de_loss_old,
                    self.summary_op,),
                    feed_dict=feed_dict
                  )
    if np.isnan(res[2]) or np.isnan(res[3]):
      print("old nan found")
      quit()
    return ["re_old:",str(res[1]),"kl:",str(res[2]),
            "p1:",str(res[3]),"p2:",str(res[4]),
            "z_old:",str(res[5]),"de_old:",str(res[6]),
            ]
  
  def partial_fit_CL_new(self,sess,xs,inc_c,step,epoch,glob_step,batch_class=None,summary_writer=None):
    feed_dict={self.x : xs,
               self.increasing_weight:inc_c,
               }
    feed_dict[self.som_sigma_anneal]=self._calc_anneal_som_sigma(glob_step)
    feed_dict[self.task_flag]=batch_class
    if glob_step%500==0 or step==0:
      save_folder=os.path.join(self.flags.checkpoint_dir,"CL_debug")
      if not os.path.exists(save_folder):
        os.mkdir(save_folder)
      debug_sd=sess.run(self.shared_dims_mask,feed_dict=feed_dict)
      x_out_zcomb_logit=sess.run(tf.nn.sigmoid(self.x_out_zcomb_logit),feed_dict=feed_dict)
      x_out_oldd_zcomb_logit=sess.run(tf.nn.sigmoid(self.x_out_oldd_zcomb_logit),feed_dict=feed_dict)
      x_prototype=sess.run(tf.nn.sigmoid(self.x_prototype_logit),feed_dict=feed_dict)
      old_som_mask=sess.run(self.old_som_mask,feed_dict=feed_dict)
      imageio.imwrite(save_folder+"/newdata_vs_proto_e{}_step{}.png".format(epoch,step), 
                        np.hstack((np.vstack((xss for xss in xs)),np.vstack((xss for xss in x_prototype)))))
      # blank image mean that new data is mapped to a node that has no data on it before
      imageio.imwrite(save_folder+"/generate_de_e{}_comb{}.png".format(epoch,step),
                          np.hstack((np.vstack((xss for xss in x_out_oldd_zcomb_logit)),
                          np.vstack((xss for xss in x_out_zcomb_logit*np.reshape(old_som_mask,[-1,1,1,1]))))))
      print("step:",step,"shared_dims:",debug_sd[0])
      print("alpha range",stat(sess.run(tf.exp(self.embeddings[:,:,:,2]))))

    for i in range(2):
      res = sess.run(self.optimizer_new,feed_dict=feed_dict)
    if self.flags.update_som:
      for j in range(1):
        for i in range(len(xs)):
          feed_dict[self.x]=[xs[i]]
          sess.run(self.optimizer_som_alt,feed_dict=feed_dict)
          feed_dict[self.x]=xs

    res = sess.run((self.env_spike,
                    self.reconstr_loss,
                    self.latent_loss,
                    self.prior1,
                    self.prior2,
                    self.z_new_loss,
                    self.x_de_loss_new,
                    self.increasing_weight,
                    self.posterior_loss,
                    self.summary_op,),
                    feed_dict=feed_dict
                  )
    if np.isnan(res[2]) or np.isnan(res[3]):
      print("new nan found")
      quit()
    if summary_writer is not None:
      summary_writer.add_summary(res[-1], glob_step)
    return ["re_new:",str(res[1]),"kl:",str(res[2]),
            "p1:",str(res[3]),"p2:",str(res[4]),"z_new:",str(res[5]),"de_new:",str(res[6]),
            "w:",str(res[7]),"pos:",str(res[8])
            ]

  def reconstruct(self, sess, xs):
    """ Reconstruct given data. """
    z_mean,z_logvar=self.transform(sess,xs)
    return sess.run(self.x_out, 
                    feed_dict={self.z: z_mean, self.lam:1,
                               self.this_env:np.ones([len(xs),1])*self.flags.task})

  def transform(self, sess, xs):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_logvar],
                     feed_dict={self.x: xs} )
  
  def transform_old(self, sess, xs):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean_old, self.z_logvar_old],
                     feed_dict={self.x: xs,
                     #self.increasing_weight:2
                    }) 

  def inference_z(self, sess, xs, increasing_weight=2, batch_class=None):
    if batch_class is not None:
      return sess.run(self.z,feed_dict={self.x: xs,self.task_flag:batch_class,self.increasing_weight:increasing_weight,self.lam:1})
    else:
      return sess.run(self.z,feed_dict=
                  {self.x: xs,self.increasing_weight:increasing_weight,self.lam:1,
                  self.task_flag:np.ones([len(xs),1])*(self.flags.task-1)})

  
  def inference_spike(self,sess,xs,increasing_weight=2,batch_class=None):
    if batch_class is not None:
      spike=np.exp(sess.run(self.z_log_spike,feed_dict={self.x: xs,self.task_flag:batch_class,self.increasing_weight:increasing_weight}))
    else:
      spike=np.exp(sess.run(self.z_log_spike,
                  feed_dict={self.x: xs,self.increasing_weight:increasing_weight,
                  self.task_flag:np.ones([len(xs),1])*(self.flags.task-1)}))
    return spike

  def inference_spike_old(self,sess,xs,increasing_weight=2):
    spike=np.exp(sess.run(self.z_log_spike_old,feed_dict={self.x: xs,self.increasing_weight:increasing_weight}))
    return spike

  def som_weights(self,sess):
    return sess.run(self.embeddings,feed_dict={})
  
  def winner_from_x(self,sess,xs,batch_class=None):
    if batch_class is not None:
      w_index=sess.run(self.w_index,feed_dict={self.x: xs,self.task_flag:batch_class,self.increasing_weight:2,self.lam:1})
    else:
      w_index=sess.run(self.w_index,feed_dict={self.x: xs,self.increasing_weight:2})
    return w_index
  
  def get_som_prior(self,sess):
    mixture_prior_nor=sess.run(self.get_normalize_prior(self.mixture_prior))
    self.flatten_prior=tf.reshape(mixture_prior_nor,[-1])
    return sess.run([self.mixture_prior,self.flatten_prior])

  def sampling_from_mixture(self,sess,number_of_samples):
    flat_prior=sess.run(self.flatten_prior,feed_dict={})
    all_choice=np.arange(self.flags.som_dim[0]*self.flags.som_dim[1])
    for i in range(number_of_samples):
      this_ind=np.random.choice(all_choice, p=flat_prior)
    return

  def generate(self, sess, zs, env_z=None,env=None):
    """ Generate data by sampling from latent space. """
    if env is None:
      env=np.ones([1,1])
    return sess.run(self.x_out,
                    feed_dict={self.z: zs,self.lam:1,
                               self.this_env:env})
  
  def get_recons_loss(self, sess, xs, env_z=None,segs=None,batch_class=None):
        if segs is None:
            feed_dict={self.x: xs,
                        self.increasing_weight:2,
                        self.this_env:np.ones([len(xs),1])*self.flags.task}
            if batch_class is not None:
              feed_dict[self.task_flag]=batch_class
            reconstr_loss, latent_loss = sess.run((self.reconstr_loss, self.latent_loss,),feed_dict=feed_dict)
        else:
          feed_dict={self.x: xs,
                    self.seg:segs,
                    self.fadein: 1.,
                    self.env_z:env_z}
          reconstr_loss, latent_loss = sess.run((self.reconstr_loss, self.latent_loss,),feed_dict=feed_dict)
        return reconstr_loss, latent_loss

  def copy_model_weights(self,sess):
    allvars = tf.compat.v1.trainable_variables()
    c_vars=[x for x in allvars if 
            'oldcopy' not in x.name and ('vae' in x.name or 'som' in x.name)]
    p_vars=[x for x in allvars if 
            'oldcopy' in x.name and ('vae' in x.name or 'som' in x.name)]
    copy_ops = [p_vars[ix].assign(var.value()) for ix, var in enumerate(c_vars)]
    sess.run(copy_ops)

  def able_to_generate(self,sess):
    som_env=sess.run(self.som_env,feed_dict={})
    if np.sum(som_env)==0:
      return False
    else:
      return True

  def get_most_close_one_hot_global_alpha(self,alpha,alpha1,alpha2,alpha3):
    dis=[0,0,0]
    dis[0]=np.sum(np.abs(alpha-alpha1))
    dis[1]=np.sum(np.abs(alpha-alpha2))
    dis[2]=np.sum(np.abs(alpha-alpha3))
    return np.argmin(dis)