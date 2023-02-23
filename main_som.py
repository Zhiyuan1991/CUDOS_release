from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
import imageio
from shutil import copyfile
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from model_som import VAE
from metric_MIG import *
from elbo_decomposition import *
from data_manager import *
from utils import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
np.set_printoptions(precision=2)

tf.compat.v1.disable_v2_behavior()

#Training
tf.compat.v1.flags.DEFINE_integer("epoch_size",  100, "epoch size")
tf.compat.v1.flags.DEFINE_integer("epoch_size_t2", 15, "epoch size for task 2")
tf.compat.v1.flags.DEFINE_integer("epoch_size_t3", 10, "epoch size for task 3")
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.compat.v1.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.compat.v1.flags.DEFINE_float("SOM_learning_rate", 5e-4, "learning rate for SOM")
tf.compat.v1.flags.DEFINE_integer("seed", 0, "random seed")
tf.compat.v1.flags.DEFINE_integer("train_test_mode", 0, "0: all data; 1: train data; 2:test_data")
tf.compat.v1.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.compat.v1.flags.DEFINE_integer("min_step",0, "for step function")
tf.compat.v1.flags.DEFINE_integer("max_step",3000, "for step function")
tf.compat.v1.flags.DEFINE_integer("gpu_index",0, "which GPU card to use")
#Tasks
tf.compat.v1.flags.DEFINE_integer("start_task", 2, "")
tf.compat.v1.flags.DEFINE_integer("max_task", 2, "")
tf.compat.v1.flags.DEFINE_integer("replay_num",50000, "how many generative replay data to use")
tf.compat.v1.flags.DEFINE_boolean("known_task_boundary", True, "")
#model
tf.compat.v1.flags.DEFINE_boolean("update_som",   True, "")
tf.compat.v1.flags.DEFINE_boolean("som_model_spike", True, "")
tf.compat.v1.flags.DEFINE_boolean("sparse_coding",True, "")
tf.compat.v1.flags.DEFINE_boolean("Bayesian_SOM", True, "")
tf.compat.v1.flags.DEFINE_boolean("use_replay", True, "")

tf.compat.v1.flags.DEFINE_integer("z_dim", 15, "z_dim")
tf.compat.v1.flags.DEFINE_list("som_dim",[20,20],"som_dim")
#som tuning
tf.compat.v1.flags.DEFINE_float("gamma", 1., "gamma param for latent loss")
tf.compat.v1.flags.DEFINE_float("comp_alpha_init", .2, "")
tf.compat.v1.flags.DEFINE_float("delta_threshold", 0.1, "")
tf.compat.v1.flags.DEFINE_float("delta_value", 10., "")
#testing
tf.compat.v1.flags.DEFINE_integer("task", 2, "testing task")
tf.compat.v1.flags.DEFINE_boolean("sparse_coding_MIG", False, "")
tf.compat.v1.flags.DEFINE_boolean("use_statistics", False, "")
tf.compat.v1.flags.DEFINE_integer("test_img_ind", 277, "")
#mode
tf.compat.v1.flags.DEFINE_float("gpu_usage", 1., "TF GPU usage fraction") #change to 0.5 if using mode 6
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "checkpoints/3dshapes_bn_cudos_r17_copy", "checkpoint directory")
tf.compat.v1.flags.DEFINE_integer("mode", 1, "mode. 1: Train; 2: Check disentanglement and SOM generation; 3: Run metrics")

flags = tf.compat.v1.flags.FLAGS
epoch_list=[flags.epoch_size,flags.epoch_size_t2,flags.epoch_size_t3]

save_img_ind_flag=True
add_ind_number=True
gif_nums=7
maxi=2

reorder_dims=[]
assert reorder_dims==[] or len(reorder_dims)==flags.z_dim

z_dim=flags.z_dim

save_every_eporch=1

image_chn=3
image_size=64

def concept_shifter(manager,cur_task):
  manager.load(train_test_split_mode=flags.train_test_mode,seed=flags.seed,task=cur_task)

def get_batch_class(sess,xs,model,last_new_step,last_new_env,glob_step):
  recons_loss_all=[]
  already_used_env=sess.run(model.already_used_env)
  for env in range(3):
    recons_loss,_ = model.get_recons_loss(sess, xs, batch_class=np.ones([len(xs),1])*env)
    recons_loss_all.append(recons_loss)
  if glob_step-last_new_step<1000 and last_new_step>-1:
    return np.ones([len(xs),1])*last_new_env,last_new_env,last_new_step
  if (len(recons_loss_all)==0 or np.min(recons_loss_all)>4000) and (glob_step-last_new_step>1000 or last_new_step==-1):
    ind_most_close_zero=np.argmin(already_used_env)
    already_used_env[ind_most_close_zero]=1
    sess.run(model.already_used_env.assign(already_used_env))
    last_new_env=ind_most_close_zero
    return np.ones([len(xs),1])*last_new_env,last_new_env,glob_step
  else:
    return np.ones([len(xs),1])*np.argmin(recons_loss_all),last_new_env,last_new_step

def train_som_mixture_CL(sess,model,manager,saver):
  summary_writer = tf.compat.v1.summary.FileWriter(flags.checkpoint_dir, sess.graph)
  glob_step=0
  last_new_step=-1
  last_new_env=-1
  for cur_task in range(flags.start_task,flags.max_task+1):
    flags.task=cur_task
    model.copy_model_weights(sess)
    replay_flag=model.able_to_generate(sess)
    if cur_task==1:
      replay_flag=False
    print("**** replay flag:", replay_flag)

    # load new data
    concept_shifter(manager,cur_task)
    print("**** training data size: ",manager.n_samples)
    n_samples = manager.sample_size
    indices = list(range(n_samples))
    step = 0
    batch_size=flags.batch_size
    # save training data images
    imageio.imwrite(flags.checkpoint_dir+"/training_data_task{}.png".format(cur_task),
                np.vstack((xs for xs in manager.get_images(range(64)))))

    # generating replay data
    if replay_flag and flags.use_replay:
      replay_num=flags.replay_num
      replay_samples,replay_zs,replay_envs=sampling_from_som_mixture(sess,model,replay_num)
      imageio.imwrite(flags.checkpoint_dir+"/training_data_task_re.png",
                np.vstack((xs for xs in replay_samples[:64])))
      indices_re = list(range(len(replay_samples)))
      total_batch_re = len(indices_re)//batch_size
    else:
      replay_samples=None
      replay_envs=None
    
    this_epoch_range=epoch_list[cur_task-1]
    for epoch in range(this_epoch_range):
      random.shuffle(indices)
      if replay_samples is not None:
        random.shuffle(indices_re)
      total_batch = n_samples // batch_size

      for i in range(total_batch):
        inc_c=calc_increasing_capacity(step,flags.min_step,flags.max_step)
        batch_indices = indices[batch_size*i : batch_size*(i+1)]
        if len(batch_indices)<batch_size:
          continue
        #batch_xs = manager.get_images(batch_indices)
        batch_xs,batch_class = manager.get_images(batch_indices,with_label=True)

        if not flags.known_task_boundary:
          batch_class,last_new_env,last_new_step=get_batch_class(sess,batch_xs,model,last_new_step,last_new_env,glob_step)

        if replay_samples is None:
          model_output = model.partial_fit(sess, batch_xs, inc_c, step,glob_step,epoch,batch_class,summary_writer)
        else:
          i_re=i%total_batch_re
          batch_indices_re = indices_re[batch_size*i_re : batch_size*(i_re+1)]
          batch_replays=replay_samples[batch_indices_re]
          batch_old_zs=replay_zs[batch_indices_re]
          batch_old_envs=replay_envs[batch_indices_re]

          # learning new data
          model_output_new = model.partial_fit_CL_new(sess,batch_xs,inc_c,step,epoch,glob_step,batch_class,summary_writer)

          # learning replay data
          model_output_old = model.partial_fit_CL_old(sess,batch_replays,batch_old_zs,batch_old_envs,step,glob_step)

        step += 1
        glob_step +=1

        if glob_step%500==0 or step==0:
          disentangle_check_image_row(sess,model,manager,epoch=epoch,step=step,task=cur_task)        
        
        if i%100==0:
          if not replay_flag:
            if np.isnan(np.float(model_output[1])):
              print("nan found")
              quit()
            res_str=" ".join(model_output)
            print("e: "+str(epoch)+" "+res_str)
          else:
            if np.isnan(np.float(model_output_old[1])) or np.isnan(np.float(model_output_new[1])):
              print("nan found")
              quit()
            res_str=" ".join(model_output_old)
            print("e: "+str(epoch)+" "+res_str)
            res_str=" ".join(model_output_new)
            print("e: "+str(epoch)+" "+res_str)
      
      if (epoch)%2==0:
        disentangle_check_image_row(sess,model,manager,epoch=epoch,step=step,task=cur_task)
        aggregate_on_mixture_som(sess,model,manager,saver,glob_step,cur_task,replay_samples,replay_envs,save_flag=False,epoch=epoch)

      # Save checkpoint
      if (epoch+1)%save_every_eporch==0:
        saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = glob_step)
    
    aggregate_on_mixture_som(sess,model,manager,saver,glob_step,cur_task,replay_samples,replay_envs,save_flag=True)
    
    copy_checkpoints_and_som_to_new_folder(flags,cur_task)

def aggregate_on_mixture_som(sess,model,manager,saver,glob_step=None,cur_task=None,replay_samples=None,replay_envs=None,save_flag=False,epoch=None,draw_hist=True):
  if cur_task is not None:
    save_folder=os.path.join(flags.checkpoint_dir,str(cur_task),"vae_disentangle")
    if not os.path.exists(save_folder):
      os.mkdir(os.path.join(flags.checkpoint_dir,str(cur_task)))
      os.mkdir(save_folder)
  else:
    cur_task=2
  """ count all training data """
  som_dim=flags.som_dim
  n_samples = manager.sample_size
  indices = list(range(n_samples))
  if replay_samples is not None:
    n_samples_re=len(replay_samples)
    indices_re = list(range(n_samples_re))
  batch_size=flags.batch_size
  total_batch = n_samples // batch_size
  som_count=np.zeros(som_dim+[cur_task+1])
  spike_count=sess.run(model.env_spike,feed_dict={})
  inferred_zs_array=None
  inferred_spike_array=None
  for i in range(total_batch):
    batch_indices = indices[batch_size*i : batch_size*(i+1)]
    batch_xs,batch_class = manager.get_images(batch_indices,with_label=True)
    winners=np.swapaxes(np.array(model.winner_from_x(sess,batch_xs,batch_class=batch_class)),0,1)
    batch_spikes=model.inference_spike(sess,batch_xs,batch_class=batch_class)
    batch_inferred_zs,_=model.transform(sess,batch_xs)
    if len(batch_spikes)==1:
      batch_spikes=np.tile(batch_spikes,[len(batch_xs),1])
    if inferred_zs_array is None:
      inferred_zs_array=batch_inferred_zs
      inferred_spike_array=batch_spikes
    else:
      inferred_zs_array=np.vstack((inferred_zs_array,batch_inferred_zs))
      inferred_spike_array=np.vstack((inferred_spike_array,batch_spikes))
    for j in range(len(winners)):
      w=winners[j]
      som_count[w[0]][w[1]][cur_task]+=1
      spike_count[cur_task]+=batch_spikes[j]
  if replay_samples is not None:
    for i in range(n_samples_re//batch_size):
      batch_indices = indices_re[batch_size*i : batch_size*(i+1)]
      batch_xs = replay_samples[batch_indices]
      batch_env=replay_envs[batch_indices]
      batch_class=batch_env
      winners=np.swapaxes(np.array(model.winner_from_x(sess,batch_xs,batch_class)),0,1)
      for j in range(len(winners)):
        w=winners[j]
        som_count[w[0]][w[1]][batch_env[j]]+=1

  spike_count[cur_task]=spike_count[cur_task]/(total_batch*batch_size)
  print("spike count: ")
  for spike_count_i in spike_count[0:3]:
    print(spike_count_i)
  spike_count[1]=(spike_count[1]>0.8)*1.
  
  same_spike_count=0
  for i in range(len(inferred_spike_array)):
    this_spike=inferred_spike_array[i]
    this_spike=(this_spike>0.8)*1.
    same_spike_bool=(this_spike==spike_count[cur_task])
    if False not in same_spike_bool:
      same_spike_count+=1
  print("same as aggregate:", same_spike_count,'/', i+1)

  """ get the som_env for each node"""
  som_env=np.zeros(som_dim)
  for i in range(som_dim[0]):
    for j in range(som_dim[1]):
      som_env[i][j]=np.argmax(som_count[i][j])

  """ assign new values of som_env and env_spike and save """  
  allvars = tf.compat.v1.trainable_variables()
  som_env_vars=[x for x in allvars if 'oldcopy' not in x.name and 'som_env' in x.name]
  env_spike_vars=[x for x in allvars if 'oldcopy' not in x.name and 'som_env_spike' in x.name]
  tf_ops = [som_env_vars[0].assign(som_env),
            env_spike_vars[0].assign(spike_count)]
  if save_flag:
    sess.run(tf_ops)
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = glob_step+1)

  """ draw statistic z with spike """
  fig, ax = plt.subplots()
  x_values=np.arange(flags.z_dim)
  ax.errorbar(x_values,np.mean(inferred_zs_array,axis=0),np.std(inferred_zs_array,axis=0),lw=2,fmt='.k',ecolor='b',alpha=0.5)
  ax.bar(x_values, np.mean(inferred_spike_array,axis=0), width=0.25, color='r',alpha=0.5)
  print("inferrred spike std:",np.std(inferred_spike_array,axis=0))
  if epoch is not None:
    plt.savefig(os.path.join(save_folder,"check_spike_with_z_e{}_step{}.png".format(epoch,glob_step)))
  else:
    plt.savefig(os.path.join(save_folder,"check_spike_with_z.png"))
  plt.close()
  for i in range(z_dim):
    fig, ax = plt.subplots()
    ax.hist(inferred_zs_array[:,i],bins=100)
    plt.savefig(os.path.join(save_folder,"hist_z{}.png".format(i)))
    plt.close()

  """ draw histrogram """
  if draw_hist:
    draw_som_histogram(som_count,som_env,cur_task,epoch)

def draw_som_histogram(som_count,som_env=None,task=None,epoch=None):
  save_folder=os.path.join(flags.checkpoint_dir,"som_hist")
  if not os.path.exists(save_folder):
    os.mkdir(save_folder)
  if len(som_count.shape)==3:
    som_count=np.sum(som_count,2)
  fig, ax = plt.subplots()
  color_data=[]
  draw_data=[]
  env_cord=[]
  env_data=[]
  value_max=np.max(som_count)
  if som_env is not None:
    for i in range(flags.som_dim[0]):
      for j in range(flags.som_dim[1]):
        env_data.append(som_env[i][j])
        env_cord.append([i,j])
  for i in range(flags.som_dim[0]):
    for j in range(flags.som_dim[1]):
      draw_color=som_count[i][j]/value_max
      if draw_color==0:
        continue
      color_data.append(draw_color)
      draw_data.append([i,j])
  draw_data=np.array(draw_data)
  env_cord=np.array(env_cord)
  if som_env is not None:
    plt.scatter(env_cord[:,0],env_cord[:,1],c=env_data,cmap=plt.cm.binary,marker="s")
  plt.scatter(draw_data[:,0],draw_data[:,1],c=color_data,vmin=0,vmax=1,cmap=plt.cm.rainbow,marker=".")
  plt.axis([-1, flags.som_dim[0], -1, flags.som_dim[1]])
  if task is None:
    plt.savefig(os.path.join(save_folder,"som_histogram.png"))
  elif epoch is None:
    plt.savefig(os.path.join(save_folder,"som_histogram_t{}.png".format(int(task))))
  else:
    plt.savefig(os.path.join(save_folder,"som_histogram_t{}_e{}.png".format(int(task),int(epoch))))
  plt.close()

def disentangle_check_image_row(sess,model,manager,use_statistics=flags.use_statistics,save_original=True,add_ind_number=add_ind_number,epoch=None,step=None,task=flags.task):
  if not os.path.exists("disentangle_img_row"):
    os.mkdir("disentangle_img_row")
  if flags.test_img_ind==-1: #get a random image
    test_img_ind = random.randint(0,manager.sample_size)
  else:
    test_img_ind=flags.test_img_ind
  img,batch_class = manager.get_images([test_img_ind],with_label=True)
  if save_original:
    try:
      imageio.imwrite(flags.checkpoint_dir+"/original.png",img[0])
    except:
      imageio.imwrite(flags.checkpoint_dir+"/original.png", np.vstack([img[0][:,:,i] for i in range(img[0].shape[2])]))

  x_reconstruct = model.reconstruct(sess, img)
  reconstr_img = x_reconstruct[0]
  try:
    imageio.imwrite("disentangle_img_row/recon.png", reconstr_img)
    imageio.imwrite(flags.checkpoint_dir+"/recon.png",reconstr_img)
  except:
    print(np.min(reconstr_img))
    imageio.imwrite(flags.checkpoint_dir+"/recon.png", np.vstack([reconstr_img[:,:,i] for i in range(reconstr_img.shape[2])]))
    if img[0].shape[2]>3:
      quit()
  imageio.imwrite(flags.checkpoint_dir+"/original_vs_recon.png",np.hstack((img[0],reconstr_img)))

  z_mean, z_log_sigma_sq = model.transform(sess, img)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  if use_statistics: #use statistics traverse range rather than fixed range
    n_samples = manager.sample_size
    indices = list(range(n_samples))
    batch_size=flags.batch_size
    total_batch = n_samples // batch_size
    inferred_zs_array=None
    for i in range(total_batch):
      batch_indices = indices[batch_size*i : batch_size*(i+1)]
      batch_xs,batch_class = manager.get_images(batch_indices,with_label=True)
      batch_spikes=model.inference_spike(sess,batch_xs,batch_class=batch_class)
      batch_inferred_zs,_=model.transform(sess,batch_xs)
      if inferred_zs_array is None:
        inferred_zs_array=batch_inferred_zs
        inferred_spike_array=batch_spikes
      else:
        inferred_zs_array=np.vstack((inferred_zs_array,batch_inferred_zs))
        inferred_spike_array=np.vstack((inferred_spike_array,batch_spikes))
      inferred_zs_array=np.array(inferred_zs_array)
      z_m=np.mean(inferred_zs_array,axis=0)
      z_std=np.std(inferred_zs_array,axis=0)
  else:
    z_m = np.zeros([z_dim])
    z_std=np.ones([z_dim])

  n_z = z_dim
  maxi_range=maxi*z_std

  save_folder=os.path.join(flags.checkpoint_dir,str(task),"vae_disentangle")
  if not os.path.exists(save_folder):
    if not os.path.exists(os.path.join(flags.checkpoint_dir,str(task))):
      os.mkdir(os.path.join(flags.checkpoint_dir,str(task)))
    os.mkdir(save_folder)

  samples_total=[]
  if reorder_dims==[]:
    loop_dims=range(z_dim)
  else:
    loop_dims=reorder_dims
  for target_z_index in loop_dims:
    samples=[]
    for ri in range(gif_nums+1):
      value=-maxi_range[target_z_index]+2*maxi_range[target_z_index]/gif_nums*ri+z_m[target_z_index]
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_mean[0][i]
      reconstr_img = model.generate(sess, z_mean2,)
      rimg = reconstr_img[0]
      if image_chn==1:
        rimg = rimg.reshape(image_size, image_size)
      samples.append(np.array(rimg)*255)
    imgs_comb = np.hstack((img for img in samples))

    if add_ind_number:
      ind_img = Image.new('L', (image_size,image_size), (0))
      draw = ImageDraw.Draw(ind_img)
      font = ImageFont.truetype("arial.ttf", 50)
      draw.text((0,0),str(target_z_index),(255),font=font)
      img_np=np.array(ind_img)
      if image_chn==3:
        img_np=np.repeat(np.expand_dims(img_np,2),3,axis=2)
      imgs_comb=np.hstack((img_np,imgs_comb))
    samples_total.append(imgs_comb)
  imgs_total = np.vstack((img for img in samples_total))
  imageio.imwrite("disentangle_img_row/0check_z_total.png",imgs_total)
  if (epoch is not None) and (step is not None):
    imageio.imwrite(save_folder+"/0trav_t{}_e{}_step{}.png".format(flags.task,epoch,step),imgs_total)
  elif epoch is not None:
    imageio.imwrite(save_folder+"/0trav_t{}_e{}.png".format(flags.task,epoch),imgs_total)
  elif save_img_ind_flag:
    imageio.imwrite(save_folder+"/0trav_t{}_s{}_reorder.png".format(flags.task,test_img_ind),imgs_total)
  else:
    imageio.imwrite(save_folder+"/0trav_t{}_reorder.png".format(flags.task),imgs_total)
  som_params,z_saved_spike=sess.run([model.embeddings,model.env_spike],feed_dict={})
  som_means=som_params[:,:,:,0]
  som_vars=np.exp(som_params[:,:,:,1])
  som_spike=np.exp(som_params[:,:,:,2])
  print("som mean:",np.min(som_means),np.max(som_means),np.median(som_means))
  print("som var:",np.min(som_vars),np.max(som_vars),np.median(som_vars))
  print("som spike:",np.min(som_spike),np.max(som_spike),np.median(som_spike))
  print(test_img_ind)

def sampling_from_som_mixture(sess,model,num_of_samples):
  _,flatten_prior=model.get_som_prior(sess)
  flatten_som_env=np.reshape(sess.run(model.som_env,feed_dict={}),[-1])

  total_samples=[]
  total_zs=[]
  total_envs=[]
  som_size=flags.som_dim[0]*flags.som_dim[1]
  sample_count=0
  alpha1=sess.run(tf.exp(model.global_log_alpha1))
  alpha2=sess.run(tf.exp(model.global_log_alpha2))
  alpha3=sess.run(tf.exp(model.global_log_alpha3))
  while sample_count<num_of_samples:
    sample_ind=np.random.choice(np.arange(som_size), p=flatten_prior)
    if flatten_som_env[sample_ind]==0:
      continue
    sample_count+=1
    this_z,this_sample_alpha=sess.run([model.z_mixture_sample,model.z_mixture_sample_params_alpha],
                              feed_dict={model.component_flatten_ind:[sample_ind],
                                         model.increasing_weight:2})
    this_sample=model.generate(sess,this_z)
    last_task=model.get_most_close_one_hot_global_alpha(this_sample_alpha,alpha1,alpha2,alpha3)
    total_samples.append(this_sample[0])
    total_zs.append(this_z[0])
    total_envs.append(last_task)
  return np.array(total_samples),np.array(total_zs),np.reshape(np.array(total_envs),[-1,1])

def generate_from_som(sess,model,epoch=-1,use_mean=True,load_saved=False):
  save_folder=os.path.join(flags.checkpoint_dir,"som_hist")
  if not os.path.exists(save_folder):
    os.mkdir(save_folder)
  if load_saved:
    som_weights=np.load("SOM_params.npy",allow_pickle=True)
  else:
    som_weights=model.som_weights(sess)
    print("saved som weights not found")
  for i in [0,9,10,19]:
    for j in [0,9,10,19]:
      fig, ax = plt.subplots()
      x_values=np.arange(flags.z_dim)
      ax.bar(x_values, np.exp(som_weights[i,j,:,2]), width=0.25, color='r',alpha=0.5)
      plt.savefig(os.path.join(save_folder,"check_som_spike_{}_{}.png".format(i,j)))
      plt.close()
  sample_total=[]
  subname=""
  for i in range(flags.som_dim[0]):
    samples_row=[]
    for j in range(flags.som_dim[1]):
      if use_mean:
        prototype=som_weights[i][j][:,0] #mean
        subname="mean"
      else:
        prototype=sess.run(model.z_mixture_sample,
                          feed_dict={model.z_mixture_sample_params:[som_weights[i][j]],
                                     model.increasing_weight:2})[0]
        subname="sample"
      if 0:
        reconstr_img = np.zeros([1,64,64,1])
      else:
        reconstr_img = model.generate(sess, [prototype])
      rimg = reconstr_img[0]
      samples_row.append(np.array(rimg)*255.)
    samples_row.reverse()
    imgs_comb_col = np.vstack((img for img in samples_row))
    sample_total.append(imgs_comb_col)
  imgs_comb=np.hstack((img for img in sample_total))
  if image_chn==1:
    imgs_comb=imgs_comb.reshape(imgs_comb.shape[0],imgs_comb.shape[1])
  imageio.imwrite(flags.checkpoint_dir+"/prototypes_all_epoch{}_".format(epoch)+subname+".png", imgs_comb)

def main(argv):
  gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=flags.gpu_usage,visible_device_list = str(flags.gpu_index))
  
  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False,gpu_options=gpu_options))
  
  model = VAE(flags=flags,image_size=image_size,imgs_chn=image_chn)
  
  manager = DataManager_3dshapes()
  
  sess.run(tf.compat.v1.global_variables_initializer())
  saver = load_checkpoints(sess,model,flags)

  if flags.mode==1:
    train_som_mixture_CL(sess,model,manager,saver)
    generate_from_som(sess,model,epoch=-1)
        
  elif flags.mode==2:
    manager.load(train_test_split_mode=flags.train_test_mode,set_condition=True,seed=flags.seed,task=flags.task)
    disentangle_check_image_row(sess,model,manager)
    generate_from_som(sess,model,epoch=-1)

  elif flags.mode==3:
    shared_dim=[]
    ns_dim=[]
    manager.load(train_test_split_mode=0,set_condition=True,seed=flags.seed,task=1) #check previous active dimensions first
    xs=manager.get_random_images(128)
    if flags.sparse_coding_MIG:
      spike=np.mean(model.inference_spike(sess, xs, batch_class=np.ones([xs.shape[0],1])*(flags.task-2)),axis=0)
      z_mean, z_log_sigma_sq = model.transform_old(sess, xs)
      z_std=np.std(z_mean,0)
      for ind in range(flags.z_dim):
        if z_std[ind]>=1 and spike[ind]>0.5:
          shared_dim.append(ind)
        else:
          ns_dim.append(ind)
    else:
      z_mean, z_log_sigma_sq = model.transform_old(sess, xs)
      z_sigma_sq = np.mean(np.exp(z_log_sigma_sq),axis=0)
      mu_var=np.std(z_mean,0)**2
      for ind in range(flags.z_dim):
        if mu_var[ind]>1e-2:
          shared_dim.append(ind)
        else:
          ns_dim.append(ind)

    print("shared_dim:", shared_dim) # always double check the traverse results to confirm the shared dimensions
    manager.load(train_test_split_mode=0,set_condition=True,seed=flags.seed,task=flags.task) #load and check current task
    metric,_,_,active_units=mutual_info_metric_3dshapes(sess, model, manager,task=flags.task,flags=flags)
    MIG=metric[0].cpu().detach().numpy()
    MIG_sup=metric[1].cpu().detach().numpy()
    mi_normed=metric[2].cpu().detach().numpy()

    print('MIG:',np.mean(MIG),"MIG-sup",np.mean(MIG_sup[active_units]),
          "MI(new_fac,shared):",np.sum(mi_normed[0][shared_dim]))
    np.savetxt(flags.checkpoint_dir+"/check_MIG_t{}.txt".format(flags.task),
              [np.mean(MIG),np.mean(MIG_sup[active_units]),
              np.sum(mi_normed[0][shared_dim])]+[-9999]+shared_dim,'%1.5f')
    
if __name__ == '__main__':
  tf.compat.v1.app.run()
