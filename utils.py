import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy
import glob
from shutil import copyfile

def stat(x):
  print("min:",np.min(x),"max:",np.max(x),"mean:",np.mean(x),"std:",np.std(x))
  return ["min:",np.min(x),"max:",np.max(x),"mean:",np.mean(x),"std:",np.std(x)]

def load_checkpoints(sess,model,flags,check_point=None):
  if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
  saver = tf.compat.v1.train.Saver(max_to_keep=1)
  if check_point is not None:
    checkpoint = tf.train.get_checkpoint_state(check_point)
  else:
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    #saver.restore(sess, checkpoint.model_checkpoint_path)
    try:
      saver.restore(sess, checkpoint.model_checkpoint_path)
    except:
      print("***** direct restoration failed, try loading existing parameters only ****")
      if flags.mode!=1:
        print("**** Not in train mode, better check model structure ****")
      optimistic_restore(sess,checkpoint.model_checkpoint_path)
    print("**** loaded checkpoint: {0} ****".format(checkpoint.model_checkpoint_path))
  else:
    print("**** Could not find old checkpoint ****")
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  return saver

def optimistic_restore(session, save_file):
    #Adapt code from https://github.com/tensorflow/tensorflow/issues/312
    #only load those variables that exist in the checkpoint file
    reader = tf.compat.v1.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = [(var.name, var.name.split(':')[0]) for var in tf.compat.v1.global_variables()
                        if var.name.split(':')[0] in saved_shapes]
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.compat.v1.global_variables()), tf.compat.v1.global_variables()))
    with tf.compat.v1.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.compat.v1.train.Saver(restore_vars)
    saver.restore(session, save_file)

def one_hot(labels, n_class):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels].T
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y

def calc_increasing_capacity(step,min_step,max_step,start_value=1,maxi_value=2):
    if step <min_step:
      c=0
    elif step > max_step:
      c = maxi_value
    else:
      c = start_value+(maxi_value-start_value) * ((step-min_step) / (max_step-min_step))
    return c

def copy_checkpoints_and_som_to_new_folder(flags,cur_task):
  #After training, copy checkpoints to a new folder
  if not os.path.exists(os.path.join(flags.checkpoint_dir,str(cur_task))):
    os.mkdir(os.path.join(flags.checkpoint_dir,str(cur_task)))
  for file in glob.glob(os.path.join(flags.checkpoint_dir,"checkpoint*")):
    copyfile(file, os.path.join(flags.checkpoint_dir,str(cur_task),os.path.basename(file)))
