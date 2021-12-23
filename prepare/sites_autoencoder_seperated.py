import os
import random
import numpy as np
import tensorflow as tf
import pickle

###################################################################function
#####general
def mkdir(path): #define a function to create non-existed folder automatically
 folder = os.path.exists(path)
 if not folder:
  os.makedirs(path)
  print("---  new folder...  ---")
  print("---  OK  ---")
 else:
  print("---  There is this folder!  ---")

def train_test_split(path,train_ratio,validation_ratio,test_ratio,whether_from_text=True,text_directory='../database/20201009version/2_3_properties_primitive_cell_selected_cL10_aN20.txt'):
 if whether_from_text==False:
  filename=os.listdir(path)
  name_list=[]
  for eachnpyfile in filename:
   if eachnpyfile.endswith('.npy'):
    name_list.append(eachnpyfile[:-4])
 else:
  f=open(text_directory,'r')
  lines=f.readlines()
  f.close()
  lines=lines[1:]
  name_list=[]
  for line in lines:
   name_list.append(line.split()[0])
 test_size=round(test_ratio*len(name_list))
 validation_size=round(validation_ratio*len(name_list))
 random.seed(1)
 random.shuffle(name_list)
 train_name_list=name_list[test_size+validation_size:]
 validation_name_list=name_list[test_size:test_size+validation_size]
 test_name_list=name_list[:test_size]
 return name_list,train_name_list,validation_name_list,test_name_list
 
#####activation functions
def lrelu(x, leak=0.2):
 return tf.maximum(x, leak*x)

#####round
def threshold(x, val=0.5):
 x = tf.clip_by_value(x,0.5,0.5001) - 0.5
 x = tf.minimum(x * 10000,1) 
 return x

class sites_autoencoder():
 def __init__(self,learning_rate,element_idx):
  tf.reset_default_graph()
  self.a_axis = 64
  self.b_axis = 64
  self.c_axis = 64
  self.channels = 1
  self.voxel_shape = [1, self.a_axis, self.b_axis, self.c_axis, self.channels]
  self.latent_dim = 200
  self.img_shape = [1, 1, 1, 1, self.latent_dim]
  self.learning_rate= learning_rate
  self.element_index=element_idx
  
  self.weights = self.initialiseWeights()
  self.sites_voxel = tf.placeholder(shape=self.voxel_shape,dtype=tf.float32)
  self.encoded_sites = tf.placeholder(shape=self.img_shape,dtype=tf.float32) 
  
  with tf.variable_scope('encoders') as self.variable_scope_encoder:
   self.encoded_sites_to_train = self.build_encoder(self.sites_voxel, phase_train=True, reuse=False)
   self.variable_scope_encoder.reuse_variables()
   self.encoded_sites_to_check = self.build_encoder(self.sites_voxel, phase_train=False, reuse=True)

  with tf.variable_scope('decoders') as self.variable_scope_decoder:
   self.restored_sites_voxel_to_train = self.build_decoder(self.encoded_sites_to_train, phase_train=True, reuse=False)
   self.variable_scope_decoder.reuse_variables()
   self.restored_sites_voxel_to_check = self.build_decoder(self.encoded_sites_to_check,phase_train=False, reuse=True)

  self.restored_sites_voxel_to_train = threshold(self.restored_sites_voxel_to_train)
  self.restored_sites_voxel_to_check = threshold(self.restored_sites_voxel_to_check)

  self.mse_loss_to_train = tf.reduce_mean(tf.pow(self.sites_voxel - self.restored_sites_voxel_to_train, 2))
  self.mse_loss_to_check = tf.reduce_mean(tf.pow(self.sites_voxel - self.restored_sites_voxel_to_check, 2))
  self.para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae','wg'])]
  self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="Adam_AE").minimize(self.mse_loss_to_train,var_list=self.para_ae)

 def train(self, epochs_number, batch_size, train_name_list,validation_name_list,test_name_list, sites_voxel_path, model_folder_path):

  mkdir(model_folder_path)
  g=open(model_folder_path+'sites_autoencoder_learning_curve'+str(self.element_index)+'.txt','w')
  g.write('epoch train_loss validation_loss')
  g.write('\n')
  best_loss=1
  
  saver = tf.train.Saver() 
  restore_saver = tf.train.Saver() 
  with tf.Session() as self.sess:  
   self.sess.run(tf.global_variables_initializer())    
   restore_saver.restore(self.sess,'./autoencoder_model/sites/sites_autoencoder.ckpt')

   for epoch in range(epochs_number):
    # Select a random training mini-batch
    np.random.seed(epoch)
    idx = np.random.randint(0, len(train_name_list), batch_size)
    mse_tr = 0; mse_val = 0; train_len=0; validation_len=0;
    # Train the model
    for i in range(batch_size):
     pkl_file = open(sites_voxel_path+train_name_list[idx[i]]+'.pkl','rb')
     [extraction_list,compressed_image] = pickle.load(pkl_file)
     pkl_file.close()
     inputs_batch=compressed_image[extraction_list.index(self.element_index)].reshape(1,64,64,64,1)
     mse_l, _ = self.sess.run([self.mse_loss_to_train, self.optimizer],feed_dict={self.sites_voxel:inputs_batch})
     mse_tr += mse_l
    train_loss=mse_tr/batch_size
    # Validate the model
    
    for i in range(len(validation_name_list)):
     pkl_file = open(sites_voxel_path+validation_name_list[i]+'.pkl','rb')
     [extraction_list,compressed_image] = pickle.load(pkl_file)
     pkl_file.close()
     validation_len += len(extraction_list)
     inputs_batch=compressed_image[extraction_list.index(self.element_index)].reshape(1,64,64,64,1)
     mse_v = self.sess.run(self.mse_loss_to_check,feed_dict={self.sites_voxel:inputs_batch})#,o_vector:test_object_batch})
     mse_val += mse_v
    validation_loss=mse_val/len(validation_name_list)
    
    # Plot the progress
    print ("[learning_rate: %f] [epoch: %d] [train loss: %f] [validation loss: %f]" % (self.learning_rate, epoch, train_loss, validation_loss))
    new_line=str(epoch)+' '+str(train_loss)+' '+str(validation_loss)+'\n'
    g.write(new_line)
    # which model should be saved
    if validation_loss<best_loss:
     best_loss=validation_loss
     saver.save(self.sess, save_path = model_folder_path + 'sites_autoencoder'+str(self.element_index)+'.ckpt')
   
   # Test the model
   mse_test=0; len_test=0;
   for i in range(len(test_name_list)):
    pkl_file = open(sites_voxel_path+test_name_list[i]+'.pkl','rb')
    [extraction_list,compressed_image] = pickle.load(pkl_file)
    pkl_file.close()
    len_test += len(extraction_list)
    inputs_batch=compressed_image[extraction_list.index(self.element_index)].reshape(1,64,64,64,1)
    mse_t = self.sess.run(self.mse_loss_to_check,feed_dict={self.sites_voxel:inputs_batch})
    mse_test += mse_t
   test_loss=mse_test/len(test_name_list)
   
  g.write('\n'+'learning_rate: '+str(self.learning_rate))
  g.write('\n'+'validation_loss: '+str(best_loss))
  g.close()
  return best_loss, test_loss

 def initialiseWeights(self):
  global weights
  weights = {}
  xavier_init = tf.contrib.layers.xavier_initializer()
  weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 64, self.latent_dim], initializer=xavier_init)
  weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
  weights['wae1'] = tf.get_variable("wae1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
  weights['wae2'] = tf.get_variable("wae2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wae3'] = tf.get_variable("wae3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wae4'] = tf.get_variable("wae4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
  weights['wae5'] = tf.get_variable("wae5", shape=[4, 4, 4, 64, self.latent_dim], initializer=xavier_init)    
  return weights

 def build_decoder(self, z, batch_size=1, phase_train=True, reuse=False):
  with tf.variable_scope("gen",reuse=reuse):
   z = tf.reshape(z,[1,1,1,1,self.latent_dim])
   g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,64), strides=[1,1,1,1,1], padding="VALID")
   g_1 = lrelu(g_1)
   g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,64), strides=[1,2,2,2,1], padding="SAME")
   g_2 = lrelu(g_2)
   g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,64), strides=[1,2,2,2,1], padding="SAME")
   g_3 = lrelu(g_3)
   g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,64), strides=[1,2,2,2,1], padding="SAME")
   g_4 = lrelu(g_4)
   g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,64,64,64,1), strides=[1,2,2,2,1], padding="SAME")
   g_5 = tf.nn.sigmoid(g_5)
   return g_5

 def build_encoder(self, inputs, phase_train=True, reuse=False):
  with tf.variable_scope("enc",reuse=reuse):
   inputs = tf.reshape(inputs,[1,64,64,64,1])
   d_1 = tf.nn.conv3d(inputs, weights['wae1'], strides=[1,2,2,2,1], padding="SAME")
   d_1 = lrelu(d_1)
   d_2 = tf.nn.conv3d(d_1, weights['wae2'], strides=[1,2,2,2,1], padding="SAME") 
   d_2 = lrelu(d_2)      
   d_3 = tf.nn.conv3d(d_2, weights['wae3'], strides=[1,2,2,2,1], padding="SAME")  
   d_3 = lrelu(d_3) 
   d_4 = tf.nn.conv3d(d_3, weights['wae4'], strides=[1,2,2,2,1], padding="SAME")     
   d_4 = lrelu(d_4)
   d_5 = tf.nn.conv3d(d_4, weights['wae5'], strides=[1,1,1,1,1], padding="VALID")
   d_5 = tf.nn.tanh(d_5)
   return d_5

 def rebuild_autoencoder(self,previous_model_path,name_list,task,sites_voxel_path='./',encoded_sites_path='./',output_sites_voxel_path='./',to_be_restored_encoded_sites_path='./',decode_mode='reconstruct', target_system=None):
  restore_saver = tf.train.Saver() 
  with tf.Session() as self.sess:  
   self.sess.run(tf.global_variables_initializer())    
   restore_saver.restore(self.sess,previous_model_path+'sites_autoencoder'+str(self.element_index)+'.ckpt')
   if task=='encode':
    mkdir(encoded_sites_path)
    for i in range(len(name_list)):
     try:
      encoded_sites=np.load(encoded_sites_path+name_list[i]+'.npy')
     except:
      encoded_sites = np.zeros((self.latent_dim,118))
     pkl_file = open(sites_voxel_path+name_list[i]+'.pkl','rb')
     [extraction_list,compressed_image] = pickle.load(pkl_file)
     pkl_file.close()
     inputs_batch=compressed_image[extraction_list.index(self.element_index)].reshape(1,64,64,64,1)
     encoded_sites[:,self.element_index]=self.generate_encoded(inputs_batch)
     np.save(encoded_sites_path+name_list[i]+'.npy',encoded_sites)
   if task=='decode':
    mkdir(output_sites_voxel_path)
    name_list=os.listdir(to_be_restored_encoded_sites_path)
    for i in range(len(name_list)):
     inputs_batch=np.load(to_be_restored_encoded_sites_path+name_list[i]).reshape(self.latent_dim,118)
     if decode_mode=='reconstruct':
      pkl_file = open(sites_voxel_path+name_list[i][:-4]+'.pkl','rb')
      [extraction_list, _] = pickle.load(pkl_file)
      pkl_file.close()
     if decode_mode=='generate':
      if target_system==None:
       number_element = np.random.randint(2, 4, 1)
       extraction_list = np.random.randint(0, 118, number_element).tolist()
      if type(target_system)==list:
       if len(target_system)>=2 and len(target_system)<=8:
        elementlist = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar','K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr','Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        extraction_list=[]
        for ele in target_system:
         extraction_list.append(elementlist.index(ele))
     if self.element_index in extraction_list:
      try:
       pkl_file = open(output_sites_voxel_path+name_list[i][:-4]+'.pkl','rb')
       [_, compressed_image] = pickle.load(pkl_file)
       pkl_file.close()
      except:
       compressed_image=extraction_list
      compressed_image[extraction_list.index(self.element_index)]=self.generate_voxel(inputs_batch[:,self.element_index])
      output = open(output_sites_voxel_path+name_list[i][:-4]+'.pkl', 'wb')
      sites_voxel=[extraction_list,compressed_image]
      pickle.dump(sites_voxel, output, -1)
      output.close()

 def generate_encoded(self,input_sites_voxel):
  encoded_sites = self.encoded_sites_to_check.eval(feed_dict={self.sites_voxel:input_sites_voxel.reshape(1,64,64,64,1)}).reshape(self.latent_dim)
  return encoded_sites

 def generate_voxel(self,input_encoded_sites):
  sites_voxel = self.restored_sites_voxel_to_check.eval(feed_dict={self.encoded_sites_to_check:input_encoded_sites.reshape(1,1,1,1,self.latent_dim)}).reshape(64,64,64)
  return sites_voxel

class find_best_autoencoder():
 def __init__(self, sites_voxel_path='./', model_folder_path='./', encoded_sites_path='./', epochs=200, batch_size=8,train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1,whether_training=True,whether_learning_rate_tune=True,learning_rate=0.0003,whether_encoded_generate=True,whether_from_text=True,text_directory='../database/20201009version/2_3_properties_primitive_cell_selected_cL10_aN20.txt',whether_voxel_generate=True,output_sites_voxel_path='./',to_be_restored_encoded_sites_path='./',decode_mode='reconstruct', target_system=None,element_index=0):
  # Load the dataset
  name_list,train_name_list,validation_name_list,test_name_list = train_test_split(sites_voxel_path,train_ratio,validation_ratio,test_ratio,whether_from_text,text_directory)

  if whether_training==True:
   if whether_learning_rate_tune==True:
    # Compare to get the best model
    best_loss=100
    learning_rate_list=[0.001,0.0003,0.0001,0.00003,0.00001]
    for i in range(10):
     if i<5:
      ae=sites_autoencoder(learning_rate_list[i],element_index)
      validation_loss, test_loss=ae.train(epochs, batch_size, train_name_list,validation_name_list,test_name_list, sites_voxel_path, model_folder_path+'lr_'+str(learning_rate_list[i])+'/')
      if validation_loss<best_loss:
       best_loss=validation_loss
       best_lr=learning_rate_list[i]
    command='cp '+model_folder_path+'lr_'+str(best_lr)+'/* '+model_folder_path
    os.system(command)

   else:
    ae=sites_autoencoder(learning_rate,element_index)
    validation_loss, test_loss=ae.train(epochs, batch_size, name_list,validation_name_list,test_name_list, sites_voxel_path, model_folder_path)
   
  #Generate encoded sites
  if whether_encoded_generate==True:
   ae=sites_autoencoder(learning_rate,element_index)
   ae.rebuild_autoencoder(model_folder_path,name_list,'encode',sites_voxel_path,encoded_sites_path,output_sites_voxel_path,to_be_restored_encoded_sites_path,decode_mode, target_system)

  #Generate sites voxel
  if whether_voxel_generate==True:
   ae=sites_autoencoder(learning_rate,element_index)
   ae.rebuild_autoencoder(model_folder_path,name_list,'decode',sites_voxel_path,encoded_sites_path,output_sites_voxel_path,to_be_restored_encoded_sites_path,decode_mode, target_system)

