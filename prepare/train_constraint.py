import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
import os
from keras import backend as K
import random

def mkdir(path): #define a function to create non-existed folder automatically
 folder = os.path.exists(path)
 if not folder:
  os.makedirs(path)
  print("---  new folder...  ---")
  print("---  OK  ---")
 else:
  print("---  There is this folder!  ---")

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class constrain_model():
 def __init__(self, learning_rate,whether_from_trained_model=False, trained_formation_energy_directory='./formation_energy.h5',trained_formation_energy_constants_directory='./formation_energy.npy'):
  # Input shape
  self.img_rows = 160
  self.img_cols = 160
  self.channels = 1
  self.img_shape = (self.img_rows, self.img_cols, self.channels)
  self.learning_rate= learning_rate
  optimizer = Adam(learning_rate,beta_1=0.9,beta_2=0.999)

  # Build and compile the constrain network
  if whether_from_trained_model==True:
   self.constrain = self.rebuild_constrain(trained_formation_energy_directory)
   constants=np.load(trained_formation_energy_constants_directory)
   self.mu=constants[0]
   self.delta=constants[1]
  else:
   self.constrain = self.build_constrain()
  self.constrain.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=[r2_score,keras.metrics.MeanSquaredError()])

 def build_constrain(self):
  model = Sequential()
  model.add(Conv2D(2, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  model.add(Conv2D(4, kernel_size=4, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  model.add(Conv2D(8, kernel_size=4, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  model.add(Conv2D(16, kernel_size=4, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  model.add(Conv2D(32, kernel_size=4, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0))
  model.add(Dropout(0.25))
  #model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  #model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  #model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  #model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(Dropout(0.25))
  model.add(Dense(512,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(Dropout(0.25))
  model.add(Dense(512,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(Dropout(0.25))
  model.add(Dense(512,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(Dropout(0.25))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(256,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(128,activation='relu'))
  #model.add(BatchNormalization(momentum=0.9))
  #model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(64,activation='relu'))
  #model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(1,activation='tanh'))
  #model.add(Dense(1,activation='relu'))
  model.add(LeakyReLU(alpha=0))
  model.summary()
  img = Input(shape=self.img_shape)
  validity = model(img)
  return Model(img, validity)

 def rebuild_constrain(self,trained_constrain_directory):
  model=self.build_constrain()
  model.load_weights(trained_constrain_directory)
  print('weights loaded from '+trained_constrain_directory)
  return model

 def train(self, text_directory, combined_graph_path, combined_graph_name, property_directory, constrain_model_folder_path, constrain_model_name):
  f=open(text_directory,'r')
  lines=f.readlines()
  f.close()
  lines=lines[1:]
  random.Random(1).shuffle(lines)
  name_list=[]
  for line in lines:
   name_list.append(line.split()[0])
  total_number=len(name_list)
  large_batch_size=total_number//40
  batch_size=128
  best_mae=100
  best_r2=-100
  mkdir(constrain_model_folder_path+'best/')
  mkdir(constrain_model_folder_path+'final/')
  f=open(constrain_model_folder_path+constrain_model_name+'_learning_curve.txt','w')
  f.write('train_mae train_r2 test_mae test_r2\n')
  y_o=np.load(property_directory)
  self.mu=np.min(y_o)
  self.delta=np.ptp(y_o)
  y=(y_o-self.mu)/self.delta
  y=y.reshape(total_number,1)
  validation_set=np.load(combined_graph_path+str(38)+'_'+combined_graph_name)
  validation_set = np.expand_dims(validation_set, axis=3)
  y_validation=y[38*large_batch_size:38*large_batch_size+large_batch_size]
  print(y_validation)
  test_set=np.load(combined_graph_path+str(39)+'_'+combined_graph_name)
  test_set = np.expand_dims(test_set, axis=3)
  y_test=y[39*large_batch_size:]

  for random_epoch in range(1000):
   # Load the dataset
   epoch=random.Random(random_epoch).randint(0,37)
   large_batch=np.load(combined_graph_path+str(epoch)+'_'+combined_graph_name)
   large_batch = np.expand_dims(large_batch, axis=3)
   
   #large_batch = np.append(large_batch, validation_set[epoch*10:epoch*10+10], axis=0)
   #large_batch = np.append(large_batch, test_set[epoch*10:epoch*10+10], axis=0)

   y_large_batch=y[epoch*large_batch_size:epoch*large_batch_size+large_batch_size]

   #y_large_batch = np.append(y_large_batch, y_validation[epoch*10:epoch*10+10], axis=0)
   #y_large_batch = np.append(y_large_batch, y_test[epoch*10:epoch*10+10], axis=0)
   
   for iteration in range(100):
    np.random.seed(random_epoch+iteration)
    idx = np.random.randint(0, large_batch.shape[0], batch_size)
    imgs = large_batch[idx]
    valid = y_large_batch[idx]
    # Train the constrain
    d_loss_train = self.constrain.train_on_batch(imgs, valid)
    d_loss=self.constrain.test_on_batch(validation_set, y_validation)        
    # Plot the progress
    print ("[%d | %d] [validation mae: %f, acc.: %.2f%%, mse: %f] [train mae: %f, acc.: %.2f%%, mse: %f]" % (random_epoch, iteration, d_loss[0]*self.delta, 100*d_loss[1], d_loss[2]*self.delta*self.delta, d_loss_train[0]*self.delta, 100*d_loss_train[1], d_loss_train[2]*self.delta*self.delta))
    f.write(str(d_loss_train[0]*self.delta)+' '+str(d_loss_train[1])+' '+str(d_loss[0]*self.delta)+' '+str(d_loss[1])+'\n')
   if d_loss[0]*self.delta < best_mae:
    best_mae=d_loss[0]*self.delta
    best_r2=d_loss[1]
    self.constrain.save(constrain_model_folder_path+'best/'+constrain_model_name+'.h5')
    d_loss_test=self.constrain.test_on_batch(test_set, y_test)

  self.constrain.save(constrain_model_folder_path+'final/'+constrain_model_name+'.h5')
  f.write('best validation mae: '+str(best_mae)+' best validation r2: '+str(best_r2)+'\n')
  f.write('best test mae: '+str(d_loss_test[0]*self.delta)+'best test r2: '+str(d_loss_test[1])+'\n')
  
  d_loss_test=self.constrain.test_on_batch(test_set, y_test)
  f.write('final validation mae: '+str(d_loss[0]*self.delta)+' final validation r2: '+str(d_loss[1])+'\n')
  f.write('final test mae: '+str(d_loss_test[0]*self.delta)+'final test r2: '+str(d_loss_test[1])+'\n')
  f.write('learning_rate: '+str(self.learning_rate))
  f.close()
  constants=np.array([self.mu,self.delta])
  np.save(constrain_model_folder_path+constrain_model_name+'.npy',constants)
  return best_mae,d_loss_test[0]*self.delta

 def make_prediction(self,img):
  formation_energy = self.constrain.predict(img)
  formation_energy = formation_energy*self.delta +self.mu
  return formation_energy

 def get_test_details(self, text_directory, combined_graph_path, combined_graph_name, property_directory, constrain_model_folder_path, constrain_model_name):
  f=open(text_directory,'r')
  lines=f.readlines()
  f.close()
  lines=lines[1:]
  random.Random(1).shuffle(lines)
  name_list=[]
  for line in lines:
   name_list.append(line.split()[0])
  total_number=len(name_list)
  large_batch_size=total_number//40
  batch_size=512
  best_mae=100
  best_r2=-100
  mkdir(constrain_model_folder_path+'best/')
  mkdir(constrain_model_folder_path+'final/')
  f=open(constrain_model_folder_path+constrain_model_name+'_learning_curve.txt','w')
  f.write('train_mae train_r2 test_mae test_r2\n')
  y_o=np.load(property_directory)
  self.mu=np.min(y_o)
  self.delta=np.ptp(y_o)
  y=(y_o-self.mu)/self.delta
  y=y.reshape(total_number,1)
  validation_set=np.load(combined_graph_path+str(38)+'_'+combined_graph_name)
  validation_set = np.expand_dims(validation_set, axis=3)
  y_validation=y[38*large_batch_size:38*large_batch_size+large_batch_size]
  test_set=np.load(combined_graph_path+str(39)+'_'+combined_graph_name)
  test_set = np.expand_dims(test_set, axis=3)
  y_test=y[39*large_batch_size:]
  g=open(constrain_model_folder_path+constrain_model_name+'_details.txt','w')
  g.write('test_real test_predict\n')
  y_test_predict=self.make_prediction(test_set)
  #print(y_test_predict)
  for i in range(len(y_test)):
   #g.write(str(y_test[i]*self.delta +self.mu)[1:-1]+' '+str(y_test_predict[i]).split()[0][11:-2]+'\n')
   g.write(str(y_test[i]*self.delta +self.mu)[1:-1]+' '+str(y_test_predict[i])[1:-1]+'\n')

class find_best_constrain_model():
 def __init__(self, whether_from_trained_model=False, trained_formation_energy_directory='./formation_energy.h5', trained_formation_energy_constants_directory='./formation_energy.npy', whether_training=True,whether_learning_rate_tune=True,learning_rate=0.0003,text_directory='../database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt', combined_graph_path='../data_preprocessing/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_other_primitive_cell_selected_cL10_aN20.npy', property_directory='./data_to_train_constraint/formation_energy.npy',constrain_model_folder_path='./',constrain_model_name='formation_energy'):
  if whether_training==True:
   if whether_learning_rate_tune==True:
    # Compare to get the best model
    best_loss=100
    learning_rate_list=[0.001,0.0003,0.0001,0.00003,0.00001]
    for i in range(10):
     if i<5:
      con=constrain_model(learning_rate_list[i],whether_from_trained_model,trained_formation_energy_directory,trained_formation_energy_constants_directory)
      validation_loss, test_loss=con.train(text_directory, combined_graph_path, combined_graph_name, property_directory, constrain_model_folder_path+'lr_'+str(learning_rate_list[i])+'/', constrain_model_name)
      if validation_loss<best_loss:
       best_loss=validation_loss
       best_lr=learning_rate_list[i]
    command='cp '+constrain_model_folder_path+'lr_'+str(best_lr)+'/* '+constrain_model_folder_path
    os.system(command)

   else:
    con=constrain_model(learning_rate,whether_from_trained_model,trained_formation_energy_directory,trained_formation_energy_constants_directory)
    validation_loss, test_loss=con.train(text_directory, combined_graph_path, combined_graph_name, property_directory, constrain_model_folder_path, constrain_model_name)
  else:
   con=constrain_model(learning_rate,whether_from_trained_model,trained_formation_energy_directory,trained_formation_energy_constants_directory)
   con.get_test_details(text_directory, combined_graph_path, combined_graph_name, property_directory, constrain_model_folder_path, constrain_model_name)
