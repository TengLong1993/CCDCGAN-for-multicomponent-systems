from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

import numpy as np
import os

from keras import backend as K

def mkdir(path): #define a function to create non-existed folder automatically
 folder = os.path.exists(path)
 if not folder:
  os.makedirs(path)
  print("---  new folder...  ---")
  print("---  OK  ---")
 else:
  print("---  There is this folder!  ---")

def min_formation_energy(y_true,y_pred):
 return K.mean(K.exp(y_pred))

class CCDCGAN():
 def __init__(self,whether_from_trained_model=False, trained_discriminator_directory='./discriminator.h5', trained_generator_directory='./generator.h5', whether_formation_energy_constrained=False,trained_formation_energy_directory='./formation_energy.h5'):
  # Input shape
  self.img_rows = 160
  self.img_cols = 160
  self.channels = 1
  self.img_shape = (self.img_rows, self.img_cols, self.channels)
  self.latent_dim = 800
  self.whether_from_trained_model=whether_from_trained_model
  self.whether_formation_energy_constrained=whether_formation_energy_constrained

  optimizer = Adam(0.0002)

  # Build and compile the discriminator
  if whether_from_trained_model==True:
   self.discriminator = self.rebuild_discriminator(trained_discriminator_directory)
  else:
   self.discriminator = self.build_discriminator()
  self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  # Build the generator
  if whether_from_trained_model==True:
   self.generator = self.rebuild_generator(trained_generator_directory)
  else:
   self.generator = self.build_generator()
  # The generator takes noise as input and generates imgs
  z = Input(shape=(self.latent_dim,))
  img = self.generator(z)
  # For the combined model we will only train the generator
  self.discriminator.trainable = False
  # The discriminator takes generated images as input and determines validity
  valid = self.discriminator(img)
  if whether_formation_energy_constrained==True:
   # Reuild the constrain
   self.constrain = self.rebuild_constrain(trained_formation_energy_directory)
   # For the combined model 2 we will only train the generator
   self.constrain.trainable = False
   # The discriminator takes generated images as input and give formation energy
   formation_energy = self.constrain(img)
   self.combined = Model(inputs=z,outputs=[valid, formation_energy])
   losses = ["binary_crossentropy", min_formation_energy]
   lossWeights = [ 1.0, 0.1]
   self.combined.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)
  else:
   self.combined = Model(z, valid)
   self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

 def build_generator(self):
  model = Sequential()
  model.add(Dense(128 * 5 * 5, activation="relu", input_dim=self.latent_dim))
  model.add(Reshape((5, 5, 128)))
  model.add(UpSampling2D())
  model.add(Conv2D(128, kernel_size=3, padding="same"))
  model.add(BatchNormalization(momentum=0.9))
  model.add(Activation("relu"))
  model.add(UpSampling2D())
  model.add(Conv2D(64, kernel_size=3, padding="same"))
  model.add(BatchNormalization(momentum=0.9))
  model.add(Activation("relu"))
  model.add(UpSampling2D())
  model.add(Conv2D(32, kernel_size=3, padding="same"))
  model.add(BatchNormalization(momentum=0.9))
  model.add(Activation("relu"))
  model.add(UpSampling2D())
  model.add(Conv2D(16, kernel_size=3, padding="same"))
  model.add(BatchNormalization(momentum=0.9))
  model.add(Activation("relu"))
  model.add(UpSampling2D())
  model.add(Conv2D(8, kernel_size=3, padding="same"))
  model.add(BatchNormalization(momentum=0.9))
  model.add(Activation("relu"))
  model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
  model.add(Activation("tanh"))
  model.summary()
  #model.name="generator"
  noise = Input(shape=(self.latent_dim,))
  img = model(noise)
  return Model(noise, img)

 def rebuild_generator(self,trained_generator_directory):
  model=load_model(trained_generator_directory)
  model.summary()
  #model.name="generator"
  return model

 def build_discriminator(self):
  model = Sequential()
  model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
  #model.add(BatchNormalization(momentum=0.9))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.summary()
  #model.name="discriminator"
  img = Input(shape=self.img_shape)
  validity = model(img)
  return Model(img, validity)

 def rebuild_discriminator(self,trained_discriminator_directory):
  model=load_model(trained_discriminator_directory)
  model.summary()
  #model.name="discriminator"
  return model

 def rebuild_constrain(self,trained_constrain_directory):
  model=load_model(trained_constrain_directory)
  model.summary()
  #model.name=trained_formation_energy_directory.split('/')[-1][:-3]
  return model

 def train(self, text_directory, combined_graph_path, combined_graph_name, GAN_model_folder_path, learning_process_text_name):
  f=open(text_directory,'r')
  lines=f.readlines()
  f.close()
  lines=lines[1:]
  name_list=[]
  for line in lines:
   name_list.append(line.split()[0])
  total_number=len(name_list)
  large_batch_size=total_number//40
  batch_size=128
  valid = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))
  mkdir(GAN_model_folder_path)
  g=open(GAN_model_folder_path+learning_process_text_name,'w')
  g.write('epoch discriminator_loss generator_loss mean_formation_energy')
  g.write('\n')
  
  for epoch in range(41):
   # Load the dataset
   large_batch=np.load(combined_graph_path+str(epoch)+'_'+combined_graph_name)
   large_batch = np.expand_dims(large_batch, axis=3)
   
   for iteration in range(2500):
    np.random.seed(iteration)
    idx = np.random.randint(0, large_batch.shape[0], batch_size)
    imgs = large_batch[idx]
    # Sample noise and generate a batch of new images
    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
    gen_imgs = self.generator.predict(noise)
    # Train the discriminator (real classified as ones and generated as zeros)
    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # Train the generator (wants discriminator to mistake images as real)
    if self.whether_formation_energy_constrained==True:
     g_loss = self.combined.train_on_batch(noise, [valid,valid])
     print ("[%d | %d] [D loss: %f, acc.: %.2f%%] [G loss: %f, from generator: %f, from constrain: %f] " % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss[0],g_loss[1],g_loss[2]))
    else:
     g_loss = self.combined.train_on_batch(noise, valid)
     print ("[%d | %d] [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
   
   if self.whether_formation_energy_constrained==True:
    new_line=str(epoch)+' '+str(d_loss[0])+' '+str(g_loss[0])+' '+str(g_loss[1])+' '+str(g_loss[2])+'\n'
    g.write(new_line)
   else:
    new_line=str(epoch)+' '+str(d_loss[0])+' '+str(g_loss)+'\n'
    g.write(new_line)
   mkdir(GAN_model_folder_path+'/epoch_%d/' % epoch)
   self.generator.save(GAN_model_folder_path+'/epoch_%d/generator.h5' % epoch)
   self.discriminator.save(GAN_model_folder_path+'/epoch_%d/discriminator.h5' % epoch)
  g.close()

 def predict(self, begin_epochs,epochs, generated_graph_folder_path):
  mkdir(generated_graph_folder_path)
  for epoch in range(begin_epochs,epochs):
   np.random.seed(epoch)
   noise = np.random.normal(0, 1, (1, self.latent_dim))
   gen_imgs = self.generator.predict(noise)
   np.save(generated_graph_folder_path+'epoch_%d.npy' % epoch,gen_imgs.reshape(160,160))

ccdcgan = CCDCGAN(whether_from_trained_model=False, trained_discriminator_directory='./dcgan_model/epoch_40/discriminator.h5', trained_generator_directory='./dcgan_model/epoch_40/generator.h5', whether_formation_energy_constrained=False,trained_formation_energy_directory='./formation_energy.h5')
ccdcgan.train(text_directory='../database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt', combined_graph_path='../data_preprocessing/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_other_primitive_cell_selected_cL10_aN20.npy', GAN_model_folder_path='./dcgan_model/', learning_process_text_name='DCGAN_learning_curve.txt')
#for i in range(10):
# ccdcgan.predict(begin_epochs=5*i,epochs=5*i+5,generated_graph_folder_path='./dcgan_generated_graph/')
