import os
import numpy as np
import random
import math

def mkdir(path): #define a function to create non-existed folder automatically
 folder = os.path.exists(path)
 if not folder:
  os.makedirs(path)
  print("---  new folder...  ---")
  print("---  OK  ---")
 else:
  print("---  There is this folder!  ---")

def get_graph(text_directory,encoded_lattice_path,encoded_sites_path,graph_path):
 mkdir(graph_path)
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 for i in range(len(lines)):
  line=lines[i]
  name=line.split()[0]+'.npy'
  encoded_lattice_name=encoded_lattice_path+name
  encoded_lattice=np.load(encoded_lattice_name).reshape(50)
  encoded_sites_name=encoded_sites_path+name
  encoded_sites=np.load(encoded_sites_name).reshape(200,118)
  graph=np.zeros([160,160])
  graph[0,:50]=encoded_lattice
  for j in range(118):
   for k in range(200):
    row=(j*200+k)//160+1
    column=(j*200+k)%160
    graph[row,column]=encoded_sites[k,j]
  graph_name=graph_path+name
  np.save(graph_name,graph)

def restore_encoded_lattice_sites_from_graph(graph_path,whether_restore_encoded_lattice=True,encoded_lattice_path='./',whether_restore_encoded_sites=True,encoded_sites_path='./'):
 if whether_restore_encoded_lattice==True:
  mkdir(encoded_lattice_path)
 if whether_restore_encoded_sites==True:
  mkdir(encoded_sites_path)
 filename=os.listdir(graph_path)
 for eachfile in filename:
  graph_name=graph_path+eachfile
  graph=np.load(graph_name).reshape(160,160)
  if whether_restore_encoded_lattice==True:
   encoded_lattice_name=encoded_lattice_path+eachfile
   encoded_lattice=graph[0,:25].reshape(25,1)
   np.save(encoded_lattice_name,encoded_lattice)
  if whether_restore_encoded_sites==True:
   encoded_sites_name=encoded_sites_path+eachfile
   encoded_sites=np.zeros([200,118])
   for j in range(118):
    for k in range(200):
     row=(j*200+k)//160+1
     column=(j*200+k)%160
     encoded_sites[k,j]=graph[row,column]
   np.save(encoded_sites_name,encoded_sites) 

def get_large_batch_graph(text_directory,graph_path,data_X_path,data_X_name):
 mkdir(data_X_path)
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 random.Random(1).shuffle(lines)
 total_number=len(lines)
 large_batch_size=total_number//40
 for epoch in range(40):
  data_X=int(0)
  if epoch<39:
   for i in range(epoch*large_batch_size, epoch*large_batch_size+large_batch_size):
    line=lines[i]
    graphname=graph_path+line.split()[0]+'.npy'
    graph=np.load(graphname).reshape(1,160,160)
    if type(data_X)==int:
     data_X=graph
    else:
     data_X=np.append(data_X, graph, axis=0)
   np.save(data_X_path+str(epoch)+'_'+data_X_name,data_X)
  else:
   for i in range(epoch*large_batch_size, total_number):
    line=lines[i]
    graphname=graph_path+line.split()[0]+'.npy'
    graph=np.load(graphname).reshape(1,160,160)
    if type(data_X)==int:
     data_X=graph
    else:
     data_X=np.append(data_X, graph, axis=0)
   np.save(data_X_path+str(epoch)+'_'+data_X_name,data_X)

def get_data_X(text_directory,graph_path,data_X_path,data_X_name):
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 data_X=int(0)
 for i in range(len(lines)):
  line=lines[i]
  graphname=graph_path+line.split()[0]+'.npy'
  graph=np.load(graphname).reshape(1,160,160)
  if type(data_X)==int:
   data_X=graph
  else:
   data_X=np.append(data_X, graph, axis=0)
 mkdir(data_X_path)
 np.save(data_X_path+data_X_name,data_X)

def get_data_y(text_directory,property_name,data_y_path,data_y_name,process_type='Normal'):
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 random.Random(1).shuffle(lines)
 property_index=title.index(property_name)
 y=[]
 for i in range(len(lines)):
  line=lines[i]
  if process_type=='log10':
   property_value=math.log10(float(line.split()[property_index]))######################modified
  if process_type=='Normal':
   property_value=float(line.split()[property_index])
  y.append(property_value)
 data_y=np.array(y)
 mkdir(data_y_path)
 np.save(data_y_path+data_y_name,data_y)
