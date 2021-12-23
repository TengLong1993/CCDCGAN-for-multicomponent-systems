import os
from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from joblib import Parallel, delayed
import pickle

def mkdir(path): #define a function to create non-existed folder automatically
 folder = os.path.exists(path)
 if not folder:
  os.makedirs(path)
  print("---  new folder...  ---")
  print("---  OK  ---")
 else:
  print("---  There is this folder!  ---")

def get_scale(sigma):
 scale = 1.0/(2*sigma**2)
 return scale

def basis_translate(atoms):
 N = len(atoms)
 pos = atoms.positions
 cg = np.mean(pos,0)
 dr = 7.5 - cg #move to center of 15A-cubic box
 dpos = np.repeat(dr.reshape(1,3),N,0)
 new_pos = dpos + pos
 atoms_ = atoms.copy()
 atoms_.cell = 15.0*np.identity(3)
 atoms_.positions = new_pos
 return atoms_
 
def get_fakeatoms_grid(atoms,nbins):
 atomss = []
 scaled_positions = []
 ijks = []
 grid = np.array([float(i)/float(nbins) for i in range(nbins)])
 yv,xv,zv = np.meshgrid(grid,grid,grid)
 pos = np.zeros((nbins**3,3))
 pos[:,0] = xv.flatten()
 pos[:,1] = yv.flatten()
 pos[:,2] = zv.flatten()
 atomss = Atoms('H'+str(nbins**3))
 atomss.set_cell(atoms.get_cell())#making pseudo-crystal containing H positioned at pre-defined fractional coordinate
 atomss.set_pbc(True)
 atomss.set_scaled_positions(pos)
 fakeatoms_grid = atomss
 return fakeatoms_grid

def get_image_one_atom(atom,fakeatoms_grid,nbins,scale):
 grid_copy = fakeatoms_grid.copy()
 ngrid = len(grid_copy)
 image = np.zeros((1,nbins**3))
 grid_copy.append(atom)
 drijk = grid_copy.get_distances(-1,range(0,nbins**3),mic=True)
 pijk = np.exp(-scale*drijk**2)
 image[:,:] = pijk.flatten()
 return image.reshape(nbins,nbins,nbins)

def get_atomlist_atomindex(atomlisttype='all element',a_list=None):#atomlisttype indicate which kind of list to be used; 'all element' is to use the whole peroidic table, 'specified' is to give a list on our own
 if atomlisttype=='all element':
  all_atomlist = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
 else:
  if atomlisttype=='specified':
   all_atomlist=a_list
  else:
   print('atom list type is not acceptable')
   return
 cod_atomlist=all_atomlist
 cod_atomindex = {}
 for i,symbol in enumerate(all_atomlist):
  cod_atomindex[symbol] = i
 return cod_atomlist,cod_atomindex
	
def get_image_all_atoms_sites(atoms,nbins,scale,num_cores,atomlisttype,a_list):
 fakeatoms_grid = get_fakeatoms_grid(atoms,nbins)
 cell = atoms.get_cell()
 imageall_gen = Parallel(n_jobs=num_cores)(delayed(get_image_one_atom)(atom,fakeatoms_grid,nbins,scale) for atom in atoms)
 imageall_list = list(imageall_gen)
 cod_atomlist,cod_atomindex = get_atomlist_atomindex(atomlisttype,a_list)
 nchannel = len(cod_atomlist)
 channellist = list(range(118))
 shape = (nbins,nbins,nbins,nchannel)
 image = np.zeros(shape,dtype=np.float32)
 for i,atom in enumerate(atoms):
  nnc = cod_atomindex[atom.symbol]
  img_i = imageall_list[i]
  image[:,:,:,nnc] += img_i * (img_i>=0.02)
 return image,channellist

def get_atoms(inputfile):
 atoms = read(inputfile)
 return atoms

def generate_sites_voxel(text_directory,cif_path,sites_voxel_path,atomlisttype,a_list):
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 mkdir(sites_voxel_path)
 scale=get_scale(0.26)
 for i in range(len(lines)):
  line=lines[i]
  filename=cif_path+line.split()[0]+'.cif'
  atoms=get_atoms(filename)
  atoms_=basis_translate(atoms)
  image,channellist=get_image_all_atoms_sites(atoms_,64,scale,8,atomlisttype,a_list)
  output = open(sites_voxel_path+line.split()[0]+'.pkl', 'wb')
  extraction_list=[]
  compressed_image=[]
  for j in range(len(channellist)):
   if image[:,:,:,j].any():#numpy.zeros([64,64,64],dtype=np.float32):
    extraction_list.append(j)
    compressed_image.append(image[:,:,:,j])   
  matrix=[extraction_list,compressed_image]
  pickle.dump(matrix, output, -1)
  output.close()

def extract_cell(atoms):
 cell = atoms.cell
 atoms_ = Atoms('Bi')
 atoms_.cell = cell
 atoms_.set_scaled_positions([0.5,0.5,0.5])
 return atoms_

def get_image_all_atoms(atoms,nbins,scale,num_cores,atomlisttype,a_list):
 fakeatoms_grid = get_fakeatoms_grid(atoms,nbins)
 cell = atoms.get_cell()
 imageall_gen = Parallel(n_jobs=num_cores)(delayed(get_image_one_atom)(atom,fakeatoms_grid,nbins,scale) for atom in atoms)
 imageall_list = list(imageall_gen)
 cod_atomlist,cod_atomindex = get_atomlist_atomindex(atomlisttype,a_list)
 nchannel = len(cod_atomlist)
 channellist = []
 for i,atom in enumerate(atoms):
  channel = cod_atomindex[atom.symbol]
  channellist.append(channel)
 channellist = list(set(channellist))
 nc = len(channellist)
 shape = (nbins,nbins,nbins,nc)
 image = np.zeros(shape,dtype=np.float32)
 for i,atom in enumerate(atoms):
  nnc = channellist.index(cod_atomindex[atom.symbol])
  img_i = imageall_list[i]
  image[:,:,:,nnc] += img_i * (img_i>=0.02)
 return image,channellist

def generate_lattice_voxel(text_directory,cif_path,lattice_voxel_path,atomlisttype,a_list):
 f=open(text_directory,'r')
 lines=f.readlines()
 f.close()
 title=lines[0].split()
 lines=lines[1:]
 mkdir(lattice_voxel_path)
 scale=get_scale(0.26)
 for i in range(len(lines)):
  line=lines[i]
  filename=cif_path+line.split()[0]+'.cif'
  atoms=get_atoms(filename)
  length_list=atoms.get_cell_lengths_and_angles()[:3]
  atoms_=extract_cell(atoms)
  image,channellist=get_image_all_atoms(atoms_,32,scale,8,atomlisttype,a_list)
  image=image.reshape(32,32,32)
  savefilename=lattice_voxel_path+line.split()[0]+'.npy'
  np.save(savefilename,image)

def voxel_generation(text_directory,cif_path,lattice_voxel_path,sites_voxel_path,atomlisttype,a_list=None,whether_lattice_voxel=True,whether_sites_voxel=True):
 if whether_lattice_voxel==True:
  print('generating lattice voxel')
  generate_lattice_voxel(text_directory,cif_path,lattice_voxel_path,atomlisttype,a_list)
 if whether_sites_voxel==True:
  print('generating sites voxel')
  generate_sites_voxel(text_directory,cif_path,sites_voxel_path,atomlisttype,a_list)

#voxel_generation(text_directory='../database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt',cif_path='../database/20201009version/other_primitive_cell/',lattice_voxel_path='./temp/original_lattice_voxel/',sites_voxel_path='./temp/original_sites_voxel/',atomlisttype='all element',a_list=None,whether_lattice_voxel=True,whether_sites_voxel=True)

