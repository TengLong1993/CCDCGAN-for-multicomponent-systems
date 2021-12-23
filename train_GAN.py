import prepare.generate_voxel_from_structure as gvfs
import prepare.lattice_autoencoder as la
import prepare.sites_autoencoder as sa
import prepare.sites_autoencoder_seperated as sas
import data_processing_for_GAN as dpfg
import train_constraint as tc
import gan.ccdcgan as gan

##### 1. generate voxels from crystal structures
print('generate voxels from crystal structures')
gvfs.voxel_generation(text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt',cif_path='./database/20201009version/other_primitive_cell/',lattice_voxel_path='./calculation/original_lattice_voxel/',sites_voxel_path='./calculation/original_sites_voxel/',atomlisttype='all element',a_list=None,whether_lattice_voxel=True,whether_sites_voxel=True)

##### 2. generate crystal images for crystal structures
#### 2.1. train lattice autoencoder and generate encoded lattice
print('training lattice autoencoder')
la.find_best_autoencoder(lattice_voxel_path='./calculation/original_lattice_voxel/', model_folder_path='./calculation/autoencoder_model/lattice/', encoded_lattice_path='./calculation/original_encoded_lattice/', epochs=400, batch_size=1024, train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1,whether_training=True,whether_learning_rate_tune=True,learning_rate=0.0001,whether_encoded_generate=True,whether_from_text=True,text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt')
print('removing lattice voxel to release hard disk')
command='rm -r ./calculation/original_lattice_voxel/;'
os.system(command)
#### 2.2. train general sites autoencoder and generate encoded sites
print('training sites autoencoder')
sa.find_best_autoencoder(sites_voxel_path='./calculation/original_sites_voxel/', model_folder_path='./calculation/autoencoder_model/sites/', encoded_sites_path='./calculation/original_encoded_sites/', epochs=100, batch_size=1024, train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1,whether_training=True,whether_learning_rate_tune=True,learning_rate=0.00003,whether_encoded_generate=False,whether_from_text=True,text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt',whether_voxel_generate=False)
#### 2.3. train elemental sites autoencoder and generate seperated encoded sites
train_list=[1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
for element in train_list:
 print('element_number:',element)
 element_idx=element-1
 find_best_autoencoder(sites_voxel_path='./calculation/original_sites_voxel/', model_folder_path='./calculation/autoencoder_model/sites_seperated/', encoded_sites_path='./calculation/original_encoded_sites_seperated/', epochs=100, batch_size=256, train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1,whether_training=True,whether_learning_rate_tune=True,learning_rate=0.00003,whether_encoded_generate=True,whether_from_text=True,text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20_no_inert/'+str(element)+'.txt',whether_voxel_generate=False,element_index=element_idx)
print('removing sites voxel to release hard disk')
command='rm -r ./calculation/original_sites_voxel/;'
os.system(command)
#### 2.4. combine them as crystal images
print('generating crystal images')
dpfg.get_large_batch_graph(text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20_no_inert.txt',graph_path='./calculation/original_graph_seperated/',data_X_path='./calculation/data_to_train_GAN/',data_X_name='graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy')
print('removing encoded vectors to release hard disk')
command='rm -r ./calculation/original_encoded_sites/;'
os.system(command)
command='rm -r ./calculation/original_encoded_lattice/;'
os.system(command)

##### 3. train constraint model
#### 3.1. get property as output
print('generating property vector')
dpfg.get_data_y(text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20_no_inert.txt',property_name='formation_energy_per_atom',data_y_path='./calculation/data_to_train_constraint/',data_y_name='shuffled_formation_energy.npy')
#### 3.2. train constraint 
print('train constraint model')
tc.find_best_constrain_model(whether_from_trained_model=False, whether_training=True,whether_learning_rate_tune=True,learning_rate=3e-4,text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20_no_inert.txt', combined_graph_path='./calculation/data_preprocessing/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_shuffled_other_primitive_cell_selected_cL10_aN20_no_inert.npy', property_directory='./calculation/data_to_train_constraint/shuffled_formation_energy.npy',constrain_model_folder_path='./calculation/constrain_model/formation_energy/',constrain_model_name='formation_energy')

##### 4. train CCDCGAN
print('train ccdcgan model')
ccdcgan = gan.CCDCGAN(whether_from_trained_model=False, whether_formation_energy_constrained=True,trained_formation_energy_directory='./calculation/constrain_model/formation_energy/best/formation_energy.h5')
ccdcgan.train(text_directory='./database/20201009version/other_properties_primitive_cell_selected_cL10_aN20.txt', combined_graph_path='./calculation/data_preprocessing/data_to_train_GAN/', combined_graph_name='graphs_MP_20201009version_other_primitive_cell_selected_cL10_aN20.npy', GAN_model_folder_path='./calculation/dcgan_model/', learning_process_text_name='DCGAN_learning_curve.txt')

print('the training process has successfully finished')