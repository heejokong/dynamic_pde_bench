import os

DATASET_DICT = {}
SOURCE_PATH = '../datasets/post_datasets'


""" USING ON THIS PAPER """
################################################################
# Classic NS2D Benchmarks FROM FNO
################################################################
name = 'ns2d_fno_1e-5'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-5_train.hdf5'),
    'test_path':  os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-5_test.hdf5')
    }
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['num_train']  = 1000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 20
DATASET_DICT[name]['raw_res']    = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['pred_seq']   = 10
DATASET_DICT[name]['test_seq']   = 10
DATASET_DICT[name]['in_res']     = (64, 64)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns2d_fno_1e-3'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-3_train.hdf5'),
    'test_path':  os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-3_train.hdf5')
    }
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['num_train']  = 1000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 50
DATASET_DICT[name]['raw_res']    = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['pred_seq']   = 20
DATASET_DICT[name]['test_seq']   = 40
DATASET_DICT[name]['in_res']     = (64, 64)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench NS2D Benchmarks
################################################################
name = 'ns2d_pdb_M1_eta1e-2_zeta1e-2'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['pred_seq']   = 11
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench 2D DR Benchmarks
################################################################
name = 'dr_pdb'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/dr_pdb/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/dr_pdb/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 900
DATASET_DICT[name]['num_test']   = 100
DATASET_DICT[name]['total_seq']  = 101
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 2
DATASET_DICT[name]['in_seq']     = 20
# DATASET_DICT[name]['test_seq']   = 81
DATASET_DICT[name]['pred_seq']   = 50
DATASET_DICT[name]['test_seq']   = 50
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 10
# DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEArena Incompressible 2D NS Benchmarks
################################################################
name = 'ns2d_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdearena/ns2d_pda/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdearena/ns2d_pda/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 6500
DATASET_DICT[name]['num_test']   = 650
DATASET_DICT[name]['total_seq']  = 14
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 4
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEArena 2D SWE Benchmarks
################################################################
name = 'sw2d_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdearena/sw2d_pda/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdearena/sw2d_pda/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
# DATASET_DICT[name]['num_train']  = 7000
DATASET_DICT[name]['num_train']  = 5600
# DATASET_DICT[name]['num_test']   = 400
DATASET_DICT[name]['num_test']   = 10
DATASET_DICT[name]['total_seq']  = 88
DATASET_DICT[name]['raw_res']    = (96, 192)
# DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['n_channels'] = 2
DATASET_DICT[name]['in_seq']     = 16
# DATASET_DICT[name]['test_seq']   = 68
DATASET_DICT[name]['pred_seq']   = 40
DATASET_DICT[name]['test_seq']   = 40
DATASET_DICT[name]['in_res']     = (96, 192)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 8
# DATASET_DICT[name]['downsample_t'] = 1
