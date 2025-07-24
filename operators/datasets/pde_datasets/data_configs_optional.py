import os

DATASET_DICT = {}
SOURCE_PATH = '../datasets/post_datasets'


################################################################
# Classic NS2D Benchmarks FROM FNO
################################################################
name = 'ns2d_fno_1e-4'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-4_train.hdf5'),
    'test_path':  os.path.join(SOURCE_PATH, 'fno/ns2d_fno_1e-4_train.hdf5')
    }
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 1000
DATASET_DICT[name]['total_seq']  = 30
DATASET_DICT[name]['raw_res']    = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 20
DATASET_DICT[name]['in_res']     = (64, 64)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench NS2D Benchmarks
################################################################
name = 'ns2d_pdb_M1_eta1e-1_zeta1e-1'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench Turbulence PDE Benchmarks
################################################################
name = 'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 900
DATASET_DICT[name]['num_test']   = 20
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 900
DATASET_DICT[name]['num_test']   = 20
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench 3D CFD Benchmarks
################################################################
name = 'ns3d_pdb_M1_rand'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1_rand/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1_rand/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 90
DATASET_DICT[name]['num_test']   = 10
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128, 128)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128, 128)
DATASET_DICT[name]['downsample'] = (1, 1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns3d_pdb_M1e-1_rand'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1e-1_rand/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1e-1_rand/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 90
DATASET_DICT[name]['num_test']   = 10
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (128, 128, 128)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (128, 128, 128)
DATASET_DICT[name]['downsample'] = (1, 1, 1)
DATASET_DICT[name]['downsample_t'] = 1

name = 'ns3d_pdb_M1_turb'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1_turb/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/ns3d_pdb_M1_turb/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 540
DATASET_DICT[name]['num_test']   = 60
DATASET_DICT[name]['total_seq']  = 21
DATASET_DICT[name]['raw_res']    = (64, 64, 64)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 11
DATASET_DICT[name]['in_res']     = (64, 64, 64)
DATASET_DICT[name]['downsample'] = (1, 1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEBench 2D SWE & 2D DR Benchmarks
################################################################
name = 'swe_pdb'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdebench/swe_pdb/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdebench/swe_pdb/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 900
DATASET_DICT[name]['num_test']   = 60
DATASET_DICT[name]['total_seq']  = 101
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 91
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
# DATASET_DICT[name]['downsample_t'] = 5
DATASET_DICT[name]['downsample_t'] = 1

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
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 91
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
# DATASET_DICT[name]['downsample_t'] = 10
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEArena 1D Kuramoto-Sivashinsky Benchmarks
################################################################
name = 'ks1d_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdearena/ks1d_pda/ks1d_pda_fixed_train.hdf5'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdearena/ks1d_pda/ks1d_pda_fixed_test.hdf5')
    }
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['num_train']  = 2048
DATASET_DICT[name]['num_test']   = 128
DATASET_DICT[name]['total_seq']  = 140
DATASET_DICT[name]['raw_res']    = (256,)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 640
DATASET_DICT[name]['in_res']     = (256,)
DATASET_DICT[name]['downsample'] = (1,)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# PDEArena Compressible & Incompressible 2D NS Benchmarks
################################################################
name = 'ns2d_cond_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'pdearena/ns2d_cond_pda/train'),
    'test_path':  os.path.join(SOURCE_PATH, 'pdearena/ns2d_cond_pda/test')
    }
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['num_train']  = 3100
DATASET_DICT[name]['num_test']   = 200
DATASET_DICT[name]['total_seq']  = 56
DATASET_DICT[name]['raw_res']    = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 46
DATASET_DICT[name]['in_res']     = (128, 128)
DATASET_DICT[name]['downsample'] = (1, 1)
# DATASET_DICT[name]['downsample_t'] = 2
DATASET_DICT[name]['downsample_t'] = 1

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
DATASET_DICT[name]['num_train']  = 7000
DATASET_DICT[name]['num_test']   = 400
DATASET_DICT[name]['total_seq']  = 88
DATASET_DICT[name]['raw_res']    = (96, 192)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 78
DATASET_DICT[name]['in_res']     = (96, 192)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1

################################################################
# CFDBench Compressible 2D NS Benchmarks
################################################################
name = 'cfdbench'
DATASET_DICT[name] = {
    'train_path': os.path.join(SOURCE_PATH, 'cfdbench/ns2d_cdb_train.hdf5'),
    'test_path':  os.path.join(SOURCE_PATH, 'cfdbench/ns2d_cdb_test.hdf5')
    }
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['num_train']  = 9000
DATASET_DICT[name]['num_test']   = 1000
DATASET_DICT[name]['total_seq']  = 20
DATASET_DICT[name]['raw_res']    = (64, 64)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['in_seq']     = 10
DATASET_DICT[name]['test_seq']   = 10
DATASET_DICT[name]['in_res']     = (64, 64)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['downsample_t'] = 1
