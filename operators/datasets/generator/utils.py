import os
import h5py
import pickle
import scipy
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from operators.datasets.generator.cfdbench import get_auto_dataset


################################################################
# For FNO NS2D Benchmarks
################################################################
def preprocess_fno_benchmark(load_path, save_path, n_train=9000, n_test=1000):
    try:
        data = h5py.File(load_path)
        data = np.array(data['u'])
        data = np.transpose(data, (3, 1, 2, 0))  ## N, X, Y, T
    except:
        data = scipy.io.loadmat(load_path)
        data = np.array(data['u'])  ## N, X, Y, T

    train_u = data[:n_train]
    test_u = data[n_train:n_train+n_test]
    print(train_u.shape, test_u.shape)

    pickle.dump(train_u, open(save_path + "_train.pkl", 'wb'))
    pickle.dump(test_u, open(save_path + "_test.pkl", 'wb'))

    def convert_pickle_to_hdf5(fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            data = np.expand_dims(data, axis=-1)  ## N, X, Y, T, 1
        hdf5_name = os.path.splitext(fname)[0] + '.hdf5'
        with h5py.File(hdf5_name, 'w') as hf:
            hf.create_dataset('data', data=data)
        print("Conversion completed!")

    convert_pickle_to_hdf5(save_path + "_train.pkl")
    convert_pickle_to_hdf5(save_path + "_test.pkl")


################################################################
# For PDEBench 2D CFD Benchmarks
################################################################
def preprocess_pdebench_cfd_2d(load_path, save_path, n_train=9000, n_test=1000):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164693&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    with h5py.File(load_path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)   ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        print('Content loaded:', vx.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y ,C]
        data = np.stack([vx, vy, density, pressure], axis=-1).transpose(0,2,3,1,4)
        print(data.shape)   # B, X, Y, T, C
    del vx, vy, density, pressure

    ##
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids', train_ids)
    print('test ids', test_ids)

    os.mkdir(save_path)
    os.mkdir(save_path + '/train')
    os.mkdir(save_path + '/test' )
    print('path created')

    for i in range(n_train):
        with h5py.File(save_path + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)

    for i in range(n_test):
        with h5py.File(save_path + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

    print('file saved')


################################################################
# For PDEBench SWE Benchmarks
################################################################
def preprocess_pdebench_swe_2d(load_path, save_path, n_train=900, n_test=100):
    ## t: 0~ 5, [101], x, y: -1~1. [128]
    data = []
    with h5py.File(load_path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])

        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # [1000, 128, 128, 101, 2]
        print(data.shape)

    ##
    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    os.mkdir(save_path)
    os.mkdir(save_path + '/train')
    os.mkdir(save_path + '/test')
    print('path created')

    for i in range(n_train):
        with h5py.File(save_path + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)

    for i in range(n_test):
        with h5py.File(save_path + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

    print('file saved')


################################################################
# For PDEBench DR Benchmarks
################################################################
def preprocess_pdebench_dr_2d(load_path, save_path, n_train=900, n_test=100):
    ## t: 0~1, [101], x, y: -2.5~2.5 [128]
    data = []
    with h5py.File(load_path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])

        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # 1000,128,128,101,2
        print(data.shape)

    ##
    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    os.mkdir(save_path)
    os.mkdir(save_path + '/train')
    os.mkdir(save_path + '/test')
    print('path created')

    for i in range(n_train):
        with h5py.File(save_path + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)

    for i in range(n_test):
        with h5py.File(save_path + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

    print('file saved')


################################################################
# For PDEBench 3D CFD Benchmarks
################################################################
def preprocess_pdebench_cfd_3d(load_path, save_path, n_train=90, n_test=10):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164693&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    with h5py.File(load_path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        vz = f['Vz']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)
        print('Content loaded:', vx.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape, z.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y, Z ,C]
        data = np.stack([vx, vy, vz, density, pressure],axis=-1).transpose(0,2,3,4,1,5)
        print(data.shape)   # B, X, Y, T, C
    del vx, vy, vz, density, pressure

    ##
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids',train_ids)
    print('test ids',test_ids)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.mkdir(save_path + '/train')
    os.mkdir(save_path + '/test' )
    print('path created')

    for i in range(n_train):
        with h5py.File(save_path + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)

    for i in range(n_test):
        with h5py.File(save_path + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

    print('file saved')


################################################################
# For PDEArena NS2D Benchmarks
################################################################
def preprocess_pdearena_ns2d(load_path, save_path):
    """
    Preprocess the Navier-Stokes 2D dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 14, 3)
    """
    # 
    os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
    # 
    test_tot = 0
    train_tot = 0
    # Traverse the file in LOAD_PATH
    for root, dirs, files in os.walk(load_path):
        for file in tqdm(files):
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.h5'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in file:
                        key = 'test'
                        path = os.path.join(save_path, 'test')
                    elif 'train' in file:
                        key = 'train'
                        path = os.path.join(save_path, 'train')
                    elif 'valid' in file:
                        key = 'valid'
                        path = os.path.join(save_path, 'train')
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f[key]['u'][:]
                    vx = f[key]['vx'][:]
                    vy = f[key]['vy'][:]

                    out = np.stack([u, vx, vy], axis=-1)
                    out = np.transpose(out, (0, 2, 3, 1, 4))

                    # Create the destination file
                    for data in out:
                        if key == 'test':
                            idx = test_tot
                            test_tot += 1
                        else:
                            idx = train_tot
                            train_tot += 1
                        dst_file = 'data_{}.hdf5'.format(idx)
                        with h5py.File(os.path.join(path, dst_file), 'w') as g:
                            # Write data as a hdf5 dataset
                            # with key 'data'
                            g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue


################################################################
# For PDEArena SWE Benchmarks
################################################################
def preprocess_pdearena_swe(load_path, save_path):
    """
    Preprocess the Shallow Water dataset from PDEArena

    there are 5 channels in the dataset:
        u, v, div, vor, pres
    data shape: (N, 96, 192, 88, 5)
    """
    # 
    os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
    # 
    test_tot = 0
    train_tot = 0
    # Traverse the file in LOAD_PATH
    for root, dirs, files in tqdm(os.walk(load_path)):
        for file in files:
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.nc'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in root:
                        key = 'test'
                        path = os.path.join(save_path, 'test')
                    elif 'train' in root:
                        key = 'train'
                        path = os.path.join(save_path, 'train')
                    elif 'valid' in root:
                        key = 'valid'
                        path = os.path.join(save_path, 'train')
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f['u'][:]
                    u = u[:, 0, ...]
                    v = f['v'][:]
                    v = v[:, 0, ...]
                    div = f['div'][:]
                    div = div[:, 0, ...]
                    vor = f['vor'][:]
                    vor = vor[:, 0, ...]
                    pres = f['pres'][:]

                    # data = np.stack([u, v, div, vor, pres], axis=-1)
                    data = np.stack([vor, pres], axis=-1)
                    data = np.transpose(data, (1, 2, 0, 3))

                    # Create the destination file
                    if key == 'test':
                        idx = test_tot
                        test_tot += 1
                    else:
                        idx = train_tot
                        train_tot += 1
                    dst_file = 'data_{}.hdf5'.format(idx)
                    with h5py.File(os.path.join(path, dst_file), 'w') as g:
                        # Write data as a hdf5 dataset
                        # with key 'data'
                        g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue


################################################################
# For CFDBench Benchmarks
################################################################
def preprocess_cfdbench(load_path, save_path):
    train_data_cavity, dev_data_cavity, test_data_cavity = get_auto_dataset(
        data_dir=Path(load_path),
        data_name='cavity_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    train_data_cylinder, dev_data_cylinder, test_data_cylinder = get_auto_dataset(
        data_dir=Path(load_path),
        data_name='cylinder_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    train_data_tube, dev_data_tube, test_data_tube = get_auto_dataset(
        data_dir=Path(load_path),
        data_name='tube_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )

    cavity_lens = [data.shape[0] for data in train_data_cavity.all_features]
    cylinder_lens = [data.shape[0] for data in train_data_cylinder.all_features]
    tube_lens = [data.shape[0] for data in train_data_tube.all_features]

    train_cavity_feats, train_cylinder_feats, train_tube_feats = train_data_cavity.all_features, train_data_cylinder.all_features, train_data_tube.all_features
    test_cavity_feats, test_cylinder_feats, test_tube_feats = test_data_cavity.all_features, test_data_cylinder.all_features, test_data_tube.all_features

    train_feats = train_cavity_feats + train_cylinder_feats + train_tube_feats
    test_feats = test_cavity_feats + test_cylinder_feats + test_tube_feats

    print(cavity_lens)
    print(cylinder_lens)
    print(tube_lens)

    infer_steps = 20

    def split_trajectory(data_list, time_step, grid_size=64):
        traj_split = []
        for i, x in enumerate(data_list):
            T = x.shape[0]
            num_segments = int(np.ceil(T / time_step))
            padded_length = num_segments * time_step
            padded_array = np.zeros((padded_length, *x.shape[1:]))

            # Copy the original data into the padded array
            padded_array[:T, ...] = x

            # If needed, pad the last segment with the last frame of the original array
            if T % time_step != 0:
                last_frame = x[-1, ...]
                padded_array[T:, ...] = last_frame

            # Reshape the array into segments
            padded_array = F.interpolate(torch.from_numpy(padded_array),size=(grid_size,grid_size),mode='bilinear',align_corners=True).numpy()
            padded_array = padded_array.reshape((num_segments, time_step, *padded_array.shape[1:]))

            traj_split.append(padded_array)

        traj_split = np.concatenate(traj_split, axis=0)
        return traj_split

    train_data = split_trajectory(train_feats, infer_steps,grid_size=64)
    test_data = split_trajectory(test_feats, infer_steps,grid_size=64)
    train_data, test_data = train_data.transpose(0,3,4,1,2), test_data.transpose(0, 3, 4, 1, 2) # B, X, Y, T, C
    print(train_data.shape, test_data.shape)

    with h5py.File(os.path.join(save_path, 'ns2d_cdb_train.hdf5'),'w') as fp:
        fp.create_dataset('data',data=train_data,compression=None)

    with h5py.File(os.path.join(save_path, 'ns2d_cdb_test.hdf5'),'w') as fp:
        fp.create_dataset('data',data=test_data,compression=None)
