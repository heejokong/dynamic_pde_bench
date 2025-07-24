import os
from operators.datasets.generator import preprocess_fno_benchmark, preprocess_pdebench_cfd_2d, \
            preprocess_pdebench_swe_2d, preprocess_pdebench_dr_2d, preprocess_pdebench_cfd_3d, \
            preprocess_pdearena_ns2d, preprocess_pdearena_swe, preprocess_cfdbench


if __name__ == '__main__':

    # DATA_PATH = "/home/scv1/hj_kong/physics_ml/operator_works/datasets/prev_datasets"
    # SAVE_PATH = "/home/scv1/hj_kong/physics_ml/operator_works/datasets/post_datasets"
    DATA_PATH = "/home/hj_kong/physics_ml/operator_works/datasets/prev_datasets"
    SAVE_PATH = "/home/hj_kong/physics_ml/operator_works/datasets/post_datasets"

    prepare_fno = False
    prepare_pdearena = True # False
    prepare_cfdbench = False

    prepare_pdebench_cfd_2d = False
    prepare_pdebench_cfd_3d = False # True
    prepare_pdebench_cfd_turb = False # True
    prepare_pdebench_swe_2d = False
    prepare_pdebench_dr_2d = False


    # %%
    #### FNO datasets ####
    if prepare_fno:
        d_path = os.path.join(DATA_PATH, "fno/ns_V1e-3_N5000_T50.mat")
        s_path = os.path.join(SAVE_PATH, "fno/ns2d_fno_1e-3")
        preprocess_fno_benchmark(load_path=d_path, save_path=s_path, n_train=4800, n_test=200)

        d_path = os.path.join(DATA_PATH, "fno/ns_V1e-4_N10000_T30.mat")
        s_path = os.path.join(SAVE_PATH, "fno/ns2d_fno_1e-4")
        preprocess_fno_benchmark(load_path=d_path, save_path=s_path, n_train=9000, n_test=1000)

        d_path = os.path.join(DATA_PATH, "fno/NavierStokes_V1e-5_N1200_T20.mat")
        s_path = os.path.join(SAVE_PATH, "fno/ns2d_fno_1e-5")
        preprocess_fno_benchmark(load_path=d_path, save_path=s_path, n_train=1000, n_test=200)


    # %%
    #### PDEBench 2D CFD datasets ####
    if prepare_pdebench_cfd_2d:
        # 
        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=9000, n_test=1000)
        
        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=9000, n_test=1000)
        
        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=9000, n_test=1000)

        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=9000, n_test=1000)

        # 
        # d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
        # s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512")
        # preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)
        
        # d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
        # s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512")
        # preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)


    # %%
    #### PDEBench 2D SWE datasets ####
    if prepare_pdebench_swe_2d:
        d_path = os.path.join(DATA_PATH, "pdebench/2D/shallow-water/2D_rdb_NA_NA.h5")
        s_path = os.path.join(SAVE_PATH, "pdebench/swe_pdb")
        preprocess_pdebench_swe_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)


    # %%
    #### PDEBench 2D DR datasets ####
    if prepare_pdebench_dr_2d:
        d_path = os.path.join(DATA_PATH, "pdebench/2D/diffusion-reaction/2D_diff-react_NA_NA.h5")
        s_path = os.path.join(SAVE_PATH, "pdebench/dr_pdb")
        preprocess_pdebench_dr_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)


    # %%
    #### PDEBench 2D Turbulence CFD datasets ####
    if prepare_pdebench_cfd_turb:
        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Turb/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)
        
        d_path = os.path.join(DATA_PATH, "pdebench/2D/CFD/2D_Train_Turb/2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512")
        preprocess_pdebench_cfd_2d(load_path=d_path, save_path=s_path, n_train=900, n_test=100)


    # %%
    #### PDEBench 3D CFD datasets ####
    if prepare_pdebench_cfd_3d:
        d_path = os.path.join(DATA_PATH, "pdebench/3D/Train/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns3d_pdb_M1_rand")
        preprocess_pdebench_cfd_3d(load_path=d_path, save_path=s_path, n_train=90, n_test=10)

        d_path = os.path.join(DATA_PATH, "pdebench/3D/Train/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns3d_pdb_M1_turb")
        preprocess_pdebench_cfd_3d(load_path=d_path, save_path=s_path, n_train=540, n_test=60)

        d_path = os.path.join(DATA_PATH, "pdebench/3D/Train/3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5")
        s_path = os.path.join(SAVE_PATH, "pdebench/ns3d_pdb_M1e-1_rand")
        preprocess_pdebench_cfd_3d(load_path=d_path, save_path=s_path, n_train=90, n_test=10)


    # %%
    #### PDEArena datasets ####
    if prepare_pdearena:
        # d_path = os.path.join(DATA_PATH, "pdearena/NavierStokes-2D")
        # s_path = os.path.join(SAVE_PATH, "pdearena/ns2d_pda")
        # preprocess_pdearena_ns2d(load_path=d_path, save_path=s_path)

        # d_path = os.path.join(DATA_PATH, "pdearena/NavierStokes-2D-conditoned")
        # s_path = os.path.join(SAVE_PATH, "pdearena/ns2d_cond_pda")
        # preprocess_pdearena_ns2d(load_path=d_path, save_path=s_path)

        d_path = os.path.join(DATA_PATH, "pdearena/ShallowWater-2D")
        s_path = os.path.join(SAVE_PATH, "pdearena/sw2d_pda")
        preprocess_pdearena_swe(load_path=d_path, save_path=s_path)


    # %%
    #### CFDBench datasets ####
    if prepare_cfdbench:
        d_path = os.path.join(DATA_PATH, "cfdbench")
        s_path = os.path.join(SAVE_PATH, "cfdbench")
        preprocess_cfdbench(load_path=d_path, save_path=s_path)

