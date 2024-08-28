import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from CaloQVAE import logging
logger = logging.getLogger(__name__)

import jetnet
from utils.plotting.HighLevelFeatures import HighLevelFeatures as HLF
# from jetnet.evaluation import fpd, kpd

def extract_shower_and_energy(given_file, which):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(which))
    if which == 0.:
        shower = given_file['showers'][:]
        energy = given_file['incident_energies'][:]
    else:
        shower = given_file['showers'][:]
        energy = given_file['incidence energy'][:]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower, energy

def prepare_high_data_for_classifier(test, e_inc, hlf_class, label):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """
    # voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    voxel, E_inc = test, e_inc
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

def check_and_replace_nans_infs(data):
    if np.isnan(data).any() or np.isinf(data).any():
        logger.info("Data contains NaNs or Infs. Handling them...")
        # Replace NaNs and Infs with zeros (or you can choose a different strategy)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def get_fpd_kpd_metrics(test_data, gen_data, syn_bool, hlf, ref_hlf):
    # print("TESTING HELLO")
    if syn_bool == True:
        data_showers = (np.array(test_data['showers']))
        energy = (np.array(test_data['incident_energies']))
        gen_showers = (np.array(gen_data['showers'], dtype=float))
        hlf.Einc = energy
    else:
        data_showers = test_data
        gen_showers = gen_data
    hlf.CalculateFeatures(data_showers)
    ref_hlf.CalculateFeatures(gen_showers)
    hlf_test_data = prepare_high_data_for_classifier(test_data, hlf.Einc, hlf, 0.)[:, :-1]
    hlf_gen_data = prepare_high_data_for_classifier(gen_data, hlf.Einc, ref_hlf, 1.)[:, :-1]
    hlf_test_data = check_and_replace_nans_infs(hlf_test_data)
    hlf_gen_data = check_and_replace_nans_infs(hlf_gen_data)
    fpd_val, fpd_err = jetnet.evaluation.fpd(hlf_test_data, hlf_gen_data)
    kpd_val, kpd_err = jetnet.evaluation.kpd(hlf_test_data, hlf_gen_data)
    
    result_str = (
        f"FPD (x10^3): {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f}\n" 
        f"KPD (x10^3): {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f}"
    )
    
    logger.info(result_str)
    return fpd_val, fpd_err, kpd_val, kpd_err


def search_HEPMetric(run_path, engine, dev, step = 10, data="caloqvae"):
    train_loader,test_loader,val_loader = engine.data_mgr.create_dataLoader()
    
    fn = create_filenames_dict(run_path, data)
    en_list = []
    fpd_recon, fpd_sample = [], []
    kpd_recon, kpd_sample = [], []
    hlf, ref_hlf = None, None
    
    R = engine._config.engine.r_param
    reducedata = engine._config.reducedata
    scaled=engine._config.data.scaled
    
    for i in range(step,(1+fn["size"])*step,step):

        _right_dir = get_right_dir(i, fn)
        # _pattern = get_right_pattern(i, fn)
        model_path = fn["prefix"] + "/" + _right_dir + '/files/'
        # modelCreator.load_state(model_path + get_right_pattern(i, fn), dev)
        full_path = model_path + get_right_pattern(i, fn)
        logger.info(f'Loading model {full_path}')
        engine.model_creator.load_state(full_path, dev)
        engine.model.eval();
        
        # get the samples
        xtarget_samples = []
        xrecon_samples = []
        xgen_samples = []
        xgen_samples_qpu = []
        n_samples4_qpu = 200
        entarget_samples = []
        for data_loader in [val_loader,test_loader,train_loader]:
            arr = gen_synth_data(engine, R, reducedata, data_loader)
            xtarget_samples.append(arr[0])
            xrecon_samples.append(arr[1])
            xgen_samples.append(arr[2])
            entarget_samples.append(arr[3])
            
            
        xtarget_samples = torch.cat(xtarget_samples, dim=0)
        xrecon_samples = torch.cat(xrecon_samples, dim=0)
        xgen_samples = torch.cat(xgen_samples, dim=0)
        # xgen_samples_qpu = torch.cat(xgen_samples_qpu, dim=0)
        entarget_samples = torch.cat(entarget_samples, dim=0)
        
        # xrecon_samples_2 = torch.cat(xrecon_samples_2, dim=0)
        if i == step:
            logger.info("First epoch")
            hlf = HLF('electron', filename=engine._config.data.binning_xml_electrons, wandb=False)
            ref_hlf = HLF('electron', filename=engine._config.data.binning_xml_electrons, wandb=False)
            hlf.Einc = entarget_samples

        recon_HEPMetrics = get_fpd_kpd_metrics(np.array(xtarget_samples), np.array(xrecon_samples), False, hlf, ref_hlf)
        sample_HEPMetrics = get_fpd_kpd_metrics(np.array(xtarget_samples), np.array(xgen_samples), False, hlf, ref_hlf)

        en_list.append(i)
        fpd_recon.append(recon_HEPMetrics[0])
        kpd_recon.append(recon_HEPMetrics[2])
        fpd_sample.append(sample_HEPMetrics[0])
        kpd_sample.append(sample_HEPMetrics[2])
        logger.info("Finished generating HEP Metrics for epoch " + str(i) + " ...")
    return en_list, fpd_recon, kpd_recon, fpd_sample, kpd_sample

def gen_synth_data(engine, R, reducedata, data_loader):
    xtarget_samples = []
    xrecon_samples = []
    xgen_samples = []
    xgen_samples_qpu = []
    entarget_samples = []
    
    with torch.no_grad():
        for xx in data_loader:
            in_data, true_energy, in_data_flat = engine._preprocess(xx[0],xx[1])
            ###############################################
            # true_energy = true_energy[:n_samples4_qpu,:]
            # in_data = in_data[:n_samples4_qpu,:]
            ##############################################
            # print(in_data.shape)
            if reducedata:
                in_data = engine._reduce(in_data, true_energy, R=R)
            fwd_output = engine.model((in_data, true_energy), False)
            if reducedata:
                in_data = engine._reduceinv(in_data, true_energy, R=R)
                recon_data = engine._reduceinv(fwd_output.output_activations, true_energy, R=R)
                engine._model.sampler._batch_size = true_energy.shape[0]
                if True:
                    sample_energies, sample_data = engine._model.generate_samples_cond(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True)
                    # sample_energies_qpu, sample_data_qpu = engine.model.generate_samples_qpu_cond(true_energy=true_energy, num_samples=1, thrsh=30, beta=1/beta0)
                else:
                    sample_energies, sample_data = engine._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True)
                    # sample_energies_qpu, sample_data_qpu = engine._model.generate_samples_qpu(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True, beta=1/beta0)
                engine._model.sampler._batch_size = engine._config.engine.rbm_batch_size
                sample_data = engine._reduceinv(sample_data, sample_energies, R=R)
                # sample_data_qpu = engine._reduceinv(sample_data_qpu, sample_energies_qpu, R=R)
            elif scaled:
                in_data = torch.tensor(engine._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                recon_data = torch.tensor(engine._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))
                # recon_data_2 = torch.tensor(engine._data_mgr.inv_transform(fwd_just_act.detach().cpu().numpy()))
                # recon_data_2 = torch.tensor(engine._data_mgr.inv_transform(fwd_energy_shift.detach().cpu().numpy()))
                engine._model.sampler._batch_size = true_energy.shape[0]

                if True:
                    sample_energies, sample_data = engine._model.generate_samples_cond(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True)
                    # sample_energies_qpu, sample_data_qpu = engine.model.generate_samples_qpu_cond(true_energy=true_energy[:100,:], num_samples=1, thrsh=30, beta=1/beta0)
                else:
                    sample_energies, sample_data = engine._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True)
                    # sample_energies_qpu, sample_data_qpu = engine._model.generate_samples_qpu(num_samples=true_energy.shape[0], true_energy=true_energy, measure_time=True, beta=1/beta0)
                engine._model.sampler._batch_size = engine._config.engine.rbm_batch_size
                sample_data = torch.tensor(engine._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
                # sample_data_qpu = torch.tensor(engine._data_mgr.inv_transform(sample_data_qpu.detach().cpu().numpy()))
            else:
                in_data = in_data.detach().cpu()*1000
                recon_data = fwd_output.output_activations.detach().cpu()*1000
                engine._model.sampler._batch_size = true_energy.shape[0]
                sample_energies, sample_data = engine._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy)
                engine._model.sampler._batch_size = engine._config.engine.rbm_batch_size
                # sample_energies, sample_data = engine._model.generate_samples(num_samples=2048)
                sample_data = sample_data.detach().cpu()*1000


            xtarget_samples.append(in_data.detach().cpu())
            xrecon_samples.append( recon_data.detach().cpu())
            xgen_samples.append( sample_data.detach().cpu())
            # xgen_samples_qpu.append( sample_data_qpu.detach().cpu())
            entarget_samples.append(true_energy.detach().cpu())
            
        xtarget_samples = torch.cat(xtarget_samples, dim=0)
        xrecon_samples = torch.cat(xrecon_samples, dim=0)
        xgen_samples = torch.cat(xgen_samples, dim=0)
        # xgen_samples_qpu = torch.cat(xgen_samples_qpu, dim=0)
        entarget_samples = torch.cat(entarget_samples, dim=0)


    return xtarget_samples, xrecon_samples, xgen_samples, entarget_samples



def save_plot(HEPMetric_output, run_path):
    path = run_path.split('files')[0] + 'files/'
    en_list, fpd_recon, kpd_recon, fpd_sample, kpd_sample = HEPMetric_output[0], HEPMetric_output[1], HEPMetric_output[2], HEPMetric_output[3], HEPMetric_output[4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot fpd_recon and fpd_sample on ax1
    ax1.scatter(en_list, fpd_recon, color='blue', label='FPD Recon')
    ax1.scatter(en_list, fpd_sample, color='green', label='FPD Sample')
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('FPD Values')
    ax1.set_title('FPD Recon vs FPD Sample')
    ax1.legend()

    # Plot kpd_recon and kpd_sample on ax2
    ax2.scatter(en_list, kpd_recon, color='blue', label='KPD Recon')
    ax2.scatter(en_list, kpd_sample, color='green', label='KPD Sample')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('KPD Values')
    ax2.set_title('KPD Recon vs KPD Sample')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path + f'KPD_and_FPD.png')
    # plt.show()
    np.savez(path + 'JetData.npz', array1=en_list, array2=fpd_recon, array3=kpd_recon, array4=fpd_sample, array5=kpd_sample)

def create_filenames_dict(run_path, data="caloqvae"):
    pattern = r'\d+.pth$'
    if data=="caloqvae":
        filenames = {}
        file = run_path.split("/")[-3]
        # filenames[file] = list(np.sort(os.listdir(run_path.split("files")[0] + f'files/RBM/')))
        fn_ = list(np.sort(os.listdir(run_path.split("wandb")[0] + f'wandb/{file}/files/')))
        filenames[file] = [word for word in fn_ if re.search(pattern, word)]
        filenames["size"] = int(len(filenames[file]))
        # filenames["prefix"] = run_path.split("outputs_sym")[0] + "outputs_sym"
        filenames["prefix"] = run_path.split("wandb")[0] + "wandb"
    else:
        filenames = {}
        files = os.listdir(run_path.split("wandb")[0] + "wandb")
        trueInd = [ "run" in file for file in files]
        for i, file in enumerate(files):
            if trueInd[i] and "latest" not in file:
                try:
                    fn_ = list(np.sort(os.listdir(run_path.split("wandb")[0] + f'wandb/{file}/files/')))
                    # filenames[file] = [word for word in fn_ if word.endswith('pth')]
                    filenames[file] = [word for word in fn_ if re.search(pattern, word)]
                    
                    
                except:
                    logger.warning(f'Directory {run_path.split("wandb")[0]}' + f'wandb/{file}/files/ might not exist.')


        list_of_files = []
        for key in filenames.keys():
            list_of_files = list_of_files + filenames[key]
        filenames["size"] = int(len(list_of_files))
        filenames["prefix"] = run_path.split("wandb")[0] + "wandb"
        filenames = {key: value for key, value in filenames.items() if value}
    return filenames

def get_right_dir(i, filenames):
    pattern = get_right_pattern(i, filenames)
    logger.info(f'Model {pattern}')
    
    for key in filenames.keys():
        # if f'RBM_{i}_9_weights.pth' in filenames[key]:
        if pattern in filenames[key]:
            _right_dir = key
            break
    return _right_dir

def get_right_pattern(i, filenames):
    first_key = list(filenames)[0]
    pattern = filenames[first_key][-1].split('default_')[0] + f'default_{i}.pth'
    return pattern