from io import BytesIO
from PIL import Image
import wandb
import torch
import matplotlib.pyplot as plt

# Encoded data and RBM
def rbm_energy_hist(engine, model_config, val_loader, reducedata=False):
    partition_size = model_config.n_latent_nodes_per_p
    energy_encoded_data = []
    energy_rbm_data = []

    engine.model.eval()
    with torch.no_grad():
        for xx in val_loader:
            in_data, true_energy, in_data_flat = engine._preprocess(xx[0],xx[1])
            if reducedata:
                in_data = engine._reduce(in_data, true_energy, R=engine.R)
            # enIn = torch.cat((in_data, true_energy), dim=1)
            # beta, post_logits, post_samples = engine.model.encoder(enIn, False)
            beta, post_logits, post_samples = engine.model.encoder(in_data, true_energy, False)
            post_samples = torch.cat(post_samples, 1)
            post_samples_energy = engine.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                     post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )
            energy_encoded_data.append(post_samples_energy.detach().cpu())

            #RBM
            p1, p2, p3, p4 = engine.model.sampler.block_gibbs_sampling()
            rbm_samples_energy = engine.model.stater.energy_samples(p1, p2, p3, p4, 1.0)
            energy_rbm_data.append(rbm_samples_energy.detach().cpu())

    energy_encoded_data = torch.cat(energy_encoded_data, dim=0)
    energy_rbm_data = torch.cat(energy_rbm_data, dim=0)
    return energy_encoded_data, energy_rbm_data

def rbm_energy_hist_cond(engine, model_config, val_loader, reducedata=False):
    partition_size = model_config.n_latent_nodes_per_p
    energy_encoded_data = []
    energy_rbm_data = []

    engine.model.eval()
    with torch.no_grad():
        for xx in val_loader:
            in_data, true_energy, in_data_flat = engine._preprocess(xx[0],xx[1])
            if reducedata:
                in_data = engine._reduce(in_data, true_energy, R=engine.R)
            # enIn = torch.cat((in_data, true_energy), dim=1)
            # beta, post_logits, post_samples = engine.model.encoder(enIn, False)
            beta, post_logits, post_samples = engine.model.encoder(in_data, true_energy, False)
            post_samples = torch.cat(post_samples, 1)
            post_samples_energy = engine.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                     post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )
            energy_encoded_data.append(post_samples_energy.detach().cpu())

            #RBM
            u = engine.model.encoder.binary_energy(true_energy).to(dtype=torch.float32)
            p1, p2, p3, p4 = engine.model.sampler.block_gibbs_sampling_cond(u)
            rbm_samples_energy = engine.model.stater.energy_samples(p1, p2, p3, p4, 1.0)
            energy_rbm_data.append(rbm_samples_energy.detach().cpu())

    energy_encoded_data = torch.cat(energy_encoded_data, dim=0)
    energy_rbm_data = torch.cat(energy_rbm_data, dim=0)
    return energy_encoded_data, energy_rbm_data


def rbm_energy_hist_fcn(engine, model_config, val_loader, reducedata=False):
    partition_size = 512 #model_config.n_latent_nodes
    energy_encoded_data = []
    energy_rbm_data = []

    engine.model.eval()
    with torch.no_grad():
        for xx in val_loader:
            in_data, true_energy, in_data_flat = engine._preprocess(xx[0],xx[1])
            if reducedata:
                in_data = engine._reduce(in_data, true_energy, R=engine.R)
            enIn = torch.cat((in_data, true_energy), dim=1)
            beta, post_logits, post_samples = engine.model.encoder(enIn, False)
            # beta, post_logits, post_samples = engine.model.encoder(in_data, true_energy, False)
            post_samples = torch.cat(post_samples, 1)
            post_samples_energy = engine.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                     post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )
            energy_encoded_data.append(post_samples_energy.detach().cpu())

            #RBM
            p1, p2, p3, p4 = engine.model.sampler.block_gibbs_sampling()
            rbm_samples_energy = engine.model.stater.energy_samples(p1, p2, p3, p4, 1.0)
            energy_rbm_data.append(rbm_samples_energy.detach().cpu())

    energy_encoded_data = torch.cat(energy_encoded_data, dim=0)
    energy_rbm_data = torch.cat(energy_rbm_data, dim=0)
    return energy_encoded_data, energy_rbm_data


def plot_RBM_energy(energy_encoded_data, energy_rbm_data, _wandb=True, title=None, filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    plt.hist(energy_encoded_data.numpy(), bins=70, linewidth=2.5, color="b", density=True)
    plt.hist(energy_rbm_data.numpy(), bins=20, linewidth=2.5, color="cyan", density=True, fc=(1, 0, 1, 0.5))

    plt.xlabel("RBM Energy")
    plt.ylabel("PDF")
    plt.legend(["Encoded Data", "RBM"])
    plt.grid("True")
    if _wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=500)
        buf.seek(0)
        image = wandb.Image(Image.open(buf))
        buf.close()
        plt.close(fig)
        return image
    else:
        if title is not None:
            plt.gcf().suptitle(title)
        if filename is not None:
            plt.savefig(filename, facecolor='white')
        else:
            plt.show()
        plt.close()

    
    
def generate_rbm_energy_hist(engine, model_config, val_loader, _wandb=True):
    if "PRBMCNN" in model_config.model_type:
        energy_encoded_data, energy_rbm_data = rbm_energy_hist(engine, model_config, val_loader, reducedata=engine._config.reducedata)
        image = plot_RBM_energy(energy_encoded_data, energy_rbm_data, _wandb)
    elif "PRBMFCN" in model_config.model_type:
        energy_encoded_data, energy_rbm_data = rbm_energy_hist_fcn(engine, model_config, val_loader, reducedata=engine._config.reducedata)
        image = plot_RBM_energy(energy_encoded_data, energy_rbm_data, _wandb)
    elif "QVAE" in model_config.model_type:    
        energy_encoded_data, energy_rbm_data = rbm_energy_hist_cond(engine, model_config, val_loader, reducedata=engine._config.reducedata)
        image = plot_RBM_energy(energy_encoded_data, energy_rbm_data, _wandb)
    return image