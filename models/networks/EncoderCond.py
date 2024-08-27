"""
Encoder CNN Conditionalized

This encoder uses hierachies such that 
the binary energy is used as the first encoded partition.

"""
import torch
import torch.nn as nn  
from models.networks.hierarchicalEncoder import HierarchicalEncoder
import torch.nn.functional as F
import numpy as np

class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, 0,0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x

class EncoderBlockSmallPBHv2(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallPBHv2, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv2d(45, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
    
                   PeriodicConv2d(128, 512, (3,5), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv2d(513, 1024, (3,5), 1, 0),
                           nn.BatchNorm2d(1024),
                           nn.PReLU(1024, 0.02),

                           PeriodicConv2d(1024, self.n_latent_nodes, (3,4), 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        # self.seq3 = nn.Sequential(
        #                    nn.Linear(self.num_input_nodes, 4*self.n_latent_nodes),
        #                    nn.PReLU(4*self.n_latent_nodes, 0.02),
        #                    nn.Linear(4*self.n_latent_nodes, self.n_latent_nodes),
        #                    nn.Dropout(0.2),
        #             )
        self.seq3 = nn.Sequential( #northern sunset 406
                           nn.Linear(self.num_input_nodes, 4*self.n_latent_nodes),
                           nn.Linear(4*self.n_latent_nodes, self.n_latent_nodes),
                           nn.Dropout(0.2),
                    )
        # self.seq3 = nn.Sequential(
        #                    nn.Linear(self.num_input_nodes, self.n_latent_nodes),
        #                    nn.Dropout(0.2),
        #             )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        x = torch.cat([x] + post_samples,1)
        self.x_current = x
        x = self.seq3(x)
        
        return x
    
    
class EncoderBlockSmallPBHv2Small(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallPBHv2Small, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv2d(45, 64, (3,5), 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   PeriodicConv2d(64, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv2d(129, 256, (3,5), 1, 0),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv2d(256, self.n_latent_nodes, (3,4), 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        # self.seq3 = nn.Sequential(
        #                    nn.Linear(self.num_input_nodes, 4*self.n_latent_nodes),
        #                    nn.PReLU(4*self.n_latent_nodes, 0.02),
        #                    nn.Linear(4*self.n_latent_nodes, self.n_latent_nodes),
        #                    nn.Dropout(0.2),
        #             )
        self.seq3 = nn.Sequential( #swift-cosmos
                           nn.Linear(self.num_input_nodes, 4*self.n_latent_nodes),
                           nn.Linear(4*self.n_latent_nodes, self.n_latent_nodes),
                           nn.Dropout(0.2),
                    )
        # self.seq3 = nn.Sequential(
        #                    nn.Linear(self.num_input_nodes, self.n_latent_nodes),
        #                    nn.Dropout(0.2),
        #             )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        x = torch.cat([x] + post_samples,1)
        self.x_current = x
        x = self.seq3(x)
        
        return x
    
    
class EncoderBlockPBHv3(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockPBHv3, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv2d(45, 64, (3,5), 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   PeriodicConv2d(64, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv2d(129, 256, (3,5), 1, 0),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv2d(256, self.n_latent_nodes, (3,4), 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        pos_enc_samples = self._pos_enc(post_samples)
        x = x + pos_enc_samples.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)

#     def _pos_enc(self, post_samples):
#         post_samples = torch.cat(post_samples,1)
#         M = post_samples.shape[1]
#         post_samples_cpu = post_samples.clone().detach().cpu()

#         pres = [(torch.arange(0,M).multiply(i*np.pi/M).cos() * post_samples_cpu + torch.arange(0,M).multiply(i*np.pi/M).sin() *(1 - post_samples_cpu).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M-1,1)] # np.arange(1,M-1,1)
#         pos_enc = torch.cat(pres,2).transpose(1,2);
#         res = pos_enc.sum([1,2])
#         return res.unsqueeze(1).to(post_samples.device)
    
    
class EncoderBlockPBHv4(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockPBHv4, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv2d(45, 64, (3,4), 1, 1),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   PeriodicConv2d(64, 128, (3,3), (1,2), 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv2d(129, 256, (3,3), (1,2), 1),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv2d(256, self.n_latent_nodes, (3,4), 1, 0),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        pos_enc_samples = self._pos_enc(post_samples)
        x = x + pos_enc_samples.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M/4-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0):
        return (torch.log(x0) - log_e_min)/(log_e_max - log_e_min)

#     def _pos_enc(self, post_samples):
#         post_samples = torch.cat(post_samples,1)
#         M = post_samples.shape[1]
#         post_samples_cpu = post_samples.clone().detach().cpu()

#         pres = [(torch.arange(0,M).multiply(i*np.pi/M).cos() * post_samples_cpu + torch.arange(0,M).multiply(i*np.pi/M).sin() *(1 - post_samples_cpu).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M,M/4-1)]
#         pos_enc = torch.cat(pres,2).transpose(1,2);
#         res = pos_enc.sum([1,2])
#         return res.unsqueeze(1).to(post_samples.device)


class EncoderBlockPBHv4_0(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockPBHv4_0, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv2d(45, 64, (3,4), 1, 1),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   PeriodicConv2d(64, 128, (3,3), (1,2), 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv2d(129, 256, (3,3), (1,2), 1),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv2d(256, self.n_latent_nodes, (3,4), 1, 0),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        # pos_enc_samples = self._pos_enc(post_samples)
        x = x #+ pos_enc_samples.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M/4-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0):
        return (torch.log(x0) - log_e_min)/(log_e_max - log_e_min)
    
    
class EncoderHierarchyPB_BinEv2(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderHierarchyPB_BinEv2, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        
        layers = [self.n_latent_nodes + ((level+1)*self.n_latent_nodes)] + [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        if self.encArch == "SmallPB":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockSmallPBHv2(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv2":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockSmallPBHv2Small(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv3":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockPBHv3(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv4":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockPBHv4(layers[l], layers[l+1]))
        elif self.encArch == "SmallPB3Dv1":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockPBH3Dv1(layers[l], layers[l+1]))
        elif self.encArch == "SmallPB3Dv2":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockPBH3Dv2(layers[l], layers[l+1]))

        sequential = sequentialMultiInput(*moduleLayers)
        return sequential

    def forward(self, x, x0, is_training=True, beta_smoothing_fct=5):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent nodes in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        # logger.debug("ERROR Encoder::hierarchical_posterior")
        
        post_samples = []
        post_logits = []
        
        #loop hierarchy levels. apply previously defined network to input.
        #input is concatenation of data and latent variables per layer.
        
        post_samples.append(self.binary_energy(x0))
        
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            current_net=self._networks[lvl]
            # if type(x) is tuple:
            #     current_input=torch.cat([x[0]]+post_samples,dim=1)
            # else:
            #     current_input=torch.cat([x]+post_samples,dim=1)
            current_input = x

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0, post_samples), min=-88., max=88.)
            # logits=torch.clamp(current_net(current_input, x0, post_logits), min=-88., max=88.)
            
            # logits=torch.clamp(current_net(current_input, x0, post_samples), min=-10., max=10.)

            post_logits.append(logits)

            # Scalar tensor - device doesn't matter but made explicit
            # beta = torch.tensor(self._config.model.beta_smoothing_fct,
            #                     dtype=torch.float, device=logits.device,
            #                     requires_grad=False)
            beta = torch.tensor(beta_smoothing_fct,
                                dtype=torch.float, device=logits.device,
                                requires_grad=False)

            samples=self.smoothing_dist_mod(logits, beta, is_training)

            if type(x) is tuple:
                samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)

            post_samples.append(samples)
              
        return beta, post_logits, post_samples
    
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.bitwise_and(mask).ne(0).byte()
    
    def binary_energy(self, x, lin_bits=20, sqrt_bits=20, log_bits=20):
        reps = int(np.floor(self.n_latent_nodes/(lin_bits+sqrt_bits+log_bits)))
        residual = self.n_latent_nodes - reps*(lin_bits+sqrt_bits+log_bits)
        x = torch.cat((self.binary(x.int(),lin_bits), 
                       self.binary((x.sqrt() * torch.sqrt(torch.tensor(10))).int(),sqrt_bits), 
                       self.binary((x.log() * torch.tensor(10).exp()).int(),log_bits)), 1)
        return torch.cat((x.repeat(1,reps), torch.zeros(x.shape[0],residual).to(x.device, x.dtype)), 1)


    
class EncoderHierarchyPB_BinEv2Off(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderHierarchyPB_BinEv2Off, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        
        layers = [self.n_latent_nodes + ((level)*self.n_latent_nodes)] + [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        if self.encArch == "SmallPB":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockSmallPBHv2(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv2":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockSmallPBHv2Small(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv3":
            for l in range(len(layers)-1):
                moduleLayers.append(EncoderBlockPBHv3(layers[l], layers[l+1]))
        elif self.encArch == "SmallPBv4":
            if level != 0:
                for l in range(len(layers)-1):
                    moduleLayers.append(EncoderBlockPBHv4(layers[l], layers[l+1]))
            else:
                for l in range(len(layers)-1):
                    moduleLayers.append(EncoderBlockPBHv4_0(layers[l], layers[l+1]))
                    

        sequential = sequentialMultiInput(*moduleLayers)
        return sequential

    def forward(self, x, x0, is_training=True, beta_smoothing_fct=5):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent nodes in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        # logger.debug("ERROR Encoder::hierarchical_posterior")
        
        post_samples = []
        post_logits = []
        
        #loop hierarchy levels. apply previously defined network to input.
        #input is concatenation of data and latent variables per layer.
        
        # post_samples.append(self.binary_energy(x0))
        
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            current_net=self._networks[lvl]
            # if type(x) is tuple:
            #     current_input=torch.cat([x[0]]+post_samples,dim=1)
            # else:
            #     current_input=torch.cat([x]+post_samples,dim=1)
            current_input = x

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0, post_samples), min=-88., max=88.)
            # logits=torch.clamp(current_net(current_input, x0, post_logits), min=-88., max=88.)
            
            # logits=torch.clamp(current_net(current_input, x0, post_samples), min=-10., max=10.)

            post_logits.append(logits)

            # Scalar tensor - device doesn't matter but made explicit
            # beta = torch.tensor(self._config.model.beta_smoothing_fct,
            #                     dtype=torch.float, device=logits.device,
            #                     requires_grad=False)
            beta = torch.tensor(beta_smoothing_fct,
                                dtype=torch.float, device=logits.device,
                                requires_grad=False)

            samples=self.smoothing_dist_mod(logits, beta, is_training)

            if type(x) is tuple:
                samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)

            post_samples.append(samples)
              
        return beta, post_logits, post_samples
    
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.bitwise_and(mask).ne(0).byte()
    
    def binary_energy(self, x, lin_bits=20, sqrt_bits=20, log_bits=20):
        reps = int(np.floor(self.n_latent_nodes/(lin_bits+sqrt_bits+log_bits)))
        residual = self.n_latent_nodes - reps*(lin_bits+sqrt_bits+log_bits)
        x = torch.cat((self.binary(x.int(),lin_bits), 
                       self.binary((x.sqrt() * torch.sqrt(torch.tensor(10))).int(),sqrt_bits), 
                       self.binary((x.log() * torch.tensor(10).exp()).int(),log_bits)), 1)
        return torch.cat((x.repeat(1,reps), torch.zeros(x.shape[0],residual).to(x.device, x.dtype)), 1)
    
    
##########################
class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv3d, self).__init__()
        self.padding = padding
        # try 3x3x3 cubic convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        # Pad input tensor with periodic boundary and circle-center conditions
        if self.padding == 1:
            mid = x.shape[-1] // 2
            shift = torch.cat((x[..., [-1], mid:], x[..., [-1], :mid]), -1)
            x = torch.cat((x, shift), dim=-2)
        x = F.pad(x, (self.padding, self.padding, 0, 0, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x

class EncoderBlockPBH3Dv1(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockPBH3Dv1, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv3d(1, 64, (5,3,5), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),
    
                   PeriodicConv3d(64, 128, (5,3,3), (2,1,2), 1),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv3d(129, 256, (5,3,3), (2,1,2), 0),
                           nn.BatchNorm3d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv3d(256, self.n_latent_nodes, (3,3,3), (1,1,1), 0),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        # 1 channel of a 3d object / shower
        x = x.reshape(x.shape[0], 1, self.z, self.r,self.phi) 
        pos_enc_samples = self._pos_enc(post_samples)
        x = x + pos_enc_samples.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
            
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M/4-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

#     def _pos_enc(self, post_samples):
#         post_samples = torch.cat(post_samples,1)
#         M = post_samples.shape[1]
#         post_samples_cpu = post_samples.clone().detach().cpu()

#         pres = [(torch.arange(0,M).multiply(i*np.pi/M).cos() * post_samples_cpu + torch.arange(0,M).multiply(i*np.pi/M).sin() *(1 - post_samples_cpu).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M,M/4-1)]
#         pos_enc = torch.cat(pres,2).transpose(1,2);
#         res = pos_enc.sum([1,2])
#         return res.unsqueeze(1).to(post_samples.device)


class EncoderBlockPBH3Dv2(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockPBH3Dv2, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        
        self.seq1 = nn.Sequential(
                   # nn.Linear(self.num_input_nodes, 24*24),
                   # nn.Unflatten(1, (1,24, 24)),
    
                   PeriodicConv3d(1, 32, (3,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),
    
                   PeriodicConv3d(32, 64, (3,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConv3d(64, 128, (3,3,3), (1,1,2), 1),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv3d(129, 256, (3,3,3), (2,1,2), 0),
                           nn.BatchNorm3d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv3d(256, self.n_latent_nodes, (3,3,3), (1,2,1), 0),
                           nn.PReLU(self.n_latent_nodes, 1.0),
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        # 1 channel of a 3d object / shower
        x = x.reshape(x.shape[0], 1, self.z, self.r,self.phi) 
        pos_enc_samples = self._pos_enc(post_samples)
        x = x + pos_enc_samples.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
            
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M/4-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

#     def _pos_enc(self, post_samples):
#         post_samples = torch.cat(post_samples,1)
#         M = post_samples.shape[1]
#         post_samples_cpu = post_samples.clone().detach().cpu()

#         pres = [(torch.arange(0,M).multiply(i*np.pi/M).cos() * post_samples_cpu + torch.arange(0,M).multiply(i*np.pi/M).sin() *(1 - post_samples_cpu).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M,M/4-1)]
#         pos_enc = torch.cat(pres,2).transpose(1,2);
#         res = pos_enc.sum([1,2])
#         return res.unsqueeze(1).to(post_samples.device)