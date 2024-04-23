"""
Encoder CNN

Changes the way the encoder is constructed wrt V2.

"""
import torch
import torch.nn as nn  
from models.networks.hierarchicalEncoder import HierarchicalEncoder
import torch.nn.functional as F
import numpy as np

class EncoderUCNN(HierarchicalEncoder):
    def __init__(self, **kwargs):
        super(EncoderUCNN, self).__init__(**kwargs)
        self.minEnergy = 256.0
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 16, 3, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),
                )
        self.seq2 = nn.Sequential(
                           nn.Conv2d(17, 32, 3, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(32, 0.02),
                           nn.BatchNorm2d(32),
                        )
        
        self.seq3 = nn.Sequential(
                           nn.Conv2d(33, 64, 3, 1, 0),
                           nn.BatchNorm2d(64),
                           nn.PReLU(64, 0.02),
                        )
        
        self.seq4 = nn.Sequential(
                           nn.Conv2d(65, 128, 3, 1, 0),
                           nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(128, 0.02),
                           nn.BatchNorm2d(128),
                           
        
                           nn.Conv2d(128, 256, 3, 1, 0),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),
                           
        
                           nn.Conv2d(256, 512, 2, 1, 0),
                           nn.BatchNorm2d(512),
                           nn.PReLU(512, 0.02),
                           
            
                           nn.Conv2d(512, 1024, 2, 1, 0),
                           nn.MaxPool2d(2,stride=2),
                           nn.BatchNorm2d(1024),
                           nn.PReLU(1024, 0.02),
        
                            nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           # nn.PReLU(self.n_latent_nodes, 0.02),
                           nn.Sigmoid(),
                           # nn.BatchNorm2d(self.n_latent_nodes),
            
                           nn.Flatten(),
                        )
        
        
        return nn.Sequential(self.seq1, self.seq2)   #<--- this goes to an unused place

    def forward2(self, x, x0, is_training=True):
        """Overrides forward in HierarchicalEncoder
        :param level
        """
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq2(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq3(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq4(x)
        
        # x = self.sequential(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,22,22).divide(self.minEnergy).log2()), 1)
        # x = self.sequential2(x)
        return x

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
        # for lvl in range(self.n_latent_hierarchy_lvls):
            
        # current_net=self._networks[0]
        if type(x) is tuple:
            current_input=torch.cat([x[0]]+post_samples,dim=1)
        else:
            current_input=torch.cat([x]+post_samples,dim=1)

        # Clamping logit values
        # logits=torch.clamp(current_net(current_input), min=-88., max=88.)
        logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
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
    
    
class EncoderUCNNH(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderUCNNH, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        # layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
        #     list(self._config.model.encoder_hidden_nodes) + \
        #     [self.n_latent_nodes]
        layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
            [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        for l in range(len(layers)-1):
            # moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
            if self.encArch == 'Large':
                moduleLayers.append(EncoderBlock(layers[l], layers[l+1]))
            elif self.encArch == 'Small':
                moduleLayers.append(EncoderBlockSmall(layers[l], layers[l+1]))
            elif self.encArch == 'SmallUnconditioned':
                moduleLayers.append(EncoderBlockSmallUnconditioned(layers[l], layers[l+1]))
            elif self.encArch == 'SmallPosEnc':
                moduleLayers.append(EncoderBlockSmallPosEnc(layers[l], layers[l+1]))
            elif self.encArch == 'SmallPB':
                moduleLayers.append(EncoderBlockSmallPB(layers[l], layers[l+1]))
           

        # sequential = nn.Sequential(*moduleLayers)
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
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            current_net=self._networks[lvl]
            if type(x) is tuple:
                current_input=torch.cat([x[0]]+post_samples,dim=1)
            else:
                current_input=torch.cat([x]+post_samples,dim=1)

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0), min=-88., max=88.)
            # logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
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
    
    
    
class EncoderBlock(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlock, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 16, 3, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),
                )
        self.seq2 = nn.Sequential(
                           nn.Conv2d(17, 32, 3, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(32, 0.02),
                           nn.BatchNorm2d(32),
                        )
        
        self.seq3 = nn.Sequential(
                           nn.Conv2d(33, 64, 3, 1, 0),
                           nn.BatchNorm2d(64),
                           nn.PReLU(64, 0.02),
                        )
        
        self.seq4 = nn.Sequential(
                           nn.Conv2d(65, 128, 3, 1, 0),
                           nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(128, 0.02),
                           nn.BatchNorm2d(128),
                           
        
                           nn.Conv2d(128, 256, 3, 1, 0),
                           nn.BatchNorm2d(256),
                           nn.PReLU(256, 0.02),
                           
        
                           nn.Conv2d(256, 512, 2, 1, 0),
                           nn.BatchNorm2d(512),
                           nn.PReLU(512, 0.02),
                           
            
                           nn.Conv2d(512, 1024, 2, 1, 0),
                           nn.MaxPool2d(2,stride=2),
                           nn.BatchNorm2d(1024),
                           nn.PReLU(1024, 0.02),
        
                            nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           # nn.PReLU(self.n_latent_nodes, 0.02),
                           nn.Sigmoid(),
                           # nn.BatchNorm2d(self.n_latent_nodes),
            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq2(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq3(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq4(x)
        
        return x

class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
    
class EncoderBlockSmall(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmall, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   nn.Conv2d(64, 256, 3, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                )

        self.seq2 = nn.Sequential(
                           nn.Conv2d(257, 512, 3, 1, 0),
                           nn.BatchNorm2d(512),
                           nn.PReLU(512, 0.02),

                           nn.Conv2d(512, self.n_latent_nodes, 3, 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),

                           # nn.Sigmoid(),            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        
        return x
    
    
class EncoderBlockSmallUnconditioned(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallUnconditioned, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   nn.Conv2d(64, 256, 3, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                )

        self.seq2 = nn.Sequential(
                           nn.Conv2d(256, 512, 3, 1, 0),
                           nn.BatchNorm2d(512),
                           nn.PReLU(512, 0.02),

                           nn.Conv2d(512, self.n_latent_nodes, 3, 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),

                           # nn.Sigmoid(),            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        
        return x


class EncoderBlockSmallPosEnc(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallPosEnc, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   nn.Conv2d(64, 256, 3, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                )

        self.seq2 = nn.Sequential(
                           nn.Conv2d(256, 512, 3, 1, 0),
                           nn.BatchNorm2d(512),
                           nn.PReLU(512, 0.02),

                           nn.Conv2d(512, self.n_latent_nodes, 3, 1, 0),
                           # nn.BatchNorm2d(self.n_latent_nodes),
                           nn.PReLU(self.n_latent_nodes, 1.0),

                           # nn.Sigmoid(),            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        x = x0.unsqueeze(2).unsqueeze(3).repeat(1, x.size(1), x.size(2), x.size(3)).divide(1000.0) + x
        x = self.seq2(x)
        
        return x
    
    
    
class EncoderUCNNHPosEnc(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', dev=None, lz=45,ltheta=16,lr=9, pe=1, **kwargs):
        self.encArch = encArch
        if dev == None:
            self.cyl_enc = self._cylinder_pos_enc(lz,ltheta,lr,pe)
        else:
            self.cyl_enc = self._cylinder_pos_enc(lz,ltheta,lr,pe)
            self.cyl_enc = self.cyl_enc.to(dev)
        super(EncoderUCNNHPosEnc, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        # layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
        #     list(self._config.model.encoder_hidden_nodes) + \
        #     [self.n_latent_nodes]
        layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
            [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        for l in range(len(layers)-1):
            # moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
            if self.encArch == 'Large':
                moduleLayers.append(EncoderBlock(layers[l], layers[l+1]))
            elif self.encArch == 'Small':
                moduleLayers.append(EncoderBlockSmall(layers[l], layers[l+1]))
            elif self.encArch == 'SmallUnconditioned':
                moduleLayers.append(EncoderBlockSmallUnconditioned(layers[l], layers[l+1]))
            elif self.encArch == 'SmallPosEnc':
                moduleLayers.append(EncoderBlockSmallPosEnc(layers[l], layers[l+1]))
           

        # sequential = nn.Sequential(*moduleLayers)
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
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            current_net=self._networks[lvl]
            if type(x) is tuple:
                current_input=torch.cat([x[0] + self.cyl_enc]+post_samples,dim=1)
            else:
                current_input=torch.cat([x + self.cyl_enc]+post_samples,dim=1)

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0), min=-88., max=88.)
            # logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
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
    
    def _positional_encoding(self, pos, i):
        """
        Computes the positional encoding for a given position and model dimension.
        Arguments:
        pos -- a scalar or a vector of positions
        d_model -- the dimensionality of the model
        Returns:
        pe -- the positional encoding for the given position(s)
        """
        # pe = torch.zeros((len(pos), d_model))
        # for i in range(d_model):
        div_term = torch.exp(i * -torch.log(torch.tensor([10000.0])) / 3)
        if i % 2 == 0:
            # pe[:, i] = torch.sin(pos * div_term)
            pe = torch.sin(pos * div_term)
        else:
            # pe[:, i] = torch.cos(pos * div_term)
            pe = torch.cos(pos * div_term)
        # return pe.detach().sum(dim=1)
        return pe.detach()
    
    def _positional_encoding_2(self, pos, d_model):
        """
        Computes the positional encoding for a given position and model dimension.
        Arguments:
        pos -- a scalar or a vector of positions
        d_model -- the dimensionality of the model
        Returns:
        pe -- the positional encoding for the given position(s)
        """
        pe = torch.zeros((len(pos), d_model))
        for i in range(d_model):
            div_term = torch.exp(i * -torch.log(torch.tensor([10000.0])) / d_model)
            if i % 2 == 0:
                pe[:, i] = torch.sin(pos * div_term)
            else:
                pe[:, i] = torch.cos(pos * div_term)
        return pe.detach().sum(dim=1)

    def _cylinder_pos_enc(self, lz=512,ltheta=512, lr=512, pe=1):
        """
        In cylindrical coordinates, voxels are tagged as follows:
        idx_list = torch.tensor(range(6480))

        pos_z = idx_list // 144
        pos_theta = (idx_list - 144*pos_z) // 9
        pos_r = (idx_list - 144*pos_z) % 9
        """
        if pe == 1:
            PE_z = self._positional_encoding(torch.tensor(range(45)),lz).repeat_interleave(144)
            PE_theta = self._positional_encoding(torch.tensor(range(16)),ltheta).repeat_interleave(9).repeat(45)
            PE_r = self._positional_encoding(torch.tensor(range(9)),lr).repeat(16*45)
        else:
            PE_z = self._positional_encoding_2(torch.tensor(range(45)),lz).repeat_interleave(144)
            PE_theta = self._positional_encoding_2(torch.tensor(range(16)),ltheta).repeat_interleave(9).repeat(45)
            PE_r = self._positional_encoding_2(torch.tensor(range(9)),lr).repeat(16*45)

        return PE_z + PE_theta + PE_r
    
    
    
class EncoderBlockSmallPB(nn.Module):
    """
        This only works w/o hierachy levels currently
    """
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallPB, self).__init__()
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
        

    def forward(self, x, x0):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        
        return x

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x
    
class EncoderHierarchyPB(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderHierarchyPB, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        # layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
        #     list(self._config.model.encoder_hidden_nodes) + \
        #     [self.n_latent_nodes]
        layers = [self.n_latent_nodes + (level*self.n_latent_nodes)] + [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        for l in range(len(layers)-1):
            moduleLayers.append(EncoderBlockSmallPBH(layers[l], layers[l+1]))
            # if self.encArch == 'Large':
            #     moduleLayers.append(EncoderBlock(layers[l], layers[l+1]))
            # elif self.encArch == 'Small':
            #     moduleLayers.append(EncoderBlockSmall(layers[l], layers[l+1]))
            # elif self.encArch == 'SmallUnconditioned':
            #     moduleLayers.append(EncoderBlockSmallUnconditioned(layers[l], layers[l+1]))
            # elif self.encArch == 'SmallPosEnc':
            #     moduleLayers.append(EncoderBlockSmallPosEnc(layers[l], layers[l+1]))
            # elif self.encArch == 'SmallPB':
            #     moduleLayers.append(EncoderBlockSmallPB(layers[l], layers[l+1]))
            
           

        # sequential = nn.Sequential(*moduleLayers)
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
    

class EncoderBlockSmallPBH(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmallPBH, self).__init__()
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
        self.seq3 = nn.Sequential(
                           nn.Linear(self.num_input_nodes, self.n_latent_nodes),
                    )
        

    def forward(self, x, x0, post_samples):
        x = x.reshape(x.shape[0],self.z, self.r,self.phi) 
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        x = torch.cat([x] + post_samples,1)
        self.x_current = x
        x = self.seq3(x)
        
        return x
    
    
class EncoderHierarchyPB_BinE(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderHierarchyPB_BinE, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        # layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
        #     list(self._config.model.encoder_hidden_nodes) + \
        #     [self.n_latent_nodes]
        layers = [self.n_latent_nodes + (level*self.n_latent_nodes)] + [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        for l in range(len(layers)-1):
            moduleLayers.append(EncoderBlockSmallPBH(layers[l], layers[l+1]))

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
            
        
        post_samples.append(self.binary_energy(x0))   
        return beta, post_logits, post_samples
    
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.bitwise_and(mask).ne(0).byte()
    
    def binary_energy(self, x, lin_bits=20, sqrt_bits=10, log_bits=4):
        reps = int(np.floor(512/(lin_bits+sqrt_bits+log_bits)))
        residual = 512 - reps*(lin_bits+sqrt_bits+log_bits)
        x = torch.cat((self.binary(x.int(),lin_bits), self.binary(x.sqrt().int(),sqrt_bits), self.binary(x.log().int(),log_bits)), 1)
        return torch.cat((x.repeat(1,reps), torch.zeros(x.shape[0],residual).to(x.device, x.dtype)), 1)