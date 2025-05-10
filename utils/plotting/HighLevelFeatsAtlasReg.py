import numpy as np
import torch
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import h5py

from io import BytesIO
from PIL import Image
import wandb

class HighLevelFeatures_ATLAS_regular:
    def __init__(self, particle, filename, relevantLayers=[0,1,2,3,12,13,14], wandb = True):
        """
        Initialize the PLT_ATLAS object.

        Parameters:
        - filename: Path to the raw HDF5 data file.
        - event_n: Event number to process.
        - relevantLayers: List of layer numbers to plot. Defaults to [0, 1, 2, 3, 12].
        """
        self.wandb = wandb
        self.relevantLayers = relevantLayers
        self.ATLAS_raw_dir = filename
        # self.data = h5py.File(self.ATLAS_raw_dir, 'r')
        self.bin_info()
        print(self.ATLAS_raw_dir)
        
    def bin_info(self):
        with h5py.File(self.ATLAS_raw_dir, 'r') as file:
            # List all groups
            # self.data = {}
            self.binsize_alpha = {}
            self.binstart_alpha = {}
            self.binsize_radius = {}
            self.binstart_radius = {}
            print("Keys: %s" % list(file.keys()))
            for key in file.keys():
                # self.data[key] = torch.tensor(np.array(file[key]))
                if "binsize_alpha_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binsize_alpha[layer] = torch.tensor(np.array(file[key])) # self.data[f"binsize_alpha_layer_{layer}"]
                if "binstart_alpha_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binstart_alpha[layer] = torch.tensor(np.array(file[key])) #self.data[f"binstart_alpha_layer_{layer}"]
                if "binsize_radius_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binsize_radius[layer] = torch.tensor(np.array(file[key])) #self.data[f"binsize_radius_layer_{layer}"]
                if "binstart_radius_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binstart_radius[layer] = torch.tensor(np.array(file[key])) #self.data[f"binstart_radius_layer_{layer}"]

    def get_sector_arrays(self, layer):
        # All in torch, then to numpy once
        r0 = self.binstart_radius[layer]
        r1 = r0 + self.binsize_radius[layer]
        a0 = torch.rad2deg(self.binstart_alpha[layer]).round()
        a1 = a0 + torch.rad2deg(self.binsize_alpha[layer]).round()
        e  = self.single_event_energy

        # Convert once
        return (
            r0.cpu().numpy(),
            r1.cpu().numpy(),
            a0.cpu().numpy(),
            a1.cpu().numpy(),
            e
        )

    def _make_equal_bin_transform(self, r0, r1):
        # Build sorted unique boundaries
        bounds = np.unique(np.concatenate([r0, r1]))
        N = len(bounds) - 1
        plot_bounds = np.linspace(0, 1, N+1)

        # For any radius array R, get indices of lower boundary
        def transform(R):
            idx = np.searchsorted(bounds, R, side="right") - 1
            # clip to valid range
            idx = np.clip(idx, 0, N-1)
            # fraction within each bin
            frac = (R - bounds[idx]) / (bounds[idx+1] - bounds[idx])
            return plot_bounds[idx] + frac*(plot_bounds[idx+1] - plot_bounds[idx])

        return transform

    def plot_calorimeter(self, ax, scale='equal_bin',
                         cmap="rainbow", norm=None, title=None):
        # Collect arrays
        r0, r1, a0, a1, e = self.get_sector_arrays(self.current_layer)

        # Setup normalization
        if norm is None:
            norm = LogNorm(vmin=max(e.min(),1e-4), vmax=max(e.max(),1e-4))

        # Precompute transform
        if scale=='equal_bin':
            transform = self._make_equal_bin_transform(r0, r1)
            r0p, r1p = transform(r0), transform(r1)
        else:
            transform = lambda R: R
            r0p, r1p = transform(r0), transform(r1)

        # Build all wedges and colors
        patches = []
        for inner, outer, start, end in zip(r0p, r1p, a0, a1):
            width = outer - inner
            patches.append(Wedge((0,0), outer, start, end, width=width))

        # Create and add a single PatchCollection
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="grey", linewidths=0.1)
        pc.set_array(e)
        ax.add_collection(pc)
        ax.grid(False)

        # Adjust limits
        Rmax = r1p.max()
        ax.set_xlim(-Rmax-0.1, Rmax+0.1)
        ax.set_ylim(-Rmax-0.1, Rmax+0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=15)

    def DrawSingleShower(self, data, filename=None, title=None, scale='equal_bin',
                         vmin=1e-4, vmax=1e4, cmap='rainbow'):
        """
        Plot all specified layers of the calorimeter for the given event in a composite figure.

        Parameters:
        - scale: 'linear' or 'equal_bin', determines the type of radial scale for all subplots.
        """
        num = len(self.relevantLayers)
        fig, axes = plt.subplots(1, num, figsize=(15,15), dpi=200)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        vox = 14*24
        for ax, layer, i in zip(axes, self.relevantLayers, range(num)):
            self.single_event_energy = data[i*vox:(i+1)*vox]
            self.current_layer = str(layer)
            self.plot_calorimeter(ax, scale=scale, cmap=cmap, norm=norm,
                                  title=f"Layer {layer}")
        # Add a single horizontal colorbar below the subplots
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal',
                     fraction=0.05, pad=0.1, label='Energy')
            
        if self.wandb:
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