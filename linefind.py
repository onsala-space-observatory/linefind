#!/usr/bin/env python

"""Tool to determine line / continuum channels in a cube."""

from __future__ import print_function, division
import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RADIUS = 0.8
DEFAULT_WIDTH = 0.1
DEFAULT_SIGMA = 5.75
DEFAULT_BOXCAR_WIDTH = 30

class LineFinder(object):
    """Class to identify lines / continuum in cubes."""

    def __init__(self, cube, radius=DEFAULT_RADIUS, width=DEFAULT_WIDTH, beam=''):
        self.data = fits.open(cube)[0].data
        if beam:
            self.data *= fits.open(beam)[0].data
        self.n_chan, self.n_dec, self.n_ra = self.data.shape[-3:]
        other_dims = self.data.shape[:-3]
        if not other_dims:
            # This is presumed to be the Stokes axis. If it doesn't exist, create.
            self.data = self.data[None,...]
            other_dims = (1,)
        if len(other_dims) > 1:
            raise ValueError('Unexpected cube shape. Should have 3 or 4 axes')
        self.n_stokes = other_dims[0]
        self.radius = radius
        self.width = width
        self.rms = None
        self.maxx = None
        self.snr = None
        self.peaks = [None] * self.n_stokes
        self.non_peaks = [None] * self.n_stokes
        self.peak_mask = [None] * self.n_stokes
        self.non_peak_mask = [None] * self.n_stokes
        self.annulus_1chan = None
        # We'll use this for plotting the annulus
        self.data_1chan = self.data[0,0,:,:].copy()

    def noise(self):
        """Return the per-channel noise in the cube."""
        dec_lin_vals = np.linspace(-1, 1, self.n_dec)
        ra_lin_vals = np.linspace(-1, 1, self.n_ra)
        dec_grid, ra_grid = np.meshgrid(dec_lin_vals, ra_lin_vals)
        # Construct a mask
        annulus = np.ones_like(ra_grid, dtype=np.bool)
        annulus[ra_grid**2 + dec_grid**2 > (self.radius - self.width/2)**2] = False
        annulus[ra_grid**2 + dec_grid**2 > (self.radius + self.width/2)**2] = True
        self.annulus_1chan = annulus
        annulus_cells = len(np.where(annulus == False)[0])
        annulus = annulus[None,...].repeat(self.n_chan, axis=0)[None,...]
        masked_data = np.ma.array(self.data, mask=annulus)
        # noise_vals will contain the non-masked data only. RA & DEC will be flattened.
        noise_vals = np.empty(shape=self.data.shape[:-2]+(annulus_cells,))
        for i in range(self.n_chan):
            noise_vals[-1,i,:] = masked_data[-1,i,:].compressed()
        # Compute the noise along the RA & DEC axis
        self.rms = noise_vals.std(axis=-1)
        return self.rms

    def plot_annulus(self):
        """Plot the annulus used to calculate the noise level."""
        if self.annulus_1chan is None:
            self.noise()
        self.data_1chan /= np.nanmax(self.data_1chan) - np.nanmin(self.data_1chan)
        self.data_1chan -= np.nanmin(self.data_1chan)
        img = self.data_1chan[...,None].repeat(3, axis=-1) # For RGB
        img[np.isnan(img)] = 1 # Change NaN's to white
        img[np.logical_not(self.annulus_1chan)] = [1, 0, 0] # Plot annulus in Red
        plt.imshow(img)
        plt.show()

    def max(self):
        """Return the per-channel max values in the cube."""
        tmp_shape = self.data.shape[:-2] + (-1,)
        data = self.data.reshape(tmp_shape)
        self.maxx = np.nan_to_num(data).max(axis=-1)
        return self.maxx

    def boxcar(self, line_chans, width, limit=True):
        """Return a padded version of the input array. This is a boxcar convolution."""
        kernel = np.arange(-width // 2 + 1, width // 2 + 1).reshape(-1, 1)
        out_padded = np.unique(line_chans.reshape(1, -1) + kernel)
        if limit:
            out_padded = out_padded[out_padded >= 0]
            out_padded = out_padded[out_padded < self.n_chan]
        return out_padded

    def find_lines(self, sigma=DEFAULT_SIGMA, pad=DEFAULT_BOXCAR_WIDTH, stokes_index=0):
        """Re-calculate the continuum and line channels.
        Use 'plot()' or 'spw_string()' to get results."""
        if self.rms is None:
            self.noise()
        if self.maxx is None:
            self.max()
        if self.snr is None:
            self.snr = self.maxx / self.rms
        tmp_peaks = np.where(self.snr[stokes_index] > sigma)[0]
        self.peaks[stokes_index] = self.boxcar(tmp_peaks, pad, limit=True)

        self.peak_mask[stokes_index] = np.zeros_like(self.snr[stokes_index], dtype=np.bool)
        self.peak_mask[stokes_index][self.peaks[stokes_index]] = True
        self.non_peak_mask[stokes_index] = np.logical_not(self.peak_mask[stokes_index])
        all_chans = np.arange(len(self.snr[stokes_index]))
        self.non_peaks[stokes_index] = all_chans[self.non_peak_mask[stokes_index]]

    def plot(self, inverse=False, residuals=False, stokes_index=0):
        """Plot the line channels (or continuum if inverse=True)."""
        if self.peaks[stokes_index] is None:
            raise RuntimeError('find_lines has not been called for this stokes')

        if residuals:
            residuals = self.snr[stokes_index].copy()
            # Use NaNs to create discontinuties in the line plot
            residuals[self.peaks[stokes_index]] = np.nan
            plt.plot(residuals)
        else:
            plt.plot(self.snr[stokes_index])
        # The -1 is a fudge to get the identified channels below SNR data:
        offset = self.snr[stokes_index][self.non_peak_mask[stokes_index]].mean() - 1
        if inverse:
            # Mark continuum channels
            y_offset = np.zeros_like(self.non_peaks[stokes_index]) + offset
            plt.plot(self.non_peaks[stokes_index], y_offset, '.')
        else:
            # Mark lines channels
            y_offset = np.zeros_like(self.peaks[stokes_index]) + offset
            plt.plot(self.peaks[stokes_index], y_offset, '.')
        plt.show()

    def spw_string(self, inverse=False, stokes_index=0):
        """Return a CASA SPW format string of the continuum channels
        (or lines if inverse=True)."""
        if self.peaks[stokes_index] is None:
            raise RuntimeError('find_lines has not been called for this stokes')
        if not inverse:
            return self.aggrigate(self.peaks[stokes_index])
        else:
            return self.aggrigate(self.non_peaks[stokes_index])

    @staticmethod
    def aggrigate(vals):
        """Generate a CASA spw string from a numpy array."""
        out_str = ''
        start = vals[0]
        i = 0 # initialise in case len(vals) == 1
        for i in range(1, len(vals)):
            if vals[i] != vals[i-1] + 1:
                if start == vals[i-1]:
                    out_str += '{};'.format(start)
                else:
                    out_str += '{}~{};'.format(start, vals[i-1])
                start = vals[i]
        if start == vals[i]:
            out_str += '{};'.format(start)
        else:
            out_str += '{}~{};'.format(start, vals[i])
        return out_str


def main():
    """Command line interface to ContFinder class."""
    parser = argparse.ArgumentParser(description='Identify lines / continuum in fits cubes.')
    parser.add_argument('-r', '--radius', type=float, default=DEFAULT_RADIUS,
                        help=('Radius of annulus for calculating noise (0.0 - 1.0). '
                              'Default: {}').format(DEFAULT_RADIUS))
    parser.add_argument('-w', '--width', type=float, default=DEFAULT_WIDTH,
                        help=('Width of annulus for calculating noise, (0.0 - 1.0). '
                              'Default: {}').format(DEFAULT_WIDTH))
    parser.add_argument('-b', '--beam', type=str, default='',
                        help=('Undo the specified beam by multiplying it by the cube. '
                              'This should be a FITS file in the same format as the data cube.'))
    parser.add_argument('-s', '--sigma', type=float, default=DEFAULT_SIGMA,
                        help=('SNR above which we declare a line. '
                              'Default: {}'.format(DEFAULT_SIGMA)))
    parser.add_argument('-p', '--pad', type=int, default=DEFAULT_BOXCAR_WIDTH,
                        help=('Number of channels to pad out lines. '
                              'Default: {}'.format(DEFAULT_BOXCAR_WIDTH)))
    parser.add_argument('-i', '--invert', action='store_true',
                        help='Do continuum instead of lines.')
    parser.add_argument('--annulus', action='store_true',
                        help='Plot the annulus used to determine noise.')
    parser.add_argument('--residual', action='store_true',
                        help='Plot the SNR of the continuum channels only.')
    parser.add_argument('fitsfile', type=str)
    args = parser.parse_args()

    line_finder = LineFinder(args.fitsfile, args.radius, args.width, args.beam)
    line_finder.find_lines(args.sigma, args.pad)
    if args.annulus:
        line_finder.plot_annulus()
    line_finder.plot(inverse=args.invert, residuals=args.residual)
    print(line_finder.spw_string(args.invert))


if __name__ == '__main__':
    main()
