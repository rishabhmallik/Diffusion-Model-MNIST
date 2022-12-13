# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Main dependencies
import numpy
import scipy.fftpack
import torch


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
                
        Example:
        
            print(fftind(5))
            
            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]
            
        """
    t = torch.arange(size)
    k_ind = torch.stack(torch.meshgrid(t, t)) - int( (size + 1)/2 )
    k_ind = torch.fft.fftshift(k_ind)
    
    return( k_ind )

def sigma_gauss(alpha, size):
    
        # Defines momentum indices
    k_idx = fftind(size)
    
        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    #amplitude = numpy.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    #amplitude[0,0] = 0
    
        # Defines the amplitude as a power law exp(-1/8*(alpha*|k|)^2)
    amplitude = torch.exp(-(alpha**2*(k_idx[0]**2 + k_idx[1]**2))/8.0)
    sigma = torch.fft.ifft2(amplitude).real
    return amplitude, sigma
    

def gaussian_random_field(alpha = 1.0, size = 128, flag_normalize = True, samples = 1, channels = 1):
    
    amplitude, _ = sigma_gauss(alpha, size)
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
        
    noise = torch.randn(size = (samples, channels, size ,size)) \
       + 1j * torch.randn(size = (samples, channels, size ,size))
    
        # To real space
    
    ft_gfield = noise * amplitude[None,None,:,:]
    gfield = torch.fft.ifft2(ft_gfield).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - torch.mean(gfield)
        gfield = gfield/torch.std(gfield)
         
    return gfield


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example[0], cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    main()





