from __future__ import division
from opensimplex import OpenSimplex
import numpy as np
import random,math

"""
use Perlin noise to generate the next step
http://www.redblobgames.com/articles/noise/introduction.html

non-uniform random selections change what happens in each location 
in isolation but instead we want something
where the random selection in one location is related to to the 
random selections in nearby locations. This is called coherence

Lower frequencies make wider hills and higher frequencies make narrower hills.

The wavelength is a distance, measured in pixels or tiles or meters 
or whatever you use for your maps. wavelength = map_size / frequency
High wavelength (low frequency) and low wavelength (high frequency)

pink noise is the noise where low frequency has a little higher amplitude 
than high frequency), which is better for nature landscape
blue noise is the noise where low frequency has a little lower amplitude
than high frequency, which is better for city map

You need to use different seeds for the elevation and moisture noise.

You can make the noise better by modifying and combining them.
For example, if the noise function returns [2,-1,5], then you can say
the first position is 2, the second is 2 + 1 = 1, and the third
position is 1 + 5 = 6
You could also do the inverse, and use the differences between 
noise values. 
Or min, max, average
"""

# http://www.redblobgames.com/maps/terrain-from-noise/
# generate perlin noise for each point in the grid => [0.,1.]
def compute_noise(num_row=256, num_col=256, freq=10.):
    # XXX: frequency of the noise: the higher, the more variance appears in the grid
    # e.g. we are making a map:
    #      noise_val > threshold => there is a mountain there
    #      then with a high freq, the mountain distribution will be much denser

    noise_values = np.zeros((num_row, num_col), dtype=np.float)

    random.seed()
    seed=random.randint(0,255)
    gen = OpenSimplex(seed)
    def noise(nx, ny):
        assert(nx>=-0.5 and nx<=0.5)
        assert(ny>=-0.5 and ny<=0.5)

        # XXX: octaves
        """
        a*noise(m*x,n*y):
        a is amplitude
            which represent the height of our mountain on that position
        m and n are frequency
            which represent the density of our mountains on that position
        so this following example is to 
            mix big low frequency hills and small high frequency hills
        mountain_elevation[y][x] = 1 * noise(1 * nx, 1 * ny);
                                   + 0.5 * noise(2 * nx, 2 * ny);
                                   + 0.25 * noise(4 * nx, 2 * ny);
        """
        # noise_value = gen.noise2d(x=freq*nx, y=freq*ny)
        noise_value = 1. * gen.noise2d(x=1*freq*nx, y=1*freq*ny)\
                      +0.5 * gen.noise2d(x=2*freq*nx, y=2*freq*ny)\
                      +0.25 * gen.noise2d(x=4*freq*nx, y=2*freq*ny)
        # print noise_value 
        noise_value = noise_value / (1.+0.5+0.25)
        # print noise_value 
        assert(noise_value>=-1. and noise_value<=1.)
        # Rescale from -1.:+1. to 0.0:1.0
        noise_value = noise_value / 2.0 + 0.5

        # XXX: Redistribution
        # To make flat valleys, we can raise the hill elevation to a power
        # elevation[y][x] = Math.pow(noise_value, x);
        # Higher x values push middle elevations down into valleys 
        # and lower values pull middle elevations up towards mountain peaks.
        noise_value = math.pow(noise_value, 1.);
        # print noise_value
        return noise_value

    for y in range(num_row):
        for x in range(num_col):
            nx = x/num_col - 0.5
            ny = y/num_row - 0.5
            noise_values[y][x] = noise(nx, ny)
      
    # f = open('./noise', 'wt')
    # f.write('P2\n')
    # f.write(str(num_row)+' '+str(num_col)+'\n')
    # f.write('255\n')
    # for y in range(num_row):
        # for x in range(num_row):
            # f.write("%s\n" % int(noise_values[y][x] * 127.0 + 128.0))
    # f.close()

    return noise_values

# The constant a pushes everything up, b pushes the edges down, 
# and c controls how quick the drop off is.
def island_noise(center_x,center_y,num_row,num_col,a=0.05,b=1.,c=1.5):
    noise_values = compute_noise(num_row, num_col, freq=10.)
    # print noise_values
    for y in xrange(num_row):
        for x in xrange(num_col):
            nx = x/num_col - 0.5
            ny = y/num_row - 0.5
            d = np.linalg.norm([ny,nx])
            a = 0.05
            b = 1.
            c = 1.5
            multiply_approach = (noise_values[y][x]+a) * (1-b*math.pow(d,c))
            a = 0.1
            b = 0.3
            c = 2.
            add_approach = noise_values[y][x]+a - b*math.pow(d,c)
            noise_values[y][x] = multiply_approach
            noise_values[y][x] = add_approach
            noise_values[y][x] = (add_approach+multiply_approach) / 2.
    # print noise_values
    return noise_values








