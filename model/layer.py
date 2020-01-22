import random
import numpy as np
from scipy.stats import multivariate_normal

class Layer:
    '''Defines an individual layer of a calorimeter. The properties of the layer are
    name, its material given as X0 per cm, the thickness, the response measuring the
    level of ionisation (in arbitrary units, zero for passive layer). The layer can
    keep track of the ionisation in it.'''

    def __init__(self, name, thickness, numcells, material_outer, material_inner=[]):
        '''material_inner and material_outer in the form [material, cell_height, response]'''
        self._name = name
        self._material_o = material_outer[0]
        self._thickness = thickness
        self._yield_o = material_outer[2]
        self._ionisation = 0
        self._numcells = numcells
        self._cellsize = material_outer[1]
        self._cells = np.zeros((numcells, numcells))
        self._response_o = material_outer[2]
        self._height = numcells * material_outer[1]
        self._active = material_outer[2]
        if material_inner:
            self._two_mats = True
            self._material_i = material_inner[0]
            self._yield_i = material_inner[2]
            self._innersize = material_inner[1]
            self._response_i = material_inner[2]
            self._active = max(material_outer[2], material_inner[2])
        else:
            self._two_mats = False
        self._missed = 0.0

    def ionise(self, particle, step):
        '''Records the ionisation in each layer from a particle going a certain length.'''
        if particle.ionise:
            location = self.cellpart(particle)
            if location == 'o':
                response = self._response_o
            else:
                response = self._response_i
            # Treating it as a dot
            count = response * step
            self._ionisation += count

            # Treating it as a line, total ionisation in all cells should equal
            # previous value
            if response > 0:
                y = particle.y
                x = particle.x
                p = particle.p
                cellsize = self._cellsize
                h = self._height

                if x > h or x < 0 or y < 0 or y > h:
                    self._missed += count
                else:
                    if y % cellsize == 0:
                        y_ = int(np.floor(y / cellsize) - int(np.sign(p[1]) + 1) // 2)
                    else:
                        y_ = int(y / cellsize)
                    if x % cellsize == 0:
                        x_ = int(np.floor(x / cellsize) - int(np.sign(p[0]) + 1) // 2)
                    else:
                        x_ = int(x / cellsize)
                    #print(x_, y_, cellsize)
                    self._cells[y_, x_] += count

    def interact(self, particle, std, step):
        '''Let a particle interact (bremsstrahlung or pair production). The interaction
        length for photons if 9/7 times that of an electron.'''
        location = self.cellpart(particle)
        if location == 'o':
            material = self._material_o
        else:
            material = self._material_i
        particles = [particle]
        r = random.random()
        if particle.name == 'phot':
            material *= 9/7
        if r < step / material:
            particles = particle.interact(std)

        return particles

    def cellpart(self, particle):
        if self._two_mats:
            cellsize = self._cellsize
            innersize = self._innersize
            outersize = (cellsize - innersize) / 2
            y = particle.y
            x = particle.x
            p = particle.p
            ycell = y % cellsize
            xcell = x % cellsize
            if xcell > outersize and xcell < outersize + innersize:
                if ycell > outersize and ycell < outersize + innersize:
                    return 'i'
                if (ycell == outersize and p[1] > 0) or (ycell == outersize + innersize and p[1] < 0):
                    return 'i'
            if ycell > outersize and ycell < outersize + innersize:
                if (xcell == outersize and p[0] > 0) or (xcell == outersize + innersize and p[0] < 0):
                    return 'i'
        return 'o'

    def dist_to_boundary(self, particle, z):

        x, y, p = particle.x, particle.y, particle.p
        cellsize = self._cellsize
        boundary_dist = [0,0,(self._thickness - z) / p[2]]
        if self._two_mats:
            innersize = self._innersize
            outersize = (cellsize - innersize) / 2
            coords = [x,y]
            for i in range(2):
                if p[i] == 0:
                    boundary_dist[i] = boundary_dist[2] * 100
                    continue
                cell = coords[i] / cellsize
                if cell == 0:
                    dist = outersize
                elif cell < outersize:
                    dist = cell - outersize * ((np.sign(p[0]) + 1) / 2)
                elif cell == outersize:
                    dist = outersize + innersize * (np.sign(p[0]) + 1) / 2
                elif cell < innersize:
                    dist = cell - (outersize * ((np.sign(p[0]) + 1) / 2) + innersize)
                elif cell == outersize + innersize:
                    dist = outersize + (innersize - outersize) * (np.sign(p[0]) + 1) / 2
                else:
                    dist = innersize + outersize * (np.sign(p[0]) + 3) / 2 - cell
                boundary_dist[i] = abs(dist / p[i])
        else:
            xdist = x - self._height * (np.sign(p[0]) + 1) / 2
            ydist = y - self._height * (np.sign(p[1]) + 1) / 2
            if p[0] != 0:
                boundary_dist[0] = abs(xdist / p[0])
            else:
                boundary_dist[0] = boundary_dist[2] * 100
            if p[1] != 0:
                boundary_dist[1] = abs(ydist / p[1])
            else:
                boundary_dist[1] = boundary_dist[2] * 100
        fin = min(boundary_dist)
        if fin <= 10 ** (-5):
            print('floating point error:\nx:',x,'y:',y,'z:',z,'p:',p)
            fin += 0.0001
        return fin
    def __str__(self):
        return f'{self._name:10} {self._material:.3f} {self._thickness:.2f} {self._height:.2f} cm {self._ionisation:.3f}'
