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
                cellsize = self._cellsize
                midcell = int(np.floor(self._numcells/2))
                y_ = y/cellsize
                if (y_ <= 0.0 or y_ >= 1.0) and abs(y_) <= midcell:
                    ycell = int(np.floor(y_ + midcell))
                elif (y_ >= 0.0) and y_ <= 1.0:
                    ycell = midcell
                else:
                    ycell = self._numcells + 1

                x_ = x/cellsize
                if (x_ <= 0.0 or x_ >= 1.0) and abs(x_) <= midcell:
                    xcell = int(np.floor(x_ + midcell))
                elif (x_ >= 0.0) and x_ <= 1.0:
                    xcell = midcell
                else:
                    xcell = self._numcells + 1

                if abs(xcell) < self._numcells and abs(ycell) < self._numcells:
                    self._cells[ycell, xcell] += count
                else:
                    self._missed += count

    def interact(self, particle, std, step):
        '''Let a particle interact (bremsstrahlung or pair production). The interaction
        length is assumed to be the same for electrons and photons.'''
        location = self.cellpart(particle)
        if location == 'o':
            material = self._material_o * step
        else:
            material = self._material_i * step
        particles = [particle]
        r = random.random()

        if r < material:
            particles = particle.interact(std)

        return particles

    def cellpart(self, particle):
        if self._two_mats:
            cellsize = self._cellsize
            innersize = self._innersize
            outersize = (cellsize - innersize) / 2
            y = particle.y
            x = particle.x
            ycell = y % cellsize
            xcell = x % cellsize
            if (ycell <= outersize or ycell >= outersize + innersize) and (xcell <= outersize or xcell >= outersize + innersize):
                return 'i'
        return 'o'

    def __str__(self):
        return f'{self._name:10} {self._material:.3f} {self._thickness:.2f} {self._height:.2f} cm {self._ionisation:.3f}'
