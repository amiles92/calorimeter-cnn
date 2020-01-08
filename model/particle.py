import random
import numpy as np
import csv


class Particle:
    '''Base class for particles'''

    def __init__(self, name, z, x, y, energy, ionise, cutoff, xangle, yangle):
        self.name = name
        self.z = z
        self.energy = energy
        self.ionise = ionise
        self.cutoff = cutoff
        self.y = y
        self.x = x
        self.xangle = xangle
        self.yangle = yangle

    def move(self, step):
        self.z += step

    def interact(self):
        '''This should implement the model for interaction.
        The base class particle doesn't interact at all'''
        return [self]

    def offset(self, std):
        offset_val1 = np.random.multivariate_normal((0.0, 0.0), [[std, 0], [0, std]])
        offset_val2 = np.random.multivariate_normal((0.0, 0.0), [[std, 0], [0, std]])
        # print('Offset =', offset)
        return offset_val1, offset_val2

    def __str__(self):
        return f'{self.name:10} z:{self.z:.3f} x:{self.x:.3f} y:{self.y:.3f} E:{self.energy:.3f}'


class Electron(Particle):

    def __init__(self, z, x, y, energy, xangle, yangle):
        super(Electron, self).__init__('elec', z, x, y, energy, True, 0.01, xangle, yangle)

    def interact(self, std):
        '''An electron radiates xangle photon. Make the energy split evenly.'''
        particles = []

        if self.energy > self.cutoff:

            split = random.random()
            new1, new2 = self.offset(std)
            xangle = self.xangle
            yangle = self.yangle

            particles = [Electron(self.z, self.x + new1[0] + xangle, self.y + new1[1] + yangle, split*self.energy, xangle, yangle), Photon(self.z, self.x + new2[0] + xangle,
                            self.y + new2[1] + yangle, (1.0-split)*self.energy, xangle, yangle)]
        return particles


class Photon(Particle):

    def __init__(self, z, x, y, energy, xangle, yangle):
        super(Photon, self).__init__('phot', z, x, y, energy, False, 0.01, xangle, yangle)

    def interact(self, std):
        '''A photon splits into an electron and xangle positron. Make the energy split randomly.'''
        particles = []
        if self.energy > self.cutoff:

            split = random.random()
            new1, new2 = self.offset(std)
            xangle = self.xangle
            yangle = self.yangle

            particles = [Electron(self.z, self.x + new1[0] + xangle, self.y + new1[1] + yangle, split*self.energy, xangle, yangle), Electron(self.z, self.x + new2[0] + xangle,
                            self.y + new2[1] + yangle, (1.0-split)*self.energy, xangle, yangle)]

        return particles

# class Positron(Particle):


class Muon(Particle):

    def __init__(self, z, energy):
        super(Muon, self).__init__('muon', z, x, y, energy, True, 0.01)
