import random
import numpy as np
import csv


class Particle:
    '''Base class for particles'''

    def __init__(self, name, z, x, y, energy, ionise, cutoff, p):
        self.name = name
        self.z = z
        self.energy = energy
        self.ionise = ionise
        self.cutoff = cutoff
        self.y = y
        self.x = x
        self.p = p

    def move(self, step):
        self.z += step * self.p[2]
        self.x += step * self.p[0]
        self.y += step * self.p[1]

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

    def __init__(self, z, x, y, energy, p):
        super(Electron, self).__init__('elec', z, x, y, energy, True, 0.01, p)

    def interact(self, std):
        '''An electron radiates xangle photon. Make the energy split evenly.'''
        particles = []

        if self.energy > self.cutoff:

            split = random.random()
            new = self.offset(std)
            p = self.p
            angles = np.array([[p[2] / p[0], p[2] / p[1]],
                               [p[2] / p[0], p[2] / p[1]]])
            angles[:][0] = (np.pi / 2) - angles[:][0]
            angles += new
            new_p_dir = np.array([1 - angles[:,0]**2 / 2), angles[:,0] * angles[:,1], angles[:,0] * (1 - angles[:,1]**2 / 2)])
            particles = [Electron(self.z, self.x, self.y, split*self.energy, new_p_dir[:,0]), Photon(self.z, self.x,
                            self.y, (1.0-split)*self.energy, new_p_dir[:,1])]
        return particles


class Photon(Particle):

    def __init__(self, z, x, y, energy, p):
        super(Photon, self).__init__('phot', z, x, y, energy, False, 0.01, p)

    def interact(self, std):
        '''A photon splits into an electron and xangle positron. Make the energy split randomly.'''
        particles = []
        if self.energy > self.cutoff:

            split = random.random()
            new = self.offset(std)
            p = self.p
            angles = np.array([[p[2] / p[0], p[2] / p[1]],
                               [p[2] / p[0], p[2] / p[1]]])
            angles[:][0] = (np.pi / 2) - angles[:][0]
            angles += new
            new_p_dir = np.array([1 - angles[:,0]**2 / 2), angles[:,0] * angles[:,1], angles[:,0] * (1 - angles[:,1]**2 / 2)])
            particles = [Electron(self.z, self.x, self.y, split*self.energy, new_p_dir[:,0]), Electron(self.z, self.x,
                            self.y, (1.0-split)*self.energy, new_p_dir[:,1])]

        return particles

# class Positron(Particle):


class Muon(Particle):

    def __init__(self, z, energy):
        super(Muon, self).__init__('muon', z, x, y, energy, True, 0.01)
