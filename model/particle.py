import random
import numpy as np
import csv


class Particle:
    '''Base class for particles'''

    def __init__(self, name, z, x, y, energy, ionise, cutoff, p, d):
        self.name = name
        self.z = z
        self.energy = energy
        self.ionise = ionise
        self.cutoff = cutoff
        self.y = y
        self.x = x
        self.p = p
        self.d = d

    def move(self, layer, rad_lengths, z_in_layer):
        location = layer.cellpart(self)
        if location == 'o':
            material = layer._material_o
        else:
            material = layer._material_i
        boundary = layer.dist_to_boundary(self, z_in_layer)
        step = min(boundary, rad_lengths * material)
        #print("step and coords:",step, self.x, self.y, self.z, self.p)
        self.z += step * self.p[2]
        self.x += step * self.p[0]
        self.y += step * self.p[1]
        self.d += step
        #print("coords after step:",self.x, self.y, self.z)
        return step

    def interact(self):
        '''This should implement the model for interaction.
        The base class particle doesn't interact at all'''
        return [self]

    def offset(self, std):
        offset_val1 = np.random.multivariate_normal((0.0, 0.0), [[std, 0], [0, std]])
        offset_val2 = np.random.multivariate_normal((0.0, 0.0), [[std, 0], [0, std]])
        # print('Offset =', offset)
        return np.array([offset_val1, offset_val2])

    def unit_vector(self, vectors):
        new_vecs = np.array([np.zeros(3)])
        #print('new_vecs empty:',new_vecs)
        #print(vectors)
        for v in vectors:
            mag = np.linalg.norm(v)
            #print(v,mag)
            new_vecs = np.append(new_vecs, np.array([v / mag]), axis=0)
            #print('new_vecs:',new_vecs)
        #print(new_vecs)
        return new_vecs[1:]

    def __str__(self):
        return f'{self.name:10} z:{self.z:.3f} x:{self.x:.3f} y:{self.y:.3f} E:{self.energy:.3f}'


class Electron(Particle):

    def __init__(self, z, x, y, energy, p, d):
        super(Electron, self).__init__('elec', z, x, y, energy, True, 0.01, p, d)

    def interact(self, std):
        '''An electron radiates xangle photon. Make the energy split evenly.'''
        particles = []

        if self.energy > self.cutoff:

            split = random.random()
            new = self.offset(std)
            p = self.p
            angles = np.array([[p[0] / p[2], p[1] / p[2]]] * 2)
            angles += new

            c = np.array([1.0])
            non_unit_p = np.append(np.array([np.append(angles[0],c, axis=0)]),np.array([np.append(angles[1],c,axis=0)]),axis=0)
            new_p_dir = self.unit_vector(non_unit_p)

            particles = [Electron(self.z, self.x, self.y, split*self.energy, list(new_p_dir[0]), self.d),
                        Photon(self.z, self.x, self.y, (1.0-split)*self.energy, list(new_p_dir[1]), self.d)]
        return particles


class Photon(Particle):

    def __init__(self, z, x, y, energy, p, d):
        super(Photon, self).__init__('phot', z, x, y, energy, False, 0.01, p, d)

    def interact(self, std):
        '''A photon splits into an electron and xangle positron. Make the energy split randomly.'''
        particles = []
        if self.energy > self.cutoff:

            split = random.random()
            new = self.offset(std)
            p = self.p
            angles = np.array([[p[0] / p[2], p[1] / p[2]]] * 2)
            angles += new

            c = np.array([1.0])
            non_unit_p = np.append(np.array([np.append(angles[0],c, axis=0)]),np.array([np.append(angles[1],c,axis=0)]),axis=0)
            new_p_dir = self.unit_vector(non_unit_p)

            particles = [Electron(self.z, self.x, self.y, split*self.energy, list(new_p_dir[0]), self.d),
                        Electron(self.z, self.x, self.y, (1.0-split)*self.energy, list(new_p_dir[1]), self.d)]

        return particles

# class Positron(Particle):


class Muon(Particle):

    def __init__(self, z, energy):
        super(Muon, self).__init__('muon', z, x, y, energy, True, 0.01)
