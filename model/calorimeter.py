import copy
import numpy as np

class Calorimeter:
    '''This defines the calorimeter. The model is a strict one dimensinal model,
    where layers are positioned along the positive z direction and are imagined to
    stretch infinitely into the x and y directions.'''

    class Volume:
        '''A simple volume of the detector that has a layer starting at a given z position'''

        def __init__(self, z, layer):
            self.z = z
            self.layer = layer

    def __init__(self, volumes=[]):
        self._volumes = volumes.copy()
        self._zend = 0

    def add_layer(self, layer):
        '''Add a single layer to the back of the calorimeter.'''
        self._volumes.append(self.Volume(self._zend, copy.deepcopy(layer)))
        self._zend += layer._thickness

    def add_layers(self, layers):
        '''Add a list of layers, one after the other to the back of the calorimeter.'''
        for l in layers:
            self.add_layer(l)

    def step(self, particle, std, step):
        '''Move a particle by the amount step forward in the calorimeter,
        Return a list of particles created during
        the step. If particle doesn't do anything it is just stepped forward.'''

        involume = False
        for volume in self._volumes:
            if (particle.z >= volume.z) and (particle.z < volume.z + volume.layer._thickness):
                involume = True
                break

        # Looping through every layer to check if particle within that layer
        # If found, then break
        particle.move(step)

        particles = [particle]
        if involume:
            layer = volume.layer
            layer.ionise(particle, step)
            particles = layer.interact(particle, std, step)

        return particles

    def positions(self, active=True):
        '''Provide an array of the z coordinates for the start of each layer. If active=True, only return the active layers'''
        return np.array([v.z for v in self._volumes if not active or v.layer._active>0])

    def ionisations(self, active=True):
        '''Provide a list of the ionisation deposited in each of the layers. If active=True, only return the active layers'''
        return np.array([v.layer._ionisation for v in self._volumes if not active or v.layer._active>0])

    def ions_by_layer(self, active=True):

        return np.array([v.layer._cells for v in self._volumes if not active or v.layer._active>0])

    def ions_missed(self):
        missed = []
        for v in self._volumes:
                miss = v.layer._missed
                missed.append(miss)
        return missed

    def reset(self):
        '''Clears the recorded ionisation in each layer'''
        for v in self._volumes:
            v.layer._ionisation=0
            v.layer._cells = np.zeros((v.layer._numcells, v.layer._numcells))

    def __str__(self):
        txt = 'The layers of the calorimeter:\n'
        for volume in self._volumes:
            txt += f'{volume.z:.2f} ' + str(volume.layer) + '\n'
        return txt
