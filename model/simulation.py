import copy
import numpy as np

class Simulation:
    '''A simulation is defined by a calorimeter. Then individual simulation runs can be created by
    running the same particle through the calorimter multiple times.'''
    def __init__(self, calorimeter):
        self._calorimeter = calorimeter

    def simulate(self, particle, std, number):
        '''Run a individual simulation. The ingoing particle is simulated going
        through the calorimeter "number" times. A 2D array is returned with the
        first axis the ionisation in the individual layers and the second corresponding to each
        new particle.'''
        ionisations = []
        ions_layers = []

        for i in range(number):

            self._calorimeter.reset()
            particles = [copy.copy(particle)]
            offvals = []
            iter = 0
            while iter < 1000:
                next = []
                for p in particles:
                    newparticles = self._calorimeter.step(p, std, 0.1)
                    next.extend(newparticles)
                particles = next
                iter += 1

            ionisations.append(self._calorimeter.ionisations())
            ions_layers.append(self._calorimeter.ions_by_layer())

        allionisations = np.stack(ionisations, axis=0)
        allionsbycells = np.stack(ions_layers, axis=0)
        return allionisations, allionsbycells

    def simulate_multiple(self, particles_list, std, numlayers, numcells):

        ionisations = np.zeros(numlayers)
        ions_layers = np.zeros((numlayers, numcells, numcells))

        for part in particles_list:

            particles = [copy.copy(part)]
            iter = 0
            while iter < 1000:
                next = []
                for p in particles:
                    newparticles = self._calorimeter.step(p, std, 0.1)
                    next.extend(newparticles)
                particles = next
                iter += 1

            new_ions = self._calorimeter.ionisations()
            ionisations = np.add(ionisations, new_ions)

            new_ionsbylayers = self._calorimeter.ions_by_layer()
            ions_layers = np.add(ions_layers, new_ionsbylayers)

        return ionisations, ions_layers
