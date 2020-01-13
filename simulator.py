import numpy as np
import model
import h5py
import pickle
import time

# Layer properties
print("* Initialising calorimeter *")

num_cells = 30
mycal = model.Calorimeter()
active_component = [0.01, 1.0, 1.0]
passive_component = [1, 1.5, 0.0]
# component = [material density, cell_height, response]
depth = [40, 100]
short = model.Layer('short',depth[0], num_cells, passive_component, active_component)
long = model.Layer('long', depth[1], num_cells, passive_component, active_component)
mycal.add_layers([short, long])

counts_all_runs = []
en1 = np.array([0.1])
en2 = np.arange(2.0, 22.0, 2.0)
energies = np.append(en1, en2)
direct = "simulations/single_hits/"

energies_dict = {}

# some predefined particle properties
sigma = 0.05; num_runs = 10; x = 0; y = 0

for energy in energies:

    # Define calorimeter
    mycal.reset()
    print("* Initialising incident particle *")

    # particle properties
    electron = model.Electron(0.0, x, y, energy, [0.01,0.15,np.sqrt(1-(0.01**2+0.15**2))])

    print("Energy: ", energy)
    print("* ...SIMULATING... *")
    # Run simulation
    sim = model.Simulation(mycal)
    tic = time.time()
    # counts by layer
    _ , counts_layers_run = sim.simulate(electron, sigma, num_runs)
    toc = time.time()

    nested_dict = {"Energy": energy, "num_runs": num_runs,
                  "enterx": x, "entery": y, "sigma": sigma, "time_taken": toc-tic}
    energies_dict[str(energy)] = nested_dict

    print("* SIMULATION DONE! *")
    print("That took " + str(toc-tic) + " seconds")

    # Define directory
    data_filename = '%.1fGeV_%iruns_data.h5' %(energy,num_runs)
    data_directory = direct + data_filename

    # Save data from every run
    f = h5py.File(data_directory, "w")
    f.create_dataset('dataset_1', dtype='f', data=counts_layers_run)
    f.close()
    print("* Data saved! *")

    dict_filename = '%.1fGeV_%iruns_dict.p' %(energy,num_runs)
    dict_direct = direct + dict_filename
    with open(dict_direct, 'wb') as handle:
        pickle.dump(energies_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('* Dictionary saved! *')
