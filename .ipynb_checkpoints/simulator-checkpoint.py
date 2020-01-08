import numpy as np
import model
import h5py
import pickle
import time

# Layer properties
print("* Initialising calorimeter *")
passive_depth = 1
passive_X0 = 1
passive = model.Layer('lead', passive_X0, passive_depth, 1.0, 1, 0.0)
active_depth = 1
active_width = 32
num_cells = 32
print("Cell size = " + str(active_width/num_cells))
active = model.Layer('scin', 0.01, active_depth, active_width, num_cells, 1.0)

num_layers = 30
mycal = model.Calorimeter()
for i in range(num_layers):
    mycal.add_layers([passive, active])

counts_all_runs = []
en1 = np.array([0.1])
# en2 = np.arange(2.0, 12.0, 2.0)
en2 = np.array([20.0])
energies = np.append(en1, en2)
direct = "simulations/single_hits/"

energies_dict = {}

# some predefined particle properties
sigma = 0.3; num_runs = 500; x = 0; y = 0

for energy in energies:

    # Define calorimeter
    mycal.reset()
    print("* Initialising incident particle *")

    # particle properties
    electron = model.Electron(0.0, x, y, energy, 0, 0)

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
