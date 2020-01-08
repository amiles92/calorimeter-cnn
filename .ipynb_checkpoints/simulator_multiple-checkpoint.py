from utils.multiple_hits_utils import save_multiple_hits
from utils.training_utils import get_images_single_hit, get_labels_single_hit
import numpy as np


en1 = np.array([20.0])
# en2 = np.arange(2.0, 42.0, 2.0)
en2 = np.array([32.0])
energies = np.append(en1, en2)
num_runs = 500
simulations_folder = 'simulations/single_hits/'

cents = []
for en1 in energies:
    print('Center energy:', en1)
    for en2 in energies:
        
        runs_dict = {}
        multiple_hits_images = np.zeros((num_runs, 32, 64, 64, 1))
        multiple_hits_labels = np.zeros((num_runs, 32, 64, 64, 1))
        num_iter = 0

        print('Outside energy:', en2)
        # Read images by run --> (N runs, K layers, 32, 32, 1)
        en1_images = get_images_single_hit(simulations_folder, en1, num_runs, add_noise=False)
        en1_labels = get_labels_single_hit(simulations_folder, en1, num_runs)
        en2_images = get_images_single_hit(simulations_folder, en2, num_runs, add_noise=False)

        if en1 == en2:
            random_mask = np.arange(num_runs)
            np.random.shuffle(random_mask)
            en2_images = en2_images[random_mask]

        # For every run corresponding to both energies, sum corresponding layers and cells
        r = 1.5
        for run in range(num_runs):
            num_iter += 1

            # Select images data from run
            en1_run_images = en1_images[run]
            en2_run_images = en2_images[run]
            # Get labels for first cluster only
            en1_run_labels = en1_labels[run]

            # New images with zero padding, extract 64 x 64 from this
            new_images = np.zeros((32, 128, 128, 1))
            new_labels = np.zeros((32, 128, 128, 1))

            # Limits of center --> (-32, 32)
            centx1, centy1 = 0, 0
            ix1 = centy1+64
            iy1 = centx1+64
            new_images[:, ix1-16:ix1+16, iy1-16:iy1+16] = en1_run_images
            new_labels[:, ix1-16:ix1+16, iy1-16:iy1+16] = en1_run_labels

            # Second cluster --> select random values for center within region
#                 centx2 = np.random.randint(low=-4, high=4, size=1)[0]
#                 centy2 = np.random.randint(low=-4, high=4, size=1)[0]
            theta = np.random.randint(low=0, high=360, size=1)[0]
            theta = theta * np.pi/180
            centx2 = np.round(r*np.cos(theta))
            centy2 = np.round(r*np.sin(theta))
            ix2 = int(centy2+64)
            iy2 = int(centx2+64)
            diff = np.sqrt(np.square(centx2)+np.square(centy2))
            new_images[:, ix2-16:ix2+16, iy2-16:iy2+16] += en2_run_images
            cents.append(diff)

            # Extract region for new images and labels
            new_images = new_images[:, 32:96, 32:96]
            new_images = np.expand_dims(new_images, axis=0)
            new_labels = new_labels[:, 32:96, 32:96]
            new_labels = np.expand_dims(new_labels, axis=0)

            multiple_hits_images[run] = new_images
            multiple_hits_labels[run] = new_labels

            nested_dict = {"run": run, "energy1": en1, "energy2": en2,
                           "particle 2 center": (centx2, centy2),
                           "center_differences": diff}
            runs_dict[str(num_iter-1)] = nested_dict

            if num_iter % 70 == 0:
                print('Radius:', r+0.5)
                r += 1
        
        # randomly shuffle multiple images and labels
        random_mask2 = np.arange(num_runs)
        np.random.shuffle(random_mask2)
        multiple_images = multiple_images[random_mask2]
        multiple_labels = multiple_labels[random_mask2]
    
        print('Multiple hits data size:', multiple_hits_images.shape)
        name = 'center_%.1fGeV_outside_%.1f_%iruns' %(en1, en2, num_runs)
        print(name)
        new_direct = 'simulations/multiple_hits/'
        save_multiple_hits(new_direct, name, multiple_hits_images, multiple_hits_labels, runs_dict)


