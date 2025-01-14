import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

imagefile = 'train_data/train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

# Normalize pixel values
imagearray = imagearray / 255.0

# Split the dataset
def split_data(data, split):
    split = np.array(split)
    split = split / np.sum(split)
    cum_split = np.cumsum(split)
    split_data = np.split(data, (cum_split[:-1] * len(data)).astype(int))
    return split_data

train_data, val_data, test_data = split_data(imagearray, (0.8, 0.1, 0.1))

# Visualize some samples
def visualize_samples(data, num_samples):
    fig, ax = plt.subplots(1, num_samples, figsize=(20, 5))
    for i in range(num_samples):
        ax[i].imshow(data[i], cmap='gray')
        ax[i].axis('off')
    plt.show()


visualize_samples(train_data, 5)

# Save the dataset
np.save('train_data/train_data.npy', train_data)
np.save('train_data/val_data.npy', val_data)
np.save('train_data/test_data.npy', test_data)


# Flip the images
def flip_images(data, axis):
    return np.flip(data, axis)

# Rotate the images
def rotate_images(data, angle):
    return np.rot90(data, angle)








