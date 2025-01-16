import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

imagefile = 'train_data/train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

# Normalize pixel values
imagearray = imagearray / 255.0

# Split the dataset
def split_data(data, split):
    """
    Split the dataset into multiple parts based on the given split ratios.

    Parameters:
    data (numpy.ndarray): The dataset to be split.
    split (tuple): A tuple containing the split ratios.

    Returns:
    list: A list of numpy arrays, each containing a portion of the dataset.
    """
    split = np.array(split)
    split = split / np.sum(split)
    cum_split = np.cumsum(split)
    split_data = np.split(data, (cum_split[:-1] * len(data)).astype(int))
    return split_data

train_data, val_data, test_data = split_data(imagearray, (0.8, 0.1, 0.1))

def visualize_samples(data, num_samples):
    """
    Visualize a specified number of samples from the dataset.

    Parameters:
    data (numpy.ndarray): The dataset containing image data.
    num_samples (int): The number of samples to visualize.

    Returns:
    None
    """
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
    """
    Flip the images along the specified axis.

    Parameters:
    data (numpy.ndarray): The dataset containing image data.
    axis (int): The axis along which to flip the images.

    Returns:
    numpy.ndarray: The flipped image data.
    """
    return np.flip(data, axis)

# Rotate the images
def rotate_images(data, rotation_num):
    """
    Rotate the images by the specified number of rotation by 90 degrees.

    Parameters:
    data (numpy.ndarray): The dataset containing image data.
    rotation_num (int): Number of rotations by 90 degrees

    Returns:
    numpy.ndarray: The rotated image data.
    """
    return np.rot90(data, rotation_num)








