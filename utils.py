import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


def load_svhn_dataset(train_path, test_path):
    train_data = scipy.io.loadmat(train_path)
    test_data = scipy.io.loadmat(test_path)
    
    train_images = train_data['X']
    train_labels = train_data['y'].ravel()
    test_images = test_data['X']
    test_labels = test_data['y'].ravel()

    train_images = np.transpose(train_images, (3, 0, 1, 2))
    test_images = np.transpose(test_images, (3, 0, 1, 2))
    
    train_labels[train_labels == 10] = 0
    test_labels[test_labels == 10] = 0

    return train_images, train_labels, test_images, test_labels


def visualize_class_distr(labels1, labels2, label_names=None):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot histogram for the first set of labels
    counts1, bins1, _ = axes[0].hist(labels1, bins=10, edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('# number')
    axes[0].set_title('Class Distribution (Set 1)')
    if label_names:
        axes[0].set_xticks(label_names)
    
    # Plot histogram for the second set of labels
    counts2, bins2, _ = axes[1].hist(labels2, bins=10, edgecolor='black')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('# number')
    axes[1].set_title('Class Distribution (Set 2)')
    if label_names:
        axes[1].set_xticks(label_names)
    
    # Add labels and a title
    if not label_names:
        for i, (count, x) in enumerate(zip(counts1, bins1)):
            axes[0].text(x + 0.5, count, str(int(count)), ha='center', va='bottom')
            axes[0].text(x + 0.5, count / 2, str(int(i)), ha='center', va='center')
        for i, (count, x) in enumerate(zip(counts2, bins2)):
            axes[1].text(x + 0.5, count, str(int(count)), ha='center', va='bottom')
            axes[1].text(x + 0.5, count / 2, str(int(i)), ha='center', va='center')

    # Set the x-axis ticks
    # plt.xticks(np.arange(1,10))

    # Show the plot
    plt.show()


def plot_images(images, labels, rows=3, cols=3):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(rows, cols)

    # Display the first images based on rows and columns
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            index = i * cols + j
            if index < len(images):
                ax.imshow(images[index])  # Display the image
                ax.set_title(f'Image {index + 1}, Class {labels[index]}')
                ax.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Ensure proper spacing
    plt.show()
    

def plot_transformed_images(images, feature_images, indices=None):
    # Set the number of images to display
    num_images = 3

    # Generate random indices if not provided
    if indices is None:
        indices = random.sample(range(len(images)), num_images)

    # Create subplots for each image
    fig, axes = plt.subplots(1, num_images, figsize=(16, 4))

    for i, idx in enumerate(indices):
        ax = axes[i]

        ax.imshow(images[idx])
        ax.set_title(f"Original Image {idx}")

    plt.show()

    # Create subplots for the transformed images
    fig, axes = plt.subplots(1, num_images, figsize=(16, 4))

    for i, idx in enumerate(indices):
        ax = axes[i]

        ax.imshow(feature_images[idx].squeeze(), cmap='gray')
        ax.set_title(f"Transformed Image {idx}")

    plt.show()


class Standardizer:
    def __init__(self, library='torch'):
        self.library = library
        self.mean = None
        self.std = None

    def fit(self, x):
        if self.library == 'torch':
            self.mean = torch.mean(x, dim=0)
            self.std = torch.std(x, dim=0)
        elif self.library == 'numpy':
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        else:
            raise ValueError("Library not supported. Use 'torch' or 'numpy'.")

        # Avoid division by zero
        self.std[self.std == 0] = 1

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted yet.")

        if self.library == 'torch' and not isinstance(x, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor.")
        elif self.library == 'numpy' and not isinstance(x, np.ndarray):
            raise TypeError("Input data must be a numpy.ndarray.")

        x_scaled = (x - self.mean) / self.std
        return x_scaled

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    
def create_vectors(images):

    return images.reshape(images.shape[0], -1)


def histogram(images, bins=64, eps=1e-7):
    
    images = images.reshape(images.shape[0], -1)
    hist = [np.histogram(image, bins=bins)[0] for image in images]
    
    # normalize the histogram
    hist = np.vstack(hist).astype("float")

    # hist /= (hist.sum(axis=1)[:,np.newaxis] + eps)
    
    return hist
