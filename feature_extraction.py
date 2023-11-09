from skimage import feature, io, color
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import skimage.measure
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


# technique 1
def rgb_histogram(images):
    # Create empty lists to store the histograms for each channel
    hist_r_list = []
    hist_g_list = []
    hist_b_list = []

    histograms = np.zeros((images.shape[0], 3, 256))

    for i, image in enumerate(tqdm(images)):
        
        try:
            # Split the image into color channels (R, G, B)
            r, g, b = cv2.split(image)

            # Create histograms for each channel
            hist_r = np.histogram(r, bins=256, range=(0, 256))[0]
            hist_g = np.histogram(g, bins=256, range=(0, 256))[0]
            hist_b = np.histogram(b, bins=256, range=(0, 256))[0]

            # Store the histograms in the array
            histograms[i, 0] = hist_r
            histograms[i, 1] = hist_g
            histograms[i, 2] = hist_b
            
        except Exception as e:
            print("An error occurred:", e)
            
    return histograms

# technique 2
def gray_reduced_images(images):
    
    # Create an empty array to store the processed images
    feature_images = np.empty((images.shape[0], 16, 16, 1), dtype=np.float32)

    for i, image in enumerate(tqdm(images)):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply pooling to the grayscale image
        pooled_image = skimage.measure.block_reduce(gray_image, (2, 2), np.mean)

        feature_images[i, :, :, 0] = pooled_image

    return feature_images

# technique 3
def lbp(images, radius=3, n_points=8):

    # Preallocate the lbp_images array
    feature_images = np.empty((images.shape[0], images.shape[1], images.shape[2], 1), dtype=np.float32)

    # create the lpb images
    for i, image in enumerate(tqdm(images)):

        # Convert the image to grayscale (Harris corner detection works on grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute LBP features
        feature_images[i, :, :, 0] = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    return feature_images


def plot_images(original_image, transformed_images, titles):
    plt.figure(figsize=(30, 60))
    
    # Plot the original image
    plt.subplot(1, len(transformed_images) + 1, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Plot the transformed images
    for i, (transformed_image, title) in enumerate(zip(transformed_images, titles)):
        plt.subplot(1, len(transformed_images) + 1, i + 2)
        plt.imshow(transformed_image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.show()

def grid_search_lbp(images, n_random, radius_values, n_points_values):
    
    # Randomly select 3 images for the grid search
    selected_indices = np.random.choice(range(images.shape[0]), n_random, replace=False)
    selected_images = images[selected_indices]
    
    for image in selected_images:
        transformed_images = []
        titles = []
        for radius in radius_values:
            for n_points in n_points_values:

                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                transformed_image = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')
                transformed_images.append(transformed_image)
                titles.append(f'radius: {radius}, n_points: {n_points}')
        
        # Plot the original and transformed images
        plot_images(image, transformed_images, titles)

        
def hcd(images):

    # Preallocate the feature_images array
    feature_images = np.empty((images.shape[0], images.shape[1], images.shape[2], 1), dtype=np.float32)

    # Perform Harris Corner Detection

    for i, image in enumerate(tqdm(images)):

        # Convert the image to grayscale (Harris corner detection works on grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute LBP features
        feature_images[i, :, :, 0] = cv2.cornerHarris(gray_image, blockSize=1, ksize=3, k=0.04)

    return feature_images



# Oriented FAST and Rotated BRIEF

# function to receive keypoints and descriptors
def orb_sift(images, method='orb'):
    
    # Create an ORB object
    if method == 'orb':
        obj = cv2.ORB_create(edgeThreshold=1)
    
    if method == 'sift':
        obj = cv2.xfeatures2d.SIFT_create()


    # Initialize lists to store keypoints and descriptors for all images
    all_keypoints = []
    all_descriptors = []

    for image in tqdm(images):
        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = obj.detectAndCompute(image_gray, None)

        # Append keypoints and descriptors to the lists
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    # Convert lists of keypoints and descriptors to NumPy arrays
    all_keypoints_array = np.array(all_keypoints, dtype=object)
    all_descriptors_array = np.array(all_descriptors, dtype=object)
    
    return all_descriptors_array


# Quantization and histogram creation
def histogram_visual_words(kmeans, descriptors, bins = 100, eps=1e-7):
    visual_word_histograms = []
    
    for i, image_features in enumerate(tqdm(descriptors)):
        
        if image_features is None:
            histogram, _ = np.histogram([-1], bins=bins)
            
        else:
            visual_words = kmeans.predict(image_features)
            histogram, _ = np.histogram(visual_words, bins=bins)
        
        visual_word_histograms.append(histogram)
        
    # normalize the histogram
    hist = np.vstack(visual_word_histograms).astype("float")

    # hist /= (hist.sum(axis=1)[:,np.newaxis] + eps)
            
    return hist


def calculate_distances_in_batches(X, batch_size):
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))  # Initialize the distance matrix

    for i in tqdm(range(0, n, batch_size), 'calculating distance matrix'):
        end = min(i + batch_size, n)
        batch = X[i:end]  # Take a batch of data points
        batch_distances = pairwise_distances(batch, X)  # Calculate distances for the batch
        distance_matrix[i:end] = batch_distances  # Update the distance matrix with the batch

    return distance_matrix


def silhouette_coefficient(images, labels, subsample=1000):
    
    # select only a subset of the data
    X, _, y, _ = train_test_split(images, labels, stratify=labels, train_size=subsample, random_state=42)
    
    # Calculate pairwise distances between data points
    batch_size = 1000  # Adjust this to your needs
    distance_matrix = calculate_distances_in_batches(X, batch_size)

    n = len(X)
    silhouette_values = []

    for i in tqdm(range(n), 'calculating silhouette coefficient'):
        # average distance from the data point i to the other data points in the same cluster.
        a = np.mean(distance_matrix[i][y == y[i]])
        # minimum average distance from the data point i to the data points in a different cluster, minimizing over clusters.
        b = min([np.mean(distance_matrix[i][y != label]) for label in set(y) if label != y[i]])
        s = (b - a) / max(a, b)
        silhouette_values.append(s)
    
    silhouette_values = np.array(silhouette_values)
    
    return silhouette_values, np.mean(silhouette_values)


def print_results(results):
    
    # Create a list to store the results
    results_list = []

    max_coef = -np.inf
    best_result = None

    for key, result in results.items():
        coef = np.round(result[1], 5)

        results_list.append({'Technique': key, 'Silhouette Coefficient': coef})

        if coef > max_coef:
            max_coef = coef
            best_result = key

    # Create a DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort the DataFrame by Silhouette Coefficient in descending order
    results_df = results_df.sort_values(by='Silhouette Coefficient', ascending=False)

    # Display the results
    print(results_df)

    print(70 * '-')
    print(f'Best results obtained by feature extraction technique {best_result}: {max_coef}')

    
def plot_sh_coeff(overview):

    # plot results
    # Set the ticks and labels
    x_values = overview.index
    silhouette_coeff_values = overview['silhouette_coeff']
    plt.bar(x=x_values, height=silhouette_coeff_values, edgecolor='black')


    x_labels = [str(i) for i in x_values]
    plt.xticks(x_values, x_labels)

    # add horizontal line
    plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Customize color, linestyle, and linewidth as needed

    # Annotate the bars with their values
    for x, z in zip(x_values, silhouette_coeff_values):
        if z > 0:
            plt.text(x, z+0.0002, str(round(z, 4)), ha='center', va='bottom', fontsize=8)
        else:
            plt.text(x, z-0.0002, str(round(z, 4)), ha='center', va='top', fontsize=8)

    # Add labels and a title
    plt.xlabel('Class')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient for individual classes')

    plt.show()