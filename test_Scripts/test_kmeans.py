import numpy as np
from PIL import Image

def kmeans_quantization(image, k):
    # Convert the input image to grayscale
    image = image.convert('L')
    # Convert the image to a numpy array
    pixels = np.array(image)
    # Flatten the array
    flat_pixels = pixels.flatten()
    # Initialize the centroids randomly
    centroids = np.random.choice(flat_pixels, k)
    # Iterate until convergence
    while True:
        # Assign each pixel to the nearest centroid
        clusters = [[] for i in range(k)]
        for pixel in flat_pixels:
            distances = [np.abs(pixel - centroid) for centroid in centroids]
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(pixel)
        # Update the centroids
        new_centroids = [np.mean(cluster) for cluster in clusters]
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    # Assign each pixel to its final cluster
    final_clusters = [[] for i in range(k)]
    for pixel in flat_pixels:
        distances = [np.abs(pixel - centroid) for centroid in centroids]
        nearest_centroid = np.argmin(distances)
        final_clusters[nearest_centroid].append(pixel)
    # Compute the new pixel values for each cluster
    new_pixels = np.zeros_like(flat_pixels)
    for i, cluster in enumerate(final_clusters):
        new_value = int(np.mean(cluster))
        new_pixels[np.isin(flat_pixels, cluster)] = new_value
    # Reshape the new pixel values to the original image shape
    new_pixels = new_pixels.reshape(pixels.shape)
    # Convert the new pixel values to an image
    new_image = Image.fromarray(new_pixels)
    return new_image

input_image = Image.open('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\images\\d_r_1_.jpg')
output_image = kmeans_quantization(input_image, 4)
output_image.save('output.jpg')

