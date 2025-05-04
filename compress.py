import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def download_image_from_url(url):
    # Sending a GET request to the image URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.93 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True)
    content_type = response.headers.get("Content-Type", "")
    
    # Ensure the content is an image
    if not content_type.startswith("image"):
        raise Exception(f"URL did not return an image. Content-Type: {content_type}")
    
    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise Exception(f"Downloaded file is not a valid image. Details: {e}")

def compress_image_with_pca(image_path_or_url, k=16, n_components=3):
    """
    Compress an image using PCA and visualizes the difference.
    """
    # Load the image
    image = download_image_from_url(image_path_or_url)

    # Get original image shape and prepare data for PCA
    original_shape = image.shape
    pixel_data = image.reshape((-1, 3))  # Flatten the image into 2D array

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, 3))
    reduced_data = pca.fit_transform(pixel_data)

    # Inverse PCA to reconstruct the image
    reconstructed_pixels = pca.inverse_transform(reduced_data)
    compressed_image = np.clip(reconstructed_pixels, 0, 255).astype(np.uint8)
    compressed_image = compressed_image.reshape(original_shape)

    # Display images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (K={k}, PCA Components={n_components})")
    plt.imshow(compressed_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage with a valid direct image URL
if __name__ == "__main__":
    image_url = "https://images.unsplash.com/photo-1593642532973-d31b6557fa68"  # Valid image URL
    compress_image_with_pca(image_url, k=16, n_components=3)