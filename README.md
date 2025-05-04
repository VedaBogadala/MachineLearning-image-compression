# MachineLearning-image-compression

This project demonstrates image compression using two popular **machine learning** techniques — **Principal Component Analysis (PCA)** and **K-Means Clustering**. The script downloads an image from a given URL, reduces its dimensionality using PCA, compresses it by clustering pixel values with K-Means, and displays the original vs compressed images side by side.

## 🚀 Features

- 📥 Downloads any image directly from a web URL
- 🧠 Compresses images using **machine learning algorithms**
  - **Principal Component Analysis (PCA)** for dimensionality reduction
  - **K-Means Clustering** for pixel value compression
- 🖼️ Displays both original and compressed images using `matplotlib`

## 🧠 Machine Learning Concepts Used

### 🔹 Principal Component Analysis (PCA)
- PCA is an unsupervised **machine learning** algorithm used for reducing high-dimensional data.
- In this project, PCA is used to project RGB pixel data into a lower-dimensional space (typically 2 or 3 components).

### 🔹 K-Means Clustering
- K-Means is another unsupervised **machine learning** algorithm that partitions data into **K clusters** based on similarity.
- After reducing image data with PCA, **K-Means** groups similar pixels, and each pixel is replaced by the center of its cluster, resulting in compression.

These combined techniques enable efficient image compression by reducing redundancy while maintaining visual quality.

## 🧪 How to Run

1. Make sure Python is installed.
2. Install dependencies:

```bash
pip install requests Pillow numpy matplotlib scikit-learn
````

3. Run the script:

```bash
python compress.py
```

Inside `compress.py`, set your image URL and parameters:

```python
image_url = "https://images.unsplash.com/photo-1593642532973-d31b6557fa68"
compress_image_with_pca(image_url, k=16, n_components=3)
```

You can tune:

* `k`: Number of color clusters in K-Means
* `n_components`: Number of PCA components for dimensionality reduction

## 📸 Example Output

The script will display:

* The **original image**
* The **compressed image** using PCA and K-Means

Side by side in a `matplotlib` window.

## 📂 Files Included

* `compress.py` — Main Python script
* `README.md` — This file

## ✅ Dependencies

* `requests`
* `Pillow`
* `numpy`
* `matplotlib`
* `scikit-learn`

You can also use a `requirements.txt` file to install all:

```bash
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---
