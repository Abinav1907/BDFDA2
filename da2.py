# Install required libraries
!pip install pyspark tensorflow pillow matplotlib  

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from google.colab import files
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

# ------------------------------
# Initialize Spark Session
# ------------------------------
spark = SparkSession.builder.appName("ImageClustering").getOrCreate()

# ------------------------------
# Function to upload images in Google Colab
# ------------------------------
def upload_images():
    os.makedirs("images", exist_ok=True)
    print("Please select all your PNG images to upload")
    uploaded = files.upload()  # Opens file picker in Colab
    
    # Save each uploaded file to "images" folder
    for fname in uploaded.keys():
        with open(os.path.join("images", fname), "wb") as f:
            f.write(uploaded[fname])
    print(f"Uploaded {len(uploaded)} images to 'images/' folder")

upload_images()

# ------------------------------
# Function to load and preprocess images
# ------------------------------
def load_images(image_folder="images", size=(224,224)):
    imgs, filenames = [], []
    for file in os.listdir(image_folder):
        if file.lower().endswith(".png"):  # Only PNGs
            img_path = os.path.join(image_folder, file)
            img = image.load_img(img_path, target_size=size)  # Resize to (224,224)
            img_array = image.img_to_array(img)
            imgs.append(img_array)
            filenames.append(file)
    print(f"Loaded {len(imgs)} images")
    return np.array(imgs), filenames

images, filenames = load_images("images")

# ------------------------------
# Display random sample images
# ------------------------------
import random
num_images_to_show = 4
random_filenames = random.sample(filenames, num_images_to_show)

plt.figure(figsize=(10, 10))
for i, filename in enumerate(random_filenames):
    img_path = os.path.join("images", filename)
    img = Image.open(img_path)
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(img)
    plt.title(filename)
    plt.axis('off')
plt.show()

# ------------------------------
# Load pretrained ResNet50 for feature extraction
# ------------------------------
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Preprocess images for ResNet input
images_preprocessed = preprocess_input(images)

# Extract features (2048-dim per image)
features = model.predict(images_preprocessed, verbose=1)
print("Feature shape:", features.shape)  # -> (num_images, 2048)

# ------------------------------
# Convert extracted features into Spark DataFrame
# ------------------------------
from pyspark.sql import Row
spark_data = []
for i, feat in enumerate(features):
    spark_data.append(Row(filename=filenames[i], features=Vectors.dense(feat)))

df = spark.createDataFrame(spark_data)
df.show(5, truncate=False)

# ------------------------------
# Apply PCA (reduce features from 2048 → 2D for visualization)
# ------------------------------
pca = SparkPCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df)
df_pca = pca_model.transform(df)

# Collect PCA results for plotting
pca_features = np.array(df_pca.select("pca_features").rdd.map(lambda x: x[0]).collect())

# Scatter plot of images in PCA space
plt.scatter(pca_features[:,0], pca_features[:,1])
plt.title("PCA Plot (TensorFlow + Spark)")
plt.show()

# ------------------------------
# Elbow method (using sklearn) to find optimal number of clusters
# ------------------------------
from sklearn.cluster import KMeans
inertia = []
K = range(1, 11)  # Test k = 1..10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.xticks(K)
plt.grid(True)
plt.show()

# ------------------------------
# Apply KMeans clustering with PySpark
# ------------------------------
from pyspark.ml.clustering import KMeans
kmeans = KMeans().setK(4).setSeed(42).setFeaturesCol("features").setPredictionCol("cluster")
model = kmeans.fit(df)
df_clustered = model.transform(df)

# Show sample assignments
clusters = df_clustered.select("filename", "cluster").collect()
print("Sample cluster assignments:")
for row in clusters[:10]:
    print(row)

# ------------------------------
# Show one representative image per cluster
# ------------------------------
unique_clusters = df_clustered.select("cluster").distinct().collect()
plt.figure(figsize=(15,3))

for i, c in enumerate(unique_clusters):
    cluster_id = c["cluster"]
    rows = df_clustered.filter(df_clustered.cluster==cluster_id).select("filename").collect()
    chosen_file = random.choice(rows)["filename"]  # Pick one image from cluster
    img_path = os.path.join("images", chosen_file)
    img = Image.open(img_path)
    
    plt.subplot(1, len(unique_clusters), i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Cluster {cluster_id}")
plt.show()

# ------------------------------
# Compare raw vs processed (Resized) image per cluster
# ------------------------------
import torchvision.transforms as transforms

unique_clusters = df_clustered.select("cluster").distinct().collect()

# Torch transform for resizing and tensor conversion
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for i, c in enumerate(unique_clusters):
    cluster_id = c["cluster"]
    rows = df_clustered.filter(df_clustered.cluster == cluster_id).select("filename").collect()
    
    if not rows:
        print(f"Cluster {cluster_id} is empty. Skipping.")
        continue
    
    chosen_file = random.choice(rows)["filename"]
    
    try:
        file_index = filenames.index(chosen_file)
    except ValueError:
        print(f"Warning: Could not find index for file {chosen_file}. Skipping.")
        continue
    
    # Load original raw image
    raw_img = Image.open(os.path.join("images", chosen_file)).convert("RGB")
    
    # Attempt to access preprocessed version (⚠️ but processed_images not defined yet)
    processed_img_tensor = processed_images[file_index]
    processed_img = transforms.ToPILImage()(processed_img_tensor)
    
    # Plot raw vs processed
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Raw Image")
    axes[0].axis("off")
    
    axes[1].imshow(processed_img)
    axes[1].set_title(f"Processed Image\nCluster: {cluster_id}")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
