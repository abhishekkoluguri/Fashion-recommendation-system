import pickle
import numpy as np
import streamlit as st
import cv2
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import tensorflow  # Import TensorFlow

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a new model with Global Max Pooling
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit app
st.title("Fashion Recommender System")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Get the feature vector
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized_result])

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display similar images in a single row
    st.write("Similar Images:")

    # Create columns for similar images
    cols = st.columns(5)  # Create 5 columns for displaying images

    # Loop through the indices of similar images and display them in the columns
    for i, file in enumerate(indices[0][1:]):  # skip the first one since it will be the uploaded image itself
        with cols[i]:  # Use the corresponding column
            temp_img = cv2.imread(filenames[file])
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            st.image(temp_img, caption=f"Similar Image {file + 1}", use_column_width=True)
