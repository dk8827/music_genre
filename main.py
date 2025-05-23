import time
import numpy as np
from datetime import datetime
import warnings
import tensorflow as tf

from classifier import MusicGenreClassifier

# Setup warnings and seeds for reproducibility
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    # Record start time for performance monitoring.
    main_start_time = time.time()
    print("Music Genre Classification Neural Network")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Initialize the main classifier object.
    # This object orchestrates the entire process from data loading to model saving.
    # segment_length defines how audio files are broken down for feature extraction.
    classifier = MusicGenreClassifier(segment_length=4)

    # Create dataset: Loads audio, extracts features (e.g., Mel spectrograms),
    # and prepares them for training. Caching is used to speed up subsequent runs.
    X_data, y_data = classifier.create_dataset()

    # Check if dataset creation was successful.
    if X_data is None or y_data is None or len(X_data) == 0:
        print("Dataset creation failed or resulted in no data. Exiting.")
        exit(1)

    print(f"Dataset shape: {X_data.shape}")
    print(f"Unique Genres found in data: {np.unique(y_data)}")

    # Train the model using the prepared dataset.
    # Epochs and batch_size are key hyperparameters for the training process.
    print("Starting model training...")
    history = classifier.train_model(X_data, y_data, epochs=10, batch_size=32)

    # Save the trained model and associated metadata (e.g., label encoder, normalization stats).
    # This allows for later use without retraining.
    classifier.save_model()

    # Calculate and print the total execution time.
    total_exec_time = time.time() - main_start_time
    print(f"Total execution time: {total_exec_time/60:.2f} minutes")
    print("Processing completed! Check for saved model and plots.") 