import time
import numpy as np
from datetime import datetime
import warnings
import tensorflow as tf

from classifier import MusicGenreClassifier

# Setup warnings and seeds
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    main_start_time = time.time()
    print("Music Genre Classification Neural Network")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Initialize the classifier
    classifier = MusicGenreClassifier(segment_length=4)

    # Create dataset
    X_data, y_data = classifier.create_dataset()

    if X_data is None or y_data is None or len(X_data) == 0:
        print("Dataset creation failed or resulted in no data. Exiting.")
        exit(1)

    print(f"Dataset shape: {X_data.shape}")
    print(f"Unique Genres found in data: {np.unique(y_data)}")

    # Train the model
    print("Starting model training...")
    history = classifier.train_model(X_data, y_data, epochs=30, batch_size=32)

    # Save the trained model and metadata
    classifier.save_model()

    total_exec_time = time.time() - main_start_time
    print(f"Total execution time: {total_exec_time/60:.2f} minutes")
    print("Processing completed! Check for saved model and plots.") 