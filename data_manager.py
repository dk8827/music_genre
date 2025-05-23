import os
import time
import numpy as np
from audio_processor import AudioProcessor # Assuming AudioProcessor is in audio_processor.py

class DataManager:
    """
    Manages the creation and caching of datasets from audio files.

    Responsible for loading audio data, processing it into features using an AudioProcessor,
    and saving/loading these features to/from a cache file to speed up subsequent runs.
    """
    def __init__(self, data_path, cache_file, n_mels, segment_length, audio_processor):
        """
        Initializes the DataManager.

        Args:
            data_path (str): Path to the root directory of the music data.
            cache_file (str): Filename for storing/retrieving cached features.
            n_mels (int): Number of Mel bands (used for reshaping loaded data if necessary).
            segment_length (int): Length of audio segments in seconds.
            audio_processor (AudioProcessor): Instance of AudioProcessor for feature extraction.
        """
        self.data_path = data_path
        self.cache_file = cache_file
        self.genres = []
        self.n_mels = n_mels # Needed for reshaping in load_audio_dataset
        self.segment_length = segment_length # For print statement in load_audio_dataset
        self.audio_processor = audio_processor

    def create_dataset(self):
        """
        Creates a dataset of features and labels from audio files.

        Attempts to load from cache first. If cache is not found, invalid, or an error occurs,
        it processes audio files from scratch using `load_audio_dataset` and then caches the result.

        Returns:
            tuple: (X, y, genres)
                - X (np.ndarray): Feature data (spectrograms).
                - y (np.ndarray): Labels (genre names).
                - genres (list): List of unique genre names found.
        """
        print("Creating dataset from audio files...")
        if os.path.exists(self.cache_file):
            print("Loading cached features...")
            try:
                cached_data = np.load(self.cache_file, allow_pickle=True)
                X = cached_data['features']
                y = cached_data['labels']
                self.genres = cached_data['genres'].tolist()
                print("Successfully loaded cached features")
                print(f"Dataset shape: {X.shape}")
                unique_genres, counts = np.unique(y, return_counts=True)
                for genre, count in zip(unique_genres, counts):
                    print(f"  {genre}: {count} samples")
                return X, y, self.genres
            except Exception as e:
                print(f"Error loading cached data: {str(e)}. Processing files from scratch...")

        X, y, self.genres = self.load_audio_dataset()

        if X is not None and y is not None and len(X) > 0:
            print("Saving features to cache...")
            try:
                np.savez(self.cache_file, features=X, labels=y, genres=np.array(self.genres))
                print("Successfully cached features")
            except Exception as e:
                print(f"Error caching features: {str(e)}")
        else:
            print("No data to cache.")
        return X, y, self.genres

    def load_audio_dataset(self):
        """
        Loads audio files from the specified data path, extracts features, and organizes them.

        Iterates through genre subdirectories, processes each audio file into segments
        (using the provided AudioProcessor), and collects the features (spectrograms)
        and corresponding genre labels.

        Returns:
            tuple: (X_np, y_np, current_genres)
                - X_np (np.ndarray or None): Array of feature matrices (spectrograms).
                - y_np (np.ndarray or None): Array of genre labels.
                - current_genres (list): List of genre names processed.
        """
        start_time = time.time()
        print(f"Loading audio dataset with {self.segment_length}s segments...")
        X_data, y_labels = [], []

        genres_root_path = os.path.join(self.data_path, 'genres_original')
        if not os.path.exists(genres_root_path):
            print(f"Error: Directory not found: {genres_root_path}")
            return None, None, []

        current_genres = [d for d in os.listdir(genres_root_path) if os.path.isdir(os.path.join(genres_root_path, d))]
        if not current_genres:
            print("Error: No genre subdirectories found.")
            return None, None, []

        print(f"Found {len(current_genres)} genres: {', '.join(current_genres)}")

        total_files_to_process = sum(len(files) for _, _, files in os.walk(genres_root_path) if any(f.endswith(('.au', '.wav')) for f in files))
        processed_files_count = 0
        total_segments_created = 0

        for genre_name in current_genres:
            genre_dir_path = os.path.join(genres_root_path, genre_name)
            print(f"Processing {genre_name} files...")
            genre_segments_count = 0
            for filename in os.listdir(genre_dir_path):
                if filename.endswith(('.au', '.wav')):
                    file_full_path = os.path.join(genre_dir_path, filename)
                    # Use the audio_processor instance to extract features
                    segment_features_list = self.audio_processor.extract_features(file_full_path, use_segments=True)
                    if segment_features_list:
                        for feature_matrix in segment_features_list:
                            X_data.append(feature_matrix)
                            y_labels.append(genre_name)
                            total_segments_created += 1
                            genre_segments_count +=1
                        processed_files_count += 1

                    if processed_files_count > 0 and processed_files_count % 50 == 0 and total_files_to_process > 0 :
                        progress = (processed_files_count / total_files_to_process) * 100
                        print(f"Progress: {processed_files_count}/{total_files_to_process} files ({progress:.1f}%) - Segments: {total_segments_created}")
            print(f"Completed {genre_name}: {genre_segments_count} segments from its files.")

        if not X_data:
            print("Error: No audio files processed successfully.")
            return None, None, []

        X_np = np.array(X_data).reshape(len(X_data), self.n_mels, -1, 1)
        y_np = np.array(y_labels)

        print(f"Dataset creation completed in {(time.time() - start_time):.2f}s. Shape: {X_np.shape}")
        print(f"Total files processed: {processed_files_count}, Total segments: {total_segments_created}")
        return X_np, y_np, current_genres

    def clear_cache(self):
        """Deletes the cache file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print("Cache cleared successfully.")
            except Exception as e:
                print(f"Error clearing cache: {str(e)}")
        else:
            print("No cache file found.") 