# Music Genre Classification using Deep Learning

This project implements a CNN to classify music genres from audio, using mel-spectrograms and audio segmentation. It features advanced data augmentation, segment-based prediction with multiple aggregation strategies, and result visualization.

## Core Features

- Mel-Spectrograms & Audio Segmentation (default 4s segments)
- **Data Augmentation**:
    - Librosa-based: Time stretch, pitch shift, noise, random gain (handled in `audio_processor.py`).
    - TensorFlow-based: SpecAugment, noise, gain, spatial transforms, Mixup (handled in `augmentation_layers.py` and `trainer.py`).
- CNN with Batch Normalization, Dropout, and Residual Connections (defined in `model_builder.py`).
- **Segment-based Prediction Aggregation**: `average`, `majority_vote`, `max_confidence`, `weighted_average` (implemented in `classifier.py`).
- Caching of processed features.
- Visualization of training, results, augmentations, and segment predictions.
- Model saving/loading. Optional TensorFlow Addons.

## Project Structure

```
music_genre/
├── music_data/genres_original/ # GTZAN dataset (blues/, classical/, etc.)
├── music_genre_env/            # Virtual environment
├── .gitignore
├── main.py                     # Main script to run training
├── classifier.py               # Core MusicGenreClassifier class
├── audio_processor.py          # Audio loading and feature extraction
├── data_manager.py             # Dataset creation and caching
├── model_builder.py            # CNN model definition
├── trainer.py                  # Model training logic
├── augmentation_layers.py      # TensorFlow based augmentation layers
├── plotting_utils.py           # Plotting utilities and callbacks
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone & Environment**:
    ```bash
    git clone <your_repository_url> # Or download files
    cd music_genre
    python3 -m venv music_genre_env
    source music_genre_env/bin/activate # Windows: music_genre_env\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Dataset (GTZAN Recommended)**:
    - Download the GTZAN dataset.
    - Create `music_data/genres_original/`.
    - Organize audio files into genre subdirectories (e.g., `music_data/genres_original/blues/blues.00000.wav`).

## Usage

### Training

```bash
python main.py
```
This script processes data (caches features like `processed_features_seg4s.npz`), trains, evaluates, and saves the model (`music_genre_model.keras`, `music_genre_model_metadata.pkl`).

### Prediction

```python
from classifier import MusicGenreClassifier

# Init (use same segment_length as training)
classifier = MusicGenreClassifier(segment_length=4) 
classifier.load_model("music_genre_model.keras") # Loads .keras and associated _metadata.pkl

audio_file = "path/to/your/song.wav"
result = classifier.predict_genre(audio_file)
if result:
    genre, confidence, _, _ = result
    print(f"Predicted: {genre} (Confidence: {confidence:.2f})")

# Other utilities:
# classifier.compare_aggregation_methods(audio_file)
# classifier.visualize_segment_predictions(audio_file)
# classifier.clear_cache() # If feature extraction changes
```

## Key Components (`classifier.py`, `trainer.py`, `model_builder.py`, etc.)

The `MusicGenreClassifier` class (in `classifier.py`) encapsulates logic for:
- Initialization, loading/saving models and metadata.
- Prediction and aggregation (`predict_genre`, `predict_genre_with_segments`).
- Links to other components for feature extraction, data management, and visualization.

Key functionalities are distributed:
- `audio_processor.py`: `AudioProcessor` class for feature extraction (`extract_features`, `apply_audio_augmentation`).
- `data_manager.py`: `DataManager` class for dataset creation and caching (`create_dataset`, `load_audio_dataset`).
- `augmentation_layers.py`: TensorFlow-based augmentation pipeline (`create_data_augmentation_pipeline`).
- `model_builder.py`: CNN model architecture (`build_model`).
- `trainer.py`: `ModelTrainer` class for the training loop, including callbacks, data normalization, and TF dataset preparation (`train_model`).
- `plotting_utils.py`: Plotting callbacks and standalone visualization functions.

## Dependencies

Key dependencies (see `requirements.txt` for versions):
`tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `librosa`, `soundfile`, `matplotlib`, `seaborn`, `joblib`.
`gradio` is mentioned in requirements but not actively used in the core scripts.

## Troubleshooting

- **Dataset Not Found**: Verify `music_data/genres_original/` structure.
- **TensorFlow/CUDA**: Check TF documentation for GPU setup if applicable.
- **Memory Errors**: Reduce `batch_size` in `train_model` (via `main.py` or direct `ModelTrainer` usage).

---
This README provides a condensed overview. For detailed explanations of methods, refer to the comments within the respective Python files (`classifier.py`, `audio_processor.py`, etc.). 