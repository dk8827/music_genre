# Music Genre Classification using Deep Learning

This project implements a CNN to classify music genres from audio, using mel-spectrograms and audio segmentation. It features advanced data augmentation, segment-based prediction with multiple aggregation strategies, and result visualization.

## Core Features

- Mel-Spectrograms & Audio Segmentation (default 4s segments)
- **Data Augmentation**:
    - Librosa-based: Time stretch, pitch shift, noise, random gain.
    - TensorFlow-based: SpecAugment, noise, gain, spatial transforms, Mixup.
- CNN with Batch Normalization, Dropout, and Residual Connections.
- **Segment-based Prediction Aggregation**: `average`, `majority_vote`, `max_confidence`, `weighted_average`.
- Caching of processed features.
- Visualization of training, results, augmentations, and segment predictions.
- Model saving/loading. Optional TensorFlow Addons.

## Project Structure

```
music_genre/
├── music_data/genres_original/ # GTZAN dataset (blues/, classical/, etc.)
├── music_genre_env/            # Virtual environment
├── .gitignore
├── music_genre_nn.py           # Main script
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
python music_genre_nn.py
```
This script processes data (caches features like `processed_features_seg4s.npz`), trains, evaluates, and saves the model (`music_genre_model.h5`, `label_encoder.pkl`).

### Prediction

```python
from music_genre_nn import MusicGenreClassifier

# Init (use same segment_length as training)
classifier = MusicGenreClassifier(segment_length=4) 
classifier.load_model("music_genre_model.h5")

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

## Key Components (`music_genre_nn.py`)

The `MusicGenreClassifier` class encapsulates all logic:
- Initialization, feature extraction (`extract_features`, `apply_audio_augmentation`).
- Dataset creation and caching (`create_dataset`, `load_audio_dataset`).
- Augmentation pipeline (`create_audio_augmentation_layers`, `create_advanced_augmentation_pipeline`).
- CNN model building (`build_model`).
- Training loop with callbacks and plotting (`train_model`).
- Prediction and aggregation (`predict_genre`, `predict_genre_with_segments`).
- Visualization utilities.

## Dependencies

Key dependencies (see `requirements.txt` for versions):
`tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `librosa`, `soundfile`, `matplotlib`, `seaborn`, `gradio` (optional GUI), `joblib`, `scipy`.

## Troubleshooting

- **Dataset Not Found**: Verify `music_data/genres_original/` structure.
- **TensorFlow/CUDA**: Check TF documentation for GPU setup if applicable.
- **Memory Errors**: Reduce `batch_size` in `train_model`.

---
This README provides a condensed overview. For detailed explanations of methods, refer to the comments within `music_genre_nn.py`. 