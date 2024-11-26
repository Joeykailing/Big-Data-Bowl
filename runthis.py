import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

def get_frames_per_play():
    df = pd.read_csv("Tracking Week 1 - Colts @ Texans.csv")
    
    grouped = df.groupby(['gameId', 'playId'])
    all_plays_frames = []
    
    for (gameId, playId), group in grouped:
        group = group.sort_values('frameId')
        
        line_set_frames = group[group['event'] == 'line_set']['frameId'].values
        ball_snap_frames = group[group['event'] == 'ball_snap']['frameId'].values
        
        if len(line_set_frames) == 0 or len(ball_snap_frames) == 0:
            continue
        
        line_set_frame = line_set_frames.min()
        ball_snap_frame = ball_snap_frames[ball_snap_frames >= line_set_frame].min()
        
        if pd.isna(ball_snap_frame):
            continue
        
        trimmed_group = group[(group['frameId'] >= line_set_frame) & (group['frameId'] <= ball_snap_frame)]
        trimmed_group['x_rounded'] = (trimmed_group['x'] * 2).round() / 2
        trimmed_group['y_rounded'] = (trimmed_group['y'] * 2).round() / 2
        trimmed_group['matrix_row'] = ((trimmed_group['y_rounded'] / 53.3) * 107).round().astype(int)
        trimmed_group['matrix_col'] = ((trimmed_group['x_rounded'] / 120) * 239).round().astype(int)
        
        frame_ids = trimmed_group['frameId'].unique()
        play_frames = []
        
        for frame_id in frame_ids:
            frame_data = trimmed_group[trimmed_group['frameId'] == frame_id]
            play_frames.append(frame_data)
        
        all_plays_frames.append((gameId, playId, play_frames))
    
    return all_plays_frames

frames = get_frames_per_play()

# Read labels from plays_data.csv
plays_df = pd.read_csv('plays_data.csv')

X = []
y_success_rate = []
y_coverage_class = []

# Image dimensions
height = 108
width = 240

for (gameId, playId, play_frames) in frames:
    # Get labels
    play_label = plays_df[(plays_df['gameId'] == gameId) & (plays_df['playId'] == playId)]
    
    if play_label.empty:
        continue
    
    # Use correct column names for labels
    success_rate = play_label['Success?'].values[0]
    coverage_class = play_label['pff_passCoverage'].values[0]
    
    play_images = []
    
    for frame_data in play_frames:
        image = np.zeros((height, width))
        
        for idx, player_data in frame_data.iterrows():
            row = player_data['matrix_row']
            col = player_data['matrix_col']
            if 0 <= row < height and 0 <= col < width:
                image[row, col] = 1
        
        play_images.append(image)
    
    play_images_array = np.array(play_images)
    X.append(play_images_array)
    y_success_rate.append(success_rate)
    y_coverage_class.append(coverage_class)

# Determine the maximum sequence length
max_len = max(seq.shape[0] for seq in X)

# Pad sequences
num_samples = len(X)
X_padded = np.zeros((num_samples, max_len, height, width))

for i, seq in enumerate(X):
    seq_len = seq.shape[0]
    X_padded[i, :seq_len, :, :] = seq

# Add channel dimension
X_padded = X_padded[..., np.newaxis]

# Prepare labels
# Encode coverage_class
coverage_classes = plays_df['pff_passCoverage'].unique()
coverage_class_to_index = {label: idx for idx, label in enumerate(coverage_classes)}
num_coverage_classes = len(coverage_classes)
y_coverage_class_indices = [coverage_class_to_index[label] for label in y_coverage_class]
y_coverage_class_categorical = to_categorical(y_coverage_class_indices, num_classes=num_coverage_classes)

# Split data into training and validation sets
X_train, X_val, y_sr_train, y_sr_val, y_cc_train, y_cc_val = train_test_split(
    X_padded, y_success_rate, y_coverage_class_categorical, test_size=0.2, random_state=42)

# Define the model
input_shape = (max_len, height, width, 1)
inputs = Input(shape=input_shape)

# CNN layers
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)

# LSTM layer
x = LSTM(128, name='lstm')(x)

# Output layers
success_rate_output = Dense(1, name='success_rate_output')(x)
coverage_class_output = Dense(num_coverage_classes, activation='softmax', name='coverage_class_output')(x)

# Build the model
model = Model(inputs=inputs, outputs=[success_rate_output, coverage_class_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'success_rate_output': 'mean_squared_error',
                    'coverage_class_output': 'categorical_crossentropy'},
              metrics={'success_rate_output': 'mae',
                       'coverage_class_output': 'accuracy'})

# Train the model
model.fit(X_train,
          {'success_rate_output': np.array(y_sr_train),
           'coverage_class_output': y_cc_train},
          validation_data=(X_val,
                           {'success_rate_output': np.array(y_sr_val),
                            'coverage_class_output': y_cc_val}),
          epochs=10,
          batch_size=8)

# Step 1: Extract embeddings from the trained model
embedding_model = Model(inputs=model.input, outputs=model.get_layer('lstm').output)

# Generate embeddings for all samples
X_embeddings = embedding_model.predict(X_padded)

# Step 2: Fit a nearest neighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')  # Use cosine similarity
nn_model.fit(X_embeddings)

# Step 3: Query similar plays
def find_similar_plays(play_index, X_embeddings, nn_model):
    distances, indices = nn_model.kneighbors([X_embeddings[play_index]])
    return distances, indices

# Example usage
play_index = 0  # Index of the play to query
distances, indices = find_similar_plays(play_index, X_embeddings, nn_model)

# Display the results
print(f"Query Play Index: {play_index}")
print("Nearest Neighbors (Indices):", indices.flatten())
print("Distances to Neighbors:", distances.flatten())

# Analyze similar plays
for i, neighbor_idx in enumerate(indices.flatten()):
    print(f"Neighbor {i + 1}: Play Index {neighbor_idx}, Distance {distances[0, i]}")
