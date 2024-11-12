import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def get_frames_per_play():
    df = pd.read_csv("Tracking Week 1 - Colts @ Texans.csv")
    
    grouped = df.groupby(['gameId', 'playId'])
    all_plays_frames = []
    
    for (gameId, playId), group in grouped:
        group = group.sort_values('frameId')
        
        # Get the frame IDs for line_set and ball_snap events
        line_set_frames = group[group['event'] == 'line_set']['frameId'].values
        ball_snap_frames = group[group['event'] == 'ball_snap']['frameId'].values
        
        if len(line_set_frames) == 0 or len(ball_snap_frames) == 0:
            continue
            
        line_set_frame = line_set_frames.min()
        ball_snap_frame = ball_snap_frames[ball_snap_frames >= line_set_frame].min()
        
        if pd.isna(ball_snap_frame):
            continue
            
        # Filter frames between line_set and ball_snap
        trimmed_group = group[(group['frameId'] >= line_set_frame) & (group['frameId'] <= ball_snap_frame)]
        
        # Round coordinates to create a grid
        trimmed_group['x_rounded'] = (trimmed_group['x'] * 2).round() / 2
        trimmed_group['y_rounded'] = (trimmed_group['y'] * 2).round() / 2
        
        # Convert coordinates to matrix indices
        trimmed_group['matrix_row'] = ((trimmed_group['y_rounded'] / 53.3) * 107).round().astype(int)
        trimmed_group['matrix_col'] = ((trimmed_group['x_rounded'] / 120) * 239).round().astype(int)
        
        # Group frames
        frame_ids = trimmed_group['frameId'].unique()
        play_frames = []
        
        for frame_id in frame_ids:
            frame_data = trimmed_group[trimmed_group['frameId'] == frame_id]
            play_frames.append(frame_data)
        
        all_plays_frames.append((gameId, playId, play_frames))
    
    return all_plays_frames

def prepare_data(frames, plays_df, max_sequence_length=None):
    # Image dimensions
    height = 108
    width = 240
    
    X = []
    y_coverage_class = []
    
    # Clean the coverage classes first
    plays_df['pff_passCoverage'] = plays_df['pff_passCoverage'].fillna('Unknown')
    plays_df['pff_passCoverage'] = plays_df['pff_passCoverage'].astype(str)
    
    # Create a mapping for coverage classes to integers
    unique_coverages = sorted(plays_df['pff_passCoverage'].unique())
    coverage_to_int = {coverage: i for i, coverage in enumerate(unique_coverages)}
    num_coverage_classes = len(unique_coverages)

    for (gameId, playId, play_frames) in frames:
        # Get labels
        play_label = plays_df[(plays_df['gameId'] == gameId) & (plays_df['playId'] == playId)]
        
        if play_label.empty:
            continue
        
        coverage_class = play_label['pff_passCoverage'].values[0]
        
        # Convert coverage class to integer
        coverage_class_int = coverage_to_int[coverage_class]
        
        play_images = []
        
        for frame_data in play_frames:
            image = np.zeros((height, width))
            
            for idx, player_data in frame_data.iterrows():
                row = player_data['matrix_row']
                col = player_data['matrix_col']
                if 0 <= row < height and 0 <= col < width:
                    image[row, col] = 1
            
            play_images.append(image)
        
        # Skip plays with no frames
        if not play_images:
            continue
            
        play_images_array = np.array(play_images)
        X.append(play_images_array)
        y_coverage_class.append(coverage_class_int)
    
    # If max_sequence_length is not provided, calculate it
    if max_sequence_length is None:
        max_sequence_length = max(seq.shape[0] for seq in X)
    
    # Pad sequences
    num_samples = len(X)
    X_padded = np.zeros((num_samples, max_sequence_length, height, width))
    
    for i, seq in enumerate(X):
        seq_len = min(seq.shape[0], max_sequence_length)
        X_padded[i, :seq_len, :, :] = seq[:seq_len]
    
    # Add channel dimension
    X_padded = X_padded[..., np.newaxis]
    
    # Convert coverage classes to numpy array with shape (n_samples, 1)
    y_coverage_class = np.array(y_coverage_class, dtype=np.int32).reshape(-1, 1)
    
    # Print debugging information
    print(f"Number of samples: {len(X_padded)}")
    print(f"X shape: {X_padded.shape}")
    print(f"Coverage class shape: {y_coverage_class.shape}")
    print(f"Number of unique coverage classes: {num_coverage_classes}")
    print(f"Coverage class mapping: {coverage_to_int}")
    
    # Check if any labels are out of bounds
    if np.any(y_coverage_class >= num_coverage_classes) or np.any(y_coverage_class < 0):
        raise ValueError("Found labels in y_coverage_class that are out of bounds.")
        
    return X_padded, y_coverage_class, num_coverage_classes, coverage_to_int

def create_model(input_shape, num_coverage_classes):
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    
    # LSTM layers
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # Dense layer
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer for coverage class prediction
    coverage_class_output = Dense(num_coverage_classes, activation='softmax', name='coverage_class_output')(x)
    
    model = Model(inputs=inputs, outputs=coverage_class_output)
    return model

def main():
    # Load data
    frames = get_frames_per_play()
    plays_df = pd.read_csv('plays_data.csv')
    
    # Prepare data with a fixed maximum sequence length
    max_sequence_length = 20  # Adjust this value based on your data
    X, y_coverage_class, num_coverage_classes, coverage_mapping = prepare_data(
        frames, plays_df, max_sequence_length)
    
    # Split data
    X_train, X_val, y_cc_train, y_cc_val = train_test_split(
        X, y_coverage_class, test_size=0.2, random_state=42)
    
    # Create and compile model
    input_shape = (max_sequence_length, X.shape[2], X.shape[3], 1)
    model = create_model(input_shape, num_coverage_classes)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train,
        y_cc_train,
        validation_data=(X_val, y_cc_val),
        epochs=10,
        batch_size=8
    )
    
    return model, history, coverage_mapping

if __name__ == "__main__":
    model, history, coverage_mapping = main()
