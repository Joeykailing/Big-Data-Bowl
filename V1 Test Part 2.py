import tensorflow as tf
import pandas as pd
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('Frames_Success_V1.h5')

def getTrimmedCSV(df):
    grouped = df.groupby(['gameId', 'playId'])
    filtered_dfs = []

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
        filtered_dfs.append(trimmed_group)

    df_trimmed = pd.concat(filtered_dfs, ignore_index=True)
    df_trimmed['x_rounded'] = (df_trimmed['x'] * 2).round() / 2
    df_trimmed['y_rounded'] = (df_trimmed['y'] * 2).round() / 2
    df_trimmed['matrix_row'] = ((df_trimmed['y_rounded'] / 53.3) * 107).round().astype(int)
    df_trimmed['matrix_col'] = ((df_trimmed['x_rounded'] / 120) * 239).round().astype(int)
    
    return df_trimmed

# Load and filter data
df = pd.read_csv("tracking_week_1.csv")
df_filter = df[df["gameId"] == 2022091200]
df_filter = getTrimmedCSV(df_filter)

# Get unique play IDs
playIds = df_filter["playId"].unique()

def getFrames(df):
    frames = []
    for play_id in playIds:
        play_frames = []
        filt_df = df[df["playId"] == play_id]
        frame_Ids = filt_df["frameId"].unique()

        for frame_id in frame_Ids:
            filt_frame_df = filt_df[filt_df["frameId"] == frame_id]
            filt_frame_df = filt_frame_df[filt_frame_df["displayName"] != "football"]

            matrix = np.zeros((108, 240))  # Set shape to (108, 240) to match model expectation
            filt_frame_df['matrix_row'] = filt_frame_df['matrix_row'].astype(int).clip(0, 107)
            filt_frame_df['matrix_col'] = filt_frame_df['matrix_col'].astype(int).clip(0, 239)

            for row, col in zip(filt_frame_df['matrix_row'], filt_frame_df['matrix_col']):
                matrix[row, col] = 1  
            play_frames.append(matrix)
        
        frames.append(play_frames)
    
    return frames

frames = getFrames(df_filter)

# Set the correct dimensions
max_sequence_length = 20  # Update as per your model's training config
height, width = 108, 240

def preprocess_frames(frames, max_sequence_length, height, width):
    X_padded = np.zeros((len(frames), max_sequence_length, height, width, 1))  # Include the channel dimension
    for i, play_frames in enumerate(frames):
        play_frames = np.array(play_frames)  # Convert frames to numpy array
        seq_len = min(play_frames.shape[0], max_sequence_length)
        X_padded[i, :seq_len, :, :, 0] = play_frames[:seq_len]  # Add frames along the channel dimension
    return X_padded

# Prepare frames for prediction
X_test = preprocess_frames(frames, max_sequence_length, height, width)

# Make predictions on the preprocessed frames
predictions = model.predict(X_test)

# Display predictions
for i, play_id in enumerate(playIds):
    success_rate = predictions[i][0]  # Extract success rate prediction for this play
    print(f"Play ID: {play_id}")
    print(f"Predicted Success Rate: {success_rate:.2f}")
    print(f"Description: Play {play_id} from game 2022091200\n")
