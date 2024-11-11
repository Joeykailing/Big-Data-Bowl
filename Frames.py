import pandas as pd
import numpy as np
import warnings

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
        
        all_plays_frames.append(play_frames)
    
    return all_plays_frames

frames = get_frames_per_play()
