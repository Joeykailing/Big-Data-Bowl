import tensorflow as tf
import pandas as pd
import numpy as np


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

df = pd.read_csv("tracking_week_1.csv")

df_filter = df[df["gameId"] == 2022091200]

df_filter = getTrimmedCSV(df_filter)

playIds = df_filter["playId"].unique()

def getFrames(df):
    frames = []

    for i in range(len(playIds)):
        frames.append([])

        filt_df = df[df["playId"] == playIds[i]]
        
        frame_Ids = filt_df["frameId"].unique()
        
        for k in range(len(frame_Ids)):
            filt_frame_df = filt_df[filt_df["frameId"] == frame_Ids[k]]
            
            filt_frame_df = filt_frame_df[filt_frame_df["displayName"] != "football"]
            
            matrix = np.zeros((240, 108))
            
            filt_frame_df['matrix_row'] = filt_frame_df['matrix_row'].astype(int).clip(0, 239)
            filt_frame_df['matrix_col'] = filt_frame_df['matrix_col'].astype(int).clip(0, 107)
            
            for row, col in zip(filt_frame_df['matrix_row'], filt_frame_df['matrix_col']):
                matrix[col, row] = 1  
            
            frames[i].append(matrix)
    
    return frames

frames = getFrames(df_filter)