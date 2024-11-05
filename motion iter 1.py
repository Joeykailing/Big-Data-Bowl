import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


warnings.filterwarnings('ignore')  # kys i will slice as many dataframes as i want

ymid = 26.65
lhash = 23.3
rhash = 30.0

offensive_positions = ['QB', 'RB', 'FB', 'WR', 'TE', 'T', 'G', 'C', 'OL']

df = pd.read_csv("Tracking Week 1 - Colts @ Texans.csv")

pbp = pd.read_csv("plays.csv")

pbp['successYards'] = np.where(
    pbp['down'] <= 2,
    np.ceil(pbp['yardsToGo'] / 2),  
    pbp['yardsToGo']                 
)

pbp['Success?'] = np.where(
    pbp['yardsGained'] >= pbp['successYards'],
    1,  
    0   
)


player_data = pd.read_csv("players.csv")

player_pbp = pd.read_csv("player_play.csv")
player_pbp = player_pbp.merge(
    player_data[['nflId', 'position']],
    on='nflId',
    how='left'
)




nflid_to_name = player_data.set_index('nflId')['displayName'].to_dict()

player_pbp['Player Name'] = player_pbp['nflId'].map(nflid_to_name)  
player_pbp['IsOffense'] = player_pbp['position'].isin(offensive_positions).astype(int)

cols = player_pbp.columns.tolist()
cols.insert(0, cols.pop(cols.index('Player Name')))
player_pbp = player_pbp[cols]


game_data = pd.read_csv("games.csv")



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

def addTracking(track_df, pbp_df):

    line_set_df = track_df[track_df['event'] == 'line_set']
    line_set_pos = (
        line_set_df.groupby('nflId')
        .first()
        .reset_index()[['nflId', 'x', 'y']]
        .rename(columns={'x': 'x_start', 'y': 'y_start'})
    )
    
    ball_snap_df = track_df[track_df['event'] == 'ball_snap']
    ball_snap_pos = (
        ball_snap_df.groupby('nflId')
        .first()
        .reset_index()[['nflId', 'x', 'y']]
        .rename(columns={'x': 'x_end', 'y': 'y_end'})
    )
    
    s_snap_df = (
        ball_snap_df.groupby('nflId')
        .first()
        .reset_index()[['nflId', 's']]
        .rename(columns={'s': 's_snap'})
    )
    
    a_snap_df = (
        ball_snap_df.groupby('nflId')
        .first()
        .reset_index()[['nflId', 'a']]
        .rename(columns={'a': 'a_snap'})
        
        
        )
    
    max_s_df = (
        df1_trimmed.groupby('nflId')['s']
        .max()
        .reset_index()
        .rename(columns={'s': 'max_s'})
    )
    
    max_a_df = (
        df1_trimmed.groupby('nflId')['a']
        .max()
        .reset_index()
        .rename(columns={'a': 'max_a'})
    )
    
    player_stats_df = (
        line_set_pos
        .merge(ball_snap_pos, on='nflId', how='outer')
        .merge(s_snap_df, on='nflId', how='outer')
        .merge(a_snap_df, on = 'nflId', how = 'outer')
        .merge(max_s_df, on='nflId', how='outer')
        .merge(max_a_df, on='nflId', how='outer')
    )
    
    
    
    pbp_df = pbp_df.merge(player_stats_df, on='nflId', how='left')
    
    return pbp_df




gameId = 2022091105

game_player_pbp = player_pbp[player_pbp["gameId"] == gameId]

game_player_pbp['x_start'] = None
game_player_pbp['y_start'] = None
game_player_pbp['x_end'] = None
game_player_pbp['y_end'] = None
game_player_pbp['s_snap'] = None
game_player_pbp['a_snap'] = None
game_player_pbp['max_s'] = None
game_player_pbp['max_a'] = None
game_player_pbp['motion_type'] = None

playIds = df_trimmed[df_trimmed["gameId"] == gameId]["playId"].unique()

motion_game_pbp_list = []

for i in range(len(playIds)):
    
    playId = playIds[i]

    df1 = df[df["playId"] == playId]
    df1_trimmed = df_trimmed[df_trimmed["playId"] == playId]
    
    df1_trimmed = df1_trimmed.merge(
        player_data[['nflId', 'position']],
        on='nflId',
        how='left'
    )
    
    direction = df1_trimmed[(df1_trimmed["position"] == "QB") & (df1_trimmed["event"] == "ball_snap")]["o"].iloc[0]
    
    x_qb = df1_trimmed[(df1_trimmed["position"] == "QB") & (df1_trimmed["event"] == "ball_snap")]["x"].iloc[0]
    
    x_ball = df1_trimmed[df1_trimmed["displayName"] == "football"]["x"].iloc[0]
    y_ball = df1_trimmed[df1_trimmed["displayName"] == "football"]["y"].iloc[0]
    

    
    pbp1 = pbp[(pbp["gameId"] == gameId) & (pbp["playId"] == playId)]
    
    player_pbp1 = player_pbp[(player_pbp["gameId"] == gameId) & (player_pbp["playId"] == playId)]
    
    
    player_pbp1 = addTracking(df1_trimmed, player_pbp1)
    
    player_pbp1['motion_type'] = None
    
    formation = pbp[(pbp["gameId"] == gameId) & (pbp["playId"] == playId)]["offenseFormation"].iloc[0]
    
    player_pbp1['Formation'] = formation
    
    if(direction > 0 and direction <= 180):
        
        right_jet_conditions = (
            (player_pbp1['y_start'] > y_ball) &  
            (player_pbp1['y_end'] < player_pbp1['y_start']) &  
            (player_pbp1['y_end'] > y_ball) &  
            (player_pbp1['s_snap'] > 5)  & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['position'] == "WR")
            
        )
        
        
        left_jet_conditions = (
            (player_pbp1['y_start'] < y_ball) &  
            (player_pbp1['y_end'] > player_pbp1['y_start']) &  
            (player_pbp1['y_end'] < y_ball) &  
            (player_pbp1['s_snap'] > 5) &
            (player_pbp1['IsOffense'] == 1) &
            (player_pbp1['position'] == "WR")
        )
        
        right_fly_conditions = (
            (player_pbp1['y_start'] > y_ball) &  
            (player_pbp1['y_end'] < player_pbp1['y_start']) &  
            (player_pbp1['y_end'] < y_ball) &  
            (player_pbp1['s_snap'] > 5)  & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['position'] == "WR")
            
        )
        
        
        left_fly_conditions = (
            (player_pbp1['y_start'] < y_ball) &  
            (player_pbp1['y_end'] > player_pbp1['y_start']) &  
            (player_pbp1['y_end'] > y_ball) &  
            (player_pbp1['s_snap'] > 5) &
            (player_pbp1['IsOffense'] == 1) &
            (player_pbp1['position'] == "WR")
        )
        
        
        right_orbit_conditions = (
            (player_pbp1['x_end'] < player_pbp1['x_start']) &
            (player_pbp1['x_end'] < x_qb) & 
            (player_pbp1['y_start'] > y_ball) & 
            (player_pbp1['y_end'] < y_ball) & 
            (player_pbp1['s_snap'] > 1.5) & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['Formation'] == "SHOTGUN")
            )
        
        left_orbit_conditions = (
            (player_pbp1['x_end'] < player_pbp1['x_start']) &
            (player_pbp1['x_end'] < x_qb) & 
            (player_pbp1['y_start'] < y_ball) & 
            (player_pbp1['y_end'] > y_ball) & 
            (player_pbp1['s_snap'] > 1.5) &
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['Formation'] == "SHOTGUN")
            )
        
        
    else:
        
        left_jet_conditions = (
            (player_pbp1['y_start'] > y_ball) &  
            (player_pbp1['y_end'] < player_pbp1['y_start']) &  
            (player_pbp1['y_end'] > y_ball) &  
            (player_pbp1['s_snap'] > 5)  & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['position'] == "WR") & 
            (player_pbp1['Formation'] == "SHOTGUN")
            
        )
        
        
        right_jet_conditions = (
            (player_pbp1['y_start'] < y_ball) &  
            (player_pbp1['y_end'] > player_pbp1['y_start']) &  
            (player_pbp1['y_end'] < y_ball) &  
            (player_pbp1['s_snap'] > 5) &
            (player_pbp1['IsOffense'] == 1) &
            (player_pbp1['position'] == "WR")
        )
        
        left_fly_conditions = (
            (player_pbp1['y_start'] > y_ball) &  
            (player_pbp1['y_end'] < player_pbp1['y_start']) &  
            (player_pbp1['y_end'] < y_ball) &  
            (player_pbp1['s_snap'] > 5)  & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['position'] == "WR")
            
        )
        
        
        right_fly_conditions = (
            (player_pbp1['y_start'] < y_ball) &  
            (player_pbp1['y_end'] > player_pbp1['y_start']) &  
            (player_pbp1['y_end'] > y_ball) &  
            (player_pbp1['s_snap'] > 5) &
            (player_pbp1['IsOffense'] == 1) &
            (player_pbp1['position'] == "WR")
        )
        
        
        right_orbit_conditions = (
            (player_pbp1['x_end'] > player_pbp1['x_start']) &
            (player_pbp1['x_end'] > x_qb) & 
            (player_pbp1['y_start'] < y_ball) & 
            (player_pbp1['y_end'] > y_ball) & 
            (player_pbp1['s_snap'] > 1.5) & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['Formation'] == "SHOTGUN")
            )
        
        left_orbit_conditions = (
            (player_pbp1['x_end'] > player_pbp1['x_start']) &
            (player_pbp1['x_end'] > x_qb) & 
            (player_pbp1['y_start'] > y_ball) & 
            (player_pbp1['y_end'] < y_ball) & 
            (player_pbp1['s_snap'] > 1.5) & 
            (player_pbp1['IsOffense'] == 1) & 
            (player_pbp1['Formation'] == "SHOTGUN")
            )
    
    
    player_pbp1['x_ball'] = x_ball
    player_pbp1['y_ball'] = y_ball
    

    
    
    
    player_pbp1.loc[right_jet_conditions, 'motion_type'] = 'Right Jet'
    player_pbp1.loc[left_jet_conditions, 'motion_type'] = 'Left Jet'
    player_pbp1.loc[right_fly_conditions, 'motion_type'] = 'Right Fly'
    player_pbp1.loc[left_fly_conditions, 'motion_type'] = 'Left Fly'
    player_pbp1.loc[right_orbit_conditions, 'motion_type'] = 'Right Orbit'
    player_pbp1.loc[left_orbit_conditions, 'motion_type'] = 'Left Orbit'
        
    
    
    
    desc = pbp[(pbp["gameId"] == gameId) & (pbp["playId"] == playId)]["playDescription"].iloc[0]
    
    quarter = pbp[(pbp["gameId"] == gameId) & (pbp["playId"] == playId)]["quarter"].iloc[0]
    
    player_pbp1["Play Description"] = desc

    player_pbp1["Quarter"] = quarter
    
    motion_game_pbp_list.append(player_pbp1)
    
motion_game_pbp = pd.concat(motion_game_pbp_list, ignore_index=True)

game_player_pbp = motion_game_pbp

order = ['Player Name', 'motion_type', 'Play Description', 'Quarter', 'Formation'] + \
                [col for col in game_player_pbp.columns if col not in ['Player Name', 'motion_type', 'Play Description', 'Quarter', 'Formation']]

game_player_pbp = game_player_pbp[order]
