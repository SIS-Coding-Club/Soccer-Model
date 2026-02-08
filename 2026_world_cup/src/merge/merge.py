import pandas as pd
import numpy as np


# Convert month names to numeric for comparison
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


def get_nearest_team_stats(match_row, team_col, team_stats_df):
    """
    Get team statistics matching year and month, or closest available
    
    Args:
        match_row: Single row from match dataframe
        team_col: Column name for team ('home_team' or 'away_team')
        team_stats_df: DataFrame containing team statistics with roster dates
        
    Returns:
        Dictionary with team statistics or None values if not found
    """
    team = match_row[team_col]
    match_year = match_row['year']
    match_month = match_row['month']
    
    # Filter team stats for this team
    team_data = team_stats_df[team_stats_df['team'] == team].copy()
    
    if len(team_data) == 0:
        return {
            'attack': None, 
            'defense': None, 
            'goalkeeping': None, 
            'physical': None, 
            'team_overall': None
        }
    
    # Try to find exact year and month match
    exact_match = team_data[
        (team_data['roster_year'] == match_year) & 
        (team_data['roster_month'] == match_month)
    ]
    
    if len(exact_match) > 0:
        # Use the latest roster from that month
        nearest = exact_match.loc[exact_match['roster_date'].idxmax()]
    else:
        # Fall back to closest date by finding minimum time difference
        match_month_num = MONTH_MAP.get(match_month, 1)
        team_data['year_month'] = team_data['roster_year'] * 12 + team_data['roster_date'].dt.month
        match_year_month = match_year * 12 + match_month_num
        team_data['diff'] = abs(team_data['year_month'] - match_year_month)
        
        # Get index of minimum difference
        min_idx = team_data['diff'].idxmin()
        if pd.isna(min_idx):
            return {
                'attack': None, 
                'defense': None, 
                'goalkeeping': None, 
                'physical': None, 
                'team_overall': None
            }
        nearest = team_data.loc[min_idx]
    
    return nearest[['attack', 'defense', 'goalkeeping', 'physical', 'team_overall']].to_dict()


def prepare_team_stats(team_stats_df):
    """
    Prepare team statistics DataFrame by adding year and month columns
    
    Args:
        team_stats_df: DataFrame with team statistics and roster_date column
        
    Returns:
        Modified DataFrame with roster_year and roster_month columns
    """
    team_stats_df = team_stats_df.copy()
    team_stats_df['roster_date'] = pd.to_datetime(team_stats_df['roster_date'])
    team_stats_df['roster_year'] = team_stats_df['roster_date'].dt.year
    team_stats_df['roster_month'] = team_stats_df['roster_date'].dt.strftime('%b').str.upper()
    
    return team_stats_df


def merge_match_with_team_stats(df_match, team_stats_df, verbose=True):
    """
    Merge match data with team statistics for both home and away teams
    
    Args:
        df_match: DataFrame containing match data with year, month, home_team, away_team
        team_stats_df: DataFrame containing team statistics with roster_date
        verbose: Whether to print progress and summary (default: True)
        
    Returns:
        DataFrame with match data enriched with team statistics
    """
    # Prepare team stats
    team_stats_df = prepare_team_stats(team_stats_df)
    
    # Apply for home team
    if verbose:
        print("Merging home team statistics...")
    home_stats = df_match.apply(
        lambda row: get_nearest_team_stats(row, 'home_team', team_stats_df), 
        axis=1
    )
    home_stats_df = pd.DataFrame(
        home_stats.tolist(), 
        columns=['attack', 'defense', 'goalkeeping', 'physical', 'team_overall']
    )
    home_stats_df.columns = ['home_' + col for col in home_stats_df.columns]
    
    # Apply for away team
    if verbose:
        print("Merging away team statistics...")
    away_stats = df_match.apply(
        lambda row: get_nearest_team_stats(row, 'away_team', team_stats_df), 
        axis=1
    )
    away_stats_df = pd.DataFrame(
        away_stats.tolist(), 
        columns=['attack', 'defense', 'goalkeeping', 'physical', 'team_overall']
    )
    away_stats_df.columns = ['away_' + col for col in away_stats_df.columns]
    
    # Concatenate with original match data
    df_match_enriched = pd.concat(
        [df_match.reset_index(drop=True), home_stats_df, away_stats_df], 
        axis=1
    )
    
    # Remove rows where team stats are missing
    df_match_enriched = df_match_enriched.dropna(subset=['home_attack', 'away_attack'])
    
    if verbose:
        print(f"\nOriginal matches: {len(df_match)}")
        print(f"Matches with team stats: {len(df_match_enriched)}")
        print(f"Matches dropped: {len(df_match) - len(df_match_enriched)}")
        print(f"\nNew columns added: {[col for col in df_match_enriched.columns if col.startswith('home_') or col.startswith('away_')]}")
    
    return df_match_enriched


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    df = pd.read_csv("../../data/final_train_data/player_data.csv", index_col=False)
    df_match = pd.read_csv("../../data/final_train_data/match.csv")
    
    # You would need to generate team_stats_df from player data first
    # This is just a placeholder showing how to use the module
    print("\nTo use this module, first generate team_stats_df from player data,")
    print("then call: df_enriched = merge_match_with_team_stats(df_match, team_stats_df)")
