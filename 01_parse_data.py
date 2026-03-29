"""
01_parse_data.py
----------------
Parses Cricsheet ball-by-ball JSON files for IPL and T20I matches
into flat player-match records CSVs.

Input:  ipl_json/   → 1,169 IPL match JSON files
        t20s_json/  → 5,102 T20I match JSON files
Output: ipl_records.csv, t20_records.csv
"""
import json, os, glob, pandas as pd
from collections import defaultdict

def parse_match(path, source):
    with open(path) as f:
        d = json.load(f)
    info = d['info']
    match_id = os.path.basename(path).replace('.json', '')
    date     = info['dates'][0]
    season   = info.get('season', 'unknown')
    venue    = info.get('venue', 'unknown')
    teams    = info.get('teams', [])

    player_stats = defaultdict(lambda: {
        'runs': 0, 'balls_faced': 0,
        'wickets': 0, 'balls_bowled': 0, 'runs_conceded': 0, 'team': ''
    })

    for inning in d.get('innings', []):
        batting_team = inning['team']
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else ''

        for ov in inning.get('overs', []):
            for ball in ov.get('deliveries', []):
                batter = ball['batter']
                bowler = ball['bowler']

                player_stats[batter]['runs']        += ball['runs']['batter']
                player_stats[batter]['balls_faced'] += 1
                player_stats[batter]['team']         = batting_team

                player_stats[bowler]['balls_bowled']   += 1
                player_stats[bowler]['runs_conceded']  += ball['runs']['total']
                player_stats[bowler]['team']            = bowling_team

                if 'wickets' in ball:
                    for w in ball['wickets']:
                        if w['kind'] not in ('run out', 'retired hurt', 'obstructing the field'):
                            player_stats[bowler]['wickets'] += 1

    records = []
    for player, stats in player_stats.items():
        records.append({
            'match_id': match_id, 'source': source, 'date': date,
            'season': season, 'venue': venue, 'player': player,
            'team': stats['team'], 'runs': stats['runs'],
            'balls_faced': stats['balls_faced'], 'wickets': stats['wickets'],
            'balls_bowled': stats['balls_bowled'], 'runs_conceded': stats['runs_conceded'],
        })
    return records

def parse_all(json_dir, source, output_csv):
    all_records = []
    files = glob.glob(os.path.join(json_dir, '*.json'))
    print(f"Parsing {len(files)} {source} matches...")
    for path in files:
        all_records.extend(parse_match(path, source))
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"  → Saved {len(df)} records to {output_csv}")
    return df

if __name__ == '__main__':
    parse_all('input/ipl',  'ipl', 'output/ipl_records.csv')
    parse_all('input/t20i', 't20', 'output/t20_records.csv')
