import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def move_column_to_first(df, column_name):
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index(column_name)))
    return df[cols]


df = pd.read_csv('../data/raw/match_data_v5.csv')

df.columns = ["matchID", "blueTeamControlWardsPlaced", "blueTeamWardsPlaced", "blueTeamTotalKills",
              "blueTeamDragonKills", "blueTeamHeraldKills", "blueTeamTowersDestroyed", "blueTeamInhibitorsDestroyed",
              "blueTeamTurretPlatesDestroyed", "blueTeamFirstBlood", "blueTeamMinionsKilled", "blueTeamJungleMinions",
              "blueTeamTotalGold", "blueTeamXp", "blueTeamTotalDamageToChamps", "redTeamControlWardsPlaced",
              "redTeamWardsPlaced", "redTeamTotalKills", "redTeamDragonKills", "redTeamHeraldKills",
              "redTeamTowersDestroyed", "redTeamInhibitorsDestroyed", "redTeamTurretPlatesDestroyed",
              "redTeamMinionsKilled", "redTeamJungleMinions", "redTeamTotalGold", "redTeamXp",
              "redTeamTotalDamageToChamps", "blueWin", "empty"]

# Drop irrelevant or highly correlated features
df.drop(['blueTeamControlWardsPlaced', 'redTeamControlWardsPlaced',
         'blueTeamMinionsKilled', 'redTeamMinionsKilled',
         'blueTeamWardsPlaced', 'redTeamWardsPlaced', 'empty'], axis=1, inplace=True)

# df.rename(columns={'blueTeamTurretPlatesDestroyed': 'redTeamTurretPlatesDestroyed',
#                   'redTeamTurretPlatesDestroyed': 'blueTeamTurretPlatesDestroyed'}, inplace=True)

# remove
df['blueTeamTurretPlatesDestroyed'] = df['blueTeamTurretPlatesDestroyed'].apply(lambda x: 9 if x > 15 else x)
df['redTeamTurretPlatesDestroyed'] = df['redTeamTurretPlatesDestroyed'].apply(lambda x: 9 if x > 15 else x)

df.rename(columns={'blueTeamFirstBlood': 'firstBloodBlueTeam'}, inplace=True)

# Interaction features
df['gold_xp_ratio_blue'] = df['blueTeamTotalGold'] / (df['blueTeamXp'] + 1e-5)
df['gold_xp_ratio_red'] = df['redTeamTotalGold'] / (df['redTeamXp'] + 1e-5)
df['total_objectives_blue'] = (df['blueTeamDragonKills'] + df['blueTeamHeraldKills'] + df['blueTeamTowersDestroyed'] +
                               df['blueTeamInhibitorsDestroyed'])
df['total_objectives_red'] = (df['redTeamDragonKills'] + df['redTeamHeraldKills'] + df['redTeamTowersDestroyed'] +
                              df['redTeamInhibitorsDestroyed'])

df['gold_per_minute_blue'] = df['blueTeamTotalGold'] / 15
df['gold_per_minute_red'] = df['redTeamTotalGold'] / 15
df['xp_per_minute_blue'] = df['blueTeamXp'] / 15
df['xp_per_minute_red'] = df['redTeamXp'] / 15

# Advanced interaction features
df['objective_control_blue'] = (df['blueTeamDragonKills'] + df['blueTeamHeraldKills'] + df[
    'blueTeamTowersDestroyed']) / (df['blueTeamTotalGold'] + 1e-5)
df['objective_control_red'] = (df['redTeamDragonKills'] + df['redTeamHeraldKills'] + df['redTeamTowersDestroyed']) / (
            df['redTeamTotalGold'] + 1e-5)

df['damage_efficiency_blue'] = df['blueTeamTotalDamageToChamps'] / (df['blueTeamTotalGold'] + 1e-5)
df['damage_efficiency_red'] = df['redTeamTotalDamageToChamps'] / (df['redTeamTotalGold'] + 1e-5)

df['gold_xp_dominance_blue'] = df['blueTeamTotalGold'] - df['redTeamTotalGold'] + df['blueTeamXp'] - df['redTeamXp']
df['gold_xp_dominance_red'] = df['redTeamTotalGold'] - df['blueTeamTotalGold'] + df['redTeamXp'] - df['blueTeamXp']

df['aggressiveness_blue'] = (df['blueTeamTotalKills'] + df['blueTeamTotalDamageToChamps']) / (
            df['blueTeamTotalGold'] + 1e-5)
df['aggressiveness_red'] = (df['redTeamTotalKills'] + df['redTeamTotalDamageToChamps']) / (
            df['redTeamTotalGold'] + 1e-5)

df['objective_efficiency_blue'] = df['total_objectives_blue'] / (df['blueTeamTotalGold'] + df['blueTeamXp'] + 1e-5)
df['objective_efficiency_red'] = df['total_objectives_red'] / (df['redTeamTotalGold'] + df['redTeamXp'] + 1e-5)

# Feature Scaling
features_to_scale = ['blueTeamTotalGold', 'redTeamTotalGold', 'blueTeamXp', 'redTeamXp',
                     'blueTeamTotalDamageToChamps', 'redTeamTotalDamageToChamps',
                     'gold_xp_ratio_blue', 'gold_xp_ratio_red',
                     'total_objectives_blue', 'total_objectives_red',
                     'gold_per_minute_blue', 'gold_per_minute_red',
                     'xp_per_minute_blue', 'xp_per_minute_red',
                     'objective_control_blue', 'objective_control_red',
                     'damage_efficiency_blue', 'damage_efficiency_red',
                     'gold_xp_dominance_blue', 'gold_xp_dominance_red',
                     'aggressiveness_blue', 'aggressiveness_red',
                     'objective_efficiency_blue', 'objective_efficiency_red']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Move the target variable to the first column
df = move_column_to_first(df, 'blueWin')

df.to_csv('../data/processed/processed_match_data.csv', index=False)

train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

train_data.drop("matchID", axis=1, inplace=True)
test_data.drop("blueWin", axis=1, inplace=True)

combine = [train_data, test_data]

train_data.to_csv('../data/processed/processed_match_data_train.csv', index=False)
test_data.to_csv('../data/processed/processed_match_data_test.csv', index=False)

"""
print('_'*40)
print(train_data.info())
print('_'*40)
print(test_data.info())

correlation_matrix = df.iloc[:, list(range(1, df.shape[1], 1))].corr()

# Visualize
plt.figure(figsize=(30, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
"""

if __name__ == '__main__':
    pass
