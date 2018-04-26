import sys
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# sklearn.preprocessing.OneHotEncoder() used for categorical features to one-hots
# commands for converting to Category: pd.Categorical(__somecolumn___)
# sklearn.preprocessing.LabelEncoder() normalizes labels by hashing

#logreg = LogisticRegression()
TOURNEY_COLS = ['Season', 'Wteam', 'Wscore', 'Lteam', 'Lscore', 'Wscorediff', 'Lscorediff', 'WSeed', 'LSeed', 'Round']
REG_COLS = ['Team', 'Season', 'Score', 'Numot', 'Fgm', 'Fga', 'Fgm3', 'Fga3', 'Ftm', 'Fta', 'Or', 'Dr', 'Ast', 'To', 'Stl', 'Blk', 'Pf', 'Scorediff', 'Win']
# Name of whatever label we choose
LABEL = "HigherSeedWon"

def main():
    data = merge_input_frames("AllRegularSeason.csv", "AllTourneyResults.csv")
    # manual = pd.read_csv(open(reg_filename), usecols=[FILL ME IN], skipinitialspace=True, skiprows=0, engine="python")
    generate_model(data)
    # generate_model(manual)

def merge_input_frames(reg_filename, tour_filename):
    reg = pd.read_csv(open(reg_filename), usecols=REG_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    tourney = pd.read_csv(open(tour_filename), usecols=TOURNEY_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    # This is whhatver label we choose
    # Need both positive and negative labels tp train so whatever you choose
    # This is not a good metric beecause sometimes during Final Fours/Championships teams have the same seed
    tourney[LABEL] = np.where((tourney['WSeed'] <= tourney['LSeed']), 1, 0)
    winners = reg.rename(columns={'Team': 'Wteam', 'Score': 'WScore', 'Scorediff': 'WScorediff',
                            'Fgm': 'WFgm', 'Fga': 'WFga', 'Fgm3': 'WFgm3',
                            'Fga3': 'WFga3', 'Ftm': 'WFtm', 'Fta': 'WFta', 'Or': 'WOr',
                            'Dr': 'WDr', 'Ast': 'WAst', 'To': 'WTo', 'Stl': 'WStl',
                            'Blk': 'WBlk', 'Pf': 'WPf', 'Win': 'WWin', 'Numot': 'WNumot'})
    losers = reg.rename(columns={'Team': 'Lteam', 'Score': 'LScore', 'Scorediff': 'LScorediff',
                            'Fgm': 'LFgm', 'Fga': 'LFga', 'Fgm3': 'LFgm3',
                            'Fga3': 'LFga3', 'Ftm': 'LFtm', 'Fta': 'LFta', 'Or': 'LOr',
                            'Dr': 'LDr', 'Ast': 'LAst', 'To': 'LTo', 'Stl': 'LStl',
                            'Blk': 'LBlk', 'Pf': 'LPf', 'Win': 'LWin', 'Numot': 'LNumot'})
    tourney = pd.DataFrame.merge(tourney, winners, on=['Season', 'Wteam'])
    tourney = pd.DataFrame.merge(tourney, losers, on=['Season', 'Lteam'])
    tourney.rename(columns={'Wscore': 'T1Score', 'Lscore': 'T2Score', 'Wscorediff': 'T1-T2Score', 'Wteam': 'Team1', 'Lteam': 'Team2'}, inplace=True)
    tourney["WSeed"] = tourney["WSeed"].apply(lambda x: int(x[1:3]))
    tourney["LSeed"] = tourney["LSeed"].apply(lambda x: int(x[1:3]))
    tourney["WSeed-LSeed"] = tourney["WSeed"] - tourney["LSeed"]
    tourney.drop(['Lscorediff', "WSeed", "LSeed"], axis = 1, inplace=True)
    for feature in REG_COLS[2:]:
        tourney[feature + "AvgDiff"] = tourney['W' + feature] - tourney['L' + feature]
        tourney.drop(['W' + feature], axis = 1, inplace=True)
        tourney.drop(['L' + feature], axis = 1, inplace=True)
    #tourney['Fgm'] = tourney['Wfgm'] - tourney['Lfgm']
    tourney.to_csv("MergedFinalData.csv")
    return tourney

def generate_model(data):
    # Can manipulate and bucket features if necessary either here or the function above
    X = data.drop(LABEL,axis=1)
    Y = data[LABEL]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_train.shape)
    #print(Y_test.shape)

    # This doesnt work right now cause u will need to bucket the rounds or like give them indexes
    LR = LogisticRegression(penalty='l2', C=100)
    LR.fit(X_train,Y_train)
    #print(LR.score(X_train,Y_train))
    #print(LR.score(X_test,Y_test))
    
if __name__ == "__main__":
    main()



