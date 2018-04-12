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
    generate_model(data)
    

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
    winners = reg.rename(columns={'Team': 'Wteam', 'Score': 'WscoreAvg', 'Scorediff': 'WscorediffAvg',
                            'Fgm': 'Wfgm', 'Fga': 'Wfga', 'Fgm3': 'Wfgm3',
                            'Fga3': 'Wfga3', 'Ftm': 'Wftm', 'Fta': 'Wfta', 'Or': 'Wor',
                            'Dr': 'Wdr', 'Ast': 'Wast', 'To': 'Wto', 'Stl': 'Wstl',
                            'Blk': 'Wblk', 'Pf': 'Wpf', 'Win': 'Wwinprop', 'Numot': 'Wnumot'})
    losers = reg.rename(columns={'Team': 'Lteam', 'Score': 'LscoreAvg', 'Scorediff': 'LscorediffAvg',
                            'Fgm': 'Lfgm', 'Fga': 'Lfga', 'Fgm3': 'Lfgm3',
                            'Fga3': 'Lfga3', 'Ftm': 'Lftm', 'Fta': 'Lfta', 'Or': 'Lor',
                            'Dr': 'Ldr', 'Ast': 'Last', 'To': 'Lto', 'Stl': 'Lstl',
                            'Blk': 'Lblk', 'Pf': 'Lpf', 'Win': 'Lwinprop', 'Numot': 'Lnumot'})
    
    tourney = pd.DataFrame.merge(tourney, winners, on=['Season', 'Wteam'])
    tourney = pd.DataFrame.merge(tourney, losers, on=['Season', 'Lteam'])
    #tourney.to_csv("MergedFinalData.csv")
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



