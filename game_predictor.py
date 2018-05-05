import sys
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# sklearn.preprocessing.OneHotEncoder() used for categorical features to one-hots
# commands for converting to Category: pd.Categorical(__somecolumn___)
# sklearn.preprocessing.LabelEncoder() normalizes labels by hashing

#logreg = LogisticRegression()
TOURNEY_COLS = ['Season', 'Wteam', 'Wscore', 'Lteam', 'Lscore', 'Wscorediff', 'Lscorediff', 'WSeed', 'LSeed', 'Round']
REG_COLS = ['Team', 'Season', 'Score', 'Numot', 'Fgm', 'Fga', 'Fgm3', 'Fga3', 'Ftm', 'Fta', 'Or', 'Dr', 'Ast', 'To', 'Stl', 'Blk', 'Pf', 'Scorediff', 'Win']
# Name of whatever label we choose
LABEL = "T1 WL"

def main():
    reg_df = pd.read_csv(open("AllRegularSeason.csv"), usecols=REG_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    tourney_df = pd.read_csv(open("AllTourneyResults.csv"), usecols=TOURNEY_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    data = merge_input_frames(reg_df, tourney_df)
    # manual = pd.read_csv(open(reg_filename), usecols=[FILL ME IN], skipinitialspace=True, skiprows=0, engine="python")
    generate_model(data)
    # generate_model(manual)

def merge_input_frames(reg_df, tourney_df):
    reg = reg_df.copy()
    tourney = tourney_df.copy()
    # dropping the region from the seed (leaving just 1 - 16)
    tourney["WSeed"] = tourney["WSeed"].apply(lambda x: int(x[1:3]))
    tourney["LSeed"] = tourney["LSeed"].apply(lambda x: int(x[1:3]))
    # tourney2 is a copy of tourney with the teams in switched positions
    # aka the losing team is Team
    tourney2 = tourney[['Season', 'Lteam', 'Lscore', 'Wteam', 'Wscore', 'Wscorediff', 'Lscorediff', 'WSeed', 'LSeed', 'Round']]
    # The label for both is if Team 1 won
    tourney[LABEL] = np.where((tourney['Wscore'] > tourney['Lscore']), 1, 0)
    tourney2[LABEL] = np.where((tourney['Lscore'] > tourney['Wscore']), 1, 0)
    # Drop the score and the round, which are not necessary for our model
    tourney.drop(["Lscorediff", "Round", 'Wscore', 'Lscore', 'Wscorediff'], axis = 1, inplace=True)
    tourney2.drop(["Lscorediff", "Round", 'Wscore', 'Lscore', 'Wscorediff'], axis = 1, inplace=True)

    # Renaming the regular season data to merge with the tournament data
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
    # Merging the winning and losing stats with the tourney game
    tourney = pd.DataFrame.merge(tourney, winners, on=['Season', 'Wteam'])
    tourney = pd.DataFrame.merge(tourney, losers, on=['Season', 'Lteam'])
    tourney2 = pd.DataFrame.merge(tourney2, winners, on=['Season', 'Wteam'])
    tourney2 = pd.DataFrame.merge(tourney2, losers, on=['Season', 'Lteam'])
    # Renaming the Winning and Losing team to just T1 and T2 (order depends on which dataframe)
    tourney.rename(columns={'Wteam': 'T1', 'Lteam': 'T2'}, inplace=True)
    tourney.rename(columns={'Lteam': 'T1', 'Wteam': 'T2'}, inplace=True)
    tourney2.rename(columns={'Wteam': 'T1', 'Lteam': 'T2'}, inplace=True)
    tourney2.rename(columns={'Lteam': 'T1', 'Wteam': 'T2'}, inplace=True)
    # Finding the difference in seeds
    tourney["1Seed-2Seed"] = tourney["WSeed"] - tourney["LSeed"]
    tourney2["1Seed-2Seed"] = tourney2["LSeed"] - tourney2["WSeed"]
    # Dropping the seeds as they are now useless
    tourney.drop(["WSeed", "LSeed"], axis = 1, inplace=True)
    tourney2.drop(["WSeed", "LSeed"], axis = 1, inplace=True)
    # for re-ordering the fields later
    feature_names = []
    for feature in REG_COLS[2:]:
        feature_names.append(feature + "AvgDiff")
        # Adding the difference to the dataframe
        tourney[feature + "AvgDiff"] = tourney['W' + feature] - tourney['L' + feature]
        # Then dropping the feature after the difference is calculated
        tourney.drop(['W' + feature], axis = 1, inplace=True)
        tourney.drop(['L' + feature], axis = 1, inplace=True)
        # same as before
        tourney2[feature + "AvgDiff"] = tourney2['L' + feature] - tourney2['W' + feature]
        tourney2.drop(['W' + feature], axis = 1, inplace=True)
        tourney2.drop(['L' + feature], axis = 1, inplace=True)
    #tourney['Fgm'] = tourney['Wfgm'] - tourney['Lfgm']
    # concatenating both dataframes
    tourney = pd.concat([tourney, tourney2])
    # Reordering the fields so that it is easier to read
    tourney = tourney[["Season", "T1", "T2", "1Seed-2Seed", "T1 WL"] + feature_names]
    # randomly shuffling the data
    tourney = shuffle(tourney)
    # printing the data to a csv so we can manually verify that we have the correct result
    tourney.to_csv("MergedFinalData.csv")
    return tourney

def generate_model(data):
    X = data.drop(LABEL,axis=1)
    Y = data[LABEL]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    # Scales the data so that data with on a larger scale isn't weighted more heavily
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    LR = LogisticRegression(penalty='l2', C=100)
    LR.fit(X_train,Y_train)
    predictions=LR.predict(X_test)
    #print(LR.score(X_train,Y_train))
    print(LR.score(X_test,Y_test))
    # When uncommented, this will let us see the list of predictions
    # print('True values:', Y_test.tolist())
    # print('Predictions:', predictions.tolist())
    predict_proba = LR.predict_proba(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, predictions)
    print('Accuracy: %.2f' % (accuracy_score))
    classification_report = metrics.classification_report(Y_test, predictions)
    print('Report:', classification_report)
    
if __name__ == "__main__":
    main()



