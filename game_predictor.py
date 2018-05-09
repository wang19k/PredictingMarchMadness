import sys
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics

TOURNEY_COLS = ['Season', 'Wteam', 'Wscore', 'Lteam', 'Lscore', 'Wscorediff', 'Lscorediff', 'WSeed', 'LSeed', 'Round']
REG_COLS = ['Team', 'Season', 'Score', 'Numot', 'Fgm', 'Fga', 'Fgm3', 'Fga3', 'Ftm', 'Fta', 'Or', 'Dr', 'Ast', 'To', 'Stl', 'Blk', 'Pf', 'Scorediff', 'Win']
COM_COLS = ['T1', 'T2', 'Season', 'T1Seed', 'T2Seed']
LABEL = "T1 WL"

def main():
    reg_df = pd.read_csv(open("AllRegularSeason.csv"), usecols=REG_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    tourney_df = pd.read_csv(open("AllTourneyResults.csv"), usecols=TOURNEY_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    com_df = pd.read_csv(open("AllCombinations.csv"), usecols=COM_COLS,
                         skipinitialspace=True,
                         skiprows=0, engine="python")
    data = merge_input_frames(reg_df, tourney_df)
    test_inputs = merge_combinatorial_data(reg_df, com_df)
    # manual = pd.read_csv(open(reg_filename), usecols=[FILL ME IN], skipinitialspace=True, skiprows=0, engine="python")
    model = generate_model(data)
    team_names = create_dict_from_csv("teams.csv")
    generate_predictions(model, test_inputs, team_names)
    # generate_model(manual)

def create_dict_from_csv(filename):
    """Creates dictionary mapping team ID to team names.
    Args:
        filename: The filename that contains team IDs/names ("teams.csv").
    Returns:
        A dictionary of team names keyed by team ID.
    """
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Get's rid of the headers
        team_names = {int(rows[0]):rows[1] for rows in reader}
        return team_names


def merge_input_frames(reg_df, tourney_df):
    """Creates dictionary mapping team ID to team names.
    Args:
        reg_df: Dataframe with cleaned regular season data. Created by loading "AllRegularSeason.csv".
        tourney_df: Dataframe with cleaned tourney data. Created by loading "AllTourneyResults.csv".
    Returns:
        A merged data frame with each row containing a game and the regular season stats + seeds of
            the two opposing teams. The dataframe (randomized) contains 2 copies of each row,
            with team 1 and team 2 and the corresponding stats switched. 
    """
    # Defensive copying of dataframes
    reg = reg_df.copy()
    tourney = tourney_df.copy()
    # dropping the region from the seed (leaving just 1 - 16)
    tourney["WSeed"] = tourney["WSeed"].apply(lambda x: int(x[1:3]))
    tourney["LSeed"] = tourney["LSeed"].apply(lambda x: int(x[1:3]))
    # tourney2 is a copy of tourney with the teams in switched positions
    # aka the losing team is Team
    tourney2 = tourney[['Season', 'Lteam', 'Lscore', 'Wteam', 'Wscore', 'Wscorediff', 'Lscorediff',
                        'WSeed', 'LSeed', 'Round']]
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
    # concatenating both dataframes
    tourney = pd.concat([tourney, tourney2])
    # Reordering the fields so that it is easier to read
    tourney = tourney[["Season", "T1", "T2", "1Seed-2Seed", "T1 WL"] + feature_names]
    # randomly shuffling the data
    tourney = shuffle(tourney)
    # printing the data to a csv so we can manually verify that we have the correct result
    # tourney.to_csv("MergedFinalData.csv")
    return tourney

def merge_combinatorial_data(reg_df, com_df):
    """Merges the combinatorial data with regular season statistics.
    Args:
        reg_df: Dataframe with cleaned regular season data. Created by loading "AllRegularSeason.csv".
        tourney_df: Dataframe with the combinations of every possible matchups.
            Created by loading "AllCombinations.csv".
    Returns:
        A merged data frame with each row containing a game and the regular season stats + seeds of
            the two opposing teams. The dataframe contains all possible games for that season's
            tournament.
    """
    # Defensive copying
    reg = reg_df.copy()
    com = com_df.copy()
    # Renaming the regular season data to merge with the tournament data
    first = reg.rename(columns={'Team': 'T1', 'Score': '1Score', 'Scorediff': '1Scorediff',
                            'Fgm': '1Fgm', 'Fga': '1Fga', 'Fgm3': '1Fgm3',
                            'Fga3': '1Fga3', 'Ftm': '1Ftm', 'Fta': '1Fta', 'Or': '1Or',
                            'Dr': '1Dr', 'Ast': '1Ast', 'To': '1To', 'Stl': '1Stl',
                            'Blk': '1Blk', 'Pf': '1Pf', 'Win': '1Win', 'Numot': '1Numot'})
    second = reg.rename(columns={'Team': 'T2', 'Score': '2Score', 'Scorediff': '2Scorediff',
                            'Fgm': '2Fgm', 'Fga': '2Fga', 'Fgm3': '2Fgm3',
                            'Fga3': '2Fga3', 'Ftm': '2Ftm', 'Fta': '2Fta', 'Or': '2Or',
                            'Dr': '2Dr', 'Ast': '2Ast', 'To': '2To', 'Stl': '2Stl',
                            'Blk': '2Blk', 'Pf': '2Pf', 'Win': '2Win', 'Numot': '2Numot'})
    # Merging the stats with the tourney game
    com = pd.DataFrame.merge(com, first, on=['Season', 'T1'])
    com = pd.DataFrame.merge(com, second, on=['Season', 'T2'])
    # Replacing individual features with their differences
    com["1Seed-2Seed"] = com["T1Seed"] - com["T2Seed"]
    com.drop(["T1Seed", "T2Seed"], axis=1, inplace=True)
    for feature in REG_COLS[2:]:
        # Adding the difference to the dataframe
        com[feature + "AvgDiff"] = com['1' + feature] - com['2' + feature]
        # Then dropping the feature after the difference is calculated
        com.drop(['1' + feature], axis = 1, inplace=True)
        com.drop(['2' + feature], axis = 1, inplace=True)
    return com


def generate_model(data):
    """Creates and trains our model, which is used to predict NCAA basketball game winners.
    Args:
        data: A merged data frame with each row containing a game and the regular season stats
            + seeds of the two opposing teams
    Returns:
        A trained logistic regression model. 
    """
    # X contains all features except the label
    X = data.drop(LABEL,axis=1)
    # Y contains the label
    Y = data[LABEL]
    # Splitting the data in training and testing data
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    # Creating the model
    LR = LogisticRegression(penalty='l2', C=100)
    # Fitting the model
    LR.fit(X_train,Y_train)
    # Generating predictions
    predictions=LR.predict(X_test)
    print(LR.score(X_test,Y_test))
    # When uncommented, this will let us see the list of predictions
    # print('True values:', Y_test.tolist())
    # print('Predictions:', predictions.tolist())
    # Metrics
    predict_proba = LR.predict_proba(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, predictions)
    print('Accuracy: %.2f' % (accuracy_score))
    classification_report = metrics.classification_report(Y_test, predictions)
    print('Report:', classification_report)
    return LR


def generate_predictions(model, inputs, names):
    predictions=model.predict(inputs)
    inputs["T1"] = inputs["T1"].apply(lambda x: names[x])
    inputs["T2"] = inputs["T2"].apply(lambda x: names[x])
    inputs[LABEL] = predictions
    inputs.to_csv("FinalResults.csv")
    
if __name__ == "__main__":
    main()



