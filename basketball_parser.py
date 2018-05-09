import sys
import numpy as np
import pandas as pd
import csv


def main():
    team_names = create_dict_from_csv("teams.csv")
    inputs = get_input_frames("RegularSeasonDetailedResults.csv", team_names)
    clean_inputs(inputs)

def create_dict_from_csv(filename):
    """Creates dictionary mapping team ID to team names.
    Args:
        filename: The filename that contains team IDs/names ("teams.csv").
    Returns:
        A dictionary of team names keyed by team ID.
    """
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Gets rid of the headers
        team_names = {int(rows[0]):rows[1] for rows in reader}
        return team_names


COLUMNS = ["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"]


def get_input_frames(filename):
    """Returns the file as a dataframe with modified columns.
    Args:
        filename: Filename that contains regular season results ("RegularSeasonDetailedResults.csv").
    Returns:
        A dataframe created from the input file with added columns.
    """
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    # Creating the score differential
    inputs["Wscorediff"] = inputs["Wscore"].subtract(inputs["Lscore"])
    # Lscorediff will be negative
    inputs["Lscorediff"] = inputs["Lscore"].subtract(inputs["Wscore"])
    # The reverse for the Losing team
    inputs["Lloc"] = inputs["Wloc"].apply(lambda x: "N" if x=="N" else ("A" if x=="H" else "H"))
    # For we are dropping these columns. These could be useful in the future.
    return inputs.drop(["Daynum", "Wloc", "Lloc"], axis= 1)


def clean_inputs(inputs):
    """Cleaning the inputs and printing to yearly csv files. Essentially, we are
    duplicating the rows, as we have the data once for the winners and once for the losers.
    However, each time we are only using the information relevant to the winner or the loser.
    """
    # Drop all the info about the losers
    winners = inputs.drop(["Lteam", "Lscore", "Lscorediff", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"], axis=1)
    winners["Win"] = 1
    winners["Loss"] = 0
    # Drop all the info about the winners
    losers = inputs.drop(["Wteam", "Wscore", "Wscorediff", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf"], axis=1)
    losers["Win"] = 0
    losers["Loss"] = 1
    # Rename all the columns so they can be combined
    winners.rename(columns={'Wteam': 'Team', 'Wscore': 'Score', 'Wscorediff': 'Scorediff',
                            'Wfgm': 'Fgm', 'Wfga': 'Fga', 'Wfgm3': 'Fgm3',
                            'Wfga3': 'Fga3', 'Wftm': 'Ftm', 'Wfta': 'Fta', 'Wor': 'Or',
                            'Wdr': 'Dr', 'Wast': 'Ast', 'Wto': 'To', 'Wstl': 'Stl',
                            'Wblk': 'Blk', 'Wpf': 'Pf'}, inplace=True)
    losers.rename(columns={'Lteam': 'Team', 'Lscore': 'Score', 'Lscorediff': 'Scorediff',
                            'Lfgm': 'Fgm', 'Lfga': 'Fga', 'Lfgm3': 'Fgm3',
                            'Lfga3': 'Fga3', 'Lftm': 'Ftm', 'Lfta': 'Fta', 'Lor': 'Or',
                            'Ldr': 'Dr', 'Last': 'Ast', 'Lto': 'To', 'Lstl': 'Stl',
                            'Lblk': 'Blk', 'Lpf': 'Pf'}, inplace=True)
    # Going through all the years and creating new summary files from them
    # Probably could have made the code more RFC
    for year in range(2003, 2018):
        # Adding the wins and losses back together
        yearly1 = pd.concat([winners[winners.Season == year],
                            losers[losers.Season == year]], axis=0)
        # Group by team and average all the fields
        yearly2 = yearly1.groupby('Team').agg(['mean'])
        yearly2["Count"] = yearly1.groupby('Team')["Score"].count()
        if year == 2003:
            all_years = yearly2
        else:
            all_years = pd.concat([all_years, yearly2], axis=0)
    all_years.to_csv("AllRegularSeason.csv")
    return

if __name__=="__main__":
    main()
