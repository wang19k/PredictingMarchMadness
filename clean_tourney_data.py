import sys
import pandas as pd
import csv
import itertools
from  itertools import combinations


def main():
    filter_inputs("TourneySeeds.csv", "TourneySeedsClean.csv")
    team_names = create_dict_from_csv("teams.csv")
    input_list = get_tourney_input_frames("TourneyDetailedResults.csv")
    team_list = get_teams("TourneySeedsClean.csv")
    seed_list = get_seeds("TourneySeedsClean.csv")
    clean_inputs(input_list, seed_list)
    create_combinations(team_list, seed_list, team_names, 2011, 2013)


COLUMNS = ["Season", "Seed", "Team"]


def filter_inputs(filename, outfile):
    """Returns the file as a dataframe with modified columns.
    Args:
        filename: Filename that contains the tournament seeds ("TourneySeeds.csv").
        outfile: Filename to print the filtered dataframe to ("TourneySeedsClean.csv").
    Returns:
        None: Print dataframes to outfile.
    """
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    # Only keep data from 2003 and beyond
    inputs = inputs[inputs.Season >= 2003]
    # resetting the index
    inputs.reset_index(drop=True, inplace=True)
    inputs.to_csv(outfile)


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

def get_seeds(filename):
    """Creates dictionary mapping seasons after 2003 and team ID to team names.
    Args:
        filename: Filename that contains tournament seeds starting from 2003 ("TourneySeedsClean.csv").
    Returns:
        A dictionary of team names keyed by team ID and seasons after 2003.
    """
    seed_list = [{} for _ in range(2003, 2018)]
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Gets rid of the headers
        for rows in reader:
            # 2-dimensional mapping of team names keyed by season (years after 2003)
            # and team ID
            seed_list[(int(rows[0]) - 2003)][int(rows[2])] = rows[1]
    return seed_list


TOURNEY_COLS = ["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"]


def get_tourney_input_frames(filename):
    """Returns the file as a dataframe with modified columns.
    Args:
        filename: Filename that contains the tournament results ("TourneyDetailedResults.csv").
    Returns:
        A list of dataframes, the input dataframe split by season with some added columns.
    """
    inputs = pd.read_csv(open(filename), names=TOURNEY_COLS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    # Drop tournament data, which we won't use for now. 
    inputs.drop(["Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"], axis=1, inplace=True)
    # Getting rid of the games beyond the field of 64 (the play-in games)
    inputs = inputs[inputs.Daynum != 134][inputs.Daynum != 135]
    # Creating the score differential
    inputs["Wscorediff"] = inputs["Wscore"].subtract(inputs["Lscore"])
    # Lscorediff will be negative
    inputs["Lscorediff"] = inputs["Lscore"].subtract(inputs["Wscore"])
    # Creates a list of dataframe by year.
    input_list = []
    for year in range(2003, 2018):
        input_list.append(inputs[inputs.Season == year])
    return input_list


def get_teams(filename):
    """Creates list of seeds split by year.
    Args:
        filename: Filename that contains tournament seeds starting from 2003 ("TourneySeedsClean.csv").
    Returns:
        A list of dataframes, the input dataframe (season, seed, team) split by season.
    """
    inputs = pd.read_csv(open(filename), names=["Season", "Seed", "Team"],
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    input_list = []
    for year in range(2003, 2018):
        input_list.append(inputs[inputs.Season == year])
    return input_list


ROUNDS = {136: 1, 137: 1, 138: 2, 139: 2, 143: 3, 144: 3, 145: 4,
          146: 4, 152: 5, 154: 6}


def clean_inputs(input_list, seed_list):
    """Creates list of seeds split by year.
    Args:
        inputs_list: A list of dataframes, the input dataframe split by season.
        seed_list: A dictionary of team names keyed by team ID and seasons after 2003.
    Returns:
        None, but prints the result to "AllTourneyResults.csv".
    """
    # Go through all the seasons
    for i in range(len(seed_list)):
        # Get the dataframe corresponding to that season
        data = input_list[i]
        # Get the seed corresponding to that team for that season
        data["WSeed"] = data["Wteam"].apply(lambda x: seed_list[i][x])
        data["LSeed"] = data["Lteam"].apply(lambda x: seed_list[i][x])
        # Convert the daynum to rounds and then drop daynum
        data["Round"] = data["Daynum"].apply(lambda x: ROUNDS[x])
        data.drop("Daynum", axis=1, inplace=True)
        # Recombine the dataframes and print out to csv.
        if i == 0:
            all_years = data
        else:
            all_years = pd.concat([all_years, data], axis = 0)
        all_years.to_csv("AllTourneyResults.csv")

    
def create_combinations(team_list, seed_list, names, begin, end):
    """Generates all combinations of team matchups from begin to end per year.
    Args:
        team_list: A list of dataframes, the input dataframe (season, seed, team) split by season.
        seed_list: A dictionary of team names keyed by team ID and seasons after 2003.
        names: A dictionary of team names keyed by team ID.
        begin: A year. The start of the range to create combinations for.
        end: A year. The start of the range to create combinations for. 
    Returns:
        None, but prints the result to "AllCombinations.csv".
    """
    # Convert the years to years after 2003
    begin = begin - 2003
    end = end - 2003
    # Iterating through the range
    for i in range(begin, end + 1):
        # Getting the data frame for that year for teams and seeds in the tournament.
        data = team_list[i]
        # Create a list of all combinations of pairs of teams.
        combos = list(combinations(data['Team'], 2))
        # Creating a dataframe from the above list.
        team_frame = pd.DataFrame(combos, columns=['T1', 'T2'])
        # Adding the season and seed.
        team_frame["Season"] = (i + 2003)
        team_frame["T1Seed"] = team_frame["T1"].apply(lambda x: int((seed_list[i][x])[1:3]))
        team_frame["T2Seed"] = team_frame["T2"].apply(lambda x: int((seed_list[i][x])[1:3]))
        # Combine the dataframes and print out to csv.
        if i == begin:
            all_years = team_frame
        else:
            all_years = pd.concat([all_years, team_frame], axis = 0)
        all_years.to_csv("AllCombinations.csv")
    

if __name__=="__main__":
    main()
