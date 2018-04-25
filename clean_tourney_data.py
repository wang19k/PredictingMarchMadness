import sys
import pandas as pd
import csv

COLUMNS = ["Season", "Seed", "Team"]
TOURNEY_COLS = ["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"]
ROUNDS = {136: 1, 137: 1, 138: 2, 139: 2, 143: 3, 144: 3, 145: 4,
          146: 4, 152: 5, 154: 6}


def main():
    #filter_inputs("TourneySeeds.csv", "TourneySeedsClean.csv")
    team_names = create_dict_from_csv("teams.csv")
    input_list = get_input_frames("TourneyDetailedResults.csv")
    seed_list = get_seeds("TourneySeedsClean.csv")
    clean_inputs(input_list, seed_list, team_names)


def filter_inputs(filename, outfile):
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    inputs = inputs.groupby("Season").apply(lambda x: x[x["Season"] >= 2003])
    inputs.to_csv(outfile)

def create_dict_from_csv(filename):
    """Creates dictionary mapping team ID to team names."""
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Get's rid of the headers
        team_names = {int(rows[0]):rows[1] for rows in reader}
        return team_names

def get_seeds(filename):
    seed_list = [{} for _ in range(2003, 2018)]
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Get's rid of the headers
        for rows in reader:
            seed_list[(int(rows[0]) - 2003)][int(rows[2])] = rows[1]
    return seed_list

def get_input_frames(filename):
    """Returns the file as a dataframe with modified columns."""
    inputs = pd.read_csv(open(filename), names=TOURNEY_COLS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    inputs.drop(["Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"], axis=1, inplace=True)
    # Getting rid of the games beyond the Field of 64
    inputs = inputs[inputs.Daynum != 134][inputs.Daynum != 135]
    # Creating the score differential
    inputs["Wscorediff"] = inputs["Wscore"].subtract(inputs["Lscore"])
    # Lscorediff will be negative
    inputs["Lscorediff"] = inputs["Lscore"].subtract(inputs["Wscore"])
    input_list = []
    for year in range(2003, 2018):
        input_list.append(inputs[inputs.Season == year])
    return input_list

def clean_inputs(input_list, seed_list, names):
    #winners = inputs.drop(["Lteam", "Lscore", "Lscorediff"], axis=1)
    #losers = inputs.drop(["Wteam", "Wscore", "Wscorediff"], axis=1)
    for i in range(len(seed_list)):
        data = input_list[i]
        data["WSeed"] = data["Wteam"].apply(lambda x: seed_list[i][x])
        data["LSeed"] = data["Lteam"].apply(lambda x: seed_list[i][x])
        # Apply their names
        #data["Wteam"] = data["Wteam"].apply(lambda x: names[x])
        #data["Lteam"] = data["Lteam"].apply(lambda x: names[x])
        data["Round"] = data["Daynum"].apply(lambda x: ROUNDS[x])
        data.drop("Daynum", axis=1, inplace=True)
        if i == 0:
            all_years = data
        else:
            all_years = pd.concat([all_years, data], axis = 0)
        if not data.empty:
            data.to_csv("TourneyData/" + str(i + 2003) + "TourneyResults.csv")
        all_years.to_csv("AllTourneyResults.csv")
    

if __name__=="__main__":
    main()
