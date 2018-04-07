import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split

COLUMNS = ["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc", "Numot", "Wfgm",
           "Wfga", "Wfgm3", "Wfga3", "Wftm", "Wfta", "Wor", "Wdr", "Wast", "Wto", "Wstl",
           "Wblk", "Wpf", "Lfgm", "Lfga", "Lfgm3", "Lfga3", "Lftm", "Lfta", "Lor", "Ldr",
           "Last", "Lto", "Lstl", "Lblk", "Lpf"]


def main():
    team_names = create_dict_from_csv("teams.csv")
    #print(team_names)
    inputs = get_input_frames("RegularSeasonDetailedResults.csv", team_names)
    print(inputs.head())

def create_dict_from_csv(filename):
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None) # Get's rid of the headers
        team_names = {int(rows[0]):rows[1] for rows in reader}
        return team_names

def get_input_frames(filename, names):
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    # Can drop columns here
    # features = inputs[FEATURES], once we decide what FEATURES will be
    inputs["Wteam"] = inputs["Wteam"].apply(lambda x: names[x])
    inputs["Lteam"] = inputs["Lteam"].apply(lambda x: names[x])
    return inputs

if __name__=="__main__":
    main()
