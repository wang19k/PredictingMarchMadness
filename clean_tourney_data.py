import sys
import pandas as pd
import csv

COLUMNS = ["Season", "Seed", "Team"]

def main():
    filter_inputs("TourneySeeds.csv", "TourneySeedsClean.csv")


def filter_inputs(filename, outfile):
    inputs = pd.read_csv(open(filename), names=COLUMNS,
                         skipinitialspace=True,
                         skiprows=1, engine="python")
    inputs = inputs.groupby("Season").apply(lambda x: x[x["Season"] >= 2003])
    inputs.to_csv(outfile)

if __name__=="__main__":
    main()
