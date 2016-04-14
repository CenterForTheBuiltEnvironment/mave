import glob
import ntpath
import pandas as pd
from subprocess import call

allCsv = glob.glob("./*.csv")
csvFilenames = [ntpath.basename(csv).split('.')[0] for csv in allCsv]

metaData = pd.read_csv("../meta/all_sites.csv")

for csv in csvFilenames:
    lat = metaData.LAT[metaData.SITE_ID == int(csv)].values[0]
    lng = metaData.LNG[metaData.SITE_ID == int(csv)].values[0]
    call(["mave", allCsv[csvFilenames.index(csv)], "-conf", "enernoc1.cfg", "-ad", str(lat) + " " + str(lng)])
