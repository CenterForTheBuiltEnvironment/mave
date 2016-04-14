import glob
import ntpath
import pandas as pd
from subprocess import call

allCsv = glob.glob("./*.csv")
csvFilenames = [ntpath.basename(csv).split('.')[0] for csv in allCsv]

metaData = pd.read_csv("../Meta_enernoc2014.csv")

csvFilenames = [csvFilenames[2]]
for csv in csvFilenames:
    lat = metaData.lat[metaData.siteid == csv].values[0]
    lng = metaData.lng[metaData.siteid == csv].values[0]
    call(["mave", allCsv[csvFilenames.index(csv)], "-conf", "enernoc2.cfg", "-ad", str(lat) + " " + str(lng)])
