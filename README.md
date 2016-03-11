Mave
======
Mave is a tool for automated measurement and verification (M&V). At its most 
simple, the aim is to read energy consumption data from before and after a 
retrofit (pre-retrofit and post-retrofit data) and to predict how much energy 
the retrofit saved. Mave does this by training multiple models to the data and 
using the best model to predict energy consumption what the energy consumption 
would have been in the post-retrofit period, had the retrofit not happened.

Mave automatically resolves common problems with input files (missing data, 
irregular timestamps, outliers, etc.), builds input features from the data 
(including federal holidays, downloading weather data for the location, etc.),
and normalizes the results to a Typical Meteorological Year (TMY) for a given
physical address.

Installation
------------
Assuming all of the dependencies are installed, install mave from source using:
```
    python setup.py install
```

Alternatively, install mave from pypi using:
```
    pip install mave --no-deps
```

If you run into trouble with the dependencies, see the 
[Installation page of the wiki](https://github.com/CenterForTheBuiltEnvironment/mave/wiki/Installation)

Usage
------------
Try mave out on an example file such as  [example.csv](https://raw.githubusercontent.com/CenterForTheBuiltEnvironment/mave/master/mave/data/ex1.csv). 
This file contains 8 months of preretrofit data and 8 months of postretrofit data,
with the retrofit occurring at 6/29/2013 20:15.using any of the examples methods below.

Each of the commands below will build a model on the example file data and predict the savings. 
The difference between these approaches is primaril in how the post-retrofit period is defined,
and whether or not mave normalizes the results to a typical year dataset. 
```
    mave example.csv 
```
This assumes the last 25% of the file represents the postretrofit period as the
default value of the 'test_size' argument is 0.25. 
```
    mave example.csv -ts 0.5
```
This uses the 'ts' or 'test_size' argument to explicitly specify the fraction 
of the file to use. In this example the 50% of the file represents the 
postretrofit period (which is approximately correct for example.csv).
```
    mave example.csv -cp "2013/6/30 02:30"
```
This example uses the 'cp' or 'changepoint' argument to explicitly define the
date at which the post retrofit period begins. This overrides the 'test-size' 
value. In this case all data on or after June 30, 2013 at 02:30 represents the 
post-retrofit period. 

Note that this  is the actual datetime that the postretrofit periods
begins for example.csv. If you are wondering about mave's accuracy, for the 
hypothetical scenario in this file, the savings over the postretrofit period 
is a constant value of 6 units for each measured data point (or 15% NMBE).
```
    mave example.csv -cp "2013/6/30 02:30" -ad "berkeley, california"
```
This example uses the 'ad' or 'address' argument to include a physical address
in sunny Berkeley, California. Mave will use the Google Maps API to
resolve that to a latitude and longitude, which will then be used to lookup the
nearest available historical weather data if none is provided in the input file (it 
is in the case of example.csv) and a Typical Meteorological Year for that location.

Mave has many configurable options some of which can be passed as command line 
arguments (run mave -h for details) and many more which can be passed using a 
separate configuration file. Command line arguments override those in the config file.
However, the configuration file also allows many other advanced modeling options, 
such as specifying multiple different periods to use as pre- or post-retrofit period, 
or periods to ignore entirely. The advanced modeling options also allow the user 
to control what input features are used for the model. For example, if the input 
file has a lot of data, and seasonal production or occupancy, the user may want 
to include month as an input feature. Review the wiki documentation, or the [default config file] 
(https://github.com/CenterForTheBuiltEnvironment/mave/blob/master/mave/config/default.cfg)
for detailed descriptions of the various options. 

Results
------------
The results of the analysis are contained in the log file. The 's' (or 'save') argument
will serialize and save the model(s), along with the .csv files for the measured
and predicted datasets.

A future version of mave will also include an option to plot figures and results in 
a pdf file.

Auxiliary scripts
------------
mave also comes with two additional scripts, mave-weather and mave-tmy, for downloading
weather data and TMY data for a given location, respectively. Please see examples below
and running each command with the 'h' (or 'help') argument will describe each of the 
arguments in more detail.
```
    mave-weather 'berkeley, ca' -s '2010-01-01' -e '2010-01-05' -i 15
```
The above command will download and save the nearest available historical weather data for 
Berkeley, CA, USA, from Jan 1, 2010 to Jan 5 2010, and interpolate it to 15 minute intervals.
```
     mave-tmy 'berkeley, ca' -i 15 -y 2015
```
The above command will download and save the nearest available TMY data for Berkeley, CA, USA.
It will also create a separate file containing interpolated data, and overwrite the year 
provided in the TMY file to 2015 (as it is  useful to have a continuous year of data
for modeling purposes).

To cite this tool: 
Paul Raftery & Tyler Hoyt, 2016, Mave: software automated Measurement and Verification. 
Center for the Built Environment, University of California Berkeley, 
https://github.com/CenterForTheBuiltEnvironment/mave
