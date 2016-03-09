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
Try mave out on an example file using any of the examples methods below.
Each of these will build a model on the dataset contained in ex3.csv, the 
difference between the three approaches is in how the post-retrofit period
is defined.
```
    mave ex3.csv 
```
This assumes the last 25% of the file represents the postretrofit period as the
default value of the 'test_size' argument is 0.25. 
```
    mave ex3.csv -ts 0.45
```
This uses the 'ts' or 'test_size' argument to explicitly specify the fraction 
of the file to use. In this example the last 45% of the file represents the 
postretrofit period. 
```
    mave ex3.csv -cp "1/11/2013 00:00"
```
This example uses the 'cp' or 'changepoint' argument to explicitly define the
date at which the post retrofit period begins. This overrides the 'test-size' 
value. In this case all data on or after Jan 11, 2013 at 00:00 represents the 
post-retrofit period.
```
    mave ex3.csv -cp "1/11/2013 00:00" -ad "berkeley, california"
```
This example uses the 'ad' or 'address' argument to include a physical address
in sunny Berkeley, California. Mave will use the Google Maps API to
resolve that to a latitude and longitude, which will then be used to lookup the
nearest available historical weather data and a Typical Meteorological Year 
for that location.

Mave has many configurable options some of which can be passed as command line 
arguments (run mave -h for details) and many more which can be passed using a 
separate configuration file. Command line arguments override those in the configuration file.
However, the configuration file also allows many other advanced modeling options, 
such as specifying multiple different periods to use as pre-retrofit or post-retrofit period, 
or periods to ignore entirely. Review the wiki documentation, or the default config file 
(at /mave/mave/config/default.cfg) for detailed descriptions of the various options. 

Results
------------
The results of the analysis are contained in the log file. The 's' (or 'save') argument
will serialize and save the model(s), along with the .csv files for the measured
and predicted datasets.

To cite this tool: 
Paul Raftery & Tyler Hoyt, 2016, Mave: software automated Measurement and Verification. 
Center for the Built Environment, University of California Berkeley, 
https://github.com/CenterForTheBuiltEnvironment/mave
