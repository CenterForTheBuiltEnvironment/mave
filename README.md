Mave
======
Mave is a tool for automated measurement and verification. At it's most simple, the aim
is to read energy consumption data from before and after a retrofit (pre-retrofit and 
post-retrofit data) and to predict how much energy the retrofit saved. Mave does this 
by training a model(s) to the data and using that model to predict energy consumption 
during a different period. Mave uses three different approaches to do this.


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

If you run into trouble with the dependencies, see the [Installation page of the wiki 
on github here](https://github.com/CenterForTheBuiltEnvironment/mave/wiki/Installation)

Usage
------------
Try mave out on an example file using:
```
    mave ex3.csv "1/11/2013 00:00"
```

This will build a model on the dataset contained in ex3.csv, assuming that the postretrofit 
period begins on Jan 11, 2013 at 00;00.

Mave has many configurable options (e.g. -v for verbose output) which can be passed as 
command line arguments or using a separate config file. Review the wiki documentation for more details. 

