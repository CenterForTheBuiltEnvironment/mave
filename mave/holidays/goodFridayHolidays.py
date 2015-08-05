import pickle
from datetime import datetime, date, time

# define Good Fridays 2000-2015
holidays = [date(2000, 4, 21),
            date(2001, 4, 13),
            date(2002, 3, 29),
            date(2003, 4, 18),
            date(2004, 4, 9),
            date(2005, 3, 25),
            date(2006, 4, 14),
            date(2007, 4, 6),
            date(2008, 3, 21),
            date(2009, 4, 10),
            date(2010, 4, 2),
            date(2011, 4, 22),
            date(2012, 4, 6),
            date(2013, 3, 29),
            date(2014, 4, 18),
            date(2015, 4, 3)]

fileObject = open('.\\goodFridayHolidays.p', 'w')
pickle.dump(holidays, fileObject)
fileObject.close()
