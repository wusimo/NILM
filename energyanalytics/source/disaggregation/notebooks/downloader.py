import sys
import datetime
from loadTools import *
from os import path

user = 'HIKQ'
pswd = 'HIKQ'
server = 'app.equotaenergy.com'
ID = '143'

start_dt = datetime.datetime(2016, 3, 22)
end_dt = datetime.datetime(2016, 8, 1)
date_current = start_dt

while date_current < end_dt:
    print date_current
    try:
        points_sr = load_data().get_series_as_pd(user, pswd, server, ID, date_current
                                                 , date_current+datetime.timedelta(hours=1))

        file_name = '%d-%d-%d.csv' % (date_current.month, date_current.day, date_current.hour)
        file_path = path.join( 'new_data', 'IHG', file_name )

        points_sr.to_csv(file_path)
        print 'done'
    except KeyboardInterrupt:
        raise()
    except:
        print("Unexpected error:", sys.exc_info())
        print 'failed'

    date_current += datetime.timedelta(hours=1)

