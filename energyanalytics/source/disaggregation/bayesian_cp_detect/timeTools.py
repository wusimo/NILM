# coding=utf-8
'''
Filename: convert_time.py

Description: offer time converters

Change activity:
    2016-5-5 new
'''
from __future__ import division
from __future__ import unicode_literals
from dateutil import parser
import calendar
import pytz


class convert_time:
    ''' convert_datetime_to_utc, convert_timestr_to_utc'''

    def __init__(self):
        pass

    # def __str__(self):
    #     return "convert_datetime_to_utc, convert_timestr_to_utc"

    def from_dt_to_utc(self, time_dt, time_zone='Asia/Shanghai'):
        """
        Description: convert datetime format to utc timestamp

        Parameters: time_dt: datetime,
                    time_zone: str, default 'Asia/Shanghai'

        Returns: timestamp: int,
        """
        tz = pytz.timezone(time_zone)
        if time_dt.tzinfo is None:
            time_dt_local = tz.localize(time_dt)
        else:
            time_dt_local = time_dt

        time_utc = calendar.timegm(time_dt_local.utctimetuple())

        return time_utc

    def from_str_to_utc(self, timeStr, timeZoneStr=" +8"):
        '''
        Description: convert string time to utc timestamp

        Parameters: timeStr: str, e.g. '2015-01-01 00:00:00'
                    timeZoneStr: str, e.g. ' +8', defualt timezone is Shanghai

        Returns:    timestamp: int, utc timestamp
        '''
        # set timezone
        timeStr = timeStr + timeZoneStr
        # convert datetime format
        parsedStr = parser.parse(timeStr)
        # return utc timestamp
        return calendar.timegm(parsedStr.utctimetuple())
