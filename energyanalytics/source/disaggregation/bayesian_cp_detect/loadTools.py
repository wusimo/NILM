# coding=utf-8
'''
Filename: load.py

Description: used to download, upload, remove data

Change activity:
    2016-5-11 review
'''
from __future__ import division
from __future__ import unicode_literals
from timeTools import convert_time
from base64 import encodestring
from json import loads
from os.path import split
from sys import exc_info
from sys import exit
from urllib import urlencode
from urllib2 import Request
from urllib2 import urlopen
from pandas import Series
from pandas import to_datetime


class load_data():
    '''False - error; None - blank; True - success'''

    def __init__(self):
        pass

    def _sending_request_to_DB(self, user, pswd, url, data):
        """
        Description: send request to DB

        Parameters: user: str,
                    pswd: str,
                    url: str,
                    data: urlencode,

        Returns: results: dict
        """
        login = encodestring('%s:%s' % (user, pswd)).replace('\n', '')

        additional = {'User-Agent': 'Mozilla/5.0',
                      'Authorization': 'Basic %s' % login}
        try:
            request = Request(url, data, additional)
            results = urlopen(request, timeout=5).read()
            return results
        except Exception, e:
            exc_type, exc_obj, exc_tb = exc_info()
            fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
            print exc_type, fname, exc_tb.tb_lineno
            print 'Error: ', e
            return False

        print "success to send request"

    def download_series_from_DB(self, user, pswd, addr, meter_id,
                                start_utc=None, end_utc=None,
                                limit=None, disagg=None):
        """
        Description: download data from DB directly as utc timestamp

        Parameters: user: str, -> sending_request_to_DB
                    pswd: str, -> sending_request_to_DB
                    addr: str,
                    meter_id: str,
                    start_utc: int, defaut None
                    end_utc: int, default None
                    limit: int, default None
                    disagg: str, default None, e.g. lighting, misc, cooling,
                                                    heating, motor, plug
                    tz: str,

        Returns: results: list
        """
        values = {'isExternalRequest': True, 'start_utc': start_utc,
                  'end_utc': end_utc, 'limit': limit, 'disagg': disagg}

        if start_utc is None or end_utc is None:
            values.pop('start_utc')
            values.pop('end_utc')

        if limit is None:
            values.pop('limit')

        if disagg is None:
            values.pop('disagg')

        data = urlencode(values)
        url = "http://" + addr + "/api/getseries/" + meter_id + "/"
        try:
            results = loads(self._sending_request_to_DB(user, pswd, url, data))
            if results == []:
                return None
            points = results[0]['points']
        except Exception, e:
            exc_type, exc_obj, exc_tb = exc_info()
            fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
            print exc_type, fname, exc_tb.tb_lineno
            print 'Error: ', e
            return False

        return list(reversed(points))

    def get_series_as_pd(self, user, pswd, addr, meter_id,
                         start_dt=None, end_dt=None,
                         limit=None, disagg=None, tz='Asia/Shanghai'):
        """
        Description: download data as datetime and output pandas.Series

        Parameters: user: str,
                    pswd: str,
                    addr: str,
                    meter_id: str,
                    start_dt: datetime, defaut None
                    end_dt: datetime, default None
                    limit: int, default None
                    disagg: str, default None, e.g. lighting, misc, cooling,
                                                    heating, motor, plug
                    tz: str,

        Returns: results: pandas.Series
        """
        if start_dt is not None and end_dt is not None:
            start_utc = convert_time().from_dt_to_utc(start_dt, tz)
            end_utc = convert_time().from_dt_to_utc(end_dt, tz)
        else:
            start_utc = None
            end_utc = None

        points = self.download_series_from_DB(user, pswd, addr, meter_id,
                                              start_utc, end_utc,
                                              limit, disagg)
        if points:
            timestamp = list(zip(*points)[0])
            value = list(zip(*points)[-1])
            if timestamp[-1] == end_utc:
                timestamp.pop()
                value.pop()

            index_utc = to_datetime(timestamp, unit='s', utc=True)
            index_local = index_utc.tz_convert(tz)

            points_sr = Series(value, index=index_local, name=meter_id)

            return points_sr
        else:
            return None

    def upload_series_to_DB(self, user, pswd, addr, meter_id,
                            points, disagg=None):
        """
        Description: upload data to DB as utc timestamp

        Parameters: user: str, -> sending_request_to_DB
                    pswd: str, -> sending_request_to_DB
                    addr: str,
                    meter_id: str,
                    points: list, e.g. [[t, v], [t, v]]
                    disagg: str, default None, e.g. lighting, misc, cooling,
                                                    heating, motor, plug

        Returns: True: boolean,
        """
        values = {'isExternalRequest': True, 'points': points, 'disagg': disagg}

        if disagg is None:
            values.pop('disagg')

        data = urlencode(values)

        url = "http://" + addr + "/api/putseries/" + meter_id + "/"

        try:
            result = self._sending_request_to_DB(user, pswd, url, data)
            if result == 'True':
                print "Success to upload"
                return True
        except Exception, e:
            exc_type, exc_obj, exc_tb = exc_info()
            fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
            print exc_type, fname, exc_tb.tb_lineno
            print 'Error: ', e
            return False

    def remove_on_DB(self, user, pswd, addr, meter_id, start_utc, end_utc,
                     disagg=None):
        """
        Description: remove data on DB directly as utc timestamp

        Parameters: user: str, -> sending_request_to_DB
                    pswd: str, -> sending_request_to_DB
                    addr: str,
                    meter_id: str,
                    start_utc: int, defaut None
                    end_utc: int, default None
                    disagg: str, default None, e.g. lighting, misc, cooling,
                                                    heating, motor, plug

        Returns: True: boolean,
        """
        points = []
        erase_flag = "True"
        values = {'isExternalRequest': True, 'points': points,
                  'start_utc': start_utc, 'end_utc': end_utc, 'disagg': disagg,
                  'erase_flag': erase_flag}
        if disagg is None:
            values.pop('disagg')

        data = urlencode(values)
        url = "http://" + addr + "/api/putseries/" + meter_id + "/"

        print values
        print url
        yes = raw_input("Are you sure to remove? y/n\n")
        if yes in ['yes', 'y', 'YES', 'Y', 'Yes']:
            results = self._sending_request_to_DB(user, pswd, url, data)
        else:
            exit("try again...")

        if results is None:
            exit()
        if results == "True":
            print meter_id + " success to remove"
            return True
        else:
            exit(meter_id+" failed to remove !")


if __name__ == '__main__':
    from datetime import datetime
    user = 'HIKQ'
    pswd = 'HIKQ'
    server = 'app.equotaenergy.com'
    ID = '143'
    start_dt = datetime(2016, 7, 1)
    end_dt = datetime(2016, 7, 1, 0, 30)
    # points = load_data().download_series_from_DB(user, pswd, server, ID, limit=1)
    points_sr = load_data().get_series_as_pd(user, pswd, server, ID, start_dt,
                                             end_dt)
    print points_sr
    exit()
