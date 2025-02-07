import math

class GlobalMercator(object):
    def __init__(self):
        self.originShift = 2 * math.pi * 6378137 / 2.0

    def LatLonToMeters(self, lat, lon):
        "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"

        mx = lon * self.originShift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

        my = my * self.originShift / 180.0

        return mx, my

    def MetersToLatLon(self, mx, my):
        "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"

        lon = (mx / self.originShift) * 180.0
        lat = (my / self.originShift) * 180.0

        lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)

        return lat, lon


from pyproj import Proj, transform

class GlobalMercatorPyProj(object):
    def __init__(self):
        self.proj_wgs84 = Proj(init='epsg:4326')
        self.proj_3395 = Proj(init='epsg:3395')

    def LatLonToMeters(self, lat, lon):
        x, y = transform(self.proj_wgs84, self.proj_3395, lon, lat)
        return x, y

    def MetersToLatLon(self, x, y):
        lat, lon = transform(self.proj_3395, self.proj_wgs84, x, y)
        return lat, lon
