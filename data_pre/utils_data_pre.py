import os
import pandas as pd
import numpy as np
from osgeo import osr
import geopandas
import ee
import eemont
from shapely.geometry import Polygon
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  # Setup the proxy if required
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
ee.Initialize()



class PointGeometry:
    # used for reprojection, build point with rectangle buffer
    def __init__(self, source_EPSG, target_proj):
        self.source_EPSG = source_EPSG
        self.target_proj = target_proj
        self.transform = self.build_geo_transform()

    def build_geo_transform(self):
        source = osr.SpatialReference()
        source.ImportFromEPSG(self.source_EPSG)
        target = osr.SpatialReference()
        if type(self.target_proj) == int:
            target.ImportFromEPSG(self.target_proj)
        else:
            target.ImportFromProj4(self.target_proj)
        return osr.CoordinateTransformation(source, target)

    def re_project(self, x, y):
        """
        :param x: Longitude or x
        :param y: Latitude or y
        :return:
        """
        if self.source_EPSG == 4326:
            location = self.transform.TransformPoint(y, x)
        else:
            location = self.transform.TransformPoint(x, y)
            location = [location[1], location[0]]
        return location


class grids_4_a_region:
    # used for reprojection, build point with rectangle buffer
    def __init__(self, source_EPSG, gridSize=9):
        self.source_EPSG = source_EPSG
        self.target_proj = 6933  # EASE 2.0 EPSG

        self.EASE_ulx = -17367530.45
        self.EASE_uly = 7314540.83
        if gridSize == 9:
            self.pixelWidth = 9008.05  # 36032.22
            self.pixelHeight = -9008.05  # 36032.22
        elif gridSize == 36:
            self.pixelWidth = 36032.22
            self.pixelHeight = -36032.22
        elif gridSize == 72:
            self.pixelWidth = 36032.22 * 2
            self.pixelHeight = -36032.22 * 2
        elif gridSize <= 1:
            self.pixelWidth = 1000.90*gridSize
            self.pixelHeight = -1000.90*gridSize

        self.pgeo = PointGeometry(self.source_EPSG, self.target_proj)
        self.pgeo_inv = PointGeometry(self.target_proj, self.source_EPSG)
        self.r=0
        self.c=0
        self.lon=0
        self.lat=0

    def get_boundary(self, path_2_region_shp):
        region_shp = geopandas.read_file(path_2_region_shp)
        top_left = self.pgeo.re_project(region_shp.bounds['minx'].min(), region_shp.bounds['maxy'].max())
        bottom_right = self.pgeo.re_project(region_shp.bounds['maxx'].max(), region_shp.bounds['miny'].min())
        minx, maxy = top_left[:2]
        maxx, miny = bottom_right[:2]

        minx = np.floor((minx - self.EASE_ulx) / self.pixelWidth) * self.pixelWidth + self.EASE_ulx
        maxx = np.ceil((maxx - self.EASE_ulx) / self.pixelWidth) * self.pixelWidth + self.EASE_ulx
        miny = np.ceil((miny - self.EASE_uly) / self.pixelHeight) * self.pixelHeight + self.EASE_uly
        maxy = np.floor((maxy - self.EASE_uly) / self.pixelHeight) * self.pixelHeight + self.EASE_uly

        outputBounds = (minx, maxx, miny, maxy)
        return outputBounds

    def xy_2_cr(self, x, y):
        c = []
        r = []
        for xx, yy in zip(x, y):
            c.append(int(np.floor((xx - self.EASE_ulx) / self.pixelWidth)))
            r.append(int(np.ceil((yy - self.EASE_uly) / self.pixelHeight)))
        return r, c

    def cr_2_xy(self, c, r):
        x = []
        y = []
        for cc, rr in zip(c, r):
            x.append(cc * self.pixelWidth + self.EASE_ulx)
            y.append(rr * self.pixelHeight + self.EASE_uly)
        return x, y

    def cr_gee_ring(self, c, r):
        bx = [c * self.pixelWidth + self.EASE_ulx,
              (c + 1) * self.pixelWidth + self.EASE_ulx,
              (c + 1) * self.pixelWidth + self.EASE_ulx,
              c * self.pixelWidth + self.EASE_ulx,
              c * self.pixelWidth + self.EASE_ulx]
        by = [r * self.pixelHeight + self.EASE_uly,
              r * self.pixelHeight + self.EASE_uly,
              (r - 1) * self.pixelHeight + self.EASE_uly,
              (r - 1) * self.pixelHeight + self.EASE_uly,
              r * self.pixelHeight + self.EASE_uly]
        ring = []
        for x, y in zip(bx, by):
            temp = self.pgeo_inv.re_project(x, y)
            ring.append(temp[:2])
        return ring, bx, by

    def get_wgs_grid(self, x, y):
        self.lat=y
        self.lon=x
        targetxy = self.pgeo.re_project(x, y)
        xx, yy = targetxy[:2]
        r, c = self.xy_2_cr([xx], [yy])
        self.r=r
        self.c=c
        ring, bx, by = self.cr_gee_ring(c[0], r[0])
        ring_target = [(bx[0], by[1]),
                       (bx[1], by[1]),
                       (bx[2], by[2]),
                       (bx[3], by[3])]
        return ring, ring_target

    def get_pixel_ring(self, x, y, res):
        res = res / 2
        polygon_ring = []
        for i in range(len(x)):
            loc = self.pgeo.re_project(x[i], y[i])
            polygon_ring.append([(loc[0] - res, loc[1] - res),
                                 (loc[0] - res, loc[1] + res),
                                 (loc[0] + res, loc[1] + res),
                                 (loc[0] + res, loc[1] - res)])
        return polygon_ring


def cal_pixel_weight(pixel_ring, grid_ring):
    grid_p = Polygon(grid_ring)
    area_ratio = []
    for a_ring in pixel_ring:
        pixel_p = Polygon(a_ring)
        area_ratio.append(grid_p.intersection(pixel_p).area / 100)
    return area_ratio


def parse_S1_platform_orbit(fname):
    platform = fname.split('_')[1]
    obs_orbit = int(fname.split('_')[7])
    if platform == 'S1A':
        rel_orbit = (obs_orbit - 73) % 175 + 1
        # platform = 0  # use 0 to represent the Sentinel-1A
    else:
        rel_orbit = (obs_orbit - 27) % 175 + 1  # Sentinel-1B
        # platform = -1  # use -1 to represent the Sentinel-1B
    return platform, rel_orbit


def parse_ts_s1(ts_info, sel_band):
    ts = []
    cols = ['platform', 'orbit']
    for i in range(len(ts_info['features'])):
        temp = []
        platform, orbit = parse_S1_platform_orbit(ts_info['features'][i]['id'])
        temp.append(platform)
        temp.append(orbit)
        for band in sel_band:
            try:
                temp.append(ts_info['features'][i]['properties'][band])
            except:
                temp.append(np.nan)
        ts.append(temp)
    cols.extend(sel_band)
    df = pd.DataFrame(ts, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    return df


class GeeS1TsExtractor:

    def __init__(self, start_date, end_date, geometry, orbit_properties_pass='DESCENDING',
                 dir_name='', save_file=True):
        self.product = "COPERNICUS/S1_GRD"
        self.bands = ["VV", "VH", "angle"]
        self.start_date = start_date
        self.end_date = end_date
        self.instrumentMode = 'IW'
        self.orbit_properties_pass = orbit_properties_pass
        self.geometry = geometry
        self.filtered_collection = self.sentinel1_filtered_collection()
        self.image_size = self.filtered_collection.size().getInfo()
        self.save = save_file
        if self.image_size > 0:
            self.set_output_dir(dir_name)

    def sentinel1_filtered_collection(self):

        im_collection = ee.ImageCollection(self.product).filterDate(self.start_date, self.end_date)  # product and date
        im_collection = im_collection.filter(ee.Filter.eq('orbitProperties_pass', self.orbit_properties_pass))  # orbit
        im_collection = im_collection.filter(ee.Filter.eq('instrumentMode', self.instrumentMode)).select(
            self.bands)  # IW mode and bands
        im_collection = im_collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(
            ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        im_collection = im_collection.filterBounds(self.geometry)
        return im_collection

    def set_output_dir(self, dir_name):
        if dir_name is None or dir_name == '':
            self.dir_name = ''
        else:
            self.dir_name = os.path.join(dir_name, '')

    def download_data(self):
        sel_band = ['date']
        sel_band.extend(self.bands)
        fc = ee.FeatureCollection([self.geometry])
        col = self.filtered_collection
        ts = col.getTimeSeriesByRegions(reducer=[ee.Reducer.mean()],
                                        collection=fc,
                                        bands=self.bands,
                                        scale=10)
        ts_info = ts.getInfo()
        df = parse_ts_s1(ts_info, sel_band)
        return df

    def get_and_save_data(self, site_name):
        file_name = f'{self.dir_name}{site_name}.csv'
        point_df = self.download_data()
        if self.save:
            point_df.to_csv(file_name)
        return point_df


def extract_S1(geometry, START_DATE, END_DATE, S1_DIR=''):
    ORBIT_PASS_A = 'ASCENDING'
    ORBIT_PASS_D = 'DESCENDING'

    save_to_disk = False
    df_all = []
    extractor_A = GeeS1TsExtractor(START_DATE, END_DATE, geometry, ORBIT_PASS_A, S1_DIR, save_to_disk)
    extractor_D = GeeS1TsExtractor(START_DATE, END_DATE, geometry, ORBIT_PASS_D, S1_DIR, save_to_disk)

    if extractor_A.image_size > 0:  # Extract Sentinel-1 Ascending data
        df_S1A = extractor_A.get_and_save_data('temp')
        df_all.append(df_S1A)

    if extractor_D.image_size > 0:  # Extract Sentinel-1 Descending data
        df_S1D = extractor_D.get_and_save_data('temp')
        df_all.append(df_S1D)
    if len(df_all) == 2:
        df_all = pd.concat(df_all, axis=0)
    else:
        df_all = df_all[0]
    df_all = df_all.sort_values('date')
    df_all.index = range(len(df_all))
    df_all['DOY_sin'] = np.sin(df_all.date.dt.day_of_year / 365 * np.pi * 2)
    df_all['DOY_cos'] = np.cos(df_all.date.dt.day_of_year / 365 * np.pi * 2)
    return df_all


def parse_ts(ts_info, sel_band):
    ts = []
    for i in range(len(ts_info['features'])):
        temp = []
        for band in sel_band:
            try:
                temp.append(ts_info['features'][i]['properties'][band])
            except:
                temp.append(np.nan)
        ts.append(temp)
    df = pd.DataFrame(ts, columns=sel_band)
    df['date'] = pd.to_datetime(df['date'])
    return df


class GeeTsExtractor:
    def __init__(self, product, bands, start_date, end_date, dir_name='', interp=False,save_file=True):
        self.product = product
        self.bands = bands
        self.start_date = start_date
        self.end_date = end_date
        self.collection = ee.ImageCollection(product)
        self.interp=interp
        # self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)

        self.set_output_dir(dir_name)

        self.save = save_file

    def set_date_range(self, start_date, end_date, freq='1D', gap_fill=True, max_gap=None):

        self.start_date = start_date
        self.end_date = end_date
        if gap_fill:
            date_range = pd.date_range(start_date, end_date, freq=freq, closed="left")
            self.days = pd.Series(date_range, name="id")
            self.fill = pd.DataFrame(date_range, columns=["id"])
            self.gap_fill = gap_fill
            self.max_gap = max_gap
        else:
            self.gap_fill = False

    def filtered_collection(self):
        return self.collection.filterDate(self.start_date, self.end_date)

    def set_output_dir(self, dir_name):

        if dir_name is None or dir_name == '':
            self.dir_name = ''
        else:
            self.dir_name = os.path.join(dir_name, '')

    def download_data(self, geometry, scale):
        sel_band = ['date']
        sel_band.extend(self.bands)
        fc = ee.FeatureCollection([geometry])
        col = self.filtered_collection()
        ts = col.getTimeSeriesByRegions(reducer=[ee.Reducer.mean()],
                                        collection=fc,
                                        bands=self.bands,
                                        scale=scale)
        ts_info = ts.getInfo()
        df = parse_ts(ts_info, sel_band)
        df = df.dropna()

        df = df.set_index('date')
        # List of dates for interpolation
        date_list = pd.date_range(self.start_date, self.end_date, freq='1D')

        # Reindex DataFrame using the list of dates
        df_reindexed = df.reindex(date_list)

        # Interpolate missing values
        if self.interp:
            df = df_reindexed.interpolate()
        return df

    def download_data_point(self, geometry, scale):

        data = self.filtered_collection().select(self.bands).getRegion(geometry, scale, 'EPSG:4326').getInfo()

        data_df = pd.DataFrame(data[1:], columns=data[0])
        bands_index = pd.DatetimeIndex(pd.to_datetime(data_df.time, unit='ms').dt.date)
        bands_df = data_df[self.bands].set_index(bands_index).rename_axis(index='time').sort_index()
        return bands_df

    def get_and_save_data(self, geometry, site_name, scale=463.313):
        file_name = f'{self.dir_name}{site_name}.csv'
        point_df = self.download_data(geometry, scale=scale)
        if self.save:
            point_df.to_csv(file_name)
        return point_df


def extract_terrain_soil_texture_v1(geometry, ring_wgs, pobj):
    def re_order(aux):
        soil = ['sand_0-5cm_mean', 'clay_0-5cm_mean', 'bdod_0-5cm_mean']
        terren = ['elevation', 'slope', 'aspect_sin', 'aspect_cos']
        soil.extend(terren)
        sorted_dict = {key: aux[key] for key in soil}
        return sorted_dict

    SAND_IMAGE = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')  # .select(sand);
    CLAY_IMAGE = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')  # .select(clay);
    BDOD_IMAGE = ee.Image("projects/soilgrids-isric/bdod_mean").select('bdod_0-5cm_mean')  # .select(bdod);

    AUX_IMAGE = SAND_IMAGE
    AUX_IMAGE = AUX_IMAGE.addBands(CLAY_IMAGE)
    AUX_IMAGE = AUX_IMAGE.addBands(BDOD_IMAGE)

    SOIL_TEXTURE = AUX_IMAGE.reduceRegion(ee.Reducer.mean(), geometry, 250).getInfo()
    TERRAIN_IMAGE = ee.Algorithms.Terrain(ee.Image('USGS/SRTMGL1_003')).select(['elevation', 'slope', 'aspect'])
    TERREN_FEATURES = TERRAIN_IMAGE.reduceRegion(ee.Reducer.mean(), geometry, 30).getInfo()
    try:
        TERREN_FEATURES['aspect_sin'] = np.sin(np.deg2rad(TERREN_FEATURES['aspect']))
        TERREN_FEATURES['aspect_cos'] = np.cos(np.deg2rad(TERREN_FEATURES['aspect']))
    except:
        TERREN_FEATURES['aspect_sin'] = np.nan
        TERREN_FEATURES['aspect_cos'] = np.nan
    del TERREN_FEATURES['aspect']

    soil_terrain_loc = {**SOIL_TEXTURE, **TERREN_FEATURES}

    soil_terrain_loc = re_order(soil_terrain_loc)
    loc = np.mean(ring_wgs, axis=0)

    soil_terrain_loc['r'] = pobj.r[0] / (9 / 0.05) / 1624
    soil_terrain_loc['c_sin'] = np.sin(pobj.c[0] / (9 / 0.05) / 3856 * np.pi * 2)
    soil_terrain_loc['c_cos'] = np.cos(pobj.c[0] / (9 / 0.05) / 3856 * np.pi * 2)
    return soil_terrain_loc


def prepare_grid_data_v1(geo,START_DATE, END_DATE,START_DATE_NDVI,END_DATE_NDVI,ring_wgs,pobj):
    # Sentinel-1
    df_S1=extract_S1(geo,START_DATE, END_DATE,S1_DIR='')
    # soil_terrain_loc
    df_Aux=extract_terrain_soil_texture_v1(geo,ring_wgs,pobj)
    # NDVI
    df_NDVI=extract_NDVI_13Q1(geo,START_DATE_NDVI, END_DATE_NDVI)
    # ERA5

    point_geo = ee.Geometry.Point([pobj.lon, pobj.lat], 'EPSG:4326')
    EAR5_PRODUCT='ECMWF/ERA5_LAND/DAILY_AGGR'
    bands=['temperature_2m', 'total_precipitation_sum']#,'volumetric_soil_water_layer_1'
    ex=GeeTsExtractor(EAR5_PRODUCT,bands,START_DATE_NDVI,END_DATE_NDVI)

    df_Weather = ex.download_data_point(point_geo, 11132)

    return df_S1,df_NDVI,df_Aux,df_Weather

def extract_NDVI_13Q1(polygon_grid,START_DATE_NDVI, END_DATE_NDVI):
    MOD_PRODUCT = "MODIS/006/MOD13Q1"
    bands = ['NDVI']
    ex = GeeTsExtractor(MOD_PRODUCT, bands, START_DATE_NDVI, END_DATE_NDVI)
    df_MOD = ex.get_and_save_data(polygon_grid, 'temp', 463.313)

    MOD_PRODUCT = "MODIS/006/MYD13Q1"
    ex = GeeTsExtractor(MOD_PRODUCT, bands, START_DATE_NDVI, END_DATE_NDVI)
    df_MYD = ex.get_and_save_data(polygon_grid, 'temp', 463.313)
    df_NDVI = pd.concat([df_MOD, df_MYD])
    df_NDVI = df_NDVI.sort_index()
    return df_NDVI


def samples_4_grid_v1(geo, START_DATE, END_DATE, START_DATE_NDVI, END_DATE_NDVI, ring_wgs,pobj):
    df_S1, df_NDVI, df_Aux, df_Weather = prepare_grid_data_v1(geo,START_DATE, END_DATE,START_DATE_NDVI,END_DATE_NDVI,ring_wgs,pobj)
    Aux = np.asarray(list(df_Aux.values()))
    samples = []
    for i in range(len(df_S1)):
        S1 = np.asarray(list(df_S1.loc[i]['VV':]))

        a_date = df_S1.loc[i].date

        index = (df_NDVI.index < a_date) & (df_NDVI.index > (a_date - pd.Timedelta(days=365)))
        NDVI = normalizingData(np.asarray(df_NDVI[index]['NDVI']), -2000, 10000)

        # temp = df_weather[df_weather.date<a_date][-365:]
        index = (df_Weather.index >= df_NDVI[index].index[0]) & (
                    df_Weather.index <= (df_NDVI[index].index[0] + pd.Timedelta(days=367)))
        T = df_Weather[index]['temperature_2m'].values.reshape(-1, 8).mean(axis=1)
        T = normalizingData(np.asarray(T), 263.15, 308.15)

        P = df_Weather[index]['total_precipitation_sum'].values.reshape(-1, 8).mean(axis=1)
        P = normalizingData(np.asarray(P), 0, 0.3)
        samples.append(np.concatenate([S1, Aux, NDVI, T, P]))
    samples = np.asarray(samples)
    min_per = np.asarray([-30, -35, 29.1, -1, -1, 0, 0, 100, 0, 0, -1, -1, -1, -1, -1])
    max_per = np.asarray([5, 0, 46, 1, 1, 1000, 1000, 180, 5500, 40, 1, 1, 1, 1, 1])
    samples[:, :15] = normalizingData(samples[:, :15], min_per, max_per)
    return samples, df_S1

def normalizingData(X, min_per, max_per):
    temp=(X - min_per) / (max_per - min_per)
    temp[temp>1]=1
    temp[temp<0]=0
    return temp
