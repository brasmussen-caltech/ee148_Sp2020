import gdal
import rasterio
import pandas as pd
import numpy as np
from rasterio.plot import show as rshow

def grab_raster_values(filename, points=False,
                       neighbs=False, plotit=False):
    ds = gdal.Open(filename)
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    outvals = []
    rastds = rasterio.open(filename)
    data = rastds.read()
    if plotit:
        rshow(rastds)
    if not points:
        #If coordinates not given, pull example
        points=[(-4320,299085)]
    outvals=[]
    for pttup in points:
        x,y= pttup
        row,col=rastds.index(x,y)
        if neighbs:
            extract = data[:, (row - neighbs - 1): (row + neighbs + 2),
                                 (col - neighbs - 1): (col + neighbs + 2)]
#             extract=np.swapaxes(np.swapaxes(extract,0,-1),0,1)
        else:
            extract = data[:, row, col]
        outvals.append(extract)

    return(outvals)

def sample_im_pixels(number, geoimagenm):
    rastds = rasterio.open(geoimagenm)
    min_x, min_y, max_x, max_y = rastds.bounds
    outlist = []
    rastdata = rastds.read()
    i = 0
    while i < number:
        x, y = np.random.uniform(min_x, max_x), \
            np.random.uniform(min_y, max_y)
        point = (x, y)
        row, col = rastds.index(x, y)
        if rastdata[0, row, col] > 60000:
            continue
        else:
            outlist.append(point)
        i = len(outlist)
    return(outlist)


def write_sample(outname, arr, proj, transform):
        xsize = arr.shape[2]
        ysize = arr.shape[1]
        bands = 1
        driver = gdal.GetDriverByName('GTiff')
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        out = driver.Create(outname, xsize, ysize, bands, gdal.GDT_Float32)
        out.SetProjection(srs.ExportToWkt())
        out.SetGeoTransform(transform)
        for band in range(1, bands+1):
            exportband = arr[band-1, :, :]
            out.GetRasterBand(band).WriteArray(exportband)
        del out
