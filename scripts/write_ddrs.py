import kalasiris as isis
from glob import glob
import subprocess
import os

overwritel1 = False
overwriteddr = True
overwritecub = True
overwriteds = True

imgs = glob('../data/*.IMG')
print(imgs)
for im in imgs:
    try:
        l1name = im[:-4]+'_isis.cub'
        l1dsname = im[:-4]+'_isis_ds.cub'
        ddrname = l1name[:-4]+'_ddr.cub'
        if (not os.path.exists(l1name)) | overwritel1:
            isis.mroctx2isis(from_=im, to_=l1name)
            isis.spiceinit(from_=l1name)
        if (not os.path.exists(l1dsname)):
            isis.reduce(from_=l1name, to_=l1dsname, algorithm_='average',
                        sscale_=8, lscale_=8)
        if (not os.path.exists(ddrname)) | overwriteddr:
            isis.phocube(from_=l1dsname, to_=ddrname,
                         longitude_='no', latitude_='no',
                         phase_='yes', incidence_='no',
                         localemission_='yes', localincidence_='yes',
                         sunazimuth_='yes', spacecraftazimuth_='yes',
                         offnadirangle_='yes', subspacecraftgroundazimuth_='yes',
                         subsolargroundazimuth_='yes', emission_='no')
    except subprocess.CalledProcessError as err:
        print('Had an ISIS error:')
        print(' '.join(err.cmd))
        print(err.stdout)
        print(err.stderr)
        raise(err)
ddrs = glob('../data/*ddr.cub')
for dim in ddrs:
    if (not os.path.exists(dim[:-4]+'_pds_ds.img')) | overwritecub:
        isis.isis2pds(from_=dim, to_=dim[:-4]+'_pds.img')
