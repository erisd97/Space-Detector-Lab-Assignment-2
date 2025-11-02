from astropy.io import fits
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
import ipywidgets as widgets
from scipy.optimize import curve_fit
import scipy.ndimage as ndi
import pandas as pd
#import ccd_utils as cutils
#%matplotlib widget


filelist_f336W = glob("C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/F336W/*.fits")
filelist_f555W = glob("C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/F555W/*.fits")

#taken from HST workbook 2 and combined with finding specific params to make a combined set of images from the filelist
def load_file_list(filelist):

    images = []
    fluxes = []

    for file in filelist:
        hdul = fits.open(file)
        primary_hdr = hdul[0].header
        data = hdul[1].data
        header = hdul[1].header
        images.append(data)
    flux = data.copy()
    exposure = primary_hdr['EXPTIME']
    photflam = header['PHOTFLAM']

    if header["BUNIT"].strip() == 'COUNTS':
        flux /= exposure
    flux *= photflam 
    fluxes.append(flux)
    return fluxes, header, primary_hdr, exposure,np.asarray(images)


def combine_images(filelist,kind="median",axis=0):

    stack = load_file_list(filelist)[4]
    if kind == "mean":
        return np.mean(stack,axis=axis)
    if kind == "median":
        return np.median(stack,axis=axis)

    else:
        return ValueError(f"Unsupported function: {kind}")

fluxes_336, headers_336, p_header_336, f336_exposure, f336_im = load_file_list(filelist_f336W)
fluxes_555, headers_555, p_header_555, f555_exposure, f555_im = load_file_list(filelist_f555W)
f336_median = combine_images(filelist_f336W)
f555_median = combine_images(filelist_f555W)

flux_336 = np.median(fluxes_336,axis=0)
flux_555 = np.median(fluxes_555,axis=0)


def circular_aperture(data, center, radius):
    """
    Extract a circular cutout from the input data

    Parameters:
    -----------
    data : `np.ndarray`
        the input data
    center : list or tuple
        the center of the circle to extract, in y-x order
    radius : int or float
        the radius of the circle in pixels

    Returns:
    -----------
    cutout : `np.ndarray`
        a square cutout of the data
    mask : `np.ndarray`
        the circular mask used for the operation
    from cutils
    """

    # use the radius to establish a box size and take a cutout
    box_size = int(np.ceil(2 * radius))
    cutout = square_aperture(data, center, box_size)

    ys, xs = np.indices(cutout.shape)

    # get the new center of the cutout
    xc = cutout.shape[1] // 2
    yc = cutout.shape[0] // 2

    # get the pixel start position for the mask
    y0 = yc - box_size // 2
    x0 = xc - box_size // 2

    x_coords = xs + x0
    y_coords = ys + y0

    distance = np.sqrt((x_coords - xc) ** 2 + (y_coords - yc) ** 2)

    # finally the mask...
    mask = distance <= radius

    return cutout, mask

def square_aperture(data, center, box_size):
    """
    Extract a square cutout from the input data

    Parameters:
    -----------
    data : `np.ndarray`
        the input data
    center : list or tuple
        the center of the box to extract, in y-x order
    box_size : int
        the size of the box in pixels

    Returns:
    -----------
    res : `np.ndarray`
        a cutout of the data
    """

    y, x = map(int, center)

    # ensure the box_size is odd
    if box_size % 2 == 0:
        box_size += 1

    half = box_size // 2

    # use slices
    sly = slice(y - half, y + half + 1)
    slx = slice(x - half, x + half + 1)

    return data[sly, slx]
#Gaussian fitting plots with basic assumptions made if no data given

def gauss_2d(coords, amp, x0, y0, sigma_x, sigma_y):
    y, x = coords

    gauss = amp * np.exp(
        -(((x- x0)**2) / (2*sigma_x**2) + ((y - y0)**2 / (2 * sigma_y**2))
         ))
    return gauss.ravel() #ravel flattens the array to a 1d array

def fit_gauss2d(data, p0 = None):

    if p0 is None: 
        a0 = data.max()
        x0 = data.shape[1] / 2
        y0 = data.shape[0] / 2
        sigx0 = 0.5
        sigy0 = 0.5
        p0 = [a0, x0, y0, sigx0, sigy0]

    y, x = np.indices(data.shape)

    popt, pcov = curve_fit(gauss_2d, (y, x), data.ravel(), p0 = p0)

    return popt, pcov
#create normalisation and transformation function and plot
#taken from ccd_utils directly as it wouldnt import
def normalize_image(x, pmin=1, pmax=99):
    """
    normalize data between 0 and 1, optionally clipping the original data
    between user-specified min/max values, or percentiles

    Parameters:
    -----------
    x: np.ndarray
        The data to normalize
    pmin: float or int
        the lower percentile to use for normalization
    pmax : float or int
        the upper percentile to use for normalization

    Returns:
    -----------
    norm: `matplotlib.colors.Normalize` object
        the normalized data in the interval [0, 1]
    """

    vmin, vmax = np.percentile(x, [pmin, pmax])
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    return norm(x)

def transform_by(func, x, pmin=1, pmax=99):
    """
    transform/stretch the data by a named function,
    normalizing the data using given percentiles

    Parameters:
    -----------
    func: str
        one of 'linear', 'sqrt' or 'log'
    x: np.ndarray
        data to transform
    pmin : float or int
        lower percentile
    pmax : float or int
        upper percentile
    Returns:
    ----------
    res : `np.array`
        the stretched data
    """

    x_norm = normalize_image(x, pmin=pmin, pmax=pmax)

    if func == "sqrt":
        return np.sqrt(x_norm)
    elif func == "log":
        eps = 1e-6
        return np.ma.log10(x + eps) / np.ma.log10(1 + eps)
    elif func == "linear":
        return x_norm
    else:
        return ValueError(f"Unsupported function: {func}")



def plotter(func, x, percentiles):
    vmin, vmax = np.percentile(x, percentiles)
    linear_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if func == 'sqrt':
        x_norm = np.sqrt(abs(linear_norm(x)))
    elif func == 'log':
        eps = 1e-6
        x_norm = np.log10(linear_norm(x) + eps) / np.log10(1 + eps)
    else:
        x_norm = linear_norm(x)
    fig, ax = plt.subplots(figsize=(7, 7))
    im =  ax.imshow(x_norm, origin='lower')
    return im
fig, ax = plt.subplots(3,figsize=(5,12))

transformed_1 = transform_by('sqrt', f336_im[0], pmin = 96, pmax = 99)
transformed_2 = transform_by('sqrt', f336_im[1], pmin = 96, pmax = 99)
transformed_3 = transform_by('sqrt', f336_im[2], pmin = 96, pmax = 99)

fig.suptitle('Original plots before combination, sigma clipped')

mask_1 = sigma_clip(transformed_1).mask
mask_2 = sigma_clip(transformed_2).mask
mask_3 = sigma_clip(transformed_3).mask

ax[0].set_title('Image 1')
ax[1].set_title('Image 2')
ax[2].set_title('Image 3')

ax[0].imshow(mask_1,origin='lower')
ax[1].imshow(mask_2,origin='lower')
ax[2].imshow(mask_3,origin='lower')
plt.show()

plt.close('all')

fig, ax = plt.subplots(figsize=(5,5))
transformed = transform_by('sqrt', f336_median, pmin = 96, pmax = 99)
mask = sigma_clip(transformed).mask
ax.imshow(mask,origin='lower')
ax.set_title('Sigma Clipped Median Image')
plt.show()

#function for cutting a particular square
def square_cutout(data, center, box_size):
    yc, xc = map(int, center)

    if box_size % 2 == 0:
        box_size += 1

    half_size = box_size // 2

    # take a slice
    sly = slice(yc - half_size, yc + half_size + 1)
    slx = slice(xc - half_size, xc + half_size + 1)

    return data[sly, slx]


#convert fluxes to apparent magnitude and absolute magnitude, from workbooks
def flux_to_magnitude(value, zeropoint):
    return -2.5 * np.log10(abs(value)) + zeropoint

def app_to_abs(data,distance):
    return data - (2.5 * np.log10((distance / 10)**2))
magnitudes = flux_to_magnitude(flux_336, -21.1)

def find_flux(data, coord):
    """
    Apply circular radius to the total flux around a centerpoint and mask background - help from Bruce!
    """
    #flux within aperture:
    cutout, mask = circular_aperture(data, center = coord, radius = 5)
    aperture_flux = cutout[mask].sum()

    #flux within annulus
    cutout_outer, mask_outer = circular_aperture(data, center = coord, radius = 7)
    cutout_inner, mask_inner = circular_aperture(data, center = coord, radius = 6)
    annulus_flux = cutout_outer[mask_outer].sum() - cutout_inner[mask_inner].sum()

    #returning background subtracted flux:
    return aperture_flux - annulus_flux


#the following was taken from the guidelines to calculate ellipticity
def check_ellipticity(sigma_x, sigma_y, etol=0.5):
    """
    Use Gaussian parameters to calculate the flattening or ellipticity
    """
    f = 1 - (min(sigma_x, sigma_y) / max(sigma_x, sigma_y))

    ellipse_okay = f <= etol

    if ellipse_okay:
        return True 
    else: return False


def star_finder(data, nsize = 10):
    """function to find local peaks, base code from HST W3 and help from Bruce making it a function"""

    filtered_image = ndi.gaussian_filter(data, sigma=0.5,mode='reflect')
    local_max = ndi.maximum_filter(filtered_image, size=nsize)
    local_max_mask = (local_max == filtered_image)

    coordinates = np.argwhere(local_max_mask)

    #count passes and fails:
    passes = []
    fails = []

    for coord in coordinates:
        flux = find_flux(data, coord)

        if flux > 0:
            try:
                cutout, mask = circular_aperture(data, center = coord.T, radius = 5)
                popt, pcov = fit_gauss2d(cutout)

                if check_ellipticity(popt[3], popt[4], etol=0.5):
                    passes.append(coord)


                else: 
                    fails.append(coord)

            except:
                fails.append(coord)
                continue

        else: fails.append(coord)

    return np.array(passes), np.array(fails)


passed_336, failed_336 = star_finder(f336_median)
passed_555, failed_555 = star_finder(f555_median)

#soome of the following was test running for removing blev etc
rdata_f336 = f336_median.copy()
#rdata_f554 = f555_median.copy()

def parse_blev_file(file, chip=0):
    with fits.open(file) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    odds = [9,11,13]
    evens = [8,10,12]
    rows = slice(9,790)

    odd_vals = data[chip, rows, odds]
    even_vals = data[chip, rows, evens]

    return np.mean(odd_vals), np.mean(even_vals)

def file_opener(file):
    with fits.open(file) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    return header, data

dheader,ddata = file_opener('C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/calibration/ubai2507m_darkfile.fits')
mheader,mdata = file_opener('C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/calibration/ubai2507m_maskfile.fits')
biheader,bidata = file_opener('C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/calibration/ubai2507m_biasfile.fits')
fheader,fdata = file_opener('C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/calibration/ubai2507m_flatfile.fits')

bias_corrected = rdata_f336 - bidata
dark_corrected = rdata_f336 - ddata
mask_corrected = rdata_f336 - mdata
odd, even = parse_blev_file("C:/Users/erisd/Desktop/Python files/Assignments/hst-data/hst-data/calibration/ubai2507m_blevfile.fits")
blev_corrected = rdata_f336.copy()
blev_corrected = blev_corrected.astype(float)
blev_corrected[0:800:2,:] -= odd
blev_corrected[1:800:2,:] -= even
total_corrected = blev_corrected - bidata[0] - mdata[0] - ddata[0]*f336_exposure
total_corrected *= fdata[0]

fig, ax = plt.subplots(figsize=(5, 5))
transform_total = transform_by('sqrt', total_corrected, pmin = 1, pmax = 99)
ax.imshow(transform_total, origin='lower')
ax.set_title('Fully Corrected')
plt.show()

transformed_corrected = transform_by('sqrt', total_corrected, pmin = 96, pmax = 99)
mask_c = sigma_clip(transformed_corrected).mask
fig, ax = plt.subplots(ncols=2,figsize=(10,5))
ax[0].imshow(mask_c,origin='lower')
ax[1].imshow(mask,origin='lower')
fig.suptitle("Clipped Transformed Data")
plt.show()


def catalog(data, coords):
    """
    This function will create a catalog of found stars including their coordinates, flux, and magnitudes
    """

    #creating a catalog: 
    #empty lists
    fluxes = []
    app_mags = []
    abs_mags = []

    #search and append data, distance and mag taken from workbook
    for coord in coords:
        fluxes.append(find_flux(data, coord))

    for flux in fluxes:
        app_mags.append(flux_to_magnitude(flux, -21.1))

    for mag in app_mags:
        abs_mags.append(app_to_abs(mag, 16400)) 

    #make the catalogue
    catalog_data = {
        'ID' : [(i) for i in range(len(coords))],
        'x-center':  [coord[0] for coord in coords],
        'y-center':  [coord[1] for coord in coords],
        'flux': fluxes,
        'm_app': app_mags,
        'm_abs': abs_mags,
    }

    return pd.DataFrame(catalog_data)
catalog(flux_336, passed_336)
catalog(flux_555, passed_555)


plt.close('all')
#check comparable points


fig, ax= plt.subplots(nrows = 2,figsize = (9, 9))
fig.suptitle("Plotting Detected Stars on Opposite Map")
ax[0].set_title("F555 detections on F336")
smooth_336 = ndi.gaussian_filter(f336_median, sigma = 0.5, mode = 'reflect')
smooth_norm_336 = transform_by('sqrt', smooth_336, pmin = 1, pmax = 99)
ax[0].imshow(smooth_norm_336, origin = 'lower')

y_555, x_555 = passed_555.T
ax[0].scatter(x_555, y_555, s = 5,marker = '+', color = 'r')

ax[1].set_title("F336 detections on F555")
smooth_555 = ndi.gaussian_filter(f555_median, sigma = 0.5, mode = 'reflect')
smooth_norm_555 = transform_by('sqrt', smooth_555, pmin = 1, pmax = 99)
ax[1].imshow(smooth_norm_555, origin = 'lower')

y_336, x_336 = passed_336.T
ax[1].scatter(x_336, y_336, s = 5,marker = '+', color = 'r')
plt.show()


plt.close('all')
#need to combine into one list by ignoring below zero fluxes in opposing dataset
#will apply 555 detections on 336
cat_555_336 = catalog(flux_555,passed_336)
cat_336 = catalog(flux_336,passed_336)


def compare(catalogue1,catalogue2):
    """Takes two catalogs and removes indices where there is a flux < 0"""
    #to this moment idk if its 'catalog' or catalogue' so i might have mixed them up
    adjusted_1 = catalogue1.copy()
    adjusted_2 = catalogue2.copy()

    remove = catalogue2.index[catalogue2['flux']<0].tolist()

    for i in adjusted_2['ID']:
        if i in remove:
            adjusted_2 = adjusted_2.drop(index = i)
    for i in adjusted_1['ID']:
        if i in remove:
            adjusted_1 = adjusted_1.drop(index = i)
    return adjusted_1, adjusted_2

adjusted_336, adjusted_555 = compare(cat_336,cat_555_336)

#plot histograms
plt.close('all')
fig, ax = plt.subplots(nrows = 2,figsize = (9,16))
fig.suptitle("HR Diagram")
ax[0].set_xlabel("Mag 336")
ax[0].set_ylabel("Mag (336 - 555)")
ax[0].set_title('Apparant Magnitude')
ax[0].scatter((adjusted_336["m_app"] - adjusted_555["m_app"]), adjusted_336["m_app"], s = 4)

ax[1].set_xlabel("Mag 336")
ax[1].set_ylabel("Mag (336 - 555)")
ax[1].set_title('Absolute Magnitude')
ax[1].scatter((adjusted_336["m_abs"] - adjusted_555["m_abs"]), adjusted_336["m_abs"], s = 4)
plt.show()
