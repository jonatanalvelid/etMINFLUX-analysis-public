import os
import csv
import warnings
import random
import copy
import tifffile
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d

import obf_support

# warning suppression
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore')


class DiffusionAnalysis(object):
    """ 2D lipid diffusion analysis object. Contains functions for loading data, filtering, analysing, and plotting data from etMINFLUX 2D tracking experiments. """
    def __init__(self, tag, *args, **kwargs):
        # data objects
        self.track_data = pd.DataFrame()  # for per-track parameters
        self.roi_data = pd.DataFrame()  # for per-roi parameters

        # data loading params
        self.dataset_tag = tag
        self.paths = []
        self.analyse_folders = True
        self.analyse_files = False
        self.filename_prefix_end = 10
        self._data_params = ['x','y','tim','cfr','efo','fbg']
        self.simulated_data = False
        self.compensate_confshift_x = False
        self.compensate_confshift_y = False

        # filtering params
        self.filtered = False
        self.filtered_params = []
        self.filtered_ranges = []
        
        # analysis params
        self.analysis_run = []
        self.loc_it = 3

        self.figs = [None] * 110
        self.subplot_cols = 5
        self.subplot_cols2 = 5

        plt.close('all')

    def set_analysis_parameters(self, site_rad=None, inclusion_rad=None, circle_radii=None, blob_dist=None,
                                min_time=None, split_len_thresh=None, max_time_lag=None, max_dt=None,
                                fit_len_thresh=None, meanpos_thresh=None, interval_meanpos=None,
                                meanslidestd_thresh=None, slidstd_interval=None, interval_dist=None):
        # Default values to start from:
        # site_rad = 0.15 (um), blob_dist = 0.01,
        # min_time = 10e-3, split_len_thresh = 5, max_time_lag = 0.005 (s),
        # meanpos_thresh = 0.04, interval_meanpos = 5, interval_dist = 50,
        # meanslidestd_thresh = 0.02, slidstd_interval = 40,
        if site_rad is not None:
            self.ana_site_rad = site_rad
        if inclusion_rad is not None:
            self.inclusion_rad = inclusion_rad
        if circle_radii is not None:
            self.circle_radii = circle_radii
        if blob_dist is not None:
            self.blob_dist = blob_dist
        if min_time is not None:
            self.min_time = min_time
        if split_len_thresh is not None:
            self.split_len_thresh = split_len_thresh
        if fit_len_thresh is not None:
            self.fit_len_thresh = fit_len_thresh
        if max_time_lag is not None:
            self.max_time_lag = max_time_lag
        if max_dt is not None:
            self.max_dt = max_dt
        if meanslidestd_thresh is not None:
            self.meanslidestd_thresh = meanslidestd_thresh
        if slidstd_interval is not None:
            self.slidstd_interval = slidstd_interval
        if meanpos_thresh is not None:
            self.meanpos_thresh = meanpos_thresh
        if interval_meanpos is not None:
            self.interval_meanpos = interval_meanpos
        if interval_dist is not None:
            self.interval_dist = interval_dist

        # initialize default and folder-common confocal scan params, to be used for the confocal shift compensation fit reading
        self.confocal_dwell = 2  # pixel dwell time in µs
        self.confocal_bidrectional = False  # bidirection scal boolean

    def confocal_compensation_shift_init(self, fitfiles_folder=None):
        """ Get data for compensation shift, for every possible confocal scan configuration. """
        t_lim = 0.1  # fixed time for the shift between different fitting functions for non-bidirectional shifts
        fitfiles = os.listdir(fitfiles_folder)
        fitfiles_x = [file for file in fitfiles if 'xshift' in file]
        fitfiles_y = [file for file in fitfiles if 'yshift' in file]
        
        self.shiftcompensation = {'dwell_time': [], 'bidirectional': [], 'pixel_size': [], 'fov_size': [], 'x_lim': [], 'fitfile_x': [], 'fitfile_y': [], 'shiftmode_x': [], 'shiftmode_y': [], 'shiftcoeffs_x': [], 'shiftcoeffs_y': [], 'comp_x': [], 'comp_y': []}
        for filex, filey in zip(fitfiles_x, fitfiles_y):
            pixel_size = float(filex.split('pxs')[1].split('-')[0])
            bidirectional = int(filex.split('bid')[1].split('.')[0])
            dwell_time = float(filex.split('dwell')[1].split('-')[0])
            fov_size = float(filex.split('fov')[1].split('-')[0])
            self.shiftcompensation['dwell_time'].append(dwell_time)
            self.shiftcompensation['bidirectional'].append(True if bidirectional else False)
            self.shiftcompensation['pixel_size'].append(pixel_size)
            self.shiftcompensation['fov_size'].append(fov_size)
            
            self.shiftcompensation['fitfile_x'].append(os.path.join(fitfiles_folder, filex))
            self.shiftcompensation['fitfile_y'].append(os.path.join(fitfiles_folder, filey))

            scanspeed = (pixel_size*1e-3)/(dwell_time*1e-3)  # scan speed in µm/ms
            x_lim = t_lim*scanspeed
            self.shiftcompensation['x_lim'].append(x_lim)

            with open(os.path.join(fitfiles_folder, filex), 'r') as f:
                content_x = f.read()
            coeffs = [float(i) for i in content_x.split('\n') if i != '']
            shiftfit_coeffs_x = coeffs
            shiftfit_mode_x = 'linear' if len(coeffs)==2 else 'sigmoid'  # len == 2: linear fit; len == 5: quad+sigmoid fit - sigm: [:4], quad: [4:]
            self.shiftcompensation['shiftmode_x'].append(shiftfit_mode_x)
            self.shiftcompensation['shiftcoeffs_x'].append(shiftfit_coeffs_x)

            with open(os.path.join(fitfiles_folder, filey), 'r') as f:
                content_y = f.read()
            coeffs = [float(i) for i in content_y.split('\n') if i != '']
            shiftfit_coeffs_y = coeffs  # len == 2: linear fit; len == 5: sine*linear fit
            shiftfit_mode_y = 'linear' if len(coeffs)==2 else 'sine'
            self.shiftcompensation['shiftmode_y'].append(shiftfit_mode_y)
            self.shiftcompensation['shiftcoeffs_y'].append(shiftfit_coeffs_y)

            self.shiftcompensation['comp_x'] = True
            self.shiftcompensation['comp_y'] = True  # TODO: Change this to be dependent on the scan parameters - for some parameters we do not want to correct for y-shift

        self.shiftcompensation = pd.DataFrame.from_dict(self.shiftcompensation)

    def get_shiftcompensation_idx(self, dwell, bid, pxs, fov):
        indxs = self.shiftcompensation.loc[(self.shiftcompensation['dwell_time'] == dwell) & (self.shiftcompensation['bidirectional'] == bid) & (self.shiftcompensation['pixel_size'] == pxs) & (self.shiftcompensation['fov_size'] == fov)].index
        if len(indxs) > 0:
            if self.shiftcompensation.iloc[indxs[0]]['comp_x']:
                idx = indxs[0]
            else:
                idx = np.nan  # no shift, due to decision in shiftcompensation reading
        else:
            idx = np.nan  # no shift, due to confocal parameters not found
        return idx

    def get_confocal_shift_x(self, x, conf_params):
        comp_idx = self.get_shiftcompensation_idx(*conf_params)
        if np.isnan(comp_idx):
            x_shift = 0
        else:
            shift_row = self.shiftcompensation.iloc[comp_idx]
            if shift_row['shiftmode_x'] == 'sigmoid':
                x_shift = (1-self._sigmoid(x, *[1,shift_row['x_lim'],1,0]))*self._quadratic(x, *shift_row['shiftcoeffs_x'][4:]) + self._sigmoid(x, *[1,shift_row['x_lim'],1,0])*self._sigmoid(x, *shift_row['shiftcoeffs_x'][:4])  # input: x_conf as distance in µm from left edge of confocal FOV
            elif shift_row['shiftmode_x'] == 'linear':
                x_shift = self._linear(x, *shift_row['shiftcoeffs_x'])
        return x_shift
    
    def get_confocal_shift_y(self, x, conf_params):
        comp_idx = self.get_shiftcompensation_idx(*conf_params)
        if np.isnan(comp_idx):
            y_shift = 0
        else:
            shift_row = self.shiftcompensation.iloc[comp_idx]
            if shift_row['shiftmode_y'] == 'sine':
                y_shift = self._sinusoidal(x, *shift_row['shiftcoeffs_y'])  # input: x_conf as distance in µm from left edge of confocal FOV
            elif shift_row['shiftmode_y'] == 'linear':
                y_shift = self._linear(x, *shift_row['shiftcoeffs_y'])
        return y_shift

    def print_to_file_init(self):
        # Initialize file for printing outputs to
        savename = 'analysis-output'
        self.text_file = open(os.path.join(self.paths[0],savename+'.txt'), "w")

    def print_to_file(self, str):
        self.text_file.write(str+'\n')

    def print_to_file_close(self):
        self.text_file.close()

    def set_confocal_params(self, conf_scan_params = [2, False]):
        """ Set confocal scan parameters, from supplied list of [dwelltime, bidirectional]"""
        self.confocal_dwell = conf_scan_params[0]  # dwell time in µs
        self.confocal_bidrectional = conf_scan_params[1]  # bidirectionality of scan, boolean

    def add_data(self, path, plotting=False):
        """ Add data from a folder to the analysis. """
        self.paths.append(path)
        filelist = os.listdir(path)
        filelist_rois_all = [file for file in filelist if file.endswith('.npy')]
        filelist_conf = [file for file in filelist if 'conf' in file and 'analysis' not in file and '.png' not in file]
        filelist_msr = [file for file in filelist if file.endswith('.msr')]

        self.N_rois = len(filelist_rois_all)

        roi_idx_cum = 0
        nl = '\n'

        confidxs = []
        roiidxs = []
        roinames = []
        roipospxs = []
        roiposs = []
        roisizes = []
        roiimgconfs = []
        roipxsizes = []
        roiconfoffset = []
        roiconfsize = []
        roiconfsizepx = []
        roiconfxlims = []
        roiconfylims = []
        roixconfcorr = []
        roiyconfcorr = []

        for file_idx in range(len(filelist_conf)):
            print(f'Confocal file {file_idx+1}/{len(filelist_conf)}')
            curr_img_data = {'confidx': [], 'roiidx': [], 'tridx': [], 'roisize': [], 'roipos': [], 'roipos_px': [], 'tim0': [], 'x': [], 'y': [], 'tim': [], 'pxsize': [], 'inout_flag': [], 'inout_len': [], 'inout_mask': [], 'inout_incl_flag': [], 'inout_incl_len': [], 'inout_incl_mask': [], 'filter': [], 'confimg':[], 'confimg_ext': [], 'conf_xlim': [], 'conf_ylim': [], 'conf_xcorr': [], 'conf_ycorr': []}
            file_conf = os.path.join(path, filelist_conf[file_idx])

            #TODO CHANGE FOR VIRUS BUDDING:
            file_msr = os.path.join(path, filelist_msr[file_idx])
            #file_msr = os.path.join(path, filelist_msr[0])

            print(file_conf)
            print(file_msr)
            conf_time = int(filelist_conf[file_idx].split('-')[1].split('_')[0])
            if file_idx > 0:
                conf_date_prev = copy.deepcopy(conf_date)
            else:
                conf_date_prev = 0
            conf_date = int(filelist_conf[file_idx].split('-')[0])
            if file_idx > 0 and conf_date == conf_date_prev:
                conf_time_prev = int(filelist_conf[file_idx-1].split('-')[1].split('_')[0])
            else:
                conf_time_prev = 0
            filelist_rois = [file for file in filelist_rois_all if int(file.split('-')[0])==conf_date and int(file.split('-')[1].split('_')[0])>conf_time_prev and int(file.split('-')[1].split('_')[0])<conf_time]
            image_conf = tifffile.imread(file_conf)[-1]
            
            # get metadata from confocal image in msr file (pixel size, image shape, image size, origin offset)
            msr_dataset = obf_support.File(file_msr)
            conf_msr_stack_index = 0  # in currently used imspector template file, the confocal dataset is always stack 0 in the .msr file. This might change with other templates used.
            conf_stack = msr_dataset.stacks[conf_msr_stack_index]
            pxsize = conf_stack.pixel_sizes[0]*1e6
            pxshift = pxsize/2
            conf_size_px = (conf_stack.shape[0], conf_stack.shape[1])
            conf_size = (conf_stack.lengths[0]*1e6, conf_stack.lengths[1]*1e6)
            conf_offset = (conf_stack.offsets[0]*1e6, conf_stack.offsets[1]*1e6)
            #conf_scanspeed_off = (-0.02,0)  # 70nm,1us: -110nm; 70nm,3us: -20nm
            #conf_offset = (conf_stack.offsets[0]*1e6+conf_scanspeed_off[0], conf_stack.offsets[1]*1e6+conf_scanspeed_off[1])
            #conf_offset_origin = (conf_offset[0]+conf_size[0]/2, conf_offset[1]+conf_size[1]/2)

            for roi_idx, file in enumerate(filelist_rois):
                print(file)
                roi_name = int(file.split('ROI')[1].split('-')[0])
                roi_pos = (int(file.split('[')[1].split(',')[0]),int(file.split(']')[0].split(',')[1]))
                #roi_pos_um = (roi_pos[0]*pxsize-conf_size[0]/2, roi_pos[1]*pxsize-conf_size[1]/2)
                roi_pos_um = (roi_pos[0]*pxsize+conf_offset[0], roi_pos[1]*pxsize+conf_offset[1])
                roi_size_um = (float(file.split('[')[2].split(',')[0]),float(file.split(']')[1].split(',')[1]))
                dataset = np.load(os.path.join(path, file))
                x = np.zeros((len(dataset),1))
                y = np.zeros((len(dataset),1))
                tid = np.zeros((len(dataset),1))
                tim = np.zeros((len(dataset),1))
                for i in range(len(dataset)):
                    x[i] = dataset[i][0][self.loc_it][2][0]
                    y[i] = dataset[i][0][self.loc_it][2][1]
                    tid[i] = dataset[i][4]
                    tim[i] = dataset[i][3]  # TODO Why do I not take tic anymore?
                x_raw = x * 1e6
                y_raw = y * 1e6
                tid = tid.flatten()
                tim = tim.flatten()
                track_ids = list(map(int, set(tid)))
                track_ids.sort()

                # save roi-specific parameters
                confidxs.append(file_idx)
                roiidxs.append(roi_idx)
                roinames.append(roi_name)
                roisizes.append(roi_size_um)
                roipospxs.append(roi_pos)
                roiposs.append(roi_pos_um)
                roiimgconfs.append(image_conf)
                roipxsizes.append(pxsize)
                roiconfoffset.append(conf_offset)
                roiconfsize.append(conf_size)
                roiconfsizepx.append(conf_size_px)

                # calculate compensation shift
                x_conf_correction = self.get_confocal_shift_x(roi_pos_um[0]-conf_offset[0]+pxshift, [float(self.confocal_dwell), self.confocal_bidrectional, round(pxsize,2)*1e3, round(conf_size[0],2)]) # input: roi position in x in µm from left edge of confocal FOV
                y_conf_correction = self.get_confocal_shift_y(roi_pos_um[0]-conf_offset[0]+pxshift, [float(self.confocal_dwell), self.confocal_bidrectional, round(pxsize,2)*1e3, round(conf_size[0],2)]) # input: roi position in x in µm from left edge of confocal FOV
                roixconfcorr.append(x_conf_correction)
                roiyconfcorr.append(y_conf_correction)

                # calculate x and y lims of ROI zoom
                xlim = [roi_pos_um[0]+pxshift+x_conf_correction-0.5, roi_pos_um[0]+pxshift+x_conf_correction+0.5]
                ylim = [roi_pos_um[1]+pxshift+y_conf_correction+0.5, roi_pos_um[1]+pxshift+y_conf_correction-0.5]
                roiconfxlims.append(xlim)
                roiconfylims.append(ylim)

                # get image zoom from track coordinates
                ax_onlyconf = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 1+roi_idx_cum*self.subplot_cols)
                ax_overlay = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 2+roi_idx_cum*self.subplot_cols)
                img_overlay = ax_overlay.imshow(image_conf, cmap='hot')
                extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                img_onlyconf = ax_onlyconf.imshow(image_conf, cmap='hot')
                img_onlyconf.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                cav_circ_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                ax_overlay.add_patch(cav_circ_overlay);
                ax_onlyconf.add_patch(cav_circ_onlyconf);
                ax_overlay.set_xlim(*xlim)
                ax_overlay.set_ylim(*ylim)
                ax_onlyconf.set_xlim(*xlim)
                ax_onlyconf.set_ylim(*ylim)
                for track in track_ids[::]:
                    x_track = np.array([val for val,tr in zip(x_raw,tid) if tr==track]).flatten()
                    y_track = np.array([val for val,tr in zip(y_raw,tid) if tr==track]).flatten()
                    tim_track = np.array([val for val,tr in zip(tim,tid) if tr==track]).flatten()
                    curr_img_data['tim0'].append(tim_track[0])
                    curr_img_data['confidx'].append(file_idx)
                    curr_img_data['roiidx'].append(roi_idx)
                    curr_img_data['tridx'].append(track)
                    curr_img_data['roisize'].append(roi_size_um)
                    curr_img_data['roipos'].append(roi_pos_um)
                    curr_img_data['roipos_px'].append(roi_pos)
                    curr_img_data['x'].append(x_track)
                    curr_img_data['y'].append(y_track)
                    tim_track = tim_track - tim_track[0]
                    curr_img_data['tim'].append(tim_track)
                    curr_img_data['confimg'].append(image_conf)
                    curr_img_data['confimg_ext'].append(extents_confimg)
                    curr_img_data['conf_xlim'].append(xlim)
                    curr_img_data['conf_ylim'].append(ylim)
                    curr_img_data['conf_xcorr'].append(x_conf_correction)
                    curr_img_data['conf_ycorr'].append(y_conf_correction)
                    curr_img_data['pxsize'].append(pxsize)
                    curr_img_data['inout_flag'].append(np.nan)
                    curr_img_data['inout_len'].append(np.nan)
                    curr_img_data['inout_mask'].append(np.nan)
                    curr_img_data['inout_incl_flag'].append(np.nan)
                    curr_img_data['inout_incl_len'].append(np.nan)
                    curr_img_data['inout_incl_mask'].append(np.nan)
                    curr_img_data['filter'].append(np.nan)
                    if plotting:
                        ax_overlay.plot(x_track, y_track, color='gray', linewidth=0.3);
                if plotting:
                    ax_overlay.annotate(f'Conf {file_idx}, ROI {roi_idx} {nl}ConfFile {conf_date}-{conf_time}, ROIFile {roi_name}',
                    xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
                    ax_onlyconf.annotate(f'Conf {file_idx}, ROI {roi_idx} {nl}ConfFile {conf_date}-{conf_time}, ROIFile {roi_name}',
                    xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
                roi_idx_cum += 1
            self.track_data = pd.concat([self.track_data, pd.DataFrame(curr_img_data)])
            self.track_data.reset_index(drop=True, inplace=True)
        self.roi_data['conf_idx'] = confidxs
        self.roi_data['roi_idx'] = roiidxs
        self.roi_data['roi_name'] = roinames
        self.roi_data['roi_pos_px'] = roipospxs
        self.roi_data['roi_pos'] = roiposs
        self.roi_data['roi_size'] = roisizes
        self.roi_data['pxsize'] = roipxsizes
        self.roi_data['conf_offset'] = roiconfoffset
        self.roi_data['conf_size_px'] = roiconfsizepx
        self.roi_data['conf_size'] = roiconfsize
        self.roi_data['conf_img'] = roiimgconfs
        self.roi_data['conf_xlims'] = roiconfxlims
        self.roi_data['conf_ylims'] = roiconfylims
        self.roi_data['conf_xcorr'] = roixconfcorr
        self.roi_data['conf_ycorr'] = roiyconfcorr
        print('Analysis data added...', end=' \r')
        self.subplot_rows = roi_idx_cum+1
        #self.set_fig_size()
        plt.figure(0)
        fig_ad = plt.gcf()
        fig_ad.set_size_inches(10,80)

    def mean_mfx_pos(self, shift_correction=False, plot=True):
        print('')
        dists = []
        angles = []
        conf_poss = []

        # get data for compensation shift
        fit_file = 'C:\\Users\\alvelidjonatan\\Documents\\Data\\etMINFLUX-lab\\beads-overlay\\confshift\\241128-pxs70nm\\multifitting-xshift40.txt'
        x_lim = 7.0

        with open(fit_file, 'r') as f:
            content = f.read()
        coeffs = [float(i) for i in content.split('\n') if i != '']
        coeffs_sigm = coeffs[:4]
        coeffs_quad = coeffs[4:]

        for _, roi in self.roi_data.iterrows():
            conf_idx = roi['conf_idx']
            roi_idx = roi['roi_idx']
            pxsize = roi['pxsize']
            pxshift = pxsize/2
            conf_pos = np.array(roi['roi_pos']) + [pxshift, pxshift]
            x_conf = conf_pos[0]-roi['conf_offset'][0]
            x_conf_correction = (1-self._sigmoid(x_conf, *[1,x_lim,1,0]))*self._quadratic(x_conf, *coeffs_quad) + self._sigmoid(x_conf, *[1,x_lim,1,0])*self._sigmoid(x_conf, *coeffs_sigm)  # input: x_conf as distance in µm from left edge of ROI
            x_data = []
            y_data = []
            for _, track in self.track_data.iterrows():
                if track['confidx'] == conf_idx and track['roiidx'] == roi_idx and track['inout_flag']==True:
                    x_data.append(track['x'])
                    y_data.append(track['y'])
            if len(x_data)>0:
                x_data = np.concatenate(x_data).ravel()
                y_data = np.concatenate(y_data).ravel()
                x_mean = np.mean(x_data)
                y_mean = np.mean(y_data)
                conf_mfx_mean_dist = np.sqrt((x_mean-(conf_pos[0]+x_conf_correction))**2+(y_mean-conf_pos[1])**2)
                dists.append(conf_mfx_mean_dist)
                conf_mfx_mean_angle = self._get_angle(conf_pos,(x_mean,y_mean),(conf_pos[0]+1,conf_pos[1]))
                angles.append(conf_mfx_mean_angle)
                conf_poss.append(conf_pos)
        print(dists)
        print(np.mean(dists))
        print(angles)
        print(np.mean(angles))
        if plot:
            self.plot_confmfx_dists(conf_poss, dists, angles)

    def plot_confmfx_dists(self, pos, dists, angles):
        ax = self.figs[60].add_subplot(1,1,1)
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        U = [length * np.cos(angle*np.pi/180) for length, angle in zip(dists, angles)]
        V = [length * np.sin(angle*np.pi/180) for length, angle in zip(dists, angles)]
        ax.quiver(X, Y, U, V, scale_units='xy', scale=0.04, label='Conf-MFX bead shifts')
        ax.quiver([0],[0],[0.1],[0], scale_units='xy', scale=0.04, color='red', label='Reference, 100 nm')
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        ax.legend()

        plt.figure(60)
        fig_confmfxdists = plt.gcf()
        fig_confmfxdists.set_size_inches(8, 8)
        savename = 'confmfxcoalign'
        plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")

    def plot_filtering(self):
        print('Plotting filtering...', end=' \r')
        roi_idx_cum = 0
        conf_idxs = np.unique(self.track_data['confidx'])
        for confidx in conf_idxs:
            conf_data = self.track_data[self.track_data['confidx']==confidx]
            roi_idxs = np.unique(conf_data['roiidx'])
            for roiidx in roi_idxs:
                ax_filter1 = self.figs[11].add_subplot(self.N_rois, self.subplot_cols, 1+roi_idx_cum*self.subplot_cols)
                ax_filter0 = self.figs[11].add_subplot(self.N_rois, self.subplot_cols, 2+roi_idx_cum*self.subplot_cols)
                roi_data = conf_data[conf_data['roiidx']==roiidx]
                for _,track in roi_data.iterrows():
                    x_tr = track['x']
                    y_tr = track['y']
                    if track['filter'] == 1:
                        # 1 = short
                        ax_filter0.plot(x_tr, y_tr, color='blue', linewidth=0.3);
                    elif track['filter'] == 2:
                        # 2 = blob_meandistinterval
                        ax_filter0.plot(x_tr, y_tr, color='gray', linewidth=0.3);
                    elif track['filter'] == 3:
                        # 3 = blob_meanpossliding
                        ax_filter0.plot(x_tr, y_tr, color='red', linewidth=0.3);
                    elif track['filter'] == 4:
                        # 4 = site moved
                        ax_filter0.plot(x_tr, y_tr, color='magenta', linewidth=0.3);
                    elif track['filter'] == 0:
                        # 0 = pass all
                        ax_filter1.plot(x_tr, y_tr, color='green', linewidth=0.3);
                roi_idx_cum += 1
        self.set_fig_size()

    def fit_site_position(self, mini_roi_size=0.7, offset_lim=1.2, checknext=False, checknext_distlim=0.07):
        print('Fitting site position...', end=' \r')
        pos_corr = []
        pos_corr_px = []
        site_position_fit_successes = []
        site_moveds = []
        site_moved_dists = []
        for roi_dataidx, roi in self.roi_data.iterrows():
            # get current positions and image around roi
            #pos = roi['roi_pos']
            pos_px = roi['roi_pos_px']
            pixelsize = roi['pxsize']
            confsize = roi['conf_size']
            confoffset = roi['conf_offset']
            size = int(np.ceil(mini_roi_size/pixelsize)) if np.ceil(mini_roi_size/pixelsize)%2==1 else int(np.ceil(mini_roi_size/pixelsize)+1)
            rad = int((size-1)/2)
            img_mini = roi['conf_img'][pos_px[1]-rad:pos_px[1]+rad+1, pos_px[0]-rad:pos_px[0]+rad+1].ravel()
            # fit symmetric 2D gaussian
            x = np.linspace(0, size-1, size)
            y = np.linspace(0, size-1, size)
            x, y = np.meshgrid(x, y)
            initial_guess = (np.max(img_mini),int((size-1)/2),int((size-1)/2),1,0,10)
            try:
                popt, _ = curve_fit(self._gaussian2D, (x, y), img_mini, p0=initial_guess)
                # get offsets from pixel-position
                dx = popt[1]-int((size-1)/2)
                dy = popt[2]-int((size-1)/2)
                dd = np.sqrt(dx**2+dy**2)
                # if detected offsets are too large, keep pixel-position, otherwise correct it
                if dd > offset_lim:
                    real_peak_pos = pos_px
                    real_peak_pos_um = (real_peak_pos[0]*pixelsize+confoffset[0], real_peak_pos[1]*pixelsize+confoffset[1])
                    #real_peak_pos_um = (real_peak_pos[0]*pixelsize-confsize[0]/2, real_peak_pos[1]*pixelsize-confsize[1]/2)
                    #mini_peak_pos = (int((size-1)/2),int((size-1)/2))
                else:
                    real_peak_pos = (pos_px[0]+dx, pos_px[1]+dy)
                    real_peak_pos_um = (real_peak_pos[0]*pixelsize+confoffset[0], real_peak_pos[1]*pixelsize+confoffset[1])
                    #real_peak_pos_um = (real_peak_pos[0]*pixelsize-confsize[0]/2, real_peak_pos[1]*pixelsize-confsize[1]/2)
                    #mini_peak_pos = (int((size-1)/2)+dx, int((size-1)/2)+dy)
                pos_corr.append(real_peak_pos_um)
                pos_corr_px.append(real_peak_pos)
                ### for plotting fit
                #data_fitted = self._gaussian2D((x, y), *popt)
                #_, ax = plt.subplots(1, 1, figsize=(1.5,1.5))
                #ax.imshow(img_mini.reshape(size, size))
                #ax.contour(x, y, data_fitted.reshape(size, size), 5)
                #ax.scatter(*mini_peak_pos, color='r')
                site_position_fit_successes.append(True)
                if checknext==True:
                    # check if site position moved in next confocal
                    next_roi_dataidx = roi_dataidx+1 if roi_dataidx+1 < len(self.roi_data) else 0
                    if next_roi_dataidx > 0:
                        next_roi = self.roi_data.iloc[next_roi_dataidx]
                        next_pixelsize = next_roi['pxsize']
                        next_confoffset = next_roi['conf_offset']
                        next_confsize = next_roi['conf_size']
                        if next_pixelsize==pixelsize and next_confoffset==confoffset and next_confsize==confsize:
                            img_mini_next = next_roi['conf_img'][pos_px[1]-rad:pos_px[1]+rad+1, pos_px[0]-rad:pos_px[0]+rad+1].ravel()
                            # fit symmetric 2D gaussian
                            x = np.linspace(0, size-1, size)
                            y = np.linspace(0, size-1, size)
                            x, y = np.meshgrid(x, y)
                            initial_guess = (np.max(img_mini_next),popt[1],popt[2],1,0,10)
                            try:
                                popt_next, _ = curve_fit(self._gaussian2D, (x, y), img_mini_next, p0=initial_guess)
                                # get offsets from pixel-position
                                dd_next = np.sqrt((popt[1]-popt_next[1])**2+(popt[2]-popt_next[2])**2)*pixelsize
                                site_moved_dists.append(dd_next)
                                if dd_next < checknext_distlim and popt_next[0]>popt[0]/3:
                                    site_moveds.append(False)
                                else:
                                    site_moveds.append(True)
                            except:
                                site_moveds.append(np.nan)
                                site_moved_dists.append(np.nan)
                        else:
                            site_moveds.append(np.nan)
                            site_moved_dists.append(np.nan)
                    else:
                        site_moveds.append(np.nan)
                        site_moved_dists.append(np.nan)
                else:
                    site_moveds.append(False)
                    site_moved_dists.append(np.nan)
            except:
                real_peak_pos = pos_px
                real_peak_pos_um = (real_peak_pos[0]*pixelsize-confsize[0]/2, real_peak_pos[1]*pixelsize-confsize[1]/2)
                pos_corr.append(real_peak_pos_um)
                pos_corr_px.append(real_peak_pos)
                site_position_fit_successes.append(False)
                site_moveds.append(np.nan)
                site_moved_dists.append(np.nan)
                
        self.roi_data['roi_pos_px_raw'] = self.roi_data['roi_pos_px'].copy()
        self.roi_data['roi_pos_px'] = pos_corr_px
        self.roi_data['roi_pos_raw'] = self.roi_data['roi_pos'].copy()
        self.roi_data['roi_pos'] = pos_corr
        self.roi_data['siteposition_fitsuccess'] = site_position_fit_successes
        self.roi_data['site_moved'] = site_moveds
        self.roi_data['site_moved_dists'] = site_moved_dists
        # update positions also in track_data
        self.update_track_roipos()

    def plot_sitepos_fitting(self):
        print('Plotting site position fitting...', end=' \r')
        roi_idx_cum = 0
        conf_idxs = np.unique(self.track_data['confidx'])
        for confidx in conf_idxs:
            conf_data = self.track_data[self.track_data['confidx']==confidx]
            roi_idxs = np.unique(conf_data['roiidx'])
            for roiidx in roi_idxs:
                track_data = conf_data[conf_data['roiidx']==roiidx].iloc[0]
                ax_img = self.figs[12].add_subplot(self.N_rois, self.subplot_cols, 1+roi_idx_cum*self.subplot_cols)
                img = ax_img.imshow(track_data['confimg'])
                img.set_extent(track_data['confimg_ext'])  # scale overlay image to the correct pixel size for the tracks
                ax_img.set_xlim(*track_data['conf_xlim'])
                ax_img.set_ylim(*track_data['conf_ylim'])
                roi_data = self.roi_data[(self.roi_data['conf_idx']==confidx) & (self.roi_data['roi_idx']==roiidx)]
                ax_img.scatter(roi_data['roi_pos_raw'].iloc[0][0]+roi_data['pxsize']/2+track_data['conf_xcorr'], roi_data['roi_pos_raw'].iloc[0][1]+roi_data['pxsize']/2+track_data['conf_ycorr'], color='k')
                ax_img.scatter(roi_data['roi_pos'].iloc[0][0]+roi_data['pxsize']/2+track_data['conf_xcorr'], roi_data['roi_pos'].iloc[0][1]+roi_data['pxsize']/2+track_data['conf_ycorr'], color='r')
                roi_idx_cum += 1

    def filter_site_flagging(self, plot_filtered=False):
        """ Flag tracks and localizations that passes inside sites, as well as filter tracks. """
        print('Filtering and site flagging...', end=' \r')
        roi_idx_cum = 0
        for _, roi in self.roi_data.iterrows():
            conf_idx = roi['conf_idx']
            roi_idx = roi['roi_idx']
            roi_name = roi['roi_name']
            roi_pos_um = roi['roi_pos']
            pxsize = roi['pxsize']
            pxshift = pxsize/2
            image_conf = roi['conf_img']
            conf_xlim = roi['conf_xlims']
            conf_ylim = roi['conf_ylims']
            conf_size = roi['conf_size']
            conf_offset = roi['conf_offset']
            x_conf_correction = roi['conf_xcorr']
            y_conf_correction = roi['conf_ycorr']
            site_moved = roi['site_moved']

            track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
            track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

            # get image zoom from track coordinates
            ax_onlyconf = self.figs[1].add_subplot(self.N_rois, self.subplot_cols, 1+roi_idx_cum*self.subplot_cols)
            ax_overlay = self.figs[1].add_subplot(self.N_rois, self.subplot_cols, 2+roi_idx_cum*self.subplot_cols)
            img_overlay = ax_overlay.imshow(image_conf, cmap='hot')
            extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
            img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
            img_onlyconf = ax_onlyconf.imshow(image_conf, cmap='hot')
            img_onlyconf.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
            # add circles for caveolae site to zooms
            cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
            cav_circ_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
            # add circles for caveolae site for inclusion statistics to zooms
            cav_circ_inclusion_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
            cav_circ_inclusion_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')

            ax_overlay.add_patch(cav_circ_overlay);
            ax_onlyconf.add_patch(cav_circ_onlyconf);
            ax_overlay.add_patch(cav_circ_inclusion_overlay);
            ax_onlyconf.add_patch(cav_circ_inclusion_onlyconf);
            # circle polygon from overlay in correct coordinates for test
            verts = cav_circ_overlay.get_path().vertices
            trans = cav_circ_overlay.get_patch_transform()
            cav_circ_overlay_scaled_pnts = trans.transform(verts)
            cav_circ_overlay_scaled = patches.Polygon(cav_circ_overlay_scaled_pnts)
            vertsincl = cav_circ_inclusion_overlay.get_path().vertices
            transincl = cav_circ_inclusion_overlay.get_patch_transform()
            cav_circ_inclusion_overlay_scaled_pnts = transincl.transform(vertsincl)
            cav_circ_inclusion_overlay_scaled = patches.Polygon(cav_circ_inclusion_overlay_scaled_pnts)
            ax_overlay.set_xlim(*conf_xlim)
            ax_overlay.set_ylim(*conf_ylim)
            ax_onlyconf.set_xlim(*conf_xlim)
            ax_onlyconf.set_ylim(*conf_ylim)
            for tr_idx, track in track_roi_data.iterrows():
                x_track = track['x']
                y_track = track['y']
                tim_track = track['tim']
                dists = [self._distance((x0,y0),(x1,y1)) for x1,x0,y1,y0 in zip(x_track[1:],x_track[:-1],y_track[1:],y_track[:-1])]
                #mean_x_pos = [np.mean(x_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(x_track)-self.interval_meanpos,self.interval_meanpos), np.arange(0,len(x_track)-self.interval_meanpos,self.interval_meanpos)+self.interval_meanpos)]
                #mean_y_pos = [np.mean(y_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(y_track)-self.interval_meanpos,self.interval_meanpos), np.arange(0,len(y_track)-self.interval_meanpos,self.interval_meanpos)+self.interval_meanpos)]
                slidestd_x_pos = [np.std(x_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                slidestd_y_pos = [np.std(y_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                # filter indexing: 0 = pass all, 1 = short, 2 = blob_meandistinterval, 3 = blob_meanpossliding, 4 = site moved (all tracks filtered)
                if site_moved==True:
                    filter_id = 4
                elif (tim_track[-1] - tim_track[0] < self.min_time):
                    filter_id = 1
                elif (np.mean(np.abs(dists[::self.interval_dist])) < self.blob_dist):
                    filter_id = 2
                #elif (np.std(mean_x_pos) < self.meanpos_thresh or np.std(mean_y_pos) < self.meanpos_thresh):
                #    filter_id = 3
                elif (np.mean(slidestd_x_pos) < self.meanslidestd_thresh and np.mean(slidestd_y_pos) < self.meanslidestd_thresh):
                    filter_id = 3
                else:
                    filter_id = 0
                track_roi_data.loc[tr_idx, 'filter'] = filter_id
                if filter_id == 0:
                    # site flagging of localizations
                    cont_points = cav_circ_overlay_scaled.contains_points(np.array([x_track,y_track]).T)
                    if any(cont_points):
                        track_roi_data.loc[tr_idx, 'inout_flag'] = True
                        track_roi_data.loc[tr_idx, 'inout_len'] = np.sum(cont_points)
                        try:
                            ax_overlay.plot(x_track[cont_points], y_track[cont_points], color='green', linewidth=0.3);
                            ax_overlay.plot(x_track[~cont_points], y_track[~cont_points], color='red', linewidth=0.3);
                        except:
                            pass
                    else:
                        track_roi_data.loc[tr_idx, 'inout_flag'] = False
                        track_roi_data.loc[tr_idx, 'inout_len'] = 0
                        ax_overlay.plot(x_track, y_track, color='blue', linewidth=0.3);
                    track_roi_data.loc[[tr_idx], 'inout_mask'] = pd.Series([cont_points], index=[tr_idx])
                    
                    # site flagging of localizations (inclusion statistics)
                    cont_points_incl = cav_circ_inclusion_overlay_scaled.contains_points(np.array([x_track,y_track]).T)
                    if any(cont_points_incl):
                        track_roi_data.loc[tr_idx, 'inout_incl_flag'] = True
                        track_roi_data.loc[tr_idx, 'inout_incl_len'] = np.sum(cont_points_incl)
                    else:
                        track_roi_data.loc[tr_idx, 'inout_incl_flag'] = False
                        track_roi_data.loc[tr_idx, 'inout_incl_len'] = 0
                        #ax_overlay.plot(x_track, y_track, color='blue', linewidth=0.3);
                    track_roi_data.loc[[tr_idx], 'inout_incl_mask'] = pd.Series([cont_points_incl], index=[tr_idx])
                else:
                    track_roi_data.loc[tr_idx, 'inout_flag'] = np.nan
                    track_roi_data.loc[tr_idx, 'inout_len'] = np.nan
                    track_roi_data.loc[tr_idx, 'inout_mask'] = np.nan
                    track_roi_data.loc[tr_idx, 'inout_incl_flag'] = np.nan
                    track_roi_data.loc[tr_idx, 'inout_incl_len'] = np.nan
                    track_roi_data.loc[tr_idx, 'inout_incl_mask'] = np.nan
                    if plot_filtered:
                        ax_overlay.plot(x_track, y_track, color='gray', linewidth=0.3);
            
            mask = (self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)
            self.track_data.loc[mask, 'inout_flag'] = track_roi_data['inout_flag']
            self.track_data.loc[mask, 'inout_len'] = track_roi_data['inout_len']
            self.track_data.loc[mask, 'inout_mask'] = track_roi_data['inout_mask']
            self.track_data.loc[mask, 'inout_incl_flag'] = track_roi_data['inout_incl_flag']
            self.track_data.loc[mask, 'inout_incl_len'] = track_roi_data['inout_incl_len']
            self.track_data.loc[mask, 'inout_incl_mask'] = track_roi_data['inout_incl_mask']
            self.track_data.loc[mask, 'filter'] = track_roi_data['filter']

            ax_overlay.annotate(f'Conf {conf_idx}, ROI {roi_idx}, ROIFile {roi_name}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            ax_onlyconf.annotate(f'Conf {conf_idx}, ROI {roi_idx}, ROIFile {roi_name}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            
            roi_idx_cum += 1
        #self.set_fig_size()
        plt.figure(1)
        fig_fsf = plt.gcf()
        fig_fsf.set_size_inches(10,80)

    def update_track_roipos(self):
        for tr_idx, track in self.track_data.iterrows():
            confidx = track['confidx']
            roiidx = track['roiidx']
            self.track_data.at[tr_idx, 'roipos'] = self.roi_data.loc[(self.roi_data['conf_idx'] == confidx) & (self.roi_data['roi_idx'] == roiidx)]['roi_pos'].iloc[0]
            self.track_data.at[tr_idx, 'roipos_px'] = self.roi_data.loc[(self.roi_data['conf_idx'] == confidx) & (self.roi_data['roi_idx'] == roiidx)]['roi_pos_px'].iloc[0]

    def residence_time_analysis(self):
        """ Get all delta-ts and calculate residence times inside area. """
        print('Running residence time analysis...', end=' \r')
        restimes_in_all = []
        for _, row in self.track_data.iterrows():
            restimes_in = [np.nan]
            if row['inout_incl_flag'] == True:
                restimes_in = []
                in_idx_splits = self._consecutive_bool(np.argwhere(row['inout_incl_mask']).flatten())
                for idx_list in in_idx_splits:
                    if len(idx_list) > self.split_len_thresh:
                        start_idx = idx_list[-1] if (idx_list[-1]+1)>len(row['tim'])-1 else (idx_list[-1]+1)
                        stop_idx = idx_list[0] if (idx_list[0]-1)<0 else (idx_list[0]-1)
                        restime = row['tim'][start_idx] - row['tim'][stop_idx]
                        restimes_in.append(restime)
            restimes_in_all.append(restimes_in)
        self.track_data['restimes_in'] = restimes_in_all

    def plot_residence_time(self, bin_width=0.5, xmax=20, ymin=2e-4, ymax=2e-1, format='svg'):
        print('Plotting residence time...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: Residence time')
        restimes = np.array(self.track_data.restimes_in)
        restimes = [times for times in restimes if np.any(~np.isnan(times))]
        restimes = [x*1e3 for xs in restimes for x in xs]  # in ms
        plt.figure(figsize=(8,3))
        try:
            plt.hist(restimes,bins=np.arange(0, np.max(restimes), bin_width), density=True, label=f'{np.mean(restimes):.3f} +- {np.std(restimes)/np.sqrt(len(restimes)):.3f}')
        except:
            pass
        #plt.yscale('log')
        plt.xlabel('Site residence time [ms]')
        plt.ylabel('Norm. occ. [arb.u.]')
        plt.legend()
        plt.xlim([0, xmax])
        #plt.ylim([ymin, ymax])
        savename = 'residencetime'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()
        self.print_to_file(f'Residence time = {np.mean(restimes):.3f} +- (sem) {np.std(restimes)/np.sqrt(len(restimes)):.3f} +- (std) {np.std(restimes):.3f}')

    def track_inclusion(self):
        print('Running track inclusion analysis...', end=' \r')
        confs = self.track_data['confidx'].unique()
        tracks_in_ratio = []
        for conf in confs:
            conf_data = self.track_data[self.track_data['confidx']==conf].copy()
            rois = conf_data['roiidx'].unique()
            for roi in rois:
                roi_data = conf_data[self.track_data['roiidx']==roi].copy()
                n_tracks_in = np.count_nonzero(roi_data['inout_incl_flag']==True)
                n_tracks_out = np.count_nonzero(roi_data['inout_incl_flag']==False)
                if n_tracks_in+n_tracks_out > 0:
                    tracks_in_ratio.append(n_tracks_in/(n_tracks_in+n_tracks_out))
                else:
                    tracks_in_ratio.append(np.nan)
        self.roi_data['tracks_in_ratio'] = tracks_in_ratio

    def plot_track_inclusion(self, jitter_str=0.5, format='svg'):
        print('Plotting track inclusion...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: Track inclusion')
        plt.figure(figsize=(2,3))
        track_inclusion = self.roi_data['tracks_in_ratio']
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in track_inclusion]
        plt.scatter(np.ones(len(track_inclusion)) + jitter, track_inclusion, alpha=0.4, label=f'{np.nanmean(track_inclusion):.3f} +- {np.nanstd(track_inclusion)/np.sqrt(sum(~np.isnan(track_inclusion))):.3f}')
        plt.ylabel('Track inclusion ratio [arb.u.]')
        plt.legend()
        plt.xticks([])
        plt.ylim([0,1])
        savename = 'trackinclusion'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()
        self.print_to_file(f'Track inclusion = {np.mean(track_inclusion):.3f} +- (sem) {np.std(track_inclusion)/np.sqrt(len(track_inclusion)):.3f} +- (std) {np.std(track_inclusion):.3f}')

    def get_all_dts(self):
        """ Get the full distribution of dts in the dataset. """
        dts_all = []
        for _, row in self.track_data.iterrows():
            if row['inout_flag'] == True:
                t_track = row['tim']
                dts_all.append(np.diff(t_track))
        self.dts = np.concatenate(dts_all)

    def diff_analysis(self):
        """ Get all delta-distances and delta-t for every possible dt up to self.max_time_lag for all tracks. """
        print('Running SD analysis...', end=' \r')
        dists_in_all = []
        dts_in_all = []
        dists_out_all = []
        dts_out_all = []
        dists_all = []
        dts_all = []
        for _, row in self.track_data.iterrows():
            dists_in = np.nan
            dts_in = np.nan
            dists_out = np.nan
            dts_out = np.nan
            dists = np.nan
            dts = np.nan
            # only site-passing tracks, divided in and out of site
            if row['inout_flag'] == True:
                dists_in = []
                dts_in = []
                dists_out = []
                dts_out = []
                in_idx_splits = self._consecutive_bool(np.argwhere(row['inout_mask']).flatten())
                out_idx_splits = self._consecutive_bool(np.argwhere(~row['inout_mask']).flatten())
                for idx_list in in_idx_splits:
                    if len(idx_list) > self.split_len_thresh:
                        x_track = row['x'][idx_list]
                        y_track = row['y'][idx_list]
                        t_track = row['tim'][idx_list]
                        for idx_0 in range(len(x_track)-2):
                            x0 = x_track[idx_0]
                            y0 = y_track[idx_0]
                            cumdt = 0
                            cumidx = 0
                            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                                cumidx += 1
                                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                                if cumidx==1 and cumdt>self.max_dt:
                                    break
                                if cumdt < self.max_time_lag:
                                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2)**2
                                    dists_in.append(sqd)
                                    dts_in.append(cumdt)
                for idx_list in out_idx_splits:
                    if len(idx_list) > self.split_len_thresh:
                        x_track = row['x'][idx_list]
                        y_track = row['y'][idx_list]
                        t_track = row['tim'][idx_list]
                        for idx_0 in range(len(x_track)-2):
                            x0 = x_track[idx_0]
                            y0 = y_track[idx_0]
                            cumdt = 0
                            cumidx = 0
                            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                                cumidx += 1
                                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                                if cumidx==1 and cumdt>self.max_dt:
                                    break
                                if cumdt < self.max_time_lag:
                                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2)**2
                                    dists_out.append(sqd)
                                    dts_out.append(cumdt)
            # all tracks, avoiding filtered
            if row['filter']==0:
                dists = []
                dts = []
                x_track = row['x']
                y_track = row['y']
                t_track = row['tim']
                for idx_0 in range(len(x_track)-2):
                    x0 = x_track[idx_0]
                    y0 = y_track[idx_0]
                    cumdt = 0
                    cumidx = 0
                    while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                        cumidx += 1
                        cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                        if cumdt < self.max_time_lag:
                            sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2)**2
                            dists.append(sqd)
                            dts.append(cumdt)
            dists_all.append(dists)
            dts_all.append(dts)
            dists_in_all.append(dists_in)
            dts_in_all.append(dts_in)
            #if row['inout_flag'] == True:
            #    print(f'number of d, in: {len(dists_in)}')
            dists_out_all.append(dists_out)
            dts_out_all.append(dts_out)
            #if row['inout_flag'] == True:
            #    print(f'number of d, out: {len(dists_out)}')

        self.assign_dt_bins(dts_all)
        dts_all_binned = []
        for dts_arr in dts_all:
            if np.all(~np.isnan(dts_arr)):
                dt_binned = [self._find_nearest(self.dt_bins, val) for val in dts_arr]
                dts_all_binned.append(dt_binned)
            else:
                dts_all_binned.append(np.nan)
        dts_in_all_binned = []
        for dts_arr in dts_in_all:
            if np.all(~np.isnan(dts_arr)):
                dt_binned = [self._find_nearest(self.dt_bins, val) for val in dts_arr]
                dts_in_all_binned.append(dt_binned)
            else:
                dts_in_all_binned.append(np.nan)
        dts_out_all_binned = []
        for dts_arr in dts_out_all:
            if np.all(~np.isnan(dts_arr)):
                dt_binned = [self._find_nearest(self.dt_bins, val) for val in dts_arr]
                dts_out_all_binned.append(dt_binned)
            else:
                dts_out_all_binned.append(np.nan)

        self.track_data['dists_in'] = dists_in_all
        self.track_data['dts_in'] = dts_in_all
        self.track_data['dts_in_binned'] = dts_in_all_binned
        self.track_data['dists_out'] = dists_out_all
        self.track_data['dts_out'] = dts_out_all
        self.track_data['dts_out_binned'] = dts_out_all_binned
        self.track_data['dists'] = dists_all
        self.track_data['dts'] = dts_all
        self.track_data['dts_binned'] = dts_all_binned

    def assign_dt_bins(self, dt_arr, diff_lim=6e-7):
        ## cannot set diff_lim higher than ~1e-7 if I want to be able to handle long lag times, because there the dts are all mixed up and almost all values occur, 
        ## which leads to it just being one big group after a certain dt as it does not find any difference in sorted unique dts larger than diff_lim
        dts_conc = np.concatenate([ls for ls in dt_arr if np.all(~np.isnan(ls))])
        #print(len(dts_conc))
        #print(np.max(dts_conc))
        dts_sort = np.sort(dts_conc)
        dt_unique = np.unique(dts_sort)
        #print(len(dt_unique))
        #print(np.max(dt_unique))
        dt_undiff = np.diff(dt_unique)
        self.dt_bins = dt_unique[np.insert(dt_undiff>diff_lim, 0, True)]
        #print(len(self.dt_bins))
        #print(np.max(self.dt_bins))
    
    def get_all_displacements_legacy(self, binned=True):
        dists_in_all = []
        dts_in_all = []
        dists_out_all = []
        dts_out_all = []
        dists_all = []
        dts_all = []
        key_dists_in = 'dists_in'
        key_dists_out = 'dists_out'
        key_dists_all = 'dists'
        if binned:
            key_dts_in = 'dts_in_binned'
            key_dts_out = 'dts_out_binned'
            key_dts_all = 'dts_binned'
        else:
            key_dts_in = 'dts_in'
            key_dts_out = 'dts_out'
            key_dts_all = 'dts'
        for _, row in self.track_data.iterrows():
            if row['inout_flag'] == True:
                if len(row[key_dists_in]) > 0:
                    dists_in_all.append(row[key_dists_in])
                    dts_in_all.append(row[key_dts_in])
                if len(row[key_dists_out]) > 0:
                    dists_out_all.append(row[key_dists_out])
                    dts_out_all.append(row[key_dts_out])
            dists_all.append(row[key_dists_all])
            dts_all.append(row[key_dts_all])
        return dists_in_all, dts_in_all, dists_out_all, dts_out_all, dists_all, dts_all

    def msd_analysis(self, jitter_str=0.5, y_min=0, y_max=1.0, plot=False, binned=False, format='svg', fitmodel='msd', textprint=True):
        """ binned parameter controls if to perform the MSD analysis on dt-binned datapoints,
        thus calculate MSDs in each bin, and fit that for extracting the dapp;
        or if to fit all SDs from all datapoints in a track and extract the dapp from that. """
        print('Running MSD analysis...', end=' \r')
        dapp_in = []
        dapp_out = []
        sigma_in = []
        sigma_out = []
        if plot:
            fig,ax = plt.subplots(1,2,figsize=(10,4))
        if not binned:
            if textprint:
                self.print_to_file('')
                self.print_to_file('Results: MSD analysis, non-binned')
            # fit all tracks inside site
            for rowidx, row in self.track_data.iterrows():
                if row['inout_flag'] == True:
                    d = row['dists_in']
                    t = row['dts_in']
                    if len(d) > self.fit_len_thresh:
                        x_fit = np.array([x for _, x in sorted(zip(t, d))])
                        t_fit = np.array([t for t, _ in sorted(zip(t, d))])
                        if fitmodel=='lin':
                            popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                            dapp_in.append(popt_lin[0]/4)
                            sigma_in.append(np.sqrt(popt_lin[1]/4))
                        elif fitmodel=='msd':
                            popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                            dapp_in.append(popt_msd[0])
                            sigma_in.append(popt_msd[1])
                        if plot:
                            ax[0].scatter(t_fit, x_fit, s=2, alpha=0.2)
                            if fitmodel=='lin':
                                ax[0].plot(t_fit, self._f_lin(t_fit, *popt_lin))
                            elif fitmodel=='msd':
                                ax[0].plot(t_fit, self._f_msd(t_fit, *popt_msd))
                    else:
                        dapp_in.append(np.nan)
                        sigma_in.append(np.nan)
                else:
                    dapp_in.append(np.nan)
                    sigma_in.append(np.nan)
            self.track_data['dapp_in'] = dapp_in
            self.track_data['locprec_in'] = sigma_in
            # fit all tracks outside site
            for _, row in self.track_data.iterrows():
                if row['inout_flag'] == True:
                    d = row['dists_out']
                    t = row['dts_out']
                    if len(d) > self.fit_len_thresh:
                        x_fit = np.array([x for _, x in sorted(zip(t, d))])
                        t_fit = np.array([t for t, _ in sorted(zip(t, d))])
                        if fitmodel=='lin':
                            popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                            dapp_out.append(popt_lin[0]/4)
                            sigma_out.append(np.sqrt(popt_lin[1]/4))
                        elif fitmodel=='msd':
                            popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                            dapp_out.append(popt_msd[0])
                            sigma_out.append(popt_msd[1])
                        if plot:
                            ax[1].scatter(t_fit, x_fit, s=2, alpha=0.2)
                            if fitmodel=='lin':
                                ax[1].plot(t_fit, self._f_lin(t_fit, *popt_lin))
                            elif fitmodel=='msd':
                                ax[1].plot(t_fit, self._f_msd(t_fit, *popt_msd))
                    else:
                        dapp_out.append(np.nan)
                        sigma_out.append(np.nan)
                else:
                    dapp_out.append(np.nan)
                    sigma_out.append(np.nan)
            self.track_data['dapp_out'] = dapp_out
            self.track_data['locprec_out'] = sigma_out
            if plot:
                ax[0].set_xlim([0,self.max_time_lag*1.5]);
                ax[0].set_ylim([0,0.0075]);
                ax[0].set_ylabel('SD [µm^2]')
                ax[0].set_xlabel('delta-t [s]')
                ax[1].set_xlim([0,self.max_time_lag*1.5]);
                ax[1].set_ylim([0,0.0075]);
                ax[1].set_ylabel('SD [µm^2]')
                ax[1].set_xlabel('delta-t [s]')
            savename = 'msdanalysis-sdvdt'
            #if format=='svg':
            #    plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
            #else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")

            plt.figure(figsize=(3,5))
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_in]
            plt.scatter(1*np.ones(len(dapp_in)) + jitter, dapp_in, color='green', alpha=0.4, label=f'{np.nanmean(dapp_in):.3f} +- {np.nanstd(dapp_in)/np.sqrt(sum(~np.isnan(dapp_in))):.3f} µm^2/s');
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_out]
            plt.scatter(2*np.ones(len(dapp_out)) + jitter, dapp_out, color='gray', alpha=0.4, label=f'{np.nanmean(dapp_out):.3f} +- {np.nanstd(dapp_out)/np.sqrt(sum(~np.isnan(dapp_out))):.3f} µm^2/s');
            plt.xticks([1,2]);
            plt.gca().set_xticklabels(['In site','Out of site'])
            plt.ylim(y_min,y_max);
            plt.legend()
            plt.show()
            savename = 'msdanalysis-dapp_population'
            if format=='svg':
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
            else:
                plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            if textprint:
                self.print_to_file(f'Inside: Dapp = {np.nanmean(dapp_in):.3f} +- (sem) {np.nanstd(dapp_in)/sum(~np.isnan(dapp_in)):.3f} +- (std) {np.nanstd(dapp_in):.3f} µm^2/s')
                self.print_to_file(f'Outside: Dapp = {np.nanmean(dapp_out):.3f} +- (sem) {np.nanstd(dapp_out)/sum(~np.isnan(dapp_out)):.3f} +- (std) {np.nanstd(dapp_out):.3f} µm^2/s')
                result_all = scipy.stats.ttest_ind(dapp_in, dapp_out, alternative='less')
                self.print_to_file('MSD, tracks, population, ttest')
                self.print_to_file(str(result_all))
        else:
            if textprint:
                self.print_to_file('')
                self.print_to_file('Results: MSD analysis, binned')
            ## TAKE BINNED DATAPOINTS, CALCULATE MSDS, AND FIT THOSE
            dists_in_all, dts_in_all, dists_out_all, dts_out_all, _, _ = self.get_all_displacements(binned=True)
            if plot:
                fig,ax = plt.subplots(1,2,figsize=(10,4))
            for _, row in self.track_data.iterrows():
                if row['inout_flag'] == True:
                    d = row['dists_in']
                    t = row['dts_in']
                    if len(d) > self.fit_len_thresh:
                        msds = []
                        dt_bins_track = []
                        d = np.asarray(d)
                        for dt_bin in self.dt_bins:
                            if np.any(t==dt_bin):
                                msds.append(np.mean(d[t==dt_bin]))
                                dt_bins_track.append(dt_bin)
                        msds = np.asarray(msds)
                        dt_bins_track = np.asarray(dt_bins_track)
                        if fitmodel=='lin':
                            popt_lin, _ = curve_fit(self._f_lin, dt_bins_track, msds)
                            dapp_in.append(popt_lin[0]/4)
                            sigma_in.append(np.sqrt(popt_lin[1]/4))
                        elif fitmodel=='msd':
                            popt_msd, _ = curve_fit(self._f_msd, dt_bins_track, msds)
                            dapp_in.append(popt_msd[0])
                            sigma_in.append(popt_msd[1])
                        if plot:
                            ax[0].scatter(dt_bins_track, msds, s=2, alpha=0.2)
                            if fitmodel=='lin':
                                ax[0].plot(dt_bins_track, self._f_lin(dt_bins_track, *popt_lin))
                            elif fitmodel=='msd':
                                ax[1].plot(dt_bins_track, self._f_msd(dt_bins_track, *popt_msd))
                    else:
                        dapp_in.append(np.nan)
                        sigma_in.append(np.nan)
                else:
                    dapp_in.append(np.nan)
                    sigma_in.append(np.nan)
            self.track_data['dapp_bin_in'] = dapp_in
            self.track_data['locprec_bin_in'] = sigma_in
            if plot:
                ax[0].set_xlim([0,self.max_time_lag]);
                ax[0].set_ylim([0,0.02]);
            for _, row in self.track_data.iterrows():
                if row['inout_flag'] == True:
                    d = row['dists_out']
                    t = row['dts_out']
                    if len(d) > self.fit_len_thresh:
                        msds = []
                        dt_bins_track = []
                        d = np.asarray(d)
                        for dt_bin in self.dt_bins:
                            if np.any(t==dt_bin):
                                msds.append(np.mean(d[t==dt_bin]))
                                dt_bins_track.append(dt_bin)
                        msds = np.asarray(msds)
                        dt_bins_track = np.asarray(dt_bins_track)
                        if fitmodel=='lin':
                            popt_lin, _ = curve_fit(self._f_lin, dt_bins_track, msds)
                            dapp_out.append(popt_lin[0]/4)
                            sigma_out.append(np.sqrt(popt_lin[1]/4))
                        elif fitmodel=='msd':
                            popt_msd, _ = curve_fit(self._f_msd, dt_bins_track, msds)
                            dapp_out.append(popt_msd[0])
                            sigma_out.append(popt_msd[1])
                        if plot:
                            ax[1].scatter(dt_bins_track, msds, s=2, alpha=0.2)
                            if fitmodel=='lin':
                                ax[1].plot(dt_bins_track, self._f_lin(dt_bins_track, *popt_lin))
                            elif fitmodel=='msd':
                                ax[1].plot(dt_bins_track, self._f_msd(dt_bins_track, *popt_msd))
                    else:
                        dapp_out.append(np.nan)
                        sigma_out.append(np.nan)
                else:
                    dapp_out.append(np.nan)
                    sigma_out.append(np.nan)
            self.track_data['dapp_bin_out'] = dapp_out
            self.track_data['locprec_bin_out'] = sigma_out
            if plot:
                ax[0].set_xlim([0,self.max_time_lag*1.5]);
                ax[0].set_ylim([0,0.025]);
                ax[0].set_ylabel('SD [µm^2]')
                ax[0].set_xlabel('delta-t [s]')
                ax[1].set_xlim([0,self.max_time_lag*1.5]);
                ax[1].set_ylim([0,0.02]);
                ax[1].set_ylabel('SD [µm^2]')
                ax[1].set_xlabel('delta-t [s]')
            savename = 'msdanalysis-sdvdt'
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")

            plt.figure(figsize=(3,5))
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_in]
            plt.scatter(1*np.ones(len(dapp_in)) + jitter, dapp_in, color='green', alpha=0.4, label=f'{np.mean(dapp_in):.3f} +- {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} µm^2/s');
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_out]
            plt.scatter(2*np.ones(len(dapp_out)) + jitter, dapp_out, color='gray', alpha=0.4, label=f'{np.mean(dapp_out):.3f} +- {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} µm^2/s');
            plt.xticks([1,2]);
            plt.gca().set_xticklabels(['In site','Out of site'])
            plt.ylim(y_min,y_max);
            plt.legend()
            plt.show()
            savename = 'msdanalysis-dapp_population'
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            if textprint:
                self.print_to_file(f'Inside: Dapp = {np.mean(dapp_in):.3f} +- (sem) {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} +- (std) {np.std(dapp_in):.3f} µm^2/s')
                self.print_to_file(f'Outside: Dapp = {np.mean(dapp_out):.3f} +- (sem) {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} +- (std) {np.std(dapp_out):.3f} µm^2/s')
                result_all = scipy.stats.ttest_ind(dapp_in, dapp_out, alternative='less')
                self.print_to_file('MSD, tracks, population, ttest')
                self.print_to_file(result_all)

        return dapp_in, dapp_out, sigma_in, sigma_out

    def msd_analysis_legacy(self, jitter_str=0.5, y_max=1.0, plot=False, binned=False, format='svg'):
        """ binned parameter controls if to perform the MSD analysis on dt-binned datapoints,
        thus calculate MSDs in each bin, and fit that for extracting the dapp;
        or if to fit all SDs from all datapoints in a track and extract the dapp from that. """
        print('Running MSD analysis...', end=' \r')
        dapp_in = []
        dapp_out = []
        sigma_in = []
        sigma_out = []
        if not binned:
            self.print_to_file('')
            self.print_to_file('Results: MSD analysis, non-binned')
            # FIT ALL DATAPOINTS
            dists_in_all, dts_in_all, dists_out_all, dts_out_all, _, _ = self.get_all_displacements(binned=False)
            if plot:
                fig,ax = plt.subplots(1,2,figsize=(10,4))
            for d, t in zip(dists_in_all[:], dts_in_all[:]):
                if len(d) > self.fit_len_thresh:
                    x_fit = np.array([x for _, x in sorted(zip(t, d))])
                    t_fit = np.array([t for t, _ in sorted(zip(t, d))])
                    #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                    #dapp_in.append(popt_lin[0]/4)
                    popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                    dapp_in.append(popt_msd[0])
                    #sigma_in.append(np.sqrt(popt_lin[1]/4))
                    sigma_in.append(popt_msd[1])
                    if plot:
                        ax[0].scatter(t_fit, x_fit, s=2, alpha=0.2)
                        #ax[0].plot(t_fit, self._f_lin(t_fit, *popt_lin))
                        ax[0].plot(t_fit, self._f_msd(t_fit, *popt_msd))
            if plot:
                ax[0].set_xlim([0,self.max_time_lag]);
                ax[0].set_ylim([0,0.0075]);
            for d, t in zip(dists_out_all[:], dts_out_all[:]):
                if len(d) > self.fit_len_thresh:
                    x_fit = np.array([x for _, x in sorted(zip(t, d))])
                    t_fit = np.array([t for t, _ in sorted(zip(t, d))])
                    #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                    #dapp_out.append(popt_lin[0]/4)
                    popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                    dapp_out.append(popt_msd[0])
                    #sigma_out.append(np.sqrt(popt_lin[1]/4))
                    sigma_out.append(popt_msd[1])
                    if plot:
                        ax[1].scatter(t_fit, x_fit, s=2, alpha=0.2)
                        #ax[1].plot(t_fit, self._f_lin(t_fit, *popt_lin))
                        ax[1].plot(t_fit, self._f_msd(t_fit, *popt_msd))
            if plot:
                ax[1].set_xlim([0,self.max_time_lag]);
                ax[1].set_ylim([0,0.0075]);
            savename = 'msdanalysis-sdvdt'
            #if format=='svg':
            #    plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
            #else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")

            plt.figure(figsize=(3,5))
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_in]
            plt.scatter(1*np.ones(len(dapp_in)) + jitter, dapp_in, color='green', alpha=0.4, label=f'{np.mean(dapp_in):.3f} +- {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} µm^2/s');
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_out]
            plt.scatter(2*np.ones(len(dapp_out)) + jitter, dapp_out, color='gray', alpha=0.4, label=f'{np.mean(dapp_out):.3f} +- {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} µm^2/s');
            plt.xticks([1,2]);
            plt.gca().set_xticklabels(['In site','Out of site'])
            plt.ylim(0,y_max);
            plt.legend()
            plt.show()
            savename = 'msdanalysis-dapp_population'
            if format=='svg':
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
            else:
                plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            self.print_to_file(f'Inside: Dapp = {np.mean(dapp_in):.3f} +- (sem) {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} +- (std) {np.std(dapp_in):.3f} µm^2/s')
            self.print_to_file(f'Outside: Dapp = {np.mean(dapp_out):.3f} +- (sem) {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} +- (std) {np.std(dapp_out):.3f} µm^2/s')

            result_all = scipy.stats.ttest_ind(dapp_in, dapp_out, alternative='less')
            self.print_to_file('MSD, tracks, population, ttest')
            self.print_to_file(str(result_all))
        else:
            self.print_to_file('')
            self.print_to_file('Results: MSD analysis, binned')
            ## TAKE BINNED DATAPOINTS, CALCULATE MSDS, AND FIT THOSE
            dists_in_all, dts_in_all, dists_out_all, dts_out_all, _, _ = self.get_all_displacements(binned=True)
            if plot:
                fig,ax = plt.subplots(1,2,figsize=(10,4))
            for d, t in zip(dists_in_all[:], dts_in_all[:]):
                if len(d) > self.fit_len_thresh:
                    msds = []
                    dt_bins_track = []
                    d = np.asarray(d)
                    for dt_bin in self.dt_bins:
                        if np.any(t==dt_bin):
                            msds.append(np.mean(d[t==dt_bin]))
                            dt_bins_track.append(dt_bin)
                    msds = np.asarray(msds)
                    dt_bins_track = np.asarray(dt_bins_track)
                    #popt_lin, _ = curve_fit(self._f_lin, dt_bins_track, msds)
                    #dapp_in.append(popt_lin[0]/4)
                    #sigma_in.append(np.sqrt(popt_lin[1]/4))
                    popt_msd, _ = curve_fit(self._f_msd, dt_bins_track, msds)
                    dapp_in.append(popt_msd[0])
                    sigma_in.append(popt_msd[1])
                    if plot:
                        ax[0].scatter(dt_bins_track, msds, s=2, alpha=0.2)
                        #ax[0].plot(dt_bins_track, self._f_lin(dt_bins_track, *popt_lin))
                        ax[1].plot(dt_bins_track, self._f_msd(dt_bins_track, *popt_msd))
            if plot:
                ax[0].set_xlim([0,self.max_time_lag]);
                ax[0].set_ylim([0,0.02]);
            for d, t in zip(dists_out_all[:], dts_out_all[:]):
                if len(d) > self.fit_len_thresh:
                    msds = []
                    dt_bins_track = []
                    d = np.asarray(d)
                    for dt_bin in self.dt_bins:
                        if np.any(t==dt_bin):
                            msds.append(np.mean(d[t==dt_bin]))
                            dt_bins_track.append(dt_bin)
                    msds = np.asarray(msds)
                    dt_bins_track = np.asarray(dt_bins_track)
                    #popt_lin, _ = curve_fit(self._f_lin, dt_bins_track, msds)
                    #dapp_out.append(popt_lin[0]/4)
                    #sigma_out.append(np.sqrt(popt_lin[1]/4))
                    popt_msd, _ = curve_fit(self._f_msd, dt_bins_track, msds)
                    dapp_out.append(popt_msd[0])
                    sigma_out.append(popt_msd[1])
                    if plot:
                        ax[1].scatter(dt_bins_track, msds, s=2, alpha=0.2)
                        #ax[1].plot(dt_bins_track, self._f_lin(dt_bins_track, *popt_lin))
                        ax[1].plot(dt_bins_track, self._f_msd(dt_bins_track, *popt_msd))
            if plot:
                ax[1].set_xlim([0,self.max_time_lag]);
                ax[1].set_ylim([0,0.02]);
            savename = 'msdanalysis-sdvdt'
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")

            plt.figure(figsize=(3,5))
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_in]
            plt.scatter(1*np.ones(len(dapp_in)) + jitter, dapp_in, color='green', alpha=0.4, label=f'{np.mean(dapp_in):.3f} +- {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} µm^2/s');
            jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_out]
            plt.scatter(2*np.ones(len(dapp_out)) + jitter, dapp_out, color='gray', alpha=0.4, label=f'{np.mean(dapp_out):.3f} +- {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} µm^2/s');
            plt.xticks([1,2]);
            plt.gca().set_xticklabels(['In site','Out of site'])
            plt.ylim(0,y_max);
            plt.legend()
            plt.show()
            savename = 'msdanalysis-dapp_population'
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            self.print_to_file(f'Inside: Dapp = {np.mean(dapp_in):.3f} +- (sem) {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} +- (std) {np.std(dapp_in):.3f} µm^2/s')
            self.print_to_file(f'Outside: Dapp = {np.mean(dapp_out):.3f} +- (sem) {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} +- (std) {np.std(dapp_out):.3f} µm^2/s')

            result_all = scipy.stats.ttest_ind(dapp_in, dapp_out, alternative='less')
            self.print_to_file('MSD, tracks, population, ttest')
            self.print_to_file(result_all)

        return dapp_in, dapp_out, sigma_in, sigma_out

    def msd_analysis_per_track(self, y_max=1.0, y_min=0, format='svg'):
        print('Running MSD analysis, per track...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: MSD analysis, tracks')
        dapps_all = []
        for _, row in self.track_data.iterrows():
            if row['inout_flag'] == True:
                row_dapps = [np.nan, np.nan]
                if len(row['dists_in']) > 0:
                    dists_in = row['dists_in']
                    dts_in = row['dts_in']
                    if len(dists_in) > self.fit_len_thresh:
                        x_fit = np.array([x for _, x in sorted(zip(dts_in, dists_in))])
                        t_fit = np.array([t for t, _ in sorted(zip(dts_in, dists_in))])
                        #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                        #dapp_in = popt_lin[0]/4
                        popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                        dapp_in = popt_msd[0]
                        row_dapps[0] = dapp_in
                if len(row['dists_out']) > 0:
                    dists_out = row['dists_out']
                    dts_out = row['dts_out']
                    if len(dists_out) > self.fit_len_thresh:
                        x_fit = np.array([x for _, x in sorted(zip(dts_out, dists_out))])
                        t_fit = np.array([t for t, _ in sorted(zip(dts_out, dists_out))])
                        #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                        #dapp_out = popt_lin[0]/4
                        popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                        dapp_out = popt_msd[0]
                        row_dapps[1] = dapp_out
                dapps_all.append(np.array(row_dapps))
        dapps_all = np.array(dapps_all)
        plt.figure(figsize=(3,5))
        plt.plot(dapps_all.T, '.-', color='green', alpha=0.1)
        plt.xticks([0,1]);
        plt.gca().set_xticklabels(['In site','Out of site'])
        #plt.ylim(y_min,y_max);
        plt.show()
        savename = 'dapp-trackspaired'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()
        
        dapps_ratio = [dapps[0]/dapps[1] for dapps in dapps_all]  # if dapps[0]>0 if dapps[0]<3 if dapps[1]>0 if dapps[1]<3]
        dapps_ratio = np.array([rat for rat in dapps_ratio if rat==rat])
        dapps_ratio = self._reject_outliers(dapps_ratio)
        plt.figure(figsize=(3,5))
        jitter_str = 0.5
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapps_ratio]
        plt.scatter(1*np.ones(len(dapps_ratio)) + jitter, dapps_ratio, color='green', alpha=0.4, label=f'{np.mean(dapps_ratio):.3f} +- {np.std(dapps_ratio)/np.sqrt(len(dapps_ratio)):.3f} µm^2/s');
        plt.xticks([1]);
        plt.gca().set_xticklabels(['Ratio, in/out'])
        plt.ylabel('D_app ratio, track [arb.u.]')
        #plt.ylim(y_min,y_max);
        plt.legend()
        plt.show()
        savename = 'dappratio-tracks'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()

        # ALTERNATIVE SWARM PLOTTING WITH SEABORN
        #import seaborn as sns
        #plt.figure(figsize=(2,5))
        #sns.set_style("whitegrid")
        #ax = sns.swarmplot(data=dapps_ratio)
        #ax = sns.boxplot(data=dapps_ratio,showcaps=False,boxprops={'facecolor':'None'},showfliers=False,whiskerprops={'linewidth':2})
        #ax.set_ylabel('test')
        #ax.set_ylim([0,5])
        #plt.show()
        #plt.style.use('classic')   # SET BACK PLT STANDARD STYLE, AS SEABORN CHANGES THE MATPLOTLIB STYLE

        #print(dapps_all)
        dapps_all_nonan = np.array([[ins,outs] for ins,outs in dapps_all if ins==ins if outs==outs])
        result_tracks = scipy.stats.ttest_rel(dapps_all_nonan[:,0],dapps_all_nonan[:,1],alternative='less')
        self.print_to_file('Tracks paired ttest')
        self.print_to_file(str(result_tracks))
        self.print_to_file(str(result_tracks.confidence_interval()))

        self.print_to_file(f'Dapp ratio, tracks = {np.mean(dapps_ratio):.3f} +- (sem) {np.std(dapps_ratio)/np.sqrt(len(dapps_ratio)):.3f} +- (std) {np.std(dapps_ratio):.3f} µm^2/s')

        return dapps_all
          
    def msd_analysis_per_roi(self, y_max=1.0, y_min=0, format='svg'):
        print('Running MSD analysis, per ROI...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: MSD analysis, ROIs')
        dapps_all = []
        confs = self.track_data['confidx'].unique()
        for conf in confs:
            conf_data = self.track_data[self.track_data['confidx']==conf].copy()
            rois = conf_data['roiidx'].unique()
            for roi in rois:
                roi_data = conf_data[self.track_data['roiidx']==roi].copy()
                dapps_ins = []
                dapps_outs = []
                for _, row in roi_data.iterrows():
                    if row['inout_flag'] == True:
                        if len(row['dists_in']) > 0:
                            dists_in = row['dists_in']
                            dts_in = row['dts_in']
                            if len(dists_in) > self.fit_len_thresh:
                                x_fit = np.array([x for _, x in sorted(zip(dts_in, dists_in))])
                                t_fit = np.array([t for t, _ in sorted(zip(dts_in, dists_in))])
                                #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                                #dapps_ins.append(popt_lin[0]/4)
                                popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                                dapps_ins.append(popt_msd[0])
                        if len(row['dists_out']) > 0:
                            dists_out = row['dists_out']
                            dts_out = row['dts_out']
                            if len(dists_out) > self.fit_len_thresh:
                                x_fit = np.array([x for _, x in sorted(zip(dts_out, dists_out))])
                                t_fit = np.array([t for t, _ in sorted(zip(dts_out, dists_out))])
                                #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                                #dapps_outs.append(popt_lin[0]/4)
                                popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
                                dapps_outs.append(popt_msd[0])
                dapps_all.append(np.array([np.mean(dapps_ins),np.mean(dapps_outs)]))
        dapps_all = np.array(dapps_all)
        plt.figure(figsize=(3,5))
        plt.plot(dapps_all.T, '.-', color='green', alpha=0.2)
        plt.xticks([0,1]);
        plt.gca().set_xticklabels(['In site','Out of site'])
        #plt.ylim(y_min,y_max);
        plt.show()
        savename = 'dapp-roispaired'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()

        dapps_ratio = [dapps[0]/dapps[1] for dapps in dapps_all]  # if dapps[0]>0 if dapps[0]<3 if dapps[1]>0 if dapps[1]<3]
        dapps_ratio = np.array([rat for rat in dapps_ratio if rat==rat])
        dapps_ratio = self._reject_outliers(dapps_ratio)
        plt.figure(figsize=(3,5))
        jitter_str = 0.5
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapps_ratio]
        plt.scatter(1*np.ones(len(dapps_ratio)) + jitter, dapps_ratio, color='green', alpha=0.4, label=f'{np.mean(dapps_ratio):.3f} +- {np.std(dapps_ratio)/np.sqrt(len(dapps_ratio)):.3f} µm^2/s');
        plt.xticks([1]);
        plt.gca().set_xticklabels(['Ratio, in/out'])
        plt.ylabel('D_app ratio, ROI [arb.u.]')        
        #plt.ylim(y_min,y_max);
        plt.legend()
        plt.show()
        savename = 'dappratio-rois'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()

        result_rois = scipy.stats.ttest_rel(dapps_all[:,0],dapps_all[:,1],alternative='less')
        self.print_to_file('ROIs paired ttest')
        self.print_to_file(str(result_rois))
        self.print_to_file(str(result_rois.confidence_interval()))

        self.print_to_file(f'ROIs Dapp ratio = {np.mean(dapps_ratio):.3f} +- (sem) {np.std(dapps_ratio)/np.sqrt(len(dapps_ratio)):.3f} +- (std) {np.std(dapps_ratio):.3f} µm^2/s')

        return dapps_all

    def diff_analysis_local(self, x_track, y_track, t_track):
        """ Get all delta-distances and delta-t for every possible dt up to self.max_time_lag for all tracks. """
        dists = []
        dts = []
        for idx_0 in range(len(x_track)-2):
            x0 = x_track[idx_0]
            y0 = y_track[idx_0]
            cumdt = 0
            cumidx = 0
            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                cumidx += 1
                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                if cumdt < self.max_time_lag:
                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2)**2
                    dists.append(sqd)
                    dts.append(cumdt)
        x_fit = np.array([x for _, x in sorted(zip(dts, dists))])
        t_fit = np.array([t for t, _ in sorted(zip(dts, dists))])
        #popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
        #dapp = popt_lin[0]/4
        popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
        dapp = popt_msd[0]
        return dapp

    def msd_analysis_local(self, window_pts=30):
        print('Run MSD analysis, local...', end=' \r')
        row_local_pos_all = []
        row_local_dapps_all = []
        for _, row in self.track_data.iterrows():
            row_local_pos = []
            row_local_dapps = []
            # get only non-filtered tracks
            if row['filter'] == 0:
                if len(row['x']) > window_pts:
                    for i in np.arange(0, len(row['x'])-1, 1):
                        if i > int(window_pts/2) and i < len(row['x'])-int(window_pts/2)-1:
                            lowlim = i-int(window_pts/2)
                            hilim = i+int(window_pts/2)
                        elif i <= int(window_pts/2):
                            lowlim = 0
                            hilim = i+int(window_pts/2)
                        elif i >= len(row['x'])-int(window_pts/2)-1:
                            lowlim = i-int(window_pts/2)
                            hilim = None
                        x_subtrack = row['x'][lowlim:hilim]
                        y_subtrack = row['y'][lowlim:hilim]
                        t_subtrack = row['tim'][lowlim:hilim]
                        try:
                            dapp_local = self.diff_analysis_local(x_subtrack, y_subtrack, t_subtrack)
                            row_local_pos.append([row['x'][i], row['y'][i]])
                            row_local_dapps.append(dapp_local)
                        except:
                            pass
            row_local_pos_all.append(row_local_pos)
            row_local_dapps_all.append(row_local_dapps)
            #else:
            #    row_local_pos_all.append(np.nan)
            #    row_local_dapps_all.append(np.nan)
        self.track_data['localdiff_pos'] = row_local_pos_all
        self.track_data['localdiff_dapp'] = row_local_dapps_all

    def plot_filteredtracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            x_tr = track['x']
                            y_tr = track['y']
                            if track['filter'] == 1:
                                # 1 = short
                                ax.plot(x_tr, y_tr, color='blue', linewidth=0.3);
                            elif track['filter'] == 2:
                                # 2 = blob_meandistinterval
                                ax.plot(x_tr, y_tr, color='gray', linewidth=0.3);
                            elif track['filter'] == 3:
                                # 3 = blob_meanpossliding
                                ax.plot(x_tr, y_tr, color='red', linewidth=0.3);
                            elif track['filter'] == 4:
                                # 4 = site moved
                                ax.plot(x_tr, y_tr, color='magenta', linewidth=0.3);
                            elif track['filter'] == 0:
                                # 0 = pass all
                                ax.plot(x_tr, y_tr, color='green', linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        x_tr = track['x']
                        y_tr = track['y']
                        if track['filter'] == 1:
                            # 1 = short
                            ax.plot(x_tr, y_tr, color='blue', linewidth=0.3);
                        elif track['filter'] == 2:
                            # 2 = blob_meandistinterval
                            ax.plot(x_tr, y_tr, color='gray', linewidth=0.3);
                        elif track['filter'] == 3:
                            # 3 = blob_meanpossliding
                            ax.plot(x_tr, y_tr, color='red', linewidth=0.3);
                        elif track['filter'] == 4:
                            # 4 = site moved
                            ax.plot(x_tr, y_tr, color='magenta', linewidth=0.3);
                        elif track['filter'] == 0:
                            # 0 = pass all
                            ax.plot(x_tr, y_tr, color='green', linewidth=0.3);
                plt.axis('off')
                savename = f'filteredtracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_alltracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            x_tr = track['x']
                            y_tr = track['y']
                            ax.plot(x_tr, y_tr, color='black', linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        x_tr = track['x']
                        y_tr = track['y']
                        ax.plot(x_tr, y_tr, color='black', linewidth=0.3);
                plt.axis('off')
                savename = f'alltracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_inouttracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                #img_overlay = ax.imshow(image_conf, cmap='hot')
                #extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                #img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='none', facecolor='g', alpha=0.3)
                ax.add_patch(cav_circ_overlay);
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                if track['inout_flag'] == True:
                                    x_track = track['x']
                                    y_track = track['y']
                                    ax.plot(x_track[track['inout_mask']], y_track[track['inout_mask']], color='green', linewidth=0.3);
                                    ax.plot(x_track[~track['inout_mask']], y_track[~track['inout_mask']], linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            if track['inout_flag'] == True:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track[track['inout_mask']], y_track[track['inout_mask']], color='green', linewidth=0.3);
                                ax.plot(x_track[~track['inout_mask']], y_track[~track['inout_mask']], linewidth=0.3);
                plt.axis('off')
                savename = f'inouttracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()
                
    def plot_tracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                img_overlay = ax.imshow(1000*np.ones(np.shape(image_conf)), cmap='Greys',vmin=0,vmax=1)
                extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track, y_track, linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            x_track = track['x']
                            y_track = track['y']
                            ax.plot(x_track, y_track, linewidth=0.3);
                plt.axis('off')
                savename = f'tracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_overlaytracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                img_overlay = ax.imshow(image_conf, cmap='hot')
                extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', facecolor='none', alpha=0.3)
                #ax.add_patch(cav_circ_overlay);
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track, y_track, linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            x_track = track['x']
                            y_track = track['y']
                            ax.plot(x_track, y_track, linewidth=0.3);
                plt.axis('off')
                savename = f'overlaytracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_inclusiontracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                #img_overlay = ax.imshow(image_conf, cmap='hot')
                #extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                #img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='none', facecolor='gray', alpha=0.3)
                ax.add_patch(cav_circ_overlay);
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                if track['inout_incl_flag'] == True:
                                    x_track = track['x']
                                    y_track = track['y']
                                    ax.plot(x_track, y_track, linewidth=0.3);
                                elif track['inout_incl_flag'] == False:
                                    x_track = track['x']
                                    y_track = track['y']
                                    ax.plot(x_track, y_track, color='gray', linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            if track['inout_incl_flag'] == True:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track, y_track, linewidth=0.3);
                            elif track['inout_incl_flag'] == False:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track, y_track, color='gray', linewidth=0.3);
                plt.axis('off')
                savename = f'inclusiontracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_residencetime_tracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                #img_overlay = ax.imshow(image_conf, cmap='hot')
                #extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                #img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='none', facecolor='gray', alpha=0.3)
                ax.add_patch(cav_circ_overlay);
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                if track['inout_incl_flag'] == True:
                                    x_track = track['x']
                                    y_track = track['y']
                                    ax.plot(x_track[track['inout_incl_mask']], y_track[track['inout_incl_mask']], linewidth=0.3);
                                    ax.plot(x_track[~track['inout_incl_mask']], y_track[~track['inout_incl_mask']], color='gray', linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            if track['inout_incl_flag'] == True:
                                x_track = track['x']
                                y_track = track['y']
                                ax.plot(x_track[track['inout_incl_mask']], y_track[track['inout_incl_mask']], linewidth=0.3);
                                ax.plot(x_track[~track['inout_incl_mask']], y_track[~track['inout_incl_mask']], color='gray', linewidth=0.3);
                plt.axis('off')
                savename = f'restimetracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_outside_tracks(self, roi_number=0, plot_lim_add=0.13, trackratio=0.2, tridxs=None):
        for roi_idx, roi in self.roi_data.iterrows():
            if roi_idx == roi_number:
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                roi_pos_um = roi['roi_pos']
                pxsize = roi['pxsize']
                pxshift = pxsize/2
                image_conf = roi['conf_img']
                conf_xlim = roi['conf_xlims']
                conf_ylim = roi['conf_ylims']
                conf_offset = roi['conf_offset']
                x_conf_correction = roi['conf_xcorr']
                y_conf_correction = roi['conf_ycorr']

                track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

                # get image zoom from track coordinates
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1,1,1)
                #img_overlay = ax.imshow(image_conf, cmap='hot')
                #extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift,pxshift,pxshift,pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                #img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                # add circles for caveolae site to zooms
                cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.circle_radii[2], edgecolor='none', facecolor='gray', alpha=0.3)
                ax.add_patch(cav_circ_overlay);
                verts_outer = cav_circ_overlay.get_path().vertices
                trans_outer = cav_circ_overlay.get_patch_transform()
                cav_circ_outer_scaled_pnts = trans_outer.transform(verts_outer)
                cav_circ_outer_scaled = patches.Polygon(cav_circ_outer_scaled_pnts)
                ax.set_xlim(conf_xlim[0]-plot_lim_add,conf_xlim[1]+plot_lim_add)
                ax.set_ylim(conf_ylim[0]+plot_lim_add,conf_ylim[1]-plot_lim_add)
                for tr_idx, track in track_roi_data.iterrows():
                    if tridxs is not None:
                        if track['tridx'] in tridxs:
                            if track['filter'] == 0:
                                x_track = track['x']
                                y_track = track['y']
                                cont_points_outer_raw = cav_circ_outer_scaled.contains_points(np.array([x_track,y_track]).T)
                                ax.plot(x_track[~cont_points_outer_raw], y_track[~cont_points_outer_raw], linewidth=0.3);
                                ax.plot(x_track[cont_points_outer_raw], y_track[cont_points_outer_raw], color='gray', linewidth=0.3);
                    elif tr_idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                        if track['filter'] == 0:
                            x_track = track['x']
                            y_track = track['y']
                            cont_points_outer_raw = cav_circ_outer_scaled.contains_points(np.array([x_track,y_track]).T)
                            ax.plot(x_track[~cont_points_outer_raw], y_track[~cont_points_outer_raw], linewidth=0.3);
                            ax.plot(x_track[cont_points_outer_raw], y_track[cont_points_outer_raw], color='gray', linewidth=0.3);
                plt.axis('off')
                savename = f'dtransoutsidetracks-roi{roi_number}'
                plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
                plt.show()

    def plot_fullroi_sixsites(self, cell_idx, singleroi=True, savename_suffix='', trackratio=0.2):
        roi_plots = 6
        if singleroi:
            filelist = os.listdir(self.paths[0])
            filelist_rois = [file for file in filelist if file.endswith('.npy')]
            filelist_conf = [file for file in filelist if 'conf-raw' in file and 'analysis' not in file and '.png' not in file]
            roi_times = [float(file.split('-')[1].split('_')[0]) for file in filelist_conf]
            time_between_rois = np.diff(roi_times)
            time_between_rois = np.insert(time_between_rois, 0, 0)
            time_between_bool = time_between_rois < 250
            # Find runs of consecutive confocal images below a threshold time between rois (i.e. from same cell)
            true_ranges = np.argwhere(np.diff(time_between_bool,prepend=False,append=False))
            #Conversion into list of 2-tuples
            true_ranges = true_ranges.reshape(len(true_ranges)//2,2)
            true_ranges = [tuple([r[0]-1 if r[0]>0 else r[0],r[1]]) for r in true_ranges]
            cell_range = true_ranges[cell_idx]
            num_rois = cell_range[1]-cell_range[0]
            #if num_rois < roi_plots:
            #    print(f'Confocal {cell_idx}: Not enough ROIs for six sites, only found {num_rois}.')
            #    return            
            #else:
            # create figure and grid layout
            grid_rows = 3
            grid_cols = grid_rows+int(roi_plots/3*3)
            fig = plt.figure(figsize=(roi_plots*3+grid_rows-2.8,roi_plots), dpi=200)
            gs = GridSpec(grid_rows, grid_cols, figure=fig)
            gs.update(left=0.05,right=0.995,top=0.995,bottom=0.05,wspace=0.05,hspace=0.05)
            # load data and plot in small zooms
            first = True
            roi_idx_plotting = 0
            for roi_idx_master in range(cell_range[0],cell_range[1]):
                roi = self.roi_data.iloc[roi_idx_master]
                conf_idx = roi['conf_idx']
                roi_idx = roi['roi_idx']
                if roi_idx_plotting < roi_plots:
                    roi_pos_um = roi['roi_pos']
                    pxsize = roi['pxsize']
                    pxshift = pxsize/2
                    image_conf = roi['conf_img']
                    conf_xlim = roi['conf_xlims']
                    conf_ylim = roi['conf_ylims']
                    conf_offset = roi['conf_offset']
                    x_conf_correction = roi['conf_xcorr']
                    y_conf_correction = roi['conf_ycorr']

                    track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                    track_roi_data = self.track_data.loc[track_data_roi_idxs]
                    if first:
                        # plot large confocal image
                        axlargeconf = fig.add_subplot(gs[:, :grid_rows])
                        imgplot_large = axlargeconf.imshow(image_conf, cmap='hot')
                        extents_largeconfimg = np.array(imgplot_large.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                        imgplot_large.set_extent(extents_largeconfimg)
                        axlargeconf.set_axis_off()
                        first = False
                    # get image zoom from track coordinates
                    axroizoom_onlyconf = fig.add_subplot(gs[roi_idx_plotting//2, grid_rows+roi_idx_plotting%2])
                    axroizoom_onlytracks = fig.add_subplot(gs[roi_idx_plotting//2, grid_rows+roi_idx_plotting%2+2])
                    axroizoom_overlay = fig.add_subplot(gs[roi_idx_plotting//2, grid_rows+roi_idx_plotting%2+4])
                    img_overlay = axroizoom_overlay.imshow(image_conf, cmap='gray', alpha=0.4)
                    img_onlyconf = axroizoom_onlyconf.imshow(image_conf, cmap='hot')
                    extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                    img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                    img_onlyconf.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                    # add circles for caveolae site to zooms
                    cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                    cav_circ_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                    #cav_circ_onlytracks = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                    cav_circ_incl_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                    cav_circ_incl_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                    #cav_circ_incl_onlytracks  = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                    axroizoom_overlay.add_patch(cav_circ_overlay);
                    axroizoom_onlyconf.add_patch(cav_circ_onlyconf);
                    #axroizoom_onlytracks.add_patch(cav_circ_onlytracks);
                    axroizoom_overlay.add_patch(cav_circ_incl_overlay);
                    axroizoom_onlyconf.add_patch(cav_circ_incl_onlyconf);
                    #axroizoom_onlytracks.add_patch(cav_circ_incl_onlytracks);
                    # add square for ROI position in large confocal image
                    roi_square = patches.Rectangle((conf_xlim[0], conf_ylim[0]), conf_xlim[1]-conf_xlim[0], conf_ylim[1]-conf_ylim[0], edgecolor='g', linewidth=1, facecolor='none')
                    axlargeconf.add_patch(roi_square);
                    # set axis limits for all zooms
                    axroizoom_onlytracks.set_xlim(*conf_xlim)
                    axroizoom_onlytracks.set_ylim(*conf_ylim)
                    axroizoom_onlyconf.set_xlim(*conf_xlim)
                    axroizoom_onlyconf.set_ylim(*conf_ylim)
                    axroizoom_overlay.set_xlim(*conf_xlim)
                    axroizoom_overlay.set_ylim(*conf_ylim)
                    for idx,track in track_roi_data.iterrows():
                        if idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                            #if idx<100:
                            if track['filter'] == 0:
                                axroizoom_onlytracks.plot(track['x'], track['y'], linewidth=0.3);
                                axroizoom_overlay.plot(track['x'], track['y'], linewidth=0.3);
                    axroizoom_onlytracks.set_aspect('equal', adjustable='box')
                    axroizoom_onlyconf.set_aspect('equal', adjustable='box')
                    axroizoom_overlay.set_aspect('equal', adjustable='box')
                    axroizoom_onlyconf.set_axis_off()
                    axroizoom_onlytracks.set_axis_off()
                    axroizoom_overlay.set_axis_off()
                    axlargeconf.annotate(f'ROI {roi_idx_plotting}',xy=(conf_xlim[0]-(conf_xlim[1]-conf_xlim[0])*1, conf_ylim[0]), size=10, ha='left', va='bottom', color='green')
                    axroizoom_onlyconf.annotate(f'ROI {roi_idx_plotting}', xy=(0.14, 0.93), xycoords='axes fraction', fontsize=10, ha='center', va='center', color='green')
                    roi_idx_plotting += 1
            savename = f'ROI-zooms-overlays-{self.dataset_tag}-cell{cell_idx}'+savename_suffix
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            plt.show()              
        elif not singleroi:
            num_rois = len(self.roi_data[self.roi_data['conf_idx']==cell_idx])
            #if num_rois < roi_plots:
            #    print(f'Confocal {conf_idx}: Not enough ROIs for six sites, only found {num_rois}.')
            #    return
            #else:
            # create figure and grid layout
            grid_rows = 3
            grid_cols = grid_rows+int(roi_plots/3*3)
            fig = plt.figure(figsize=(roi_plots*3+grid_rows-2.8,roi_plots), dpi=200)
            gs = GridSpec(grid_rows, grid_cols, figure=fig)
            gs.update(left=0.05,right=0.995,top=0.995,bottom=0.05,wspace=0.05,hspace=0.05)
            # load data and plot in small zooms
            first = True
            for roi_idx, roi in self.roi_data.iterrows():
                if roi['conf_idx'] == cell_idx:
                    conf_idx = roi['conf_idx']
                    roi_idx = roi['roi_idx']
                    if roi_idx < roi_plots:
                        roi_pos_um = roi['roi_pos']
                        pxsize = roi['pxsize']
                        pxshift = pxsize/2
                        image_conf = roi['conf_img']
                        conf_xlim = roi['conf_xlims']
                        conf_ylim = roi['conf_ylims']
                        conf_offset = roi['conf_offset']
                        x_conf_correction = roi['conf_xcorr']
                        y_conf_correction = roi['conf_ycorr']

                        track_data_roi_idxs = self.track_data.index[(self.track_data['confidx'] == conf_idx) & (self.track_data['roiidx'] == roi_idx)].tolist()
                        track_roi_data = self.track_data.loc[track_data_roi_idxs]
                        if first:
                            # plot large confocal image
                            axlargeconf = fig.add_subplot(gs[:, :grid_rows])
                            imgplot_large = axlargeconf.imshow(image_conf, cmap='hot')
                            extents_largeconfimg = np.array(imgplot_large.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                            imgplot_large.set_extent(extents_largeconfimg)  # scale overlay image to the correct pixel size for the tracks
                            axlargeconf.set_axis_off()
                            first = False
                        # get image zoom from track coordinates
                        axroizoom_onlyconf = fig.add_subplot(gs[roi_idx//2, grid_rows+roi_idx%2])
                        axroizoom_onlytracks = fig.add_subplot(gs[roi_idx//2, grid_rows+roi_idx%2+2])
                        axroizoom_overlay = fig.add_subplot(gs[roi_idx//2, grid_rows+roi_idx%2+4])
                        img_overlay = axroizoom_overlay.imshow(image_conf, cmap='gray', alpha=0.4)
                        img_onlyconf = axroizoom_onlyconf.imshow(image_conf, cmap='hot')
                        extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                        img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                        img_onlyconf.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                        # add circles for caveolae site to zooms
                        cav_circ_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                        cav_circ_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                        #cav_circ_onlytracks = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
                        cav_circ_incl_overlay = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                        cav_circ_incl_onlyconf = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                        #cav_circ_incl_onlytracks  = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='gray', linewidth=2, facecolor='none')
                        axroizoom_overlay.add_patch(cav_circ_overlay);
                        axroizoom_onlyconf.add_patch(cav_circ_onlyconf);
                        #axroizoom_onlytracks.add_patch(cav_circ_onlytracks);
                        axroizoom_overlay.add_patch(cav_circ_incl_overlay);
                        axroizoom_onlyconf.add_patch(cav_circ_incl_onlyconf);
                        #axroizoom_onlytracks.add_patch(cav_circ_incl_onlytracks);
                        # add square for ROI position in large confocal image
                        roi_square = patches.Rectangle((conf_xlim[0], conf_ylim[0]), conf_xlim[1]-conf_xlim[0], conf_ylim[1]-conf_ylim[0], edgecolor='g', linewidth=1, facecolor='none')
                        axlargeconf.add_patch(roi_square);
                        # set axis limits for all zooms
                        axroizoom_onlytracks.set_xlim(*conf_xlim)
                        axroizoom_onlytracks.set_ylim(*conf_ylim)
                        axroizoom_onlyconf.set_xlim(*conf_xlim)
                        axroizoom_onlyconf.set_ylim(*conf_ylim)
                        axroizoom_overlay.set_xlim(*conf_xlim)
                        axroizoom_overlay.set_ylim(*conf_ylim)
                        for idx,track in track_roi_data.iterrows():
                            if idx%int(1/trackratio)==0:  # only plot every 1/trackratioth track
                                #if idx<100:
                                if track['filter'] == 0:
                                    axroizoom_onlytracks.plot(track['x'], track['y'], linewidth=0.3);
                                    axroizoom_overlay.plot(track['x'], track['y'], linewidth=0.3);
                        axroizoom_onlytracks.set_aspect('equal', adjustable='box')
                        axroizoom_onlyconf.set_aspect('equal', adjustable='box')
                        axroizoom_overlay.set_aspect('equal', adjustable='box')
                        axroizoom_onlyconf.set_axis_off()
                        axroizoom_onlytracks.set_axis_off()
                        axroizoom_overlay.set_axis_off()
                        axlargeconf.annotate(f'ROI {roi_idx}',xy=(conf_xlim[0]-(conf_xlim[1]-conf_xlim[0])*1, conf_ylim[0]), size=10, ha='left', va='bottom', color='green')
                        axroizoom_onlyconf.annotate(f'ROI {roi_idx}', xy=(0.14, 0.93), xycoords='axes fraction', fontsize=10, ha='center', va='center', color='green')
            savename = f'ROI-zooms-overlays-{self.dataset_tag}-cell{cell_idx}'+savename_suffix
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
            plt.show()

    ######################
    def msd_analysis_local_par(self, window_pts=30):
        # seems to take the same amount of time as the non-parallelized version, oddly. not sure why this is, but somehow it does not work on this task.
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool()
        print('Run MSD analysis local, parallellized...', end=' \r')
        row_local_dapps_all = []
        rows = [row for _,row in self.track_data.iterrows()]
        row_local_dapps_all = pool.map(self.msd_par_bulk, rows)
        print(row_local_dapps_all)
        self.track_data['localdiff_dapp'] = row_local_dapps_all

    def msd_par_bulk(self, row, window_pts=125):
        row_local_pos = []
        for i in np.arange(int(window_pts/2), len(row['x'])-int(window_pts/2)-1, 1):
            x_subtrack = row['x'][i-int(window_pts/2):i+int(window_pts/2)]
            y_subtrack = row['y'][i-int(window_pts/2):i+int(window_pts/2)]
            t_subtrack = row['tim'][i-int(window_pts/2):i+int(window_pts/2)]
            try:
                dapp_local = self.diff_analysis_local(x_subtrack, y_subtrack, t_subtrack)
                row_local_pos.append(dapp_local)
            except:
                pass
        return row_local_pos
    #######################

    def create_fig(self, figidx, figsize=(6,10)):
        self.figs[figidx] = plt.figure(figidx)#, figsize=figsize)

    def set_fig_size(self):
        fig = plt.gcf()
        fig.set_size_inches(3*self.subplot_cols,3*self.subplot_rows)

    def plot_traj_pack(self, pack_lim):
        plot_roiidx = -1
        for _, row in self.track_data.iterrows():
            if row['confidx']==0:
                if row['filter'] == 0:
                    curr_roiidx = row['roiidx']
                    if curr_roiidx != plot_roiidx:
                        if plot_roiidx != -1:
                            plt.show()
                        plot_roiidx = curr_roiidx
                        fig = plt.figure(figsize=(5,5))
                        confimgplot = plt.imshow(row['confimg'])
                        confimgplot.set_extent(row['confimg_ext'])
                        plt.xlim(*row['conf_xlim'])
                        plt.ylim(*row['conf_ylim'])
                    plt.plot(row['x'][np.array(row['pc'])>pack_lim], row['y'][np.array(row['pc'])>pack_lim], color='red', linewidth=0.3)
                    plt.plot(row['x'][np.array(row['pc'])<pack_lim], row['y'][np.array(row['pc'])<pack_lim], color='black', linewidth=0.3)
        plt.show()

    def plot_dapp_map(self, binsize=0.02, bincount_thresh=5, d_max=0.7):
        print('Plot Dapp maps...', end=' \r')
        roi_idx_cum = 0
        eps = 1E-3
        ret_means = []
        ret_cnts = []

        confs = self.track_data['confidx'].unique()
        for conf in confs:
            conf_data = self.track_data[self.track_data['confidx']==conf].copy()
            rois = conf_data['roiidx'].unique()
            for roi in rois:
                roi_data = conf_data[self.track_data['roiidx']==roi].copy()
                roi_pos = roi_data.iloc[0]['roipos']
                roi_size = roi_data.iloc[0]['roisize']
                x_conf_correction = roi_data.iloc[0]['conf_xcorr']
                y_conf_correction = roi_data.iloc[0]['conf_ycorr']
                x_lims = roi_data.iloc[0]['conf_xlim']
                y_lims = roi_data.iloc[0]['conf_ylim']
                pxsize = roi_data.iloc[0]['pxsize']
                pxshift = pxsize/2
                pos_roi = []
                dapp_roi = []
                for _, row in roi_data.iterrows():
                    if row['filter'] == 0 and len(row['localdiff_dapp'])>0:
                        dapp_roi.append(np.array(row['localdiff_dapp']))
                        pos_roi.append(np.array(row['localdiff_pos']))
                if len(dapp_roi) > 1:
                    dapp_roi = np.hstack(dapp_roi)
                    x_roi = [pos[0] for pos2 in pos_roi for pos in pos2]
                    y_roi = [pos[1] for pos2 in pos_roi for pos in pos2]

                    #pxsize = roi_data.iloc[0]['pxsize']
                    x_bins = np.arange(roi_pos[0]+pxshift-roi_size[0]/2-eps, roi_pos[0]+pxshift+roi_size[0]/2+eps, binsize)
                    y_bins = np.arange(roi_pos[1]+pxshift-roi_size[1]/2-eps, roi_pos[1]+pxshift+roi_size[1]/2+eps, binsize)
                    #site_pos_px_map = [(np.abs(x_bins - roi_pos[0])).argmin(), (np.abs(y_bins - roi_pos[1])).argmin()]
                    ret_mean = binned_statistic_2d(x_roi, y_roi, dapp_roi, statistic=np.mean, bins=[x_bins, y_bins])
                    ret_cnt = binned_statistic_2d(x_roi, y_roi, dapp_roi, statistic='count', bins=[x_bins, y_bins])
                    ret_means.append(np.where(ret_cnt.statistic.T > bincount_thresh, ret_mean.statistic.T, np.nan))
                    ret_cnts.append(ret_cnt.statistic.T)

                    ax_mean = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 4+roi_idx_cum*self.subplot_cols)
                    ax_cnt = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 5+roi_idx_cum*self.subplot_cols)
                    im_mean = ax_mean.imshow(np.where(ret_cnt.statistic.T > bincount_thresh, ret_mean.statistic.T, np.nan), vmin=0, vmax=d_max, origin='lower')
                    im_cnt = ax_cnt.imshow(ret_cnt.statistic.T, vmin=0, cmap='gray', origin='lower')
                    cav_circ_overlay1 = patches.Circle((roi_pos[0]+x_conf_correction+pxshift, roi_pos[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='g', linewidth=2, facecolor='none')
                    cav_circ_overlay2 = patches.Circle((roi_pos[0]+x_conf_correction+pxshift, roi_pos[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='g', linewidth=2, facecolor='none')
                    ax_mean.add_patch(cav_circ_overlay1);
                    ax_cnt.add_patch(cav_circ_overlay2);
                    im_mean.set_extent((roi_pos[0]-roi_size[0]/2-eps, roi_pos[0]+roi_size[0]/2+eps, roi_pos[1]-roi_size[1]/2-eps, roi_pos[1]+roi_size[1]/2+eps))
                    im_cnt.set_extent((roi_pos[0]-roi_size[0]/2-eps, roi_pos[0]+roi_size[0]/2+eps, roi_pos[1]-roi_size[1]/2-eps, roi_pos[1]+roi_size[1]/2+eps))
                    ax_mean.set_xlim(*x_lims)
                    ax_mean.set_ylim(*y_lims)
                    ax_cnt.set_xlim(*x_lims)
                    ax_cnt.set_ylim(*y_lims)
                    cbar_mean = plt.colorbar(im_mean, ax=ax_mean, ticks=[0, d_max*1/4, d_max*2/4, d_max*3/4, d_max], location='bottom');
                    cbar_mean.ax.set_xlabel('D_trans (um^2/s)')
                    cbar_cnt = plt.colorbar(im_cnt, ax=ax_cnt, location='bottom');
                    cbar_cnt.ax.set_xlabel('Number of localizations')
                    roi_idx_cum += 1
                else:
                    ret_means.append([])
                    ret_cnts.append([])
        self.roi_data['dtrans_map'] = ret_means
        self.roi_data['dtrans_map_cnts'] = ret_cnts

    def dtrans_spot_analysis(self, y_max=1.0):
        print('Running Dtrans spot analysis...', end=' \r')
        dtrans_site_in_all = []
        dtrans_site_out_all = []
        dtrans_site_all = []
        for _, row in self.roi_data.iterrows():
            if len(row['dtrans_map']) > 0:
                dtrans_map = row['dtrans_map']
                roi_small_pos_px = [int(np.ceil(np.shape(dtrans_map)[0]/2)),int(np.ceil(np.shape(dtrans_map)[1]/2))]
                ### binsize 0.05, inner ring 2x2, outer ring circle at 3
                dtrans_site_full = dtrans_map[roi_small_pos_px[0]-1:roi_small_pos_px[0]+3, roi_small_pos_px[1]-1:roi_small_pos_px[1]+3]
                mask_in = np.array([[False, False, False, False],[False, True, True, False],[False, True, True, False],[False, False, False, False]])
                mask_out = np.array([[True, True, True, True],[True, False, False, True],[True, False, False, True],[True, True, True, True]])
                ### binsize 0.05, inner ring 2x2, outer ring circle at 4
                #dtrans_site_full = dtrans_map[roi_small_pos_px[0]-2:roi_small_pos_px[0]+4, roi_small_pos_px[1]-2:roi_small_pos_px[1]+4]
                #mask_in = np.array([[False, False, False, False, False, False],[False, False, False, False, False, False],[False, False, True, True, False, False],[False, False, True, True, False, False],[False, False, False, False, False, False],[False, False, False, False, False, False]])
                #mask_out = np.array([[True, True, True, True, True, True],[True, False, False, False, False, True],[True, False, False, False, False, True],[True, False, False, False, False, True],[True, False, False, False, False, True],[True, True, True, True, True, True]])
                ### binsize 0.033, inner ring 1x1, outer ring circle at 3  ## THIS IS TOO SMALL, only 33 nm. 
                #dtrans_site_full = dtrans_map[roi_small_pos_px[0]-0:roi_small_pos_px[0]+3, roi_small_pos_px[1]-0:roi_small_pos_px[1]+3]
                #mask_in = np.array([[False, False, False],[False, True, False],[False, False, False]])
                #mask_out = np.array([[True, True, True],[True, False, True],[True, True, True]])
                #print(dtrans_site_full)
                dtrans_site_in = dtrans_site_full[mask_in]
                dtrans_site_in_all.append(np.nanmean(dtrans_site_in))
                dtrans_site_out = dtrans_site_full[mask_out]
                dtrans_site_out_all.append(np.nanmean(dtrans_site_out))
                dtrans_site_all.append(np.array([np.nanmean(dtrans_site_in),np.nanmean(dtrans_site_out)]))
            else:
                dtrans_site_in_all.append(np.nan)
                dtrans_site_out_all.append(np.nan)
        self.roi_data['dtrans_site_in'] = dtrans_site_in_all
        self.roi_data['dtrans_site_out'] = dtrans_site_out_all
        # plot paired in-out graph
        dtrans_site_all = np.array(dtrans_site_all)
        plt.figure(figsize=(3,5))
        plt.plot(dtrans_site_all.T, '.-', color='green', alpha=0.2)
        plt.xticks([0,1]);
        plt.gca().set_xticklabels(['In site','Out of site'])
        plt.ylim(0,y_max);
        plt.show()
        plt.close()

    def dtrans_circle_analysis(self, sim=False):
        print('Running Dtrans circle analysis...', end=' \r')
        rad_inner = self.circle_radii[0]
        rad_peri = self.circle_radii[1]
        rad_outer = self.circle_radii[2]

        roi_idx_cum = 0
        
        dtrans_inner_all = []
        dtrans_peri_all = []
        dtrans_outer_all = []
        dtrans_outside_all = []

        confs = self.track_data['confidx'].unique()
        for conf in confs:
            conf_data = self.track_data[self.track_data['confidx']==conf].copy()
            rois = conf_data['roiidx'].unique()
            for roi in rois:
                roi_data = conf_data[self.track_data['roiidx']==roi].copy()
                ax = self.figs[3].add_subplot(self.N_rois, self.subplot_cols2, 1+roi_idx_cum*self.subplot_cols2)
                # show confocal image in background
                if not sim:
                    img_conf = ax.imshow(roi_data['confimg'].iloc[0])
                    img_conf.set_extent(roi_data['confimg_ext'].iloc[0])  # scale overlay confocal image
                # add circles for caveolae site to zooms
                if sim:
                    roi_pos_um = roi_data['roipos'][0]
                else:
                    roi_pos_um = roi_data['roipos'].iloc[0]
                #pxsize = roi_data.iloc[0]['pxsize']
                x_conf_correction = roi_data.iloc[0]['conf_xcorr']
                y_conf_correction = roi_data.iloc[0]['conf_ycorr']
                pxsize = roi_data.iloc[0]['pxsize']
                pxshift = pxsize/2

                cav_circ_inner = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), rad_inner, edgecolor='g', linewidth=2, facecolor='none')
                cav_circ_peri = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), rad_peri, edgecolor='b', linewidth=2, facecolor='none')
                cav_circ_outer = patches.Circle((roi_pos_um[0]+x_conf_correction+pxshift, roi_pos_um[1]+y_conf_correction+pxshift), rad_outer, edgecolor='r', linewidth=2, facecolor='none')
                ax.add_patch(cav_circ_inner);
                ax.add_patch(cav_circ_peri);
                ax.add_patch(cav_circ_outer);
                # circle polygon from site circle in correct coordinates for test
                verts_inner = cav_circ_inner.get_path().vertices
                trans_inner = cav_circ_inner.get_patch_transform()
                cav_circ_inner_scaled_pnts = trans_inner.transform(verts_inner)
                cav_circ_inner_scaled = patches.Polygon(cav_circ_inner_scaled_pnts)
                verts_peri = cav_circ_peri.get_path().vertices
                trans_peri = cav_circ_peri.get_patch_transform()
                cav_circ_peri_scaled_pnts = trans_peri.transform(verts_peri)
                cav_circ_peri_scaled = patches.Polygon(cav_circ_peri_scaled_pnts)
                verts_outer = cav_circ_outer.get_path().vertices
                trans_outer = cav_circ_outer.get_patch_transform()
                cav_circ_outer_scaled_pnts = trans_outer.transform(verts_outer)
                cav_circ_outer_scaled = patches.Polygon(cav_circ_outer_scaled_pnts)
                pos_roi = []
                dapp_roi = []
                for _, row in roi_data.iterrows():
                    if row['filter'] == 0 and len(row['localdiff_dapp'])>0:
                        dapp_roi.append(np.array(row['localdiff_dapp']))
                        pos_roi.append(np.array(row['localdiff_pos']))
                if len(dapp_roi) > 1:
                    dapp_roi = np.hstack(dapp_roi)
                    x_roi = np.array([pos[0] for pos2 in pos_roi for pos in pos2])
                    y_roi = np.array([pos[1] for pos2 in pos_roi for pos in pos2])
                    cont_points_inner = cav_circ_inner_scaled.contains_points(np.array([x_roi,y_roi]).T)
                    cont_points_peri_raw = cav_circ_peri_scaled.contains_points(np.array([x_roi,y_roi]).T)
                    cont_points_peri = (cont_points_peri_raw & ~cont_points_inner)
                    cont_points_outer_raw = cav_circ_outer_scaled.contains_points(np.array([x_roi,y_roi]).T)
                    cont_points_outer = (cont_points_outer_raw & ~cont_points_peri_raw)
                    cont_points_outside = (~cont_points_inner & ~cont_points_peri & ~cont_points_outer)
                    ax.plot(x_roi[cont_points_inner], y_roi[cont_points_inner], color='green', linewidth=0.3);
                    ax.plot(x_roi[cont_points_peri], y_roi[cont_points_peri], color='blue', linewidth=0.3);
                    ax.plot(x_roi[cont_points_outer], y_roi[cont_points_outer], color='red', linewidth=0.3);
                    ax.plot(x_roi[cont_points_outside], y_roi[cont_points_outside], color='gray', linewidth=0.3);
                    ax.set_xlim(roi_data.iloc[0]['conf_xlim'][0]+0.8, roi_data.iloc[0]['conf_xlim'][1]-0.8)
                    ax.set_ylim(roi_data.iloc[0]['conf_ylim'][0]-0.8, roi_data.iloc[0]['conf_ylim'][1]+0.8)
                    dtrans_inner = dapp_roi[cont_points_inner]
                    dtrans_peri = dapp_roi[cont_points_peri]
                    dtrans_outer = dapp_roi[cont_points_outer]
                    dtrans_outside = dapp_roi[cont_points_outside]
                    #print([np.nanmean(dtrans_inner), np.nanmean(dtrans_peri), np.nanmean(dtrans_outer)])
                    #print([np.nanstd(dtrans_inner)/np.sqrt(len(dtrans_inner[~np.isnan(dtrans_inner)])), np.nanstd(dtrans_peri)/np.sqrt(len(dtrans_peri[~np.isnan(dtrans_peri)])), np.nanstd(dtrans_outer)/np.sqrt(len(dtrans_outer[~np.isnan(dtrans_outer)]))])
                    dtrans_inner_all.append(dtrans_inner)
                    dtrans_peri_all.append(dtrans_peri)
                    dtrans_outer_all.append(dtrans_outer)
                    dtrans_outside_all.append(dtrans_outside)
                    roi_idx_cum += 1
                else:
                    dtrans_inner_all.append([])
                    dtrans_peri_all.append([])
                    dtrans_outer_all.append([])
                    dtrans_outside_all.append([])
        self.roi_data['dtrans_site_inner'] = dtrans_inner_all
        self.roi_data['dtrans_site_peri'] = dtrans_peri_all
        self.roi_data['dtrans_site_outer'] = dtrans_outer_all
        self.roi_data['dtrans_site_outside'] = dtrans_outside_all

    def plot_dtrans_circle_analysis(self, y_max=1.0, format='svg'):
        print('Plotting Dtrans circle analysis...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: Dtrans circle analysis')
        means = []
        sems = []
        for _, row in self.roi_data.iterrows():
            if len(row['dtrans_site_inner']) > 0:
                dtrans_inner = row['dtrans_site_inner']
                dtrans_inner_mean = np.nanmean(dtrans_inner)
                dtrans_inner_sem = np.nanstd(dtrans_inner)/np.sqrt(len(dtrans_inner[~np.isnan(dtrans_inner)]))
                dtrans_peri = row['dtrans_site_peri']
                dtrans_peri_mean = np.nanmean(dtrans_peri)
                dtrans_peri_sem = np.nanstd(dtrans_peri)/np.sqrt(len(dtrans_peri[~np.isnan(dtrans_peri)]))
                dtrans_outer = row['dtrans_site_outer']
                dtrans_outer_mean = np.nanmean(dtrans_outer)
                dtrans_outer_sem = np.nanstd(dtrans_outer)/np.sqrt(len(dtrans_outer[~np.isnan(dtrans_outer)]))
                dtrans_outside = row['dtrans_site_outside']
                dtrans_outside_mean = np.nanmean(dtrans_outside)
                dtrans_outside_sem = np.nanstd(dtrans_outside)/np.sqrt(len(dtrans_outside[~np.isnan(dtrans_outside)]))
                means.append([dtrans_inner_mean, dtrans_peri_mean, dtrans_outer_mean, dtrans_outside_mean])
                sems.append([dtrans_inner_sem, dtrans_peri_sem, dtrans_outer_sem, dtrans_outside_sem])
        means = np.array(means)
        sems = np.array(sems)
        plt.figure(figsize=(3,5))
        plt.plot(means.T, '.-', color='green', alpha=0.2)
        plt.xticks([0,1,2,3]);
        plt.gca().set_xticklabels(['Inner', 'Perimeter', 'Outer', 'Outside'])
        plt.ylim(0,y_max);
        plt.show()
        savename = 'dtrans-circleanalysis'
        if format=='svg':
            plt.savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight")
        plt.close()

        result_rois_01 = scipy.stats.ttest_rel(means[:,0],means[:,1],alternative='less')
        result_rois_02 = scipy.stats.ttest_rel(means[:,0],means[:,2],alternative='less')
        result_rois_12 = scipy.stats.ttest_rel(means[:,1],means[:,2],alternative='less')
        result_rois_03 = scipy.stats.ttest_rel(means[:,0],means[:,3],alternative='less')
        result_rois_13 = scipy.stats.ttest_rel(means[:,1],means[:,3],alternative='less')
        result_rois_23 = scipy.stats.ttest_rel(means[:,2],means[:,3],alternative='less')
        self.print_to_file('ROIs paired ttest (In V Peri, In V Outer, Peri V Outer, In V Outside, Peri V Outside, Outer V Outside)')
        self.print_to_file(str(result_rois_01))
        self.print_to_file(str(result_rois_02))
        self.print_to_file(str(result_rois_12))
        self.print_to_file(str(result_rois_03))
        self.print_to_file(str(result_rois_13))
        self.print_to_file(str(result_rois_23))
    
    def plot_dapp_tracks(self, d_max=0.7, colormap='viridis'):
        print('Plotting Dapp, tracks...', end=' \r')
        roi_idx_cum = 0
        confs = self.track_data['confidx'].unique()
        for conf in confs:
            conf_data = self.track_data[self.track_data['confidx']==conf].copy()
            rois = conf_data['roiidx'].unique()
            for roi in rois:
                roi_data = conf_data[self.track_data['roiidx']==roi].copy()
                roi_pos = roi_data.iloc[0]['roipos']
                x_conf_correction = roi_data.iloc[0]['conf_xcorr']
                y_conf_correction = roi_data.iloc[0]['conf_ycorr']
                pxsize = roi_data.iloc[0]['pxsize']
                pxshift = pxsize/2
                ax = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 3+roi_idx_cum*self.subplot_cols)
                anytracks = False
                for _, row in roi_data.iterrows():
                    if row['filter'] == 0 and len(row['localdiff_dapp'])>0:
                        d_cut = np.array(row['localdiff_dapp'].copy())
                        d_cut[d_cut < 0] = 0
                        d_cut[d_cut > d_max] = d_max
                        ax.plot([item[0] for item in row['localdiff_pos']], [item[1] for item in row['localdiff_pos']], '-', c='k', linewidth=0.4, alpha=0.2)
                        sc = ax.scatter([item[0] for item in row['localdiff_pos']], [item[1] for item in row['localdiff_pos']], c=d_cut, cmap=colormap, vmin=0, vmax=d_max, s=2, alpha=0.4)
                        anytracks = True
                if anytracks:
                    cav_circ_overlay = patches.Circle((roi_pos[0]+x_conf_correction+pxshift, roi_pos[1]+y_conf_correction+pxshift), self.inclusion_rad, edgecolor='g', linewidth=2, facecolor='none')
                    ax.add_patch(cav_circ_overlay);
                    ax.invert_yaxis()
                    ax.set_aspect('equal', adjustable='box')
                    cbar = plt.colorbar(sc, ax=ax, ticks=[0, d_max*1/4, d_max/2, d_max*3/4, d_max], location='bottom');
                    cbar.ax.set_xlabel('D_trans (um^2/s)')
                roi_idx_cum += 1

    def simulate_data(self, N_rois, N_paths, N_locs, N_locs_sg, D, dt_set, dt_random, w_n, imsize, imlimit, plotting=False, bidiffusional=False, D2=None, cav_site_rad=None, cut_outside_patt=True, save_path=None):
        print('Simulate data...', end=' \r')
        self.simulated_data = True
        self.paths.append(save_path)
        self.N_rois = N_rois
        conf_image_size = 800
        image_conf = np.zeros([conf_image_size, conf_image_size])
        pxsize = 0.1
        pxshift = pxsize/2
        confsize = conf_image_size*pxsize
        for roi_idx in range(self.N_rois):
            roi_size = (imsize*2, imsize*2)
            roi_pos = (-pxshift, -pxshift)
            traces = self._generate_traces(N_paths, D, dt_set, dt_random, N_locs, N_locs_sg, w_n, imsize, imlimit, bidiffusional, D2, cav_site_rad, cut_outside_patt)
            curr_img_data = {'confidx': [], 'roiidx': [], 'tridx': [], 'roisize': [], 'roipos': [], 'roipos_px': [], 'tim0': [], 'x': [], 'y': [], 'tim': [], 'pxsize': [], 'inout_flag': [], 'inout_len': [], 'inout_mask': [], 'confimg':[], 'confimg_ext': [], 'conf_xlim': [], 'conf_ylim': [], 'inout_incl_flag': [], 'inout_incl_len': [], 'inout_incl_mask': [], 'filter': [], 'conf_xcorr': [], 'conf_ycorr': []}
            ax_overlay = self.figs[0].add_subplot(self.N_rois, self.subplot_cols, 2+roi_idx*self.subplot_cols)
            # get image zoom from track coordinates
            img_overlay = ax_overlay.imshow(image_conf)
            img_overlay.set_extent(np.array(img_overlay.get_extent())*pxsize-confsize/2+[pxshift,pxshift,pxshift,pxshift])  # scale overlay image to the correct pixel size for the tracks
            # add circles for caveolae site to zooms
            cav_circ_overlay = patches.Circle((roi_pos[0]+pxshift, roi_pos[1]+pxshift), self.ana_site_rad, edgecolor='g', linewidth=2, facecolor='none')
            ax_overlay.add_patch(cav_circ_overlay);
            # circle polygon from overlay in correct coordinates for test
            verts = cav_circ_overlay.get_path().vertices
            trans = cav_circ_overlay.get_patch_transform()
            cav_circ_overlay_scaled_pnts = trans.transform(verts)
            cav_circ_overlay_scaled = patches.Polygon(cav_circ_overlay_scaled_pnts)
            xmin = 100
            xmax = 0
            ymin = 100
            ymax = 0
            for track_id, trace in enumerate(traces):
                x_track = np.array(trace[:,1]).flatten()
                y_track = np.array(trace[:,2]).flatten()
                tim_track = np.array(trace[:,0]).flatten()
                if np.max(x_track) > xmax:
                    xmax = np.max(x_track)
                if np.min(x_track) < xmin:
                    xmin = np.min(x_track)
                if np.max(y_track) > ymax:
                    ymax = np.max(y_track)
                if np.min(y_track) < ymin:
                    ymin = np.min(y_track)
                curr_img_data['tim0'].append(tim_track[0])
                tim_track = tim_track - tim_track[0]
                curr_img_data['confidx'].append(0)
                curr_img_data['roiidx'].append(roi_idx)
                curr_img_data['tridx'].append(track_id)
                curr_img_data['roisize'].append(roi_size)
                curr_img_data['roipos'].append(roi_pos)
                curr_img_data['roipos_px'].append(np.nan)
                curr_img_data['x'].append(x_track)
                curr_img_data['y'].append(y_track)
                curr_img_data['tim'].append(tim_track)
                curr_img_data['pxsize'].append(pxsize)
                curr_img_data['confimg'].append(image_conf)
                curr_img_data['confimg_ext'].append(np.array(img_overlay.get_extent())*pxsize-confsize/2+[pxshift,pxshift,pxshift,pxshift])
                curr_img_data['conf_xlim'].append([xmin,xmax])
                curr_img_data['conf_ylim'].append([ymax,ymin])
                curr_img_data['inout_incl_flag'].append(np.nan)
                curr_img_data['inout_incl_len'].append(np.nan)
                curr_img_data['inout_incl_mask'].append(np.nan)
                curr_img_data['conf_xcorr'].append(0)
                curr_img_data['conf_ycorr'].append(0)
                
                # filter tracks
                dists = [self._distance((x0,y0),(x1,y1)) for x1,x0,y1,y0 in zip(x_track[1:],x_track[:-1],y_track[1:],y_track[:-1])]
                slidestd_x_pos = [np.std(x_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                slidestd_y_pos = [np.std(y_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                # filter indexing: 0 = pass all, 1 = short, 2 = blob_meandistinterval, 3 = blob_meanpossliding, 4 = site moved (all tracks filtered)
                if (tim_track[-1] - tim_track[0] < self.min_time):
                    filter_id = 1
                elif (np.mean(np.abs(dists[::self.interval_dist])) < self.blob_dist):
                    filter_id = 2
                elif (np.mean(slidestd_x_pos) < self.meanslidestd_thresh and np.mean(slidestd_y_pos) < self.meanslidestd_thresh):
                    filter_id = 3
                else:
                    filter_id = 0
                curr_img_data['filter'].append(filter_id)
                if filter_id == 0:
                    cont_points = cav_circ_overlay_scaled.contains_points(np.array([x_track,y_track]).T)
                    if any(cont_points):
                        curr_img_data['inout_flag'].append(True)
                        curr_img_data['inout_len'].append(np.sum(cont_points))
                        ax_overlay.plot(x_track[cont_points], y_track[cont_points], color='green', linewidth=0.3);
                        ax_overlay.plot(x_track[~cont_points], y_track[~cont_points], color='red', linewidth=0.3);
                    else:
                        curr_img_data['inout_flag'].append(False)
                        curr_img_data['inout_len'].append(0)
                        ax_overlay.plot(x_track, y_track, color='blue', linewidth=0.3);
                    curr_img_data['inout_mask'].append(cont_points)
                else:
                    if filter_id == 1:
                        color='gray'
                    elif filter_id == 2:
                        color='yellow'
                    elif filter_id == 3:
                        color='white'
                    ax_overlay.plot(x_track, y_track, color=color, linewidth=0.3);
                    curr_img_data['inout_flag'].append(np.nan)
                    curr_img_data['inout_len'].append(np.nan)
                    curr_img_data['inout_mask'].append(np.nan)

            self.track_data = pd.concat([self.track_data, pd.DataFrame(curr_img_data)])
            self.track_data.reset_index(drop=True, inplace=True)
            ax_overlay.set_xlim(xmin,xmax)
            ax_overlay.set_ylim(ymax,ymin)
            #plt.colorbar(None, ax=ax_overlay, location='bottom');
        print('Simulated data added...')
        self.subplot_rows = self.N_rois

    def get_track_lens(self):
        return [len(track[1].x) for track in self.track_data.iterrows()]
    
    def save_pickleddata(self, format='xz'):
        savename = 'trackdata'
        self.track_data.to_pickle(os.path.join(self.paths[0], savename+'.'+format))
        savename = 'roidata'
        self.roi_data.to_pickle(os.path.join(self.paths[0], savename+'.'+format))
        savename = 'analysisparams'
        analysis_params = {'site_rad': self.ana_site_rad, 'inclusion_rad': self.inclusion_rad, 'circle_radii0': self.circle_radii[0], 'circle_radii1': self.circle_radii[1], 'circle_radii2': self.circle_radii[2], 
                           'blob_dist': self.blob_dist, 'min_time': self.min_time, 'split_len_thresh': self.split_len_thresh, 'fit_len_thresh': self.fit_len_thresh,
                           'max_time_lag': self.max_time_lag, 'max_dt': self.max_dt, 'meanpos_thresh': self.meanpos_thresh, 'interval_meanpos': self.interval_meanpos,
                           'meanslidestd_thresh': self.meanslidestd_thresh, 'slidstd_interval': self.slidstd_interval, 'interval_dist': self.interval_dist,
                           'n_rois': self.N_rois, 'subplot_cols': self.subplot_cols, 'subplot_rows': self.subplot_rows}
        with open(os.path.join(self.paths[0], savename+'.csv'), 'w') as f:
            w = csv.DictWriter(f, analysis_params.keys())
            w.writeheader()
            w.writerow(analysis_params)

    def load_pickleddata(self, folder, format='xz'):
        self.circle_radii = [np.nan, np.nan, np.nan]
        self.paths.append(folder)
        loadname = 'trackdata'
        self.track_data = pd.read_pickle(os.path.join(folder, loadname+'.'+format))
        loadname = 'roidata'
        self.roi_data = pd.read_pickle(os.path.join(folder, loadname+'.'+format))
        loadname = 'analysisparams'
        with open(os.path.join(folder, loadname+'.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                analysis_params = row
        for k,v in analysis_params.items():
            analysis_params[k] = float(v)
        for k,v in analysis_params.items():
            if k == 'site_rad':
                self.ana_site_rad = v
            elif k == 'inclusion_rad':
                self.inclusion_rad = v
            elif k == 'circle_radii0':
                self.circle_radii[0] = v
            elif k == 'circle_radii1':
                self.circle_radii[1] = v
            elif k == 'circle_radii2':
                self.circle_radii[2] = v
            elif k == 'blob_dist':
                self.blob_dist = v
            elif k == 'min_time':
                self.min_time = v
            elif k == 'split_len_thresh':
                self.split_len_thresh = v
            elif k == 'fit_len_thresh':
                self.fit_len_thresh = v
            elif k == 'max_time_lag':
                self.max_time_lag = v
            elif k == 'max_dt':
                self.max_dt = v
            elif k == 'meanpos_thresh':
                self.meanpos_thresh = v
            elif k == 'interval_meanpos':
                self.interval_meanpos = v
            elif k == 'meanslidestd_thresh':
                self.meanslidestd_thresh = v
            elif k == 'slidstd_interval':
                self.slidstd_interval = v
            elif k == 'interval_dist':
                self.interval_dist = v
            elif k == 'n_rois':
                self.N_rois = int(v)
            elif k == 'subplot_cols':
                self.subplot_cols = int(v)
            elif k == 'subplot_rows':
                self.subplot_rows = int(v)


    # UTIL FUNCTIONS

    def show_figs(self, figs=[], figsizes=None):
        if figsizes is None:
            figsizes = [(4*self.subplot_cols,3*self.subplot_rows) for fig in figs]
        for fig_idx, figsize in zip(figs, figsizes):
            if self.figs[fig_idx] is not None:
                self.figs[fig_idx].set_size_inches(*figsize, forward=True)
                self.figs[fig_idx].show()

    def save_figs(self, figs=[], format='png', name_prefix=None):
        for fig_idx in figs:
            if self.figs[fig_idx] is not None:
                if fig_idx == 0:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    savename = f'analysisResults-dtransanalysis-{folder_suffix}'
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'analysisResults-dtransanalysis-{folder_suffix}'
                elif fig_idx == 11:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'filteringResults-{folder_suffix}'
                elif fig_idx == 12:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'fitSitePositionResults-{folder_suffix}'
                else:
                    if name_prefix:
                        folder_suffix = self.paths[0].split('\\')[-1]
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'analysisResults-fig{fig_idx}'
                if format == 'pdf':
                    self.figs[fig_idx].savefig(os.path.join(self.paths[0],savename+'.pdf'), format="pdf", bbox_inches="tight")
                elif format == 'png':
                    self.figs[fig_idx].savefig(os.path.join(self.paths[0],savename+'.png'), format="png", bbox_inches="tight", dpi=200)
                elif format == 'svg':
                    self.figs[fig_idx].savefig(os.path.join(self.paths[0],savename+'.svg'), format="svg", bbox_inches="tight", dpi=200)
                plt.close(self.figs[fig_idx])
    
    def _generate_traces(self, N_paths, D, dt_set, dt_random, N_mu, N_sg, w_n, imsize, imlimit, bidiffusional, D2, cav_site_rad, cut_outside_patt):
        simlimit = imlimit/2
        all_traces = []
        while len(all_traces) < N_paths:
            trace = self._trace_gen(imsize, D, dt_set, dt_random, simlimit, N_mu, N_sg, w_n, bidiffusional, D2, cav_site_rad, cut_outside_patt)
            if trace is not None:
                all_traces.append(trace)
        return all_traces

    def _trace_gen(self, size, D, dt_set, dt_random, border_lim, N_mu, N_sg, w_n, bidiffusional, D2, cav_site_rad, cut_outside_patt):             
        """
        Generates a trajectory of 2D coordinates undergoing free diffusion.
        Input values are:
        the size of the frame (in um),
        the diffusion coefficient D,
        and the width of the time steps dt.
        The trace array contains the 1D time array and the 2D array of the 2 coordinates in time.
        """
        # from SM 0925
        dts = np.array([8.572500e-05, 1.523000e-04, 2.188750e-04, 2.853750e-04,
        3.519500e-04, 4.185250e-04, 4.851000e-04, 5.516000e-04,
        6.181750e-04, 6.847500e-04, 7.513250e-04, 8.178250e-04,
        8.844250e-04, 9.509750e-04, 1.017500e-03, 1.084075e-03,
        1.150625e-03, 1.217225e-03, 1.283750e-03, 1.350975e-03,
        1.417500e-03, 1.483475e-03, 1.550000e-03, 1.616525e-03,
        1.683725e-03, 1.749675e-03, 1.816850e-03, 1.883350e-03,
        1.949950e-03, 2.015825e-03, 2.082425e-03, 2.148925e-03,
        2.216175e-03, 2.282750e-03, 2.349250e-03, 2.415825e-03,
        2.481725e-03, 2.548975e-03, 2.615475e-03, 2.681375e-03,
        2.748625e-03, 2.815150e-03, 2.881700e-03, 2.947600e-03])
        ws = np.array([1.64228572e-01, 4.10178597e-01, 1.64720496e-01, 7.97661411e-02,
        4.53172473e-02, 2.88448437e-02, 1.92753133e-02, 1.42145040e-02,
        1.06029264e-02, 8.28486340e-03, 6.68345345e-03, 5.47487556e-03,
        4.60781382e-03, 4.00440963e-03, 3.22936260e-03, 2.91792818e-03,
        2.42069482e-03, 2.28798129e-03, 2.07563963e-03, 1.82259917e-03,
        1.58371481e-03, 1.48462204e-03, 1.32182678e-03, 1.26697185e-03,
        1.20326935e-03, 1.02808749e-03, 9.44920345e-04, 9.20147153e-04,
        9.07760556e-04, 7.92742162e-04, 7.48504318e-04, 7.57351887e-04,
        6.26407869e-04, 6.81262796e-04, 5.76861484e-04, 5.73322457e-04,
        5.22006558e-04, 5.36162668e-04, 5.45010237e-04, 4.22913788e-04,
        3.99910109e-04, 3.94601567e-04, 4.17605246e-04, 3.85753999e-04])
        
        N = np.random.exponential(250)  # roughly modelled after track length distributions from SM 0925
        roi_center = [0,0]
        x0 = random.uniform(-size,size)  # the initial coordinates are random, but within the center of the image
        y0 = random.uniform(-size,size)
        initial_coord = ([x0,y0])  
        array_coord = np.array([x0,y0])         # initialize an array for the coordinates
        array_t = [0,0]
        pattern_size = 50e-3
        
        i=0
        while i<N-1:                        # every step is created from the previous raw position, and added to the array of the coordinates:
            if dt_random:
                dt = np.random.choice(dts,p=ws)
            else:
                dt = dt_set
            if bidiffusional and self._distance(initial_coord, roi_center) < cav_site_rad:
                w = np.sqrt((2*D2*dt))           # diff in - in free diffusion every step is extracted from a normal distribution
            else:
                w = np.sqrt((2*D*dt))           # diff out - in free diffusion every step is extracted from a normal distribution
            step_raw = np.random.normal(0,w,2)    # extracts the step from the normal distribution
            coord_raw_i = initial_coord + step_raw
            if cut_outside_patt:
                if self._distance(initial_coord, coord_raw_i) > pattern_size*1.5:
                    break       # break if distance makes the localization being outside a 50 nm radius pattern (outside of MINFLUX TCP)
            meas_noise = np.random.normal(0,w_n,2)  # measurement noise to add to each step, from imprecision of measurement
            coord_i = initial_coord + step_raw + meas_noise
            if coord_i[0] > size+border_lim or coord_i[1] > size+border_lim or coord_i[0] < -size-border_lim or coord_i[1] < -size-border_lim:
                break
            initial_coord = coord_raw_i           # the obtained raw position (no noise added) will be the initial position for the next iteration
            array_coord = np.vstack((array_coord,coord_i))
            array_t.append(array_t[-1]+dt)
            i += 1
        if len(array_coord) > 30:
            array_coord = np.vstack((array_coord,array_coord[-1]))
            time = np.array(array_t)
            trace = np.c_[time, array_coord]          # creates the trace array containing the time and the coordinates
            return trace
        else:
            return None

    def _get_angle(self, p_or, p_1, p_2):
        aabs = np.sqrt((p_1[0]-p_or[0])**2+(p_1[1]-p_or[1])**2)
        babs = np.sqrt((p_2[0]-p_or[0])**2+(p_2[1]-p_or[1])**2)
        dotp = np.dot([p_1[0]-p_or[0],p_1[1]-p_or[1]],[p_2[0]-p_or[0],p_2[1]-p_or[1]])
        angle = np.arccos(dotp/(aabs*babs))*180/np.pi
        return angle

    def _distance(self, p_1, p_2):
        return np.sqrt((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)

    def _consecutive_bool(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def _reject_outliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def _f_lin(self, x, a, b):
        return a*x + b
    
    def _f_msd(self, dt, D, sigma):
        dim = 2  # dimensions of diffusion freedom
        Rblur = 1/6.2  # blurring factor (MINFLUX)
        dtmean = 85.7e-6  # smallest (and median) dt between localizations
        return 2*dim*D*dt + 2*dim*sigma**2 - 4*dim*Rblur*D*dtmean
    
    def _f_cub(self, x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    
    def _sigmoid(self, x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y)
    
    def _quadratic(self, x, a, b, c):
        y = a*x**2 + b*x + c
        return (y)

    def _sinusoidal(self, x, a, x0, f, b, c):
        y = a*np.sin(2*np.pi*f*x+x0) + b - x/c
        return (y)
    
    def _linear(self, x, a, b):
        y = a*x + b
        return (y)

    def _gaussian2D(self, xy, amplitude, xo, yo, sigma, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
        b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
        c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))
        return g.ravel()

