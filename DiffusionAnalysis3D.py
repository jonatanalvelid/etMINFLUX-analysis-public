import os
import csv
import warnings
import random
import tifffile
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
from datetime import datetime

import obf_support

# warning suppression
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore')


class DiffusionAnalysis3D(object):
    """ 3D lipid diffusion analysis object. Contains functions for loading data, filtering, analysing, and plotting data from etMINFLUX 3D tracking experiments."""
    def __init__(self, tag, *args, **kwargs):
        # data objects
        self.track_data = pd.DataFrame()
        self.roi_data = pd.DataFrame()

        # data loading params
        self.dataset_tag = tag
        self.paths = []
        self.analyse_folders = True
        self.analyse_files = False
        self.filename_prefix_end = 10
        self._data_params = ['x','y','z','tim','cfr','efo','fbg']
        self.simulated_data = False

        # filtering params
        self.filtered = False
        self.filtered_params = []
        self.filtered_ranges = []
        
        # analysis params
        self.analysis_run = []
        self.loc_it = 4

        self.figs = [None] * 110
        self.subplot_cols = 5
        self.subplot_cols2 = 5
        
        self.nl = '\n'

        plt.close('all')

    def set_analysis_parameters(self, site_rad=None, inclusion_rad=None, circle_radii=None, blob_dist=None,
                                min_time=None, split_len_thresh=None, max_time_lag=None, max_dt=None,
                                fit_len_thresh=None, meanslidestd_thresh=None, slidstd_interval=None,
                                interval_dist=None, membrane_zpos_dist=None):
        # Default values to start from:
        # site_rad = 0.15 (um), blob_dist = 0.01,
        # min_time = 10e-3, split_len_thresh = 5, max_time_lag = 0.005 (s),
        # meanpos_thresh = 0.04, interval_meanpos = 5, interval_dist = 50,
        # meanslidestd_thresh = 0.02, slidstd_interval = 30,
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
        if interval_dist is not None:
            self.interval_dist = interval_dist
        if membrane_zpos_dist is not None:
            self.membrane_zpos_dist = membrane_zpos_dist

        # initialize default and folder-common confocal scan params, to be used for the confocal shift compensation fit reading
        self.confocal_dwell = 10  # pixel dwell time in µs
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
        self.text_file = open(os.path.join(self.top_path,savename+'.txt'), "w")

    def print_to_file(self, str):
        self.text_file.write(str+'\n')

    def print_to_file_close(self):
        self.text_file.close()

    def set_confocal_params(self, conf_scan_params = [2, False]):
        """ Set confocal scan parameters, from supplied list of [dwelltime, bidirectional]"""
        self.confocal_dwell = conf_scan_params[0]  # dwell time in µs
        self.confocal_bidrectional = conf_scan_params[1]  # bidirectionality of scan, boolean

    def add_data(self, top_path, plotting=False):
        """ Add data from a folder to the analysis. """
        self.top_path = top_path
        eventfolders_all = []
        samplefolders = os.listdir(top_path)
        for samplefolder in samplefolders:
            if 'confocal' not in samplefolder:
                if os.path.isdir(os.path.join(top_path, samplefolder)):
                    eventfolders = os.listdir(os.path.join(top_path, samplefolder))
                    for eventfolder in eventfolders:
                        if os.path.isdir(os.path.join(top_path, samplefolder, eventfolder)):
                            if 'nomfx' not in eventfolder:
                                if 'manual' not in eventfolder:
                                    if any(filename.endswith('.npy') for filename in os.listdir(os.path.join(top_path, samplefolder, eventfolder))):
                                        eventfolders_all.append(os.path.join(top_path, samplefolder, eventfolder))
        print(eventfolders_all)
        
        dates = []
        sampleidxs = []
        eventidxs = []
        cycleidxs = []
        cycletstarts = []
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
        
        # loop over each eventfolder (which can contain one or multiple ROIs)
        plot_idx_cum = 0
        tot_roicycles = 0
        for folder in eventfolders_all:
            filelist = os.listdir(folder)
            filelist_npy_all = [file for file in filelist if file.endswith('.npy') and 'test' not in file]
            tot_roicycles += len(filelist_npy_all)
        for folder in eventfolders_all:
            self.paths.append(folder)
            filelist = os.listdir(folder)
            filelist_npy_all = [file for file in filelist if file.endswith('.npy') and 'test' not in file]
            filelist_msr = [file for file in filelist if file.endswith('.msr')]
            filelist_logs = [file for file in filelist if file.endswith('.txt')]
            filelist_conf = [file for file in filelist if 'conf' in file and 'analysis' not in file and '.png' not in file and 'stack' not in file]
            # get all unique ROI names
            roinames_files = np.unique([file.split('_')[1].split('-')[0] for file in filelist_npy_all])
            self.N_rois = len(roinames_files)
            # get date, sample, and event from folder name
            date = top_path.split('\\')[-1]
            sample = folder.split('\\')[-2].split('sample')[-1]
            event = folder.split('\\')[-1].split('e')[-1]
            # get all confocal image times
            conf_times = []
            for conffile in filelist_conf:
                conf_times.append(int(conffile.split('-')[1].split('_')[0]))
            conf_times = np.array(conf_times)
            for roiname in roinames_files:
                print(f'{roiname}')
                filelist_npy = [file for file in filelist_npy_all if roiname+'-' in file]
                self.N_t = len(filelist_npy)
                for cycle in range(len(filelist_npy)):
                    curr_img_data = {'date': [], 'sample': [], 'event': [], 'roiname': [], 'cycle': [], 'cycle_t_start_s': [], 'tridx': [], 'roisize': [], 'roipos': [], 'roipos_px': [], 'tim0': [], 'x': [], 'y': [], 'z': [], 'tim': [], 'pxsize': [], 'inout_flag': [], 'inout_len': [], 'inout_mask': [], 'inout_incl_flag': [], 'inout_incl_len': [], 'inout_incl_mask': [], 'filter': [], 'confimg':[], 'confimg_ext': [], 'conf_xlim': [], 'conf_ylim': [], 'conf_xcorr': [], 'conf_ycorr': []}

                    print(f'Cycle {cycle+1}/{len(filelist_npy)}')
                    file_npy = os.path.join(folder, filelist_npy[cycle])
                    file_msr = os.path.join(folder, filelist_msr[0])
                    
                    # get correct confocal image for current cycle, and cycle time
                    cycle_time = filelist_npy[cycle].split('-')[1].split('_')[0]
                    if cycle == 0:
                        base_cycle_time_hms = datetime.strptime(cycle_time, '%H%M%S')
                    cycle_time_hms = datetime.strptime(cycle_time, '%H%M%S')
                    cycle_time_since_start_s = int((cycle_time_hms - base_cycle_time_hms).total_seconds())
                    conf_cycle_idx = np.nanargmin(np.where(conf_times-int(cycle_time)<0,np.nan,conf_times-int(cycle_time)))
                    file_conf = os.path.join(folder, filelist_conf[conf_cycle_idx])
                    image_conf = tifffile.imread(file_conf)[-1]
                    
                    print(file_npy)
                    print(file_conf)
                    
                    # get metadata from confocal image in msr file (pixel size, image shape, image size, origin offset)
                    msr_dataset = obf_support.File(file_msr)
                    conf_msr_stack_index = 0  # in currently used imspector template file, the confocal dataset is always stack 0 in the .msr file. This might change with other templates used.
                    conf_stack = msr_dataset.stacks[conf_msr_stack_index]
                    pxsize = conf_stack.pixel_sizes[0]*1e6
                    pxshift = pxsize/2
                    conf_size_px = (conf_stack.shape[0], conf_stack.shape[1])
                    conf_size = (conf_stack.lengths[0]*1e6, conf_stack.lengths[1]*1e6)
                    conf_offset = (conf_stack.offsets[0]*1e6, conf_stack.offsets[1]*1e6)
                    
                    roi_name = int(file_npy.split('ROI')[1].split('-')[0])
                    roi_pos = (int(file_npy.split('[')[1].split(',')[0]),int(file_npy.split(']')[0].split(',')[1]))
                    roi_pos_um = (roi_pos[0]*pxsize+conf_offset[0], roi_pos[1]*pxsize+conf_offset[1])
                    roi_size_um = (float(file_npy.split('[')[2].split(',')[0]),float(file_npy.split(']')[1].split(',')[1]))
                    
                    dataset = np.load(os.path.join(folder, file_npy))
                    x = np.zeros((len(dataset),1))
                    y = np.zeros((len(dataset),1))
                    z = np.zeros((len(dataset),1))
                    tid = np.zeros((len(dataset),1))
                    tim = np.zeros((len(dataset),1))
                    for i in range(len(dataset)):
                        x[i] = dataset[i][0][self.loc_it][2][0]
                        y[i] = dataset[i][0][self.loc_it][2][1]
                        z[i] = dataset[i][0][self.loc_it][2][2]
                        tid[i] = dataset[i][4]
                        tim[i] = dataset[i][3]
                    x_raw = x * 1e6
                    y_raw = y * 1e6
                    z_raw = z * 1e6
                    z_raw = z_raw * 0.7  # z scaling for immersion mismatch
                    tid = tid.flatten()
                    tim = tim.flatten()
                    track_ids = list(map(int, set(tid)))
                    track_ids.sort()

                    # save roi-and-cycle-specific parameters
                    dates.append(date)
                    sampleidxs.append(sample)
                    eventidxs.append(event) 
                    roinames.append(roi_name)
                    cycleidxs.append(cycle)
                    cycletstarts.append(cycle_time_since_start_s)
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
                    ax_onlyconf = self.figs[0].add_subplot(tot_roicycles, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
                    ax_overlay = self.figs[0].add_subplot(tot_roicycles, self.subplot_cols, 2+plot_idx_cum*self.subplot_cols)
                    img_overlay = ax_overlay.imshow(image_conf, cmap='hot')
                    extents_confimg = np.array(img_overlay.get_extent())*pxsize+[pxshift, pxshift, pxshift, pxshift]+[conf_offset[0], conf_offset[0], conf_offset[1], conf_offset[1]]+[x_conf_correction, x_conf_correction, y_conf_correction, y_conf_correction]
                    img_overlay.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                    img_onlyconf = ax_onlyconf.imshow(image_conf, cmap='hot')
                    img_onlyconf.set_extent(extents_confimg)  # scale overlay image to the correct pixel size for the tracks
                    
                    # add circles for gag site to zooms
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
                        z_track = np.array([val for val,tr in zip(z_raw,tid) if tr==track]).flatten()
                        tim_track = np.array([val for val,tr in zip(tim,tid) if tr==track]).flatten()
                        curr_img_data['tim0'].append(tim_track[0])
                        curr_img_data['date'].append(date)
                        curr_img_data['sample'].append(sample)
                        curr_img_data['event'].append(event)
                        curr_img_data['roiname'].append(roi_name)
                        curr_img_data['cycle'].append(cycle)
                        curr_img_data['cycle_t_start_s'].append(cycle_time_since_start_s)
                        curr_img_data['tridx'].append(track)
                        curr_img_data['roisize'].append(roi_size_um)
                        curr_img_data['roipos'].append(roi_pos_um)
                        curr_img_data['roipos_px'].append(roi_pos)
                        curr_img_data['x'].append(x_track)
                        curr_img_data['y'].append(y_track)
                        curr_img_data['z'].append(z_track)
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
                        ax_overlay.annotate(f'ROI {roi_name}, cycle {cycle} {self.nl}event {date} - {sample} - {event}',
                        xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
                        ax_onlyconf.annotate(f'ROI {roi_name}, cycle {cycle} {self.nl}event {date} - {sample} - {event}',
                        xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
                             
                    plot_idx_cum += 1
                    self.track_data = pd.concat([self.track_data, pd.DataFrame(curr_img_data)])
                    self.track_data.reset_index(drop=True, inplace=True)
        self.roi_data['date'] = dates
        self.roi_data['sample'] = sampleidxs
        self.roi_data['event'] = eventidxs
        self.roi_data['roiname'] = roinames
        self.roi_data['cycle'] = cycleidxs
        self.roi_data['cycle_t_start_s'] = cycletstarts
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
        self.subplot_rows = plot_idx_cum+1

    def get_membrane_pos(self, membrane='bottom'):
        """ Get the position of the membrane (lowest peak in z-position histogram) for each roi, to be used for filtering. """
        self.main_membrane_pos = []
        dates = self.track_data['date'].unique()
        for date in dates:
            data_d = self.track_data[self.track_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_ds['roiname']==roiname].copy()
                        z_tracks = data_dser['z']
                        all_z_pos = np.concatenate(z_tracks.to_numpy())
                        hist = np.histogram(all_z_pos,bins=30)
                        try:
                            peaks = scipy.signal.find_peaks(hist[0], prominence=np.max(hist[0])/30)
                            peaks_pos = [hist[1][peak] for peak in peaks[0]]
                            if membrane=='bottom':
                                self.main_membrane_pos.append([[date, sample, event, roiname], np.min(peaks_pos)])
                            elif membrane=='top':
                                self.main_membrane_pos.append([[date, sample, event, roiname], np.max(peaks_pos)])
                        except:
                            self.main_membrane_pos.append([[date, sample, event, roiname], 0])

    def plot_filtering(self):
        print('Plotting filtering...', end=' \r')
        plot_idx_cum = 0
        for date, sample, event, roi, cycle in zip(self.roi_data['date'], self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            ax_filter1 = self.figs[11].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            ax_filter0 = self.figs[11].add_subplot(self.subplot_rows, self.subplot_cols, 2+plot_idx_cum*self.subplot_cols)
            for _,track in roicycle_data.iterrows():
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
                    # 4 = mean_z_pos
                    ax_filter0.plot(x_tr, y_tr, color='magenta', linewidth=0.3);
                elif track['filter'] == 0:
                    # 0 = pass all
                    ax_filter1.plot(x_tr, y_tr, color='green', linewidth=0.3);
            ax_filter0.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1
        self.set_fig_size()
        plot_idx_cum = 0
        for sample, event, roi, cycle in zip(self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            ax_filter1 = self.figs[13].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            ax_filter0 = self.figs[13].add_subplot(self.subplot_rows, self.subplot_cols, 2+plot_idx_cum*self.subplot_cols)
            for _,track in roicycle_data.iterrows():
                x_tr = track['x']
                z_tr = track['z']
                if track['filter'] == 1:
                    # 1 = short
                    ax_filter0.plot(x_tr, z_tr, color='blue', linewidth=0.3);
                elif track['filter'] == 2:
                    # 2 = blob_meandistinterval
                    ax_filter0.plot(x_tr, z_tr, color='gray', linewidth=0.3);
                elif track['filter'] == 3:
                    # 3 = blob_meanpossliding
                    ax_filter0.plot(x_tr, z_tr, color='red', linewidth=0.3);
                elif track['filter'] == 4:
                    # 4 = mean_z_pos
                    ax_filter0.plot(x_tr, z_tr, color='magenta', linewidth=0.3);
                elif track['filter'] == 0:
                    # 0 = pass all
                    ax_filter1.plot(x_tr, z_tr, color='green', linewidth=0.3);
            ax_filter0.set_ylim([-0.8,0.8])
            ax_filter1.set_ylim([-0.8,0.8])
            ax_filter0.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1
        self.set_fig_size()

    def fit_site_position(self, mini_roi_size=0.7, offset_lim=3.5):
        print('Fitting site position...', end=' \r')
        pos_corr = []
        pos_corr_px = []
        base_amplitude = np.nan
        dser_base = []
        for _, roi in self.roi_data.iterrows():
            #print(roi)
            dser_curr = [roi['date'], roi['sample'], roi['event'], roi['roiname']]
            if not dser_base:
                dser_base = dser_curr.copy()
            if dser_curr != dser_base:
                base_amplitude = np.nan
                dser_base = dser_curr.copy()
            # get current positions and image around roi
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
            initial_guess = (np.max(img_mini), int((size-1)/2), int((size-1)/2), 1, 0, 10)
            try:
                popt, _ = curve_fit(self._gaussian2D, (x, y), img_mini, p0=initial_guess)
                # get amplitude, if it is the first fit that succeeds, to get something to compare future fits with, to reject fits with a too low amplitude
                if np.isnan(base_amplitude):
                    base_amplitude = popt[0]
                if popt[0] > base_amplitude/1.2:
                    # get offsets from pixel-position
                    dx = popt[1]-int((size-1)/2)
                    dy = popt[2]-int((size-1)/2)
                    dd = np.sqrt(dx**2+dy**2)
                    # if detected offsets are too large, keep pixel-position, otherwise correct it
                    if dd > offset_lim:
                        real_peak_pos = pos_px
                        real_peak_pos_um = (real_peak_pos[0]*pixelsize+confoffset[0], real_peak_pos[1]*pixelsize+confoffset[1])
                    else:
                        real_peak_pos = (pos_px[0]+dx, pos_px[1]+dy)
                        real_peak_pos_um = (real_peak_pos[0]*pixelsize+confoffset[0], real_peak_pos[1]*pixelsize+confoffset[1])
                else:
                    # if amplitude is too low, take the last well-fitted position
                    if len(pos_corr) > 0:
                        real_peak_pos == pos_corr_px[-1]
                        real_peak_pos_um = pos_corr[-1]
                    else:
                        # if we do not have any previous positions, keep live-detected pixel position.
                        real_peak_pos = pos_px
                        real_peak_pos_um = (real_peak_pos[0]*pixelsize-confsize[0]/2, real_peak_pos[1]*pixelsize-confsize[1]/2)
                pos_corr_px.append(real_peak_pos)
                pos_corr.append(real_peak_pos_um)                    
            except:
                # if fitting fails, we likely have a spot that disappeared - if so, take the last well-fitted position
                if len(pos_corr) > 0:
                    real_peak_pos == pos_corr_px[-1]
                    real_peak_pos_um = pos_corr[-1]
                else:
                    # if we do not have any previous positions, keep live-detected pixel position
                    real_peak_pos = pos_px
                    real_peak_pos_um = (real_peak_pos[0]*pixelsize-confsize[0]/2, real_peak_pos[1]*pixelsize-confsize[1]/2)
                pos_corr_px.append(real_peak_pos)
                pos_corr.append(real_peak_pos_um)
        self.roi_data['roi_pos_px_raw'] = self.roi_data['roi_pos_px'].copy()
        self.roi_data['roi_pos_px'] = pos_corr_px
        self.roi_data['roi_pos_raw'] = self.roi_data['roi_pos'].copy()
        self.roi_data['roi_pos'] = pos_corr
        # update positions also in track_data
        self.update_track_roipos()

    def plot_sitepos_fitting(self):
        print('Plotting site position fitting...', end=' \r')
        plot_idx_cum = 0
        for date, sample, event, roi, cycle in zip(self.roi_data['date'], self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            track_data = roicycle_data.iloc[0]
            ax_img = self.figs[12].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            img = ax_img.imshow(track_data['confimg'])
            img.set_extent(track_data['confimg_ext'])  # scale overlay image to the correct pixel size for the tracks
            ax_img.set_xlim(*track_data['conf_xlim'])
            ax_img.set_ylim(*track_data['conf_ylim'])
            roi_data = self.roi_data[(self.roi_data['sample']==sample) & (self.roi_data['event']==event) & (self.roi_data['roiname']==roi) & (self.roi_data['cycle']==cycle)]
            ax_img.scatter(roi_data['roi_pos_raw'].iloc[0][0]+roi_data['pxsize']/2+track_data['conf_xcorr'], roi_data['roi_pos_raw'].iloc[0][1]+roi_data['pxsize']/2+track_data['conf_ycorr'], color='k')
            ax_img.scatter(roi_data['roi_pos'].iloc[0][0]+roi_data['pxsize']/2+track_data['conf_xcorr'], roi_data['roi_pos'].iloc[0][1]+roi_data['pxsize']/2+track_data['conf_ycorr'], color='r')
            ax_img.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1

    def filter_site_flagging(self, plot_filtered=False):
        """ Flag tracks and localizations that passes inside sites, as well as filter tracks. """
        print('Filtering and site flagging...', end=' \r')
        
        ### FILTER localizations in tracks for z-artefacts
        # set manual parameters
        window_pts = 70
        dt_loc_thresh = 550  # =500 for window_pts=40; =600 for window_pts=80; =550 for window_pts=70
        dt_artefacts_masks = []
        for _, track in self.track_data.iterrows():
            dt_track_full = np.diff(track['tim'])*1e6
            dt_track_full = np.insert(dt_track_full,0,0)
            if len(dt_track_full) > window_pts:
                local_dt_full = []
                for i in np.arange(0, len(dt_track_full)):
                    if i > int(window_pts/2) and i < len(dt_track_full)-int(window_pts/2)-1:
                        local_dt_full.append(np.mean(dt_track_full[i-int(window_pts/2):i+int(window_pts/2)]))
                    elif i <= int(window_pts/2):
                        local_dt_full.append(np.mean(dt_track_full[0:i+int(window_pts/2)]))
                    elif i >= len(dt_track_full)-int(window_pts/2)-1:
                        local_dt_full.append(np.mean(dt_track_full[i-int(window_pts/2):]))
                    else:
                        local_dt_full.append(0)
                local_dt_full = np.array(local_dt_full)
                dt_artefacts_mask = local_dt_full<dt_loc_thresh
            else:
                dt_artefacts_mask = np.full(len(dt_track_full), False)
            dt_artefacts_masks.append(dt_artefacts_mask)
        self.track_data.insert(19, 'dt_artefacts_mask', dt_artefacts_masks)

        # FILTER track lengths etc
        plot_idx_cum = 0
        for _, roicycle_data in self.roi_data.iterrows():
            date = roicycle_data['date']
            sample = roicycle_data['sample']
            event = roicycle_data['event']
            roi = roicycle_data['roiname']
            cycle = roicycle_data['cycle']
            roi_pos_um = roicycle_data['roi_pos']
            pxsize = roicycle_data['pxsize']
            pxshift = pxsize/2
            image_conf = roicycle_data['conf_img']
            conf_xlim = roicycle_data['conf_xlims']
            conf_ylim = roicycle_data['conf_ylims']
            conf_size = roicycle_data['conf_size']
            conf_offset = roicycle_data['conf_offset']
            x_conf_correction = roicycle_data['conf_xcorr']
            y_conf_correction = roicycle_data['conf_ycorr']

            track_data_roi_idxs = self.track_data.index[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)].tolist()
            track_roi_data = self.track_data.loc[track_data_roi_idxs]  # indexing with a list of indexes returns a copy

            # get image zoom from track coordinates
            ax_onlyconf = self.figs[1].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            ax_overlay = self.figs[1].add_subplot(self.subplot_rows, self.subplot_cols, 2+plot_idx_cum*self.subplot_cols)
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
            # circle polygons from overlays in correct coordinates for test
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
            
            # get mean membrane position for this roi
            track_dser = [date, sample, event, roi]
            for membrane_pos_ls in self.main_membrane_pos:
                if track_dser == membrane_pos_ls[0]:
                    membrane_pos = membrane_pos_ls[1]

            for tr_idx, track in track_roi_data.iterrows():
                x_track = track['x']
                y_track = track['y']
                z_track = track['z']
                tim_track = track['tim']
                dists = [self._distance((x0,y0,z0),(x1,y1,z1)) for x1,x0,y1,y0,z1,z0 in zip(x_track[1:],x_track[:-1],y_track[1:],y_track[:-1],z_track[1:],z_track[:-1])]
                slidestd_x_pos = [np.std(x_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(x_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                slidestd_y_pos = [np.std(y_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(y_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]
                slidestd_z_pos = [np.std(z_track[i1:i2]) for i1,i2 in zip(np.arange(0,len(z_track)-self.slidstd_interval,self.slidstd_interval), np.arange(0,len(z_track)-self.slidstd_interval,self.slidstd_interval)+self.slidstd_interval)]

                # filter indexing: 0 = pass all, 1 = short, 2 = blob_meandistinterval, 3 = blob_meanstdsliding, 4 = mean_z_pos
                if (tim_track[-1] - tim_track[0] < self.min_time):
                    filter_id = 1
                elif (np.mean(np.abs(dists[::self.interval_dist])) < self.blob_dist):
                    filter_id = 2
                elif (np.mean(slidestd_x_pos) < self.meanslidestd_thresh and np.mean(slidestd_y_pos) < self.meanslidestd_thresh and np.mean(slidestd_z_pos) < self.meanslidestd_thresh):
                    filter_id = 3
                elif np.abs(np.mean(z_track)-membrane_pos) > self.membrane_zpos_dist:
                    filter_id = 4
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
            
            mask = (self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)
            self.track_data.loc[mask, 'inout_flag'] = track_roi_data['inout_flag']
            self.track_data.loc[mask, 'inout_len'] = track_roi_data['inout_len']
            self.track_data.loc[mask, 'inout_mask'] = track_roi_data['inout_mask']
            self.track_data.loc[mask, 'inout_incl_flag'] = track_roi_data['inout_incl_flag']
            self.track_data.loc[mask, 'inout_incl_len'] = track_roi_data['inout_incl_len']
            self.track_data.loc[mask, 'inout_incl_mask'] = track_roi_data['inout_incl_mask']
            self.track_data.loc[mask, 'filter'] = track_roi_data['filter']

            ax_overlay.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            ax_onlyconf.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            
            plot_idx_cum += 1
        plt.figure(1)
        fig_fsf = plt.gcf()
        fig_fsf.set_size_inches(10,80)

    def update_track_roipos(self):
        for tr_idx, track in self.track_data.iterrows():
            sample = track['sample']
            event = track['event']
            cycle = track['cycle']
            roiname = track['roiname']
            self.track_data.at[tr_idx, 'roipos'] = self.roi_data.loc[(self.roi_data['sample'] == sample) & (self.roi_data['event'] == event) & (self.roi_data['cycle'] == cycle) & (self.roi_data['roiname'] == roiname)]['roi_pos'].iloc[0]
            self.track_data.at[tr_idx, 'roipos_px'] = self.roi_data.loc[(self.roi_data['sample'] == sample) & (self.roi_data['event'] == event) & (self.roi_data['cycle'] == cycle) & (self.roi_data['roiname'] == roiname)]['roi_pos_px'].iloc[0]

    def residence_time_analysis(self):
        """ Get all delta-ts and calculate residence times inside area. """
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

    def diff_analysis(self):
        """ Get all delta-distances and delta-t for every possible dt up to self.max_time_lag for all tracks. """
        ### Do not binning as it currently is
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
            # only site-passing tracks, that have not been filtered, divided in and out of site
            if row['inout_flag'] == True:
                dists_in = []
                dts_in = []
                dists_out = []
                dts_out = []
                # filter away marked z-artefacts here as well
                in_idx_splits = self._consecutive_bool(np.argwhere((row['inout_mask']) & (row['dt_artefacts_mask'])).flatten())
                out_idx_splits = self._consecutive_bool(np.argwhere((~row['inout_mask']) & (row['dt_artefacts_mask'])).flatten())
                for idx_list in in_idx_splits:
                    if len(idx_list) > self.split_len_thresh:
                        x_track = row['x'][idx_list]
                        y_track = row['y'][idx_list]
                        z_track = row['z'][idx_list]
                        t_track = row['tim'][idx_list]
                        for idx_0 in range(len(x_track)-2):
                            x0 = x_track[idx_0]
                            y0 = y_track[idx_0]
                            z0 = z_track[idx_0]
                            cumdt = 0
                            cumidx = 0
                            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                                cumidx += 1
                                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                                if cumidx==1 and cumdt>self.max_dt:
                                    break
                                if cumdt < self.max_time_lag:
                                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2+(z_track[idx_0+cumidx]-z0)**2)**2
                                    dists_in.append(sqd)
                                    dts_in.append(cumdt)
                for idx_list in out_idx_splits:
                    if len(idx_list) > self.split_len_thresh:
                        x_track = row['x'][idx_list]
                        y_track = row['y'][idx_list]
                        z_track = row['z'][idx_list]
                        t_track = row['tim'][idx_list]
                        for idx_0 in range(len(x_track)-2):
                            x0 = x_track[idx_0]
                            y0 = y_track[idx_0]
                            z0 = z_track[idx_0]
                            cumdt = 0
                            cumidx = 0
                            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                                cumidx += 1
                                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                                if cumidx==1 and cumdt>self.max_dt:
                                    break
                                if cumdt < self.max_time_lag:
                                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2+(z_track[idx_0+cumidx]-z0)**2)**2
                                    dists_out.append(sqd)
                                    dts_out.append(cumdt)
            # all tracks, avoiding filtered
            if row['filter']==0:
                dists = []
                dts = []
                x_track = row['x']
                y_track = row['y']
                z_track = row['z']
                t_track = row['tim']
                for idx_0 in range(len(x_track)-2):
                    x0 = x_track[idx_0]
                    y0 = y_track[idx_0]
                    z0 = z_track[idx_0]
                    cumdt = 0
                    cumidx = 0
                    while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                        cumidx += 1
                        cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                        if cumidx==1 and cumdt>self.max_dt:
                            break
                        if cumdt < self.max_time_lag:
                            sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2+(z_track[idx_0+cumidx]-z0)**2)**2
                            dists.append(sqd)
                            dts.append(cumdt)
            dists_in_all.append(dists_in)
            dts_in_all.append(dts_in)
            dists_out_all.append(dists_out)
            dts_out_all.append(dts_out)
            dists_all.append(dists)
            dts_all.append(dts)
        self.track_data['dists_in'] = dists_in_all
        self.track_data['dts_in'] = dts_in_all
        self.track_data['dists_out'] = dists_out_all
        self.track_data['dts_out'] = dts_out_all
        self.track_data['dists'] = dists_all
        self.track_data['dts'] = dts_all
 
    def get_all_displacements(self):
        dists_in_all = []
        dts_in_all = []
        dists_out_all = []
        dts_out_all = []
        dists_all = []
        dts_all = []
        key_dists_in = 'dists_in'
        key_dists_out = 'dists_out'
        key_dists_all = 'dists'
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

    def msd_analysis(self, jitter_str=0.5, y_max=1.0, plot=False, format='svg', fitmodel='msd'):
        """ Track level inside/outside MSD analysis, fitting all track parts over a certain threshold
        length inside, and the track parts outside, of the tracks that at some point passes through the site."""
        print('Running MSD analysis...', end=' \r')
        dapp_in = []
        sigma_in = []
        dapp_out = []
        sigma_out = []
        self.print_to_file('')
        self.print_to_file('Results: MSD analysis')
        if plot:
            fig,ax = plt.subplots(1,2,figsize=(10,4))
        # Fit all tracks inside site
        for _, row in self.track_data.iterrows():
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

        # Fit all tracks outside site
        for _, row in self.track_data.iterrows():
            if row['inout_flag'] == True:
                d = row['dists_out']
                t = row['dts_out']
                if len(d) > self.fit_len_thresh:
                    x_fit = np.array([x for _, x in sorted(zip(t, d))])
                    t_fit = np.array([t for t, _ in sorted(zip(t, d))])
                    if fitmodel=='lin':
                        popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
                        dapp_in.append(popt_lin[0]/4)
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
        self.track_data['locprec_out'] = dapp_out
        # get a track-level measure of the ratio between dappin and dappout
        self.track_data['dapp_ratio'] = [dapp_in/dapp_out if ~np.isnan(dapp_in) else np.nan for dapp_in,dapp_out in zip(dapp_in,dapp_out)]
        
        if plot:
            ax[0].set_xlim([0,self.max_time_lag*1.5]);
            ax[0].set_ylim([0,0.012]);
            ax[0].set_ylabel('SD [µm^2]')
            ax[0].set_xlabel('delta-t [s]')
            ax[1].set_xlim([0,self.max_time_lag*1.5]);
            ax[1].set_ylim([0,0.012]);
            ax[1].set_ylabel('SD [µm^2]')
            ax[1].set_xlabel('delta-t [s]')
        savename = 'msdanalysis-sdvdt'
        if format=='svg':
            plt.savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight")

        dapp_in = np.array(dapp_in)
        dapp_out = np.array(dapp_out)
        dapp_in = dapp_in[~np.isnan(dapp_in)]
        dapp_out = dapp_out[~np.isnan(dapp_out)]
        
        plt.figure(figsize=(3,5))
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_in]
        plt.scatter(1*np.ones(len(dapp_in)) + jitter, dapp_in, color='green', alpha=0.4, label=f'{np.mean(dapp_in):.3f} +- {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} µm^2/s');
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapp_out]
        plt.scatter(2*np.ones(len(dapp_out)) + jitter, dapp_out, color='gray', alpha=0.4, label=f'{np.mean(dapp_out):.3f} +- {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} µm^2/s');
        plt.xticks([1,2]);
        plt.gca().set_xticklabels(['In site','Out of site'])
        plt.ylim(0,y_max);
        plt.ylabel('D_app [µm^2/s]')
        plt.legend()
        plt.show()
        savename = 'msdanalysis-dapp_population'
        if format=='svg':
            plt.savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight")
        self.print_to_file(f'Inside: Dapp = {np.mean(dapp_in):.3f} +- (sem) {np.std(dapp_in)/np.sqrt(len(dapp_in)):.3f} +- (std) {np.std(dapp_in):.3f} µm^2/s')
        self.print_to_file(f'Outside: Dapp = {np.mean(dapp_out):.3f} +- (sem) {np.std(dapp_out)/np.sqrt(len(dapp_out)):.3f} +- (std) {np.std(dapp_out):.3f} µm^2/s')

        result_all = scipy.stats.ttest_ind(dapp_in, dapp_out, alternative='less')
        self.print_to_file('MSD, tracks, population, ttest')
        self.print_to_file(str(result_all))

    def msd_analysis_per_track(self, y_max=1.0, format='svg'):
        print('Running MSD analysis, per track...', end=' \r')
        
        dapps_all = []
        dser_combos = []
        dates = self.track_data['date'].unique()
        for date in dates:
            data_d = self.track_data[self.track_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_dse['roiname']==roiname].copy()
                        cycles = data_dser['cycle'].unique()
                        dser_combos.append([date, sample, event, roiname])
                        for cycle in cycles:
                            data_dserc = data_dser[data_dser['cycle']==cycle].copy()
                            dapps_all.append([[date, sample, event, roiname, cycle, data_dserc.iloc[0]['cycle_t_start_s']], [data_dserc[~np.isnan(data_dserc['dapp_in'])]['dapp_in'].tolist(), data_dserc[~np.isnan(data_dserc['dapp_in'])]['dapp_out'].tolist()], data_dserc[(~np.isnan(data_dserc['dapp_ratio']))]['dapp_ratio'].tolist()])

        cycletime_cycles_all = []
        dapps_cycles_all = []
        dappratios_cycles_all = []
        dser_combos_all = []
        for dser_combo in dser_combos:
            cycletime_dser = []
            dapps_dser = []
            dappratios_dser = []
            for row in dapps_all:
                if row[0][:-2] == dser_combo:
                    cycletime_dser.append(row[0][-1])
                    dapps_dser.append(row[1])
                    dappratios_dser.append(row[2])
            cycletime_cycles_all.append(cycletime_dser)
            dapps_cycles_all.append(dapps_dser)
            dappratios_cycles_all.append(dappratios_dser)
            dser_combos_all.append(dser_combo)
        
        plot_idx_cum = 0
        for dser, cycletimes, dapps in zip(dser_combos_all, cycletime_cycles_all, dapps_cycles_all):
            ax = self.figs[18].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            # Plot vs cycle start time
            for arr in dapps:
                arr = np.array(arr).T
                for subarr in arr:
                    ax.plot(subarr, '.-', color='green', alpha=0.2)
            ax.set_xticks([0,1],['In site','Out of site']);
            ax.set_ylabel('D_app [µm^2/s]')
            ax.set_ylim(0,y_max);
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1
        
        plot_idx_cum = 0
        for dser, cycletimes, dappratios in zip(dser_combos_all, cycletime_cycles_all, dappratios_cycles_all):
            ax = self.figs[19].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            # Plot vs cycle start time
            arr_tot = []
            for arr in dappratios:
                arr = np.array(arr)
                arr_tot.append(arr)
                jitter_str = 0.5
                jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in arr]
                ax.scatter(1*np.ones(len(arr)) + jitter, arr, color='green', alpha=0.4);
            ax.set_xticks([1],['Ratio, in/out']);
            ax.set_ylabel('D_app ratio, track [arb.u.]')
            ax.set_ylim(0,2);
            arr_tot = np.concatenate(arr_tot)
            ax.legend([f'{np.mean(arr_tot):.3f} +- {np.std(arr_tot)/np.sqrt(len(arr_tot)):.3f} µm^2/s'])
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1
            
    def msd_analysis_per_roi(self, y_max=1.0, format='svg'):
        print('Running MSD analysis, per ROI...', end=' \r')
        self.print_to_file('')
        self.print_to_file('Results: MSD analysis, ROIs')
        dapps_all = []
        dapps_allratios = []
        dates = self.track_data['date'].unique()
        for date in dates:
            data_d = self.track_data[self.track_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_ds['roiname']==roiname].copy()
                        dapps_ins = []
                        dapps_outs = []
                        dapps_ratios = []
                        for _, row in data_dser.iterrows():
                            if ~np.isnan(row['dapp_in']):
                                dapps_ins.append(row['dapp_in'])
                            if ~np.isnan(row['dapp_out']):
                                dapps_outs.append(row['dapp_out'])
                            if ~np.isnan(row['dapp_ratio']):
                                dapps_ratios.append(row['dapp_ratio'])
                        dapps_all.append(np.array([np.mean(dapps_ins),np.mean(dapps_outs)]))
                        dapps_allratios.append(np.mean(dapps_ins)/np.mean(dapps_outs))
        dapps_all = np.array(dapps_all)
        dapps_allratios = np.array(dapps_allratios)
        plt.figure(figsize=(3,5))
        plt.plot(dapps_all.T, '.-', color='green', alpha=0.2)
        plt.xticks([0,1]);
        plt.gca().set_xticklabels(['In site','Out of site'])
        plt.ylabel('D_app, ROI [µm^2/s]')        
        plt.ylim(0,y_max);
        plt.show()
        savename = 'dapp-roispaired'
        if format=='svg':
            plt.savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(3,5))
        jitter_str = 0.5
        jitter = [random.uniform(0,jitter_str)-jitter_str/2 for _ in dapps_allratios]
        plt.scatter(1*np.ones(len(dapps_allratios)) + jitter, dapps_allratios, color='green', alpha=0.4, label=f'{np.mean(dapps_allratios):.3f} +- {np.std(dapps_allratios)/np.sqrt(len(dapps_allratios)):.3f} µm^2/s');
        plt.xticks([1]);
        plt.gca().set_xticklabels(['Ratio, in/out'])
        plt.ylabel('D_app ratio, ROI [arb.u.]')        
        plt.ylim(0,2);
        plt.legend()
        plt.show()
        savename = 'dappratio-rois'
        if format=='svg':
            plt.savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight")
        plt.close()

        result_rois = scipy.stats.ttest_rel(dapps_all[:,0],dapps_all[:,1],alternative='less')
        self.print_to_file('ROIs paired ttest')
        self.print_to_file(str(result_rois))
        self.print_to_file(str(result_rois.confidence_interval()))

        self.print_to_file(f'ROIs Dapp ratio = {np.mean(dapps_allratios):.3f} +- (sem) {np.std(dapps_allratios)/np.sqrt(len(dapps_allratios)):.3f} +- (std) {np.std(dapps_allratios):.3f} µm^2/s')
          
    def msd_analysis_per_cycle(self, y_max=0.5, format='svg'):
        print('Running MSD analysis, per ROI...', end=' \r')
        dapps_all = []
        dser_combos = []
        dates = self.track_data['date'].unique()
        for date in dates:
            data_d = self.track_data[self.track_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_dse['roiname']==roiname].copy()
                        cycles = data_dser['cycle'].unique()
                        dser_combos.append([date, sample, event, roiname])
                        for cycle in cycles:
                            data_dserc = data_dser[data_dser['cycle']==cycle].copy()
                            cycletime = data_dserc.iloc[0]['cycle_t_start_s']
                            dapps_ins = []
                            dapps_outs = []
                            for _, row in data_dserc.iterrows():
                                if ~np.isnan(row['dapp_in']):
                                    dapps_ins.append(row['dapp_in'])
                                if ~np.isnan(row['dapp_out']):
                                    dapps_outs.append(row['dapp_out'])
                            dapps_all.append([[date, sample, event, roiname, cycle, cycletime], np.mean(dapps_ins), np.mean(dapps_outs), np.mean(dapps_ins)/np.mean(dapps_outs), np.std(dapps_ins)/np.sqrt(len(dapps_ins)), np.std(dapps_outs)/np.sqrt(len(dapps_outs)), np.sqrt(((np.std(dapps_ins)/np.sqrt(len(dapps_ins))/np.mean(dapps_ins))**2)+((np.std(dapps_outs)/np.sqrt(len(dapps_outs))/np.mean(dapps_outs))**2))])

        cycletime_cycles_all = []
        dapps_in_cycles_all = []
        dapps_out_cycles_all = []
        dapps_ratio_cycles_all = []
        dapps_in_sem_cycles_all = []
        dapps_out_sem_cycles_all = []
        dapps_ratio_sem_cycles_all = []
        dser_combos_all = []
        for dser_combo in dser_combos:
            cycletime_dser = []
            dapps_in_dser = []
            dapps_out_dser = []
            dapps_ratio_dser = []
            dapps_in_sem_dser = []
            dapps_out_sem_dser = []
            dapps_ratio_sem_dser = []
            for row in dapps_all:
                if row[0][:-2] == dser_combo:
                    cycletime_dser.append(row[0][-1])
                    dapps_in_dser.append(row[1])
                    dapps_out_dser.append(row[2])
                    dapps_ratio_dser.append(row[3])
                    dapps_in_sem_dser.append(row[4])
                    dapps_out_sem_dser.append(row[5])
                    dapps_ratio_sem_dser.append(row[6])
            cycletime_cycles_all.append(cycletime_dser)
            dapps_in_cycles_all.append(dapps_in_dser)
            dapps_out_cycles_all.append(dapps_out_dser)
            dapps_ratio_cycles_all.append(dapps_ratio_dser)
            dapps_in_sem_cycles_all.append(dapps_in_sem_dser)
            dapps_out_sem_cycles_all.append(dapps_out_sem_dser)
            dapps_ratio_sem_cycles_all.append(dapps_ratio_sem_dser)
            dser_combos_all.append(dser_combo)
        
        plot_idx_cum = 0
        for dser, cycletimes, dappin, dappin_sem, dappout, dappout_sem in zip(dser_combos_all, cycletime_cycles_all, dapps_in_cycles_all, dapps_in_sem_cycles_all, dapps_out_cycles_all, dapps_out_sem_cycles_all):
            ax = self.figs[16].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            # Plot vs cycle start time
            try:
                ax.errorbar(cycletimes, dappin, dappin_sem, fmt='o-', color='green', alpha=0.2)
                ax.errorbar(cycletimes, dappout, dappout_sem, fmt='o-', color='gray', alpha=0.2)
                ax.set_xlabel('Cycle start time [s]')
                ax.set_ylabel('D_app [µm^2/s]')
                ax.set_ylim(0,y_max);
                plot_idx_cum += 1
            except:
                pass
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)

        plot_idx_cum = 0
        for dser, cycletimes, dappratio, dappratio_sem in zip(dser_combos_all, cycletime_cycles_all, dapps_ratio_cycles_all, dapps_ratio_sem_cycles_all):
            ax = self.figs[17].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            # Plot vs cycle start time
            try:
                ax.errorbar(cycletimes, dappratio, dappratio_sem, fmt='o-', alpha=0.5)
                ax.set_xlabel('Cycle start time [s]')
                ax.set_ylabel('D_app ratio, ROI [arb.u.]')   
                ax.set_ylim(0,2);
            except:
                pass
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1

    def track_inclusion(self):
        print('Running track inclusion analysis...', end=' \r')
        trackinclusionratio_all = []
        dser_combos = []
        dates = self.track_data['date'].unique()
        for date in dates:
            data_d = self.track_data[self.track_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_dse['roiname']==roiname].copy()
                        cycles = data_dser['cycle'].unique()
                        dser_combos.append([date, sample, event, roiname])
                        for cycle in cycles:
                            data_dserc = data_dser[data_dser['cycle']==cycle].copy()
                            n_tracks_in = np.count_nonzero(data_dserc['inout_incl_flag']==True)
                            n_tracks_out = np.count_nonzero(data_dserc['inout_incl_flag']==False)
                            if n_tracks_in+n_tracks_out > 0:
                                trackinclusionratio_all.append(n_tracks_in/(n_tracks_in+n_tracks_out))
                            else:
                                trackinclusionratio_all.append(np.nan)
        self.roi_data['tracks_in_ratio'] = trackinclusionratio_all

    def plot_track_inclusion_per_cycle(self, y_max=1.0, format='svg'):
        tracksinratio_all = []
        dser_combos = []
        dates = self.roi_data['date'].unique()
        for date in dates:
            data_d = self.roi_data[self.roi_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_dse['roiname']==roiname].copy()
                        cycles = data_dser['cycle'].unique()
                        dser_combos.append([date, sample, event, roiname])
                        for cycle in cycles:
                            data_dserc = data_dser[data_dser['cycle']==cycle].copy()
                            try:
                                tracksinratio_all.append([[date, sample, event, roiname, cycle, data_dserc.iloc[0]['cycle_t_start_s']], data_dserc.iloc[0]['tracks_in_ratio']])
                            except:
                                tracksinratio_all.append([[date, sample, event, roiname, cycle, data_dserc.iloc[0]['cycle_t_start_s']], np.nan])

        cycletime_cycles_all = []
        tracksinratio_cycles_all = []
        dser_combos_all = []
        for dser_combo in dser_combos:
            cycletime_dser = []
            tracksinratio_dser = []
            for row in tracksinratio_all:
                if row[0][:-2] == dser_combo:
                    cycletime_dser.append(row[0][-1])
                    tracksinratio_dser.append(row[1])
            cycletime_cycles_all.append(cycletime_dser)
            tracksinratio_cycles_all.append(tracksinratio_dser)
            dser_combos_all.append(dser_combo)

        plot_idx_cum = 0
        for dser, cycletimes, tracksinratio, dser_combo in zip(dser_combos_all, cycletime_cycles_all, tracksinratio_cycles_all, dser_combos_all):
            # Plot vs cycle start time
            ax = self.figs[14].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            try:
                ax.errorbar(cycletimes, tracksinratio, fmt='o-', color='gray', alpha=0.2)
                ax.set_xlabel('Cycle start time [s]')
                ax.set_ylabel('Tracks in, ratio [a.u.]')
                ax.set_ylim(0,y_max);
            except:
                pass
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            plot_idx_cum += 1

    def diff_analysis_local(self, x_track, y_track, z_track, t_track, fitmodel='msd'):
        """ Get all delta-distances and delta-t for every possible dt up to self.max_time_lag for all tracks. """
        dists = []
        dts = []
        for idx_0 in range(len(x_track)-2):
            x0 = x_track[idx_0]
            y0 = y_track[idx_0]
            z0 = z_track[idx_0]
            cumdt = 0
            cumidx = 0
            while cumdt < self.max_time_lag and idx_0+cumidx < len(x_track)-2:
                cumidx += 1
                cumdt += t_track[idx_0+cumidx+1]-t_track[idx_0+cumidx]
                if cumdt < self.max_time_lag:
                    sqd = np.sqrt((x_track[idx_0+cumidx]-x0)**2+(y_track[idx_0+cumidx]-y0)**2+(z_track[idx_0+cumidx]-z0)**2)**2
                    dists.append(sqd)
                    dts.append(cumdt)
        x_fit = np.array([x for _, x in sorted(zip(dts, dists))])
        t_fit = np.array([t for t, _ in sorted(zip(dts, dists))])
        if fitmodel=='msd':
            popt_msd, _ = curve_fit(self._f_msd, t_fit, x_fit)
            dapp = popt_msd[0]
        elif fitmodel=='lin':
            popt_lin, _ = curve_fit(self._f_lin, t_fit, x_fit)
            dapp = popt_lin[0]/4
        return dapp

    def msd_analysis_local(self, window_pts=80, fitmodel='msd'):
        print('Run MSD analysis, local...', end=' \r')
        row_local_pos_all = []
        row_local_dapps_all = []
        for _, row in self.track_data.iterrows():
            row_local_pos = []
            row_local_dapps = []
            # get only non-filtered tracks
            if row['filter'] == 0:
                # filter away marked z-artefacts here as well
                idx_splits = self._consecutive_bool(np.argwhere(row['dt_artefacts_mask']).flatten())
                for idx_list in idx_splits:
                    if len(idx_list) > window_pts:
                        x_track_filt = row['x'][idx_list]
                        y_track_filt = row['y'][idx_list]
                        z_track_filt = row['z'][idx_list]
                        t_track_filt = row['tim'][idx_list]
                        for i in np.arange(0, len(x_track_filt)-1, 1):
                            if i > int(window_pts/2) and i < len(x_track_filt)-int(window_pts/2)-1:
                                lowlim = i-int(window_pts/2)
                                hilim = i+int(window_pts/2)
                            elif i <= int(window_pts/2):
                                lowlim = 0
                                hilim = i+int(window_pts/2)
                            elif i >= len(x_track_filt)-int(window_pts/2)-1:
                                lowlim = i-int(window_pts/2)
                                hilim = None
                            x_subtrack = x_track_filt[lowlim:hilim]
                            y_subtrack = y_track_filt[lowlim:hilim]
                            z_subtrack = z_track_filt[lowlim:hilim]
                            t_subtrack = t_track_filt[lowlim:hilim]
                            try:
                                dapp_local = self.diff_analysis_local(x_subtrack, y_subtrack, z_subtrack, t_subtrack, fitmodel=fitmodel)
                                row_local_pos.append([x_track_filt[i], y_track_filt[i], z_track_filt[i]])
                                row_local_dapps.append(dapp_local)
                            except:
                                pass
                row_local_pos_all.append(row_local_pos)
                row_local_dapps_all.append(row_local_dapps)
            else:
                row_local_pos_all.append(np.nan)
                row_local_dapps_all.append(np.nan)
        self.track_data['localdiff_pos'] = row_local_pos_all
        self.track_data['localdiff_dapp'] = row_local_dapps_all

    def create_fig(self, figidx):
        self.figs[figidx] = plt.figure(figidx)

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

    def plot_meanz_map(self, binsize=0.1, bincount_thresh=5, zlim=0.3):
        print('Plot mean-z maps...', end=' \r')
        rois = self.track_data['roiidx'].unique()
        roi_idx_cum = 0
        for roi in rois:
            roi_data = self.track_data[self.track_data['roiidx']==roi].copy()
            x_roi = np.array([])
            y_roi = np.array([])
            z_roi = np.array([])
            for _, track in roi_data.iterrows():
                x_tr = track['x']
                x_roi = np.concatenate((x_roi, x_tr))
                y_tr = track['y']
                y_roi = np.concatenate((y_roi, y_tr))
                z_tr = track['z']
                z_roi = np.concatenate((z_roi, z_tr))
                roi_size = [track['xlim'][1]-track['xlim'][0], track['ylim'][1]-track['ylim'][0]]
            x_bins = np.arange(np.mean(x_roi)-roi_size[0]/2, np.mean(x_roi)+roi_size[0]/2, binsize)
            y_bins = np.arange(np.mean(y_roi)-roi_size[1]/2, np.mean(y_roi)+roi_size[1]/2, binsize)

            ret_mean = binned_statistic_2d(x_roi, y_roi, z_roi, statistic=np.mean, bins=len(x_bins));
            ret_cnt = binned_statistic_2d(x_roi, y_roi, z_roi, statistic='count', bins=len(x_bins));
            ax = self.figs[0].add_subplot(self.subplot_rows, self.subplot_cols, 3+roi_idx_cum*self.subplot_cols)
            im = ax.imshow(np.where(ret_cnt.statistic.T > bincount_thresh, ret_mean.statistic.T, np.nan), vmin=-zlim, vmax=zlim, origin='lower');
            im.set_extent((np.mean(x_roi)-roi_size[0]/2, (np.mean(x_roi)+roi_size[0]/2), np.mean(y_roi)-roi_size[1]/2, np.mean(y_roi)+roi_size[1]/2))
            plt.colorbar(im, ax=ax, location='bottom', label='z (um)');
            roi_idx_cum += 1

    def plot_dapp_tracks(self, d_max=0.5, colormap='plasma'):
        print('Plot Dapp tracks...', end=' \r')
        plot_idx_cum = 0
        for sample, event, roi, cycle in zip(self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            roi_pos = roicycle_data.iloc[0]['roipos']
            pxsize = roicycle_data.iloc[0]['pxsize']
            pxshift = pxsize/2
            ax = self.figs[0].add_subplot(self.subplot_rows, self.subplot_cols, 3+plot_idx_cum*self.subplot_cols)
            for _, row in roicycle_data.iterrows():
                if row['filter'] == 0:
                    d_cut = np.array(row['localdiff_dapp'].copy())
                    d_cut[d_cut < 0] = 0
                    d_cut[d_cut > d_max] = d_max
                    pt = ax.plot([item[0] for item in row['localdiff_pos']], [item[1] for item in row['localdiff_pos']], '-', c='k', linewidth=0.4, alpha=0.2)
                    sc = ax.scatter([item[0] for item in row['localdiff_pos']], [item[1] for item in row['localdiff_pos']], c=d_cut, cmap=colormap, vmin=0, vmax=d_max, s=2, alpha=0.4)
            cav_circ_overlay = patches.Circle((roi_pos[0]+pxshift, roi_pos[1]+pxshift), self.inclusion_rad, edgecolor='g', linewidth=2, facecolor='none')
            ax.add_patch(cav_circ_overlay);
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')
            plt.colorbar(sc, ax=ax, ticks=[0, 0.125, 0.25, 0.375, 0.5], location='bottom');
            plot_idx_cum += 1

    def plot_dapp_map(self, binsize=0.02, bincount_thresh=5, d_max=0.5):
        print('Plot Dapp maps...', end=' \r')
        plot_idx_cum = 0
        eps = 1E-3
        ret_means = []
        ret_cnts = []

        for sample, event, roi, cycle in zip(self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            roi_pos = roicycle_data.iloc[0]['roipos']
            roi_size = roicycle_data.iloc[0]['roisize']
            x_conf_correction = roicycle_data.iloc[0]['conf_xcorr']
            y_conf_correction = roicycle_data.iloc[0]['conf_ycorr']
            x_lims = roicycle_data.iloc[0]['conf_xlim']
            y_lims = roicycle_data.iloc[0]['conf_ylim']
            pxsize = roicycle_data.iloc[0]['pxsize']
            pxshift = pxsize/2
            pos_roi = []
            dapp_roi = []
            for _, row in roicycle_data.iterrows():
                if row['filter'] == 0:
                    dapp_roi.append(np.array(row['localdiff_dapp']))
                    pos_roi.append(np.array(row['localdiff_pos']))
            try:
                dapp_roi = np.hstack(dapp_roi)
                x_roi = [pos[0] for pos2 in pos_roi for pos in pos2]
                y_roi = [pos[1] for pos2 in pos_roi for pos in pos2]

                x_bins = np.arange(roi_pos[0]+pxshift-roi_size[0]/2-eps, roi_pos[0]+pxshift+roi_size[0]/2+eps, binsize)
                y_bins = np.arange(roi_pos[1]+pxshift-roi_size[1]/2-eps, roi_pos[1]+pxshift+roi_size[1]/2+eps, binsize)
                ret_mean = binned_statistic_2d(x_roi, y_roi, dapp_roi, statistic=np.mean, bins=[x_bins, y_bins])
                ret_cnt = binned_statistic_2d(x_roi, y_roi, dapp_roi, statistic='count', bins=[x_bins, y_bins])
                ret_means.append(np.where(ret_cnt.statistic.T > bincount_thresh, ret_mean.statistic.T, np.nan))
                ret_cnts.append(ret_cnt.statistic.T)
            except:
                ret_means.append(np.nan)
                ret_cnts.append(np.nan)

            ax_mean = self.figs[0].add_subplot(self.subplot_rows, self.subplot_cols, 4+plot_idx_cum*self.subplot_cols)
            ax_cnt = self.figs[0].add_subplot(self.subplot_rows, self.subplot_cols, 5+plot_idx_cum*self.subplot_cols)
            try:
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
                plot_idx_cum += 1
            except:
                pass
        self.roi_data['dtrans_map'] = ret_means
        self.roi_data['dtrans_map_cnts'] = ret_cnts

    def dtrans_circle_analysis(self):
        print('Running Dtrans circle analysis...', end=' \r')
        rad_inner = self.circle_radii[0]
        rad_peri = self.circle_radii[1]
        rad_outer = self.circle_radii[2]

        plot_idx_cum = 0

        dtrans_inner_all = []
        dtrans_peri_all = []
        dtrans_outer_all = []
        dtrans_outside_all = []

        for date, sample, event, roi, cycle in zip(self.roi_data['date'], self.roi_data['sample'], self.roi_data['event'], self.roi_data['roiname'], self.roi_data['cycle']):
            roicycle_data = self.track_data[(self.track_data['sample']==sample) & (self.track_data['event']==event) & (self.track_data['roiname']==roi) & (self.track_data['cycle']==cycle)]
            ax = self.figs[3].add_subplot(self.subplot_rows, self.subplot_cols2, 1+plot_idx_cum*self.subplot_cols2)
            # show confocal image in background
            img_conf = ax.imshow(roicycle_data['confimg'].iloc[0])
            img_conf.set_extent(roicycle_data['confimg_ext'].iloc[0])  # scale overlay confocal image
            # add circles for caveolae site to zooms
            roi_pos_um = roicycle_data['roipos'].iloc[0]
            x_conf_correction = roicycle_data.iloc[0]['conf_xcorr']
            y_conf_correction = roicycle_data.iloc[0]['conf_ycorr']
            pxsize = roicycle_data.iloc[0]['pxsize']
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
            for _, row in roicycle_data.iterrows():
                if row['filter'] == 0:
                    dapp_roi.append(np.array(row['localdiff_dapp']))
                    pos_roi.append(np.array(row['localdiff_pos']))
            try:
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
                x_lims = [roicycle_data.iloc[0]['conf_xlim'][0]+0.2, roicycle_data.iloc[0]['conf_xlim'][1]-0.2]
                y_lims = [roicycle_data.iloc[0]['conf_ylim'][0]-0.2, roicycle_data.iloc[0]['conf_ylim'][1]+0.2]
                ax.set_xlim(*x_lims)
                ax.set_ylim(*y_lims)
                dtrans_inner = dapp_roi[cont_points_inner]
                dtrans_peri = dapp_roi[cont_points_peri]
                dtrans_outer = dapp_roi[cont_points_outer]
                dtrans_outside = dapp_roi[cont_points_outside]
                dtrans_inner_all.append(dtrans_inner)
                dtrans_peri_all.append(dtrans_peri)
                dtrans_outer_all.append(dtrans_outer)
                dtrans_outside_all.append(dtrans_outside)
            except:
                dtrans_inner_all.append(np.nan)
                dtrans_peri_all.append(np.nan)
                dtrans_outer_all.append(np.nan)
                dtrans_outside_all.append(np.nan)
            
            ax.annotate(f'ROI {roi}, cycle {cycle}, event {date} - {sample} - {event}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)
            
            plot_idx_cum += 1
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
            try:
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
            except:
                means.append([np.nan, np.nan, np.nan, np.nan])
                sems.append([np.nan, np.nan, np.nan, np.nan])
        means = np.array(means)
        sems = np.array(sems)
        plt.figure(figsize=(3,5))
        plt.plot(means.T, '.-', color='green', alpha=0.2)
        plt.xticks([0,1,2,3]);
        plt.gca().set_xticklabels(['Inner', 'Perimeter', 'Outer', 'Outside'])
        plt.ylabel('D_trans [µm^2/s]')
        plt.ylim(0,y_max);
        plt.show()
        savename = 'dtrans-circleanalysis'
        if format=='svg':
            plt.savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight")
        else:
            plt.savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight")
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

    def plot_dtrans_circle_analysis_per_cycle(self, y_max=1.0, format='svg'):
        dtrans_all = []
        dser_combos = []
        dates = self.roi_data['date'].unique()
        for date in dates:
            data_d = self.roi_data[self.roi_data['date']==date].copy()
            samples = data_d['sample'].unique()
            for sample in samples:
                data_ds = data_d[data_d['sample']==sample].copy()
                events = data_ds['event'].unique()
                for event in events:
                    data_dse = data_ds[data_ds['event']==event].copy()
                    roinames = data_dse['roiname'].unique()
                    for roiname in roinames:
                        data_dser = data_dse[data_dse['roiname']==roiname].copy()
                        cycles = data_dser['cycle'].unique()
                        dser_combos.append([date, sample, event, roiname])
                        for cycle in cycles:
                            data_dserc = data_dser[data_dser['cycle']==cycle].copy()
                            cycletime = data_dserc.iloc[0]['cycle_t_start_s']
                            dtrans_inner = data_dserc.iloc[0]['dtrans_site_inner']
                            dtrans_peri = data_dserc.iloc[0]['dtrans_site_peri']
                            dtrans_outer = data_dserc.iloc[0]['dtrans_site_outer']
                            try:
                                dtrans_all.append([[date, sample, event, roiname, cycle, cycletime], np.nanmean(dtrans_inner), np.nanmean(dtrans_peri), np.nanmean(dtrans_outer), np.nanstd(dtrans_inner)/np.sqrt(len(dtrans_inner[~np.isnan(dtrans_inner)])), np.nanstd(dtrans_peri)/np.sqrt(len(dtrans_peri[~np.isnan(dtrans_peri)])), np.nanstd(dtrans_outer)/np.sqrt(len(dtrans_outer[~np.isnan(dtrans_outer)]))])
                            except:
                                dtrans_all.append([[date, sample, event, roiname, cycle, cycletime], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        cycletime_cycles_all = []
        dtrans_inner_cycles_all = []
        dtrans_peri_cycles_all = []
        dtrans_outer_cycles_all = []
        dtrans_inner_sem_cycles_all = []
        dtrans_peri_sem_cycles_all = []
        dtrans_outer_sem_cycles_all = []
        dser_combos_all = []
        for dser_combo in dser_combos:
            cycletime_dser = []
            dtrans_inner_dser = []
            dtrans_peri_dser = []
            dtrans_outer_dser = []
            dtrans_inner_sem_dser = []
            dtrans_peri_sem_dser = []
            dtrans_outer_sem_dser = []
            for row in dtrans_all:
                if row[0][:-2] == dser_combo:
                    cycletime_dser.append(row[0][-1])
                    dtrans_inner_dser.append(row[1])
                    dtrans_peri_dser.append(row[2])
                    dtrans_outer_dser.append(row[3])
                    dtrans_inner_sem_dser.append(row[4])
                    dtrans_peri_sem_dser.append(row[5])
                    dtrans_outer_sem_dser.append(row[6])
            dser_combos_all.append(dser_combo)
            cycletime_cycles_all.append(cycletime_dser)
            dtrans_inner_cycles_all.append(dtrans_inner_dser)
            dtrans_peri_cycles_all.append(dtrans_peri_dser)
            dtrans_outer_cycles_all.append(dtrans_outer_dser)
            dtrans_inner_sem_cycles_all.append(dtrans_inner_sem_dser)
            dtrans_peri_sem_cycles_all.append(dtrans_peri_sem_dser)
            dtrans_outer_sem_cycles_all.append(dtrans_outer_sem_dser)

        plot_idx_cum = 0
        for dser, cycletimes, dtransinner, dtransinner_sem, dtransperi, dtransperi_sem, dtransouter, dtransouter_sem in zip(dser_combos_all, cycletime_cycles_all, dtrans_inner_cycles_all, dtrans_inner_sem_cycles_all, dtrans_peri_cycles_all, dtrans_peri_sem_cycles_all, dtrans_outer_cycles_all, dtrans_outer_sem_cycles_all):
            # Plot vs cycle start time
            ax = self.figs[15].add_subplot(self.subplot_rows, self.subplot_cols, 1+plot_idx_cum*self.subplot_cols)
            try:
                ax.errorbar(cycletimes, dtransinner, dtransinner_sem, fmt='o-', color='green', alpha=0.2)
                ax.errorbar(cycletimes, dtransperi, dtransperi_sem, fmt='o-', color='blue', alpha=0.2)
                ax.errorbar(cycletimes, dtransouter, dtransouter_sem, fmt='o-', color='red', alpha=0.2)
                ax.set_xlabel('Cycle start time [s]')
                ax.set_ylabel('D_trans [µm^2/s]')
                ax.set_ylim(0,y_max);
                plot_idx_cum += 1
                ax.legend(['Inner','Perimeter','Outer'])
            except:
                pass
            ax.annotate(f'ROI {dser[3]}, event {dser[0]} - {dser[1]} - {dser[2]}',
            xy=(5,5), xycoords='axes points', size=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'), annotation_clip=True)


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
                    #return
                    folder_suffix = self.paths[0].split('\\')[-1]
                    savename = f'analysisResults-dtransanalysis-{folder_suffix}'
                    #format = 'png'
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'analysisResults-dtransanalysis-{folder_suffix}'
                elif fig_idx == 11:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'filteringResults-XY-{folder_suffix}'
                elif fig_idx == 13:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'filteringResults-XZ-{folder_suffix}'
                elif fig_idx == 14:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'trackinclusion-cycles-{folder_suffix}'
                elif fig_idx == 15:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'dtrans-circleanalysis-cycles-{folder_suffix}'
                elif fig_idx == 16:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'dapp-cycles-{folder_suffix}'
                elif fig_idx == 17:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'dappratio-cycles-{folder_suffix}'
                elif fig_idx == 18:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'dapp-trackspaired-{folder_suffix}'
                elif fig_idx == 19:
                    folder_suffix = self.paths[0].split('\\')[-1]
                    if name_prefix:
                        savename = f'{name_prefix}-{folder_suffix}'
                    else:
                        savename = f'dappratio-tracks-{folder_suffix}'
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
                    self.figs[fig_idx].savefig(os.path.join(self.top_path,savename+'.pdf'), format="pdf", bbox_inches="tight")
                elif format == 'png':
                    self.figs[fig_idx].savefig(os.path.join(self.top_path,savename+'.png'), format="png", bbox_inches="tight", dpi=200)
                elif format == 'svg':
                    self.figs[fig_idx].savefig(os.path.join(self.top_path,savename+'.svg'), format="svg", bbox_inches="tight", dpi=200)
                plt.close(self.figs[fig_idx])

    def save_pickleddata(self, format='xz'):
        savename = 'trackdata-' + self.top_path.split('\\')[-1]
        self.track_data.to_pickle(os.path.join(self.top_path, savename+'.'+format))
        savename = 'roidata-' + self.top_path.split('\\')[-1]
        self.roi_data.to_pickle(os.path.join(self.top_path, savename+'.'+format))
        savename = 'analysisparams-' + self.top_path.split('\\')[-1]
        analysis_params = {'site_rad': self.ana_site_rad, 'inclusion_rad': self.inclusion_rad, 'circle_radii0': self.circle_radii[0], 'circle_radii1': self.circle_radii[1], 'circle_radii2': self.circle_radii[2],
                           'blob_dist': self.blob_dist, 'min_time': self.min_time, 'split_len_thresh': self.split_len_thresh, 'fit_len_thresh': self.fit_len_thresh, 'max_time_lag': self.max_time_lag,
                           'max_dt': self.max_dt, 'meanslidestd_thresh': self.meanslidestd_thresh, 'slidstd_interval': self.slidstd_interval,
                           'interval_dist': self.interval_dist, 'membrane_zpos_dist': self.membrane_zpos_dist, 'n_rois': self.N_rois, 'subplot_cols': self.subplot_cols, 'subplot_rows': self.subplot_rows}
        with open(os.path.join(self.top_path, savename+'.csv'), 'w') as f:
            w = csv.DictWriter(f, analysis_params.keys())
            w.writeheader()
            w.writerow(analysis_params)

    def load_pickleddata(self, folder, format='xz'):
        self.circle_radii = [np.nan, np.nan, np.nan]
        self.paths.append(folder)
        self.top_path = folder
        loadname = 'trackdata-' + self.top_path.split('\\')[-1]
        self.track_data = pd.read_pickle(os.path.join(folder, loadname+'.'+format))
        loadname = 'roidata-' + self.top_path.split('\\')[-1]
        self.roi_data = pd.read_pickle(os.path.join(folder, loadname+'.'+format))
        loadname = 'analysisparams-' + self.top_path.split('\\')[-1]
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
            elif k == 'meanslidestd_thresh':
                self.meanslidestd_thresh = v
            elif k == 'slidstd_interval':
                self.slidstd_interval = v
            elif k == 'interval_dist':
                self.interval_dist = v
            elif k == 'membrane_zpos_dist':
                self.membrane_zpos_dist = v
            elif k == 'n_rois':
                self.N_rois = int(v)
            elif k == 'subplot_cols':
                self.subplot_cols = int(v)
            elif k == 'subplot_rows':
                self.subplot_rows = int(v)
        
    def get_track_lens(self):
        return [len(track[1].x) for track in self.track_data.iterrows()]

    def _get_angle(self, p_or, p_1, p_2):
        aabs = np.sqrt((p_1[0]-p_or[0])**2+(p_1[1]-p_or[1])**2)
        babs = np.sqrt((p_2[0]-p_or[0])**2+(p_2[1]-p_or[1])**2)
        dotp = np.dot([p_1[0]-p_or[0],p_1[1]-p_or[1]],[p_2[0]-p_or[0],p_2[1]-p_or[1]])
        angle = np.arccos(dotp/(aabs*babs))*180/np.pi
        return angle

    def _distance(self, p_1, p_2):
        if len(p_1)==2:
            return np.sqrt((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)
        elif len(p_1)==3:
            return np.sqrt((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2+(p_2[2]-p_1[2])**2)
        else:
            TypeError

    def _consecutive_bool(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    def _f_lin(self, x, a, b):
        return a*x + b
    
    def _f_msd(self, dt, D, sigma):
        dim = 2  # dimensions of diffusion extension
        Rblur = 1/6.2  # blurring factor
        dtmean = 350e-6  # smallest (and median) dt between localizations
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

    