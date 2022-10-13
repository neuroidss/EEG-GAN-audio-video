#!pip install matplotlib brainflow mne librosa sounddevice absl-py pyformulas pyedflib
#!pip install diffusers transformers scipy ftfy "ipywidgets>=7,<8"
#!pip install mne mne_connectivity -U
#!pip install pyvistaqt PyQt5 darkdetect qdarkstyle

import mne
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

mne.viz.set_3d_backend('pyvistaqt')

#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from matplotlib.pyplot import draw, figure, show

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne
#from mne.connectivity import spectral_connectivity, seed_target_indices
from mne_connectivity import SpectralConnectivity
from mne_connectivity import spectral_connectivity_epochs
#from mne.viz import circular_layout, plot_connectivity_circle
from mne_connectivity.viz import plot_connectivity_circle

#import keyboard  # using module keyboard
#from pynput.mouse import Listener


import os
    
from absl import flags
FLAGS = flags.FLAGS
#  bands = [[8.,12.]]
#  methods = ['coh']
flags.DEFINE_boolean('help', False, 'help: show help and exit')
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('input_name', 'neurofeedback', 'input')
flags.DEFINE_string('serial_port', '/dev/ttyACM0', 'serial_port')
#flags.DEFINE_list('prefix', None, 'prefix')
flags.DEFINE_string('output_path', '', 'output_path')
flags.DEFINE_string('output', None, 'output: if None, used: output_path+input_name+"-%Y.%m.%d-%H.%M.%S.bdf"')
flags.DEFINE_list('ch_names', ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz'], 'ch_names')
flags.DEFINE_list('ch_names_pick', ['Fz','Cz','Pz','Oz','Fp1','Fp2','F3','F4','F7','F8','C3','C4','T7','T8','P3','P4','P7','P8','O1','O2'], 'ch_names')
#flags.DEFINE_list('ch_names_pick', ['Cz','Fz','Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2'], 'ch_names')
#flags.DEFINE_list('ch_names', ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz'], 'ch_names')
#flags.DEFINE_list('ch_names_pick', ['Cz','Fz','FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2'], 'ch_names')
#flags.DEFINE_list('ch_names_pick', ['FP1','AF3','F7','F3','FC5','T7','C3','CP5','P7','P3','PO3','O1','Oz','CP1','FC1','Fz','Cz','FC2','CP2','Pz','O2','PO4','P4','P8','CP6','C4','T8','FC6','F4','F8','AF4','FP2'], 'ch_names')
flags.DEFINE_list('bands', [8.,12.], 'bands')
#flags.DEFINE_list('bands', [4.,6.,6.5,8.,8.5,10.,10.5,12.,12.5,16.,16.5,20.,20.5,28], 'bands')
flags.DEFINE_list('methods', ['ciplv'], 'methods')
#flags.DEFINE_list('methods', ['wpli'], 'methods')
#flags.DEFINE_list('methods', ['coh'], 'methods')
flags.DEFINE_string('vmin', '0.7', 'vmin')
#flags.DEFINE_string('duration', '1', 'duration: if None, used: 5*1/bands[0]')
flags.DEFINE_string('duration', None, 'duration: if None, used: 5*1/bands[0]')
#flags.DEFINE_string('fps', '3', 'fps')
flags.DEFINE_string('fps', '10', 'fps')
#flags.DEFINE_string('fps', '20', 'fps')
#flags.DEFINE_string('fps', '30', 'fps')
flags.DEFINE_string('overlap', None, 'overlap: if None, used: duration-1/fps')
flags.DEFINE_boolean('print_freq_once', True, 'print_freq_once')
#flags.DEFINE_boolean('show_circle_cons', True, 'show_circle_cons')
flags.DEFINE_boolean('show_circle_cons', False, 'show_circle_cons')
#flags.DEFINE_boolean('show_spectrum_cons', True, 'show_spectrum_cons')
flags.DEFINE_boolean('show_spectrum_cons', False, 'show_spectrum_cons')
flags.DEFINE_boolean('sound_cons', False, 'sound_cons')
#flags.DEFINE_boolean('sound_cons', True, 'sound_cons')
flags.DEFINE_boolean('sound_cons_swap', True, 'sound_cons_swap')
flags.DEFINE_string('sound_cons_buffer_path', '', 'sound_cons_buffer_path')
flags.DEFINE_boolean('rotate', True, 'rotate')
#flags.DEFINE_boolean('show_stable_diffusion_cons', True, 'show_stable_diffusion_cons')
flags.DEFINE_boolean('show_stable_diffusion_cons', False, 'show_stable_diffusion_cons')
flags.DEFINE_string('huggingface_hub_token', 'huggingface_hub_token', 'huggingface_hub_token: token or file with token')
#flags.DEFINE_string('unet_height', '256', 'unet_height')
#flags.DEFINE_string('unet_width', '256', 'unet_width')
flags.DEFINE_string('unet_height', '512', 'unet_height')
flags.DEFINE_string('unet_width', '512', 'unet_width')
flags.DEFINE_string('num_inference_steps', '10', 'num_inference_steps')
#flags.DEFINE_string('num_inference_steps', '50', 'num_inference_steps')
flags.DEFINE_string('unet_num_inference_steps', '10', 'unet_num_inference_steps')
#flags.DEFINE_string('unet_num_inference_steps', '5', 'unet_num_inference_steps')
flags.DEFINE_string('unet_latents', None, 'unet_latents: if None, load random')
flags.DEFINE_string('unet_guidance_scale', '7.5', 'unet_guidance_scale')
flags.DEFINE_string('apply_to_latents', '10', 'apply_to_latents: closer to zero means apply more')
#flags.DEFINE_string('apply_to_latents', '0.5', 'apply_to_latents: closer to zero means apply more')
flags.DEFINE_string('apply_to_embeds', '10', 'apply_to_embeds: closer to zero means apply more')
#flags.DEFINE_string('apply_to_embeds', '1', 'apply_to_embeds: closer to zero means apply more')
flags.DEFINE_string('clip_prompt', 'villa by the sea in florence on a sunny day', 'clip_prompt')
flags.DEFINE_boolean('show_stylegan3_cons', False, 'show_stylegan3_cons')
#flags.DEFINE_boolean('show_stylegan3_cons', True, 'show_stylegan3_cons')
#flags.DEFINE_boolean('show_game_cons', True, 'show_game_cons')
flags.DEFINE_boolean('show_game_cons', False, 'show_game_cons')
#flags.DEFINE_string('game_mode', '1', 'game_mode: 1 or 3')
flags.DEFINE_string('game_mode', '3', 'game_mode: 1 or 3')
flags.DEFINE_string('n_jobs', '32', 'n_jobs')
#flags.DEFINE_boolean('cuda_jobs', True, 'cuda_jobs')
flags.DEFINE_boolean('cuda_jobs', False, 'cuda_jobs')
#flags.DEFINE_string('n_jobs', '32', "n_jobs: number of cpu jobs or 'cuda'")
flags.DEFINE_boolean('draw_fps', True, 'draw_fps')
#flags.DEFINE_string('from_bdf_file', 'neurofeedback-2022.09.20-21.50.13.bdf', 'from_bdf_file')
flags.DEFINE_string('from_bdf', None, 'from_bdf')
flags.DEFINE_string('from_edf', None, 'from_edf')
#flags.DEFINE_string('font_fname', 'fonts/freesansbold.ttf', 'font_fname')
flags.DEFINE_string('font_fname', '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', 'font_fname')
#flags.DEFINE_string('n_jobs', '8', 'n_jobs')
#flags.DEFINE_string('n_parts_one_time', None, 'n_parts_one_time')
#flags.DEFINE_string('part_len', None, 'part_len')
flags.DEFINE_boolean('show_inverse', True, 'show_inverse')
#flags.DEFINE_boolean('show_inverse', False, 'show_inverse')
#flags.DEFINE_string('fname_fwd', None, 'fname_fwd')
flags.DEFINE_string('fname_fwd', 'inverse_fwd.fif', 'fname_fwd')
#flags.mark_flag_as_required('input')
#flags.mark_flag_as_required('prefix')
#flags.mark_flag_as_required('output')
#flags.mark_flag_as_required('ch_names')
#flags.mark_flag_as_required('bands')
#flags.mark_flag_as_required('methods')
#flags.mark_flag_as_required('vmin')
#flags.mark_flag_as_required('duration')
#flags.mark_flag_as_required('overlap')
#flags.mark_flag_as_required('fps')
#flags.mark_flag_as_required('n_parts_one_time')
#flags.mark_flag_as_required('part_len')
import sys
FLAGS(sys.argv)

print(FLAGS)

if FLAGS.help:
  exit()

draw_fps=FLAGS.draw_fps
if FLAGS.cuda_jobs:
  cuda_jobs='cuda'
else:
  cuda_jobs=int(FLAGS.n_jobs)
#if FLAGS.n_jobs=='cuda':
#  n_jobs=FLAGS.n_jobs
#else:
n_jobs=int(FLAGS.n_jobs)
apply_to_latents=float(FLAGS.apply_to_latents)
apply_to_embeds=float(FLAGS.apply_to_embeds)
clip_prompt=FLAGS.clip_prompt

if os.path.isfile(FLAGS.huggingface_hub_token):
  with open(FLAGS.huggingface_hub_token, 'r') as file:
    huggingface_hub_token = file.read().replace('\n', '')
else:
  huggingface_hub_token=FLAGS.huggingface_hub_token

#print(huggingface_hub_token)
  
print_freq_once = FLAGS.print_freq_once
print_freq_once_printed = False
  
debug = FLAGS.debug
serial_port=FLAGS.serial_port

#fps=float(FLAGS.fps)
#files_path=flags.path
input_name=FLAGS.input_name
#if len(FLAGS.prefix)==0:
#  FLAGS.prefix=['']
#print('FLAGS.prefix: ',FLAGS.prefix)
#print('len(FLAGS.prefix): ',len(FLAGS.prefix))
#print('files_path: ',files_path)
#print('len(files_path): ',len(files_path))
ch_names=FLAGS.ch_names
ch_names_pick=FLAGS.ch_names_pick
bands=[{}]*int(len(FLAGS.bands)/2)
#bands[0]=FLAGS.bands
for flags_band in range(int(len(FLAGS.bands)/2)):
  bands[flags_band]=[float(FLAGS.bands[0+flags_band*2]),float(FLAGS.bands[1+flags_band*2])]
methods=FLAGS.methods
vmin=float(FLAGS.vmin)

if FLAGS.duration==None:
  duration=5*1/bands[0][0]
else:
  duration=float(FLAGS.duration)

fps=float(FLAGS.fps)  

if FLAGS.overlap==None:
  overlap=duration-1/fps
else:
  overlap=float(FLAGS.overlap)

#n_parts_one_time=int(FLAGS.n_parts_one_time)
#part_len=int(FLAGS.part_len)

if (FLAGS.from_bdf is None) and (FLAGS.from_edf is None) :
  params = BrainFlowInputParams()
  if debug:
    board_id = -1 # synthetic
    sample_rate = 512
  else:
    board_id = BoardIds.FREEEEG32_BOARD.value
    #params.serial_port = '/dev/ttyACM0'
    params.serial_port = serial_port
#    params.serial_port = '/dev/ttyS20'
    sample_rate = 512
    eeg_channels = BoardShim.get_eeg_channels(board_id)
#        if num_channels is not None:
#            eeg_channels = eeg_channels[:num_channels]

  board = BoardShim(board_id, params)
        #global board
  board.release_all_sessions()

  board.prepare_session()
  board.start_stream()

if True:

  import matplotlib.pyplot as plt
  import numpy as np
  import time
 
        #fig = plt.figure()

  show_inverse=FLAGS.show_inverse

  show_circle_cons=FLAGS.show_circle_cons
#  show_circle_cons=False
  show_spectrum_cons=FLAGS.show_spectrum_cons
#  show_spectrum_cons=True
  sound_cons=FLAGS.sound_cons
  sound_cons_swap=FLAGS.sound_cons_swap
#  sound_cons_swap=True
  sound_cons_buffer_path=FLAGS.sound_cons_buffer_path
  
  show_stable_diffusion_cons=FLAGS.show_stable_diffusion_cons
  show_stylegan3_cons=FLAGS.show_stylegan3_cons
  show_game_cons=FLAGS.show_game_cons

  import pyformulas as pf 

  if sound_cons:
    import librosa
    from librosa import load
    from librosa.core import stft, istft
    import numpy as np
    import soundfile as sf
    import sounddevice as sd

    sd.default.reset()

  if show_circle_cons:
    canvas = np.zeros((800,800))
#  canvas = np.zeros((480,640))
    screen = pf.screen(canvas, 'circle_cons')

  if show_spectrum_cons:
    canvas2 = np.zeros((800,800))
#  canvas = np.zeros((480,640))
    screen2 = pf.screen(canvas2, 'spectrum_cons')

  if show_stable_diffusion_cons:
    canvas3 = np.zeros((800,800))
#    canvas3 = np.zeros((512,512))
#  canvas = np.zeros((480,640))
    screen3 = pf.screen(canvas3, 'stable_diffusion_cons')
    import random 

  if show_stylegan3_cons:
    canvas4 = np.zeros((1024,1024))
#    canvas4 = np.zeros((800,800))
    screen4 = pf.screen(canvas4, 'stylegan3_cons')
  

  to_sum_embeds = None
  to_sum_latents = None

  buf = None
  raw = None
  input_fname_name=input_name
#  vmin=0#0.7
#  vmin=0.7
#  fps=10
#        bands = [[30.,45.]]
#bands = [[4.,7.],[8.,12.],[13.,29.],[30.,45.]]
#  bands = [[8.,12.]]
#  methods = ['coh']
#biosemi32
#  ch_names = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz']
#  ch_names_pick = ['Cz','Fz','FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2']
#  ch_names_pick = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','Cz','Fz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2']
#  ch_names_pick = ['F7','FC5','T7','CP5','P7','O1','PO3','P3','CP1','C3','FC1','F3','AF3','FP1','Fz','Cz','Pz','Oz','O2','PO4','P4','CP2','C4','FC2','F4','AF4','FP2','F8','FC6','T8','CP6','P8']
#  ch_names_pick = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5']
#Bernard's 19ch headset
#  ch_names = ["O2","T6","T4","F8","Fp2","F4","C4","P4","ch9","ch10","ch11","ch12","Pz","ch14","ch15","ch16","Fz","ch18","ch19","ch20","ch21","ch22","ch23","ch24","Fp1","F3","C3","P3","O1","T5","T3","F7"]
#  ch_names_pick = ['FP1','AF3','F7','F3','FC5','T7','C3','CP5','P7','P3','PO3','O1','Oz','CP1','FC1','Fz','Cz','FC2','CP2','Pz','O2','PO4','P4','P8','CP6','C4','T8','FC6','F4','F8','AF4','FP2']
#  ch_names_pick = ['Fz','Fp1','F7','F3','T3','C3','T5','P3','O1','Pz','O2','P4','T6','C4','T4','F4','F8','Fp2']

  label_names = ch_names
  label_names = ch_names_pick

#  cons=[]
#  for cons_index in range(int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)):
#    cons.append(np.zeros(int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)))
  cons_len=int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)
  fs_mult=3
  audio_volume_mult=200
#  cons_dur=fs_mult#fps
  cons_dur=int(fps*10)
  audio_cons_fs=int(cons_len*(fs_mult-0.0))
  cons_index=0
  cons=np.zeros((cons_dur,cons_len),dtype=float)


  cohs_tril_indices=np.zeros((2,cons_len),dtype=int)
  cohs_tril_indices_count=0
  for cons_index_diag in range(len(ch_names_pick)):
    for cons_index_diag_2 in range(2):
      for cons_index_diag_r in range(cons_index_diag+1):
        cons_index_diag_r_i=(cons_index_diag-cons_index_diag_r)
        if cons_index_diag+cons_index_diag_r+cons_index_diag_2+1<len(ch_names_pick):
          if cohs_tril_indices_count<cons_len:
            cohs_tril_indices[0][cohs_tril_indices_count]=cons_index_diag+cons_index_diag_r+cons_index_diag_2+1
            cohs_tril_indices[1][cohs_tril_indices_count]=cons_index_diag_r_i
            cohs_tril_indices_count=cohs_tril_indices_count+1

#  print(cohs_tril_indices)


  if draw_fps and (show_stylegan3_cons or show_game_cons):

    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont

    font_fname = FLAGS.font_fname
    
    font_size = 10
    font = ImageFont.truetype(font_fname, font_size)
  
    from time import perf_counter
    time000=perf_counter()
    time001=perf_counter()
    
    
  if show_inverse:

  








   if True:
       # -*- coding: utf-8 -*-
       """
       .. _tut-eeg-fsaverage-source-modeling:

       ========================================
       EEG forward operator with a template MRI
       ========================================

       This tutorial explains how to compute the forward operator from EEG data
       using the standard template MRI subject ``fsaverage``.

       .. caution:: Source reconstruction without an individual T1 MRI from the
             subject will be less accurate. Do not over interpret activity
             locations which can be off by multiple centimeters.

       Adult template MRI (fsaverage)
       ------------------------------
       First we show how ``fsaverage`` can be used as a surrogate subject.
       """

       # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
       #          Joan Massich <mailsik@gmail.com>
       #          Eric Larson <larson.eric.d@gmail.com>
       #
       # License: BSD-3-Clause

       import os.path as op
       import numpy as np

       import mne
       from mne.datasets import eegbci
       from mne.datasets import fetch_fsaverage

       # Download fsaverage files
       fs_dir = fetch_fsaverage(verbose=True)
       subjects_dir = op.dirname(fs_dir)

       # The files live in:
       subject = 'fsaverage'
       trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
       src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
       bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


       ##############################################################################
       # Load the data
       # ^^^^^^^^^^^^^
       #
       # We use here EEG data from the BCI dataset.
       #
       # .. note:: See :ref:`plot_montage` to view all the standard EEG montages
       #           available in MNE-Python.

       raw_fname, = eegbci.load_data(subject=1, runs=[6])
       raw = mne.io.read_raw_edf(raw_fname, preload=True)

       # Clean channel names to be able to use a standard 1005 montage
       new_names = dict(
           (ch_name,
            ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
           for ch_name in raw.ch_names)
       raw.rename_channels(new_names)

       # Read and set the EEG electrode locations, which are already in fsaverage's
       # space (MNI space) for standard_1020:
       montage = mne.channels.make_standard_montage('standard_1005')
       raw.set_montage(montage)
       raw.set_eeg_reference(projection=True)  # needed for inverse modeling

       # Check that the locations of EEG electrodes is correct with respect to MRI
#       mne.viz.plot_alignment(
#           raw.info, src=src, eeg=['original', 'projected'], trans=trans,
#           show_axes=True, mri_fiducials=True, dig='fiducials')

       ##############################################################################
       # Setup source space and compute forward
       # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

       if not (FLAGS.fname_fwd is None):
         if os.path.isfile(FLAGS.fname_fwd):
           fwd = mne.read_forward_solution(FLAGS.fname_fwd)
         else:
           fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                       bem=bem, eeg=True, mindist=5.0, n_jobs=None)
           mne.write_forward_solution(FLAGS.fname_fwd, fwd)
       else:
         fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                       bem=bem, eeg=True, mindist=5.0, n_jobs=None)
       print(fwd)

       ##############################################################################
       # From here on, standard inverse imaging methods can be used!
       #
       # Infant MRI surrogates
       # ---------------------
       # We don't have a sample infant dataset for MNE, so let's fake a 10-20 one:

       ch_names_ = \
           'Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2'.split()
       ch_names_=ch_names_pick
       data = np.random.RandomState(0).randn(len(ch_names_), 1000)
       info = mne.create_info(ch_names_, 1000., 'eeg')
       raw = mne.io.RawArray(data, info)

       ##############################################################################
       # Get an infant MRI template
       # ^^^^^^^^^^^^^^^^^^^^^^^^^^
       # To use an infant head model for M/EEG data, you can use
       # :func:`mne.datasets.fetch_infant_template` to download an infant template:

       subject = mne.datasets.fetch_infant_template('6mo', subjects_dir, verbose=True)

       ##############################################################################
       # It comes with several helpful built-in files, including a 10-20 montage
       # in the MRI coordinate frame, which can be used to compute the
       # MRI<->head transform ``trans``:
       fname_1020 = op.join(subjects_dir, subject, 'montages', '10-20-montage.fif')
       mon = mne.channels.read_dig_fif(fname_1020)
       mon.rename_channels(
           {f'EEG{ii:03d}': ch_name for ii, ch_name in enumerate(ch_names_, 1)})
       trans = mne.channels.compute_native_head_t(mon)
       raw.set_montage(mon)
       print(trans)

       ##############################################################################
       # There are also BEM and source spaces:

       bem_dir = op.join(subjects_dir, subject, 'bem')
       fname_src = op.join(bem_dir, f'{subject}-oct-6-src.fif')
       src = mne.read_source_spaces(fname_src)
       print(src)
 #      fname_bem = op.join(bem_dir, f'{subject}-5120-5120-5120-bem-sol.fif')
 #      bem = mne.read_bem_solution(fname_bem)

       ##############################################################################
       # You can ensure everything is as expected by plotting the result:
#       fig = mne.viz.plot_alignment(
#           raw.info, subject=subject, subjects_dir=subjects_dir, trans=trans,
#           src=src, bem=bem, coord_frame='mri', mri_fiducials=True, show_axes=True,
#           surfaces=('white', 'outer_skin', 'inner_skull', 'outer_skull'))
#       mne.viz.set_3d_view(fig, 25, 70, focalpoint=[0, -0.005, 0.01])

       ##############################################################################
       # From here, standard forward and inverse operators can be computed
       #
       # If you have digitized head positions or MEG data, consider using
       # :ref:`mne coreg` to warp a suitable infant template MRI to your
       # digitization information.   

       brain=None   
       if False:

#         fwd1=fwd
         raw1=raw
   
         raw.set_eeg_reference(projection=True)
         #events = mne.find_events(raw)
         #epochs = mne.Epochs(raw, events)
         epochs = mne.make_fixed_length_epochs(raw, 
                          duration=duration, preload=True, overlap=overlap)#, verbose='ERROR')
         cov = mne.compute_covariance(epochs, tmax=0., n_jobs=10)
  #     cov = mne.compute_covariance(epochs, tmax=0.)
         evoked = epochs['1'].average()  # trigger 1 in auditory/left
         evoked.plot_joint()
   
         inv = mne.minimum_norm.make_inverse_operator(
           evoked.info, fwd, cov, verbose=True)
         stc = mne.minimum_norm.apply_inverse(evoked, inv)
#         if not brain is None:

         data_path = mne.datasets.sample.data_path()
         subjects_dir = data_path / 'subjects'
        
         brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1, figure=1)

       raw=None
       
       if False:

           from pynput import mouse

           def on_move(x, y):
               print('Pointer moved to {0}'.format(
                   (x, y)))

           def on_click(x, y, button, pressed):
               print('{0} at {1}'.format(
                   'Pressed' if pressed else 'Released',
                   (x, y)))
               if not pressed:
                   # Stop listener
                   return False

           def on_scroll(x, y, dx, dy):
               print('Scrolled {0} at {1}'.format(
                   'down' if dy < 0 else 'up',
                   (x, y)))

           # Collect events until released
           with mouse.Listener(
                   on_move=on_move,
                   on_click=on_click,
                   on_scroll=on_scroll) as listener:
               listener.join()

           # ...or, in a non-blocking fashion:
           listener = mouse.Listener(
               on_move=on_move,
               on_click=on_click,
               on_scroll=on_scroll)
           listener.start()
       
#       while True:
#         brain.show_view()
       
       if False:
         screenshot = brain.screenshot()
         brain.close()
        
         from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)        
         nonwhite_pix = (screenshot != 255).any(-1)
         nonwhite_row = nonwhite_pix.any(1)
         nonwhite_col = nonwhite_pix.any(0)
         cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

         # before/after results
         fig = plt.figure(figsize=(4, 4))
         axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
         for ax, image, title in zip(axes, [screenshot, cropped_screenshot],
                                    ['Before', 'After']):
            ax.imshow(image)
            ax.set_title('{} cropping'.format(title))
   
   
   
   
   
   
   
   
   

   if False:
    # -*- coding: utf-8 -*-
    """
    .. _tut-eeg-mri-coords:

    ===========================================================
    EEG source localization given electrode locations on an MRI
    ===========================================================
    
    This tutorial explains how to compute the forward operator from EEG data when
    the electrodes are in MRI voxel coordinates.
    """
    
    # Authors: Eric Larson <larson.eric.d@gmail.com>
    #
    # License: BSD-3-Clause

    # %%

    import nibabel
    from nilearn.plotting import plot_glass_brain
    import numpy as np

    import mne
    from mne.channels import compute_native_head_t, read_custom_montage
    from mne.viz import plot_alignment

    ##############################################################################
    # Prerequisites
    # -------------
    # For this we will assume that you have:
    #
    # - raw EEG data
    # - your subject's MRI reconstrcted using FreeSurfer
    # - an appropriate boundary element model (BEM)
    # - an appropriate source space (src)
    # - your EEG electrodes in Freesurfer surface RAS coordinates, stored
    #   in one of the formats :func:`mne.channels.read_custom_montage` supports
    #
    # Let's set the paths to these files for the ``sample`` dataset, including
    # a modified ``sample`` MRI showing the electrode locations plus a ``.elc``
    # file corresponding to the points in MRI coords (these were `synthesized
    # <https://gist.github.com/larsoner/0ac6fad57e31cb2d9caa77350a9ff366>`__,
    # and thus are stored as part of the ``misc`` dataset).

    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / 'subjects'
    fname_raw = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    bem_dir = subjects_dir / 'sample' / 'bem'
    fname_bem = bem_dir / 'sample-5120-5120-5120-bem-sol.fif'
    fname_src = bem_dir / 'sample-oct-6-src.fif'

    misc_path = mne.datasets.misc.data_path()
    fname_T1_electrodes = misc_path / 'sample_eeg_mri' / 'T1_electrodes.mgz'
    fname_mon = misc_path / 'sample_eeg_mri' / 'sample_mri_montage.elc'

    ##############################################################################
    # Visualizing the MRI
    # -------------------
    # Let's take our MRI-with-eeg-locations and adjust the affine to put the data
    # in MNI space, and plot using :func:`nilearn.plotting.plot_glass_brain`,
    # which does a maximum intensity projection (easy to see the fake electrodes).
    # This plotting function requires data to be in MNI space.
    # Because ``img.affine`` gives the voxel-to-world (RAS) mapping, if we apply a
    # RAS-to-MNI transform to it, it becomes the voxel-to-MNI transformation we
    # need. Thus we create a "new" MRI image in MNI coordinates and plot it as:

    img = nibabel.load(fname_T1_electrodes)  # original subject MRI w/EEG
    ras_mni_t = mne.transforms.read_ras_mni_t('sample', subjects_dir)  # from FS
    mni_affine = np.dot(ras_mni_t['trans'], img.affine)  # vox->ras->MNI
    img_mni = nibabel.Nifti1Image(img.dataobj, mni_affine)  # now in MNI coords!
    plot_glass_brain(img_mni, cmap='hot_black_bone', threshold=0., black_bg=True,
                 resampling_interpolation='nearest', colorbar=True)

    ##########################################################################
    # Getting our MRI voxel EEG locations to head (and MRI surface RAS) coords
    # ------------------------------------------------------------------------
    # Let's load our :class:`~mne.channels.DigMontage` using
    # :func:`mne.channels.read_custom_montage`, making note of the fact that
    # we stored our locations in Freesurfer surface RAS (MRI) coordinates.
    #
    # .. dropdown:: What if my electrodes are in MRI voxels?
    #     :color: warning
    #     :icon: question
    #
    #     If you have voxel coordinates in MRI voxels, you can transform these to
    #     FreeSurfer surface RAS (called "mri" in MNE) coordinates using the
    #     transformations that FreeSurfer computes during reconstruction.
    #     ``nibabel`` calls this transformation the ``vox2ras_tkr`` transform
    #     and operates in millimeters, so we can load it, convert it to meters,
    #     and then apply it::
    #
    #         >>> pos_vox = ...  # loaded from a file somehow
    #         >>> img = nibabel.load(fname_T1)
    #         >>> vox2mri_t = img.header.get_vox2ras_tkr()  # voxel -> mri trans
    #         >>> pos_mri = mne.transforms.apply_trans(vox2mri_t, pos_vox)
    #         >>> pos_mri /= 1000.  # mm -> m
    #
    #     You can also verify that these are correct (or manually convert voxels
    #     to MRI coords) by looking at the points in Freeview or tkmedit.

    dig_montage = read_custom_montage(fname_mon, head_size=None, coord_frame='mri')
    dig_montage.plot()

    ##############################################################################
    # We can then get our transformation from the MRI coordinate frame (where our
    # points are defined) to the head coordinate frame from the object.

    trans = compute_native_head_t(dig_montage)
    print(trans)  # should be mri->head, as the "native" space here is MRI

#    initial_time=0.1
#    fig = None
#    figure = mne.viz.create_3d_figure(size=(800,800))
    brain = None
    while True:

    ##############################################################################
    # Let's apply this digitization to our dataset, and in the process
    # automatically convert our locations to the head coordinate frame, as
    # shown by :meth:`~mne.io.Raw.plot_sensors`.

      raw = mne.io.read_raw_fif(fname_raw)
      raw.pick_types(meg=False, eeg=True, stim=True, exclude=()).load_data()
      raw.set_montage(dig_montage)
      raw.plot_sensors(show_names=True)

    ##############################################################################
    # Now we can do standard sensor-space operations like make joint plots of
    # evoked data.

      raw.set_eeg_reference(projection=True)
      events = mne.find_events(raw)
      epochs = mne.Epochs(raw, events)
      cov = mne.compute_covariance(epochs, tmax=0., n_jobs=10)
#    cov = mne.compute_covariance(epochs, tmax=0.)
      evoked = epochs['1'].average()  # trigger 1 in auditory/left
      evoked.plot_joint()

    ##############################################################################
    # Getting a source estimate
    # -------------------------
    # New we have all of the components we need to compute a forward solution,
    # but first we should sanity check that everything is well aligned:

#      fig = plot_alignment(
#        evoked.info, trans=trans, show_axes=True, surfaces='head-dense',
#        subject='sample', subjects_dir=subjects_dir, fig=fig)

    ##############################################################################
    # Now we can actually compute the forward:

#    fwd = mne.make_forward_solution(
#        evoked.info, trans=trans, src=fname_src, bem=fname_bem, verbose=True, n_jobs=10)

      fwd = mne.make_forward_solution(
        evoked.info, trans=trans, src=fname_src, bem=fname_bem, verbose=True)

    ##############################################################################
    # Finally let's compute the inverse and apply it:

      inv = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, cov, verbose=True)
      stc = mne.minimum_norm.apply_inverse(evoked, inv)
#      if not brain is None:
        
      brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1, figure=1)
      if True:
        screenshot = brain.screenshot()
        brain.close()
        
        from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)        
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

        # before/after results
        fig = plt.figure(figsize=(4, 4))
        axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
        for ax, image, title in zip(axes, [screenshot, cropped_screenshot],
                                    ['Before', 'After']):
            ax.imshow(image)
            ax.set_title('{} cropping'.format(title))
      
#      initial_time=initial_time+0.1
#      print(brain.get_view())
#      import time
#      time.sleep(3)
#      brain.show()
#      brain = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time)









   if False:
    import matplotlib.pyplot as plt

    from nilearn import plotting

    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse

    # Set dir
    data_path = mne.datasets.sample.data_path()
    subject = 'sample'
    data_dir = data_path / 'MEG' / subject
    subjects_dir = data_path / 'subjects'
    bem_dir = subjects_dir / subject / 'bem'

    # Set file names
    fname_mixed_src = bem_dir / f'{subject}-oct-6-mixed-src.fif'
    fname_aseg = subjects_dir / subject / 'mri' / 'aseg.mgz'

    fname_model = bem_dir / f'{subject}-5120-bem.fif'
    fname_bem = bem_dir / f'{subject}-5120-bem-sol.fif'

    fname_evoked = data_dir / f'{subject}_audvis-ave.fif'
    fname_trans = data_dir / f'{subject}_audvis_raw-trans.fif'
    fname_fwd = data_dir / f'{subject}_audvis-meg-oct-6-mixed-fwd.fif'
    fname_cov = data_dir / f'{subject}_audvis-shrunk-cov.fif'

    # %%
    # Set up our source space
    # -----------------------
    # List substructures we are interested in. We select only the
    # sub structures we want to include in the source space:

    labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Brain-Stem',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']

    # %%
    # Get a surface-based source space, here with few source points for speed
    # in this demonstration, in general you should use oct6 spacing!
    src = mne.setup_source_space(subject, spacing='oct5',
                             add_dist=False, subjects_dir=subjects_dir)

    # %%
    # Now we create a mixed src space by adding the volume regions specified in the
    # list labels_vol. First, read the aseg file and the source space bounds
    # using the inner skull surface (here using 10mm spacing to save time,
    # we recommend something smaller like 5.0 in actual analyses):

    vol_src = mne.setup_volume_source_space(
        subject, mri=fname_aseg, pos=10.0, bem=fname_model,
        volume_label=labels_vol, subjects_dir=subjects_dir,
        add_interpolator=False,  # just for speed, usually this should be True
        verbose=True)

    # Generate the mixed source space
    src += vol_src
    print(f"The source space contains {len(src)} spaces and "
      f"{sum(s['nuse'] for s in src)} vertices")

    # %%
    # View the source space
    # ---------------------

    src.plot(subjects_dir=subjects_dir)

    # %%
    # We could write the mixed source space with::
    #
    #    >>> write_source_spaces(fname_mixed_src, src, overwrite=True)
    #
    # We can also export source positions to NIfTI file and visualize it again:

    nii_fname = bem_dir / f'{subject}-mixed-src.nii'
    src.export_volume(nii_fname, mri_resolution=True, overwrite=True)
    plotting.plot_img(str(nii_fname), cmap='nipy_spectral')

    # %%
    # Compute the fwd matrix
    # ----------------------
    fwd = mne.make_forward_solution(
        fname_evoked, fname_trans, src, fname_bem,
        mindist=5.0,  # ignore sources<=5mm from innerskull
        meg=True, eeg=False, n_jobs=None)
    del src  # save memory

    leadfield = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    print(f"The fwd source space contains {len(fwd['src'])} spaces and "
      f"{sum(s['nuse'] for s in fwd['src'])} vertices")

    # Load data
    condition = 'Left Auditory'
    evoked = mne.read_evokeds(fname_evoked, condition=condition,
                          baseline=(None, 0))
    noise_cov = mne.read_cov(fname_cov)

    # %%
    # Compute inverse solution
    # ------------------------
    snr = 3.0            # use smaller SNR for raw data
    inv_method = 'dSPM'  # sLORETA, MNE, dSPM
    parc = 'aparc'       # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
    loose = dict(surface=0.2, volume=1.)

    lambda2 = 1.0 / snr ** 2

    inverse_operator = make_inverse_operator(
        evoked.info, fwd, noise_cov, depth=None, loose=loose, verbose=True)
    del fwd

    stc = apply_inverse(evoked, inverse_operator, lambda2, inv_method,
                    pick_ori=None)
    src = inverse_operator['src']

    # %%
    # Plot the mixed source estimate
    # ------------------------------

    # sphinx_gallery_thumbnail_number = 3
    initial_time = 0.1
    stc_vec = apply_inverse(evoked, inverse_operator, lambda2, inv_method,
                        pick_ori='vector')
    brain = stc_vec.plot(
        hemi='both', src=inverse_operator['src'], views='coronal',
        initial_time=initial_time, subjects_dir=subjects_dir,
        brain_kwargs=dict(silhouette=True), smoothing_steps=7)

    # %%
    # Plot the surface
    # ----------------
    brain = stc.surface().plot(initial_time=initial_time,
                           subjects_dir=subjects_dir, smoothing_steps=7)
    # %%
    # Plot the volume
    # ---------------

    fig = stc.volume().plot(initial_time=initial_time, src=src,
                        subjects_dir=subjects_dir)

    # %%
    # Process labels
    # --------------
    # Average the source estimates within each label of the cortical parcellation
    # and each sub structure contained in the src space

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    labels_parc = mne.read_labels_from_annot(
        subject, parc=parc, subjects_dir=subjects_dir)

    label_ts = mne.extract_label_time_course(
        [stc], labels_parc, src, mode='mean', allow_empty=True)

    # plot the times series of 2 labels
    fig, axes = plt.subplots(1)
    axes.plot(1e3 * stc.times, label_ts[0][0, :], 'k', label='bankssts-lh')
    axes.plot(1e3 * stc.times, label_ts[0][-1, :].T, 'r', label='Brain-stem')
    axes.set(xlabel='Time (ms)', ylabel='MNE current (nAm)')
    axes.legend()
    mne.viz.tight_layout()





























  
    

  if show_game_cons:
  
    game_mode=1
#    game_mode=3
  
    dim_sg2=512

    game_out=0
    game_out_user_add = 1 << 0
    game_out_user_attack = 1 << 1
    game_out_enemy_add = 1 << 2
    game_out_enemy_attack = 1 << 3
    game_out_user_killed = 1 << 4
    game_out_enemy_killed = 1 << 5
    game_out_user_restored = 1 << 6

    #game_text=False
    game_text=True

    game=0
    game_easy=10
    game_stddev_add_user_card=0.1#1
    game_boss_enemy_cards=5#2

    game_user_add_card = 1 << 0
    game_enemy_add_card = 1 << 1
    game_user_attack_enemy = 1 << 2
    game_enemy_attack_user = 1 << 3

    game_max_user_cards=7
    game_num_user_cards=0
    #game_user_cards=np.random.rand(game_max_user_cards, dim_sg2)
    #game_user_cards_life=np.random.rand(game_max_user_cards, dim_sg2)
    game_user_cards=np.zeros((game_max_user_cards, dim_sg2))
    game_user_cards_life=np.zeros((game_max_user_cards, dim_sg2))
    game_killed_user_cards=0

    game_max_enemy_cards=7
    game_num_enemy_cards=0
    #game_enemy_cards=np.random.rand(game_max_enemy_cards, dim_sg2)
    #game_enemy_cards_life=np.random.rand(game_max_enemy_cards, dim_sg2)
    game_enemy_cards=np.zeros((game_max_enemy_cards, dim_sg2))
    game_enemy_cards_life=np.zeros((game_max_enemy_cards, dim_sg2))
    game_killed_enemy_cards=0

    game_max_possible_cards=3
    game_cur_possible_cards=0
    game_num_possible_cards=0
    #game_last_possible_cards=np.random.rand(game_max_possible_cards, dim_sg2)
    game_last_possible_cards=np.zeros((game_max_possible_cards, dim_sg2))
    #game_compare_with_possible_cards=np.random.rand(game_max_possible_cards+1, dim_sg2)
    game_compare_with_possible_cards=np.zeros((game_max_possible_cards+1, dim_sg2))

    game_user_attack_enemy_cards=np.zeros((game_max_user_cards, game_max_enemy_cards, dim_sg2))
    game_enemy_attack_user_cards=np.zeros((game_max_enemy_cards, game_max_user_cards, dim_sg2))

    game_user_attack_enemy_cards_possible=np.zeros((game_max_user_cards, game_max_enemy_cards, dim_sg2))
    game_user_stddev_compare_with_possible_cards=np.zeros((game_max_user_cards, dim_sg2))


  if show_stylegan3_cons or show_game_cons:

    sg3_models=0

#    os.chdir('/content')
#    if generate&gen_sg3_nvlabs_pt:
    import os.path
    if not os.path.isdir('stylegan3-nvlabs-pytorch'):
      os.system('git clone https://github.com/NVlabs/stylegan3.git stylegan3-nvlabs-pytorch')
#    os.chdir('/content/stylegan3-nvlabs-pytorch')
#    if generate&gen_sg3_Expl0dingCat_pt:
#    os.system('git clone https://github.com/Expl0dingCat/stylegan3-modified.git /content/stylegan3-Expl0dingCat-pytorch')
#    os.chdir('/content/stylegan3-Expl0dingCat-pytorch')

    def download_file_from_google_drive(file_id,dest_path):
      import os.path
      if not os.path.isfile(dest_path):
        os.system('mkdir -p '+os.path.dirname(dest_path))
        os.system("wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='"+file_id+" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt")
        os.system("wget --load-cookies cookies.txt -O {dest_path} 'https://docs.google.com/uc?export=download&id='"+file_id+"'&confirm='$(<confirm.txt)")

    if False:
#    if True:
      files_path = [
                ['1UP200H32RIvVYA_9TduGqIbvqfsFjpkg', 'sg3-model/', 'stylegan3-anime-faces-generator_akiyamasho', '.pkl', 'sg3_model'],
                ['1aMsP1juT3DzZpbEhcNWO_gJVw9lt7Ant', 'sg3-model/', 'stylegan3-r-afhqv2-512x512', '.pkl', 'sg3_model'],
                ['1Buunx_0kHIWdNWqRq6CBG0ILlAlcPVOb', 'sg3-model/', 'stylegan3-r-ffhq-1024x1024', '.pkl', 'sg3_model'],
                ['1YiCvVqosdRwta3qwMQHNAzuSRGKnSRp1', 'sg3-model/', 'stylegan3-r-ffhqu-256x256', '.pkl', 'sg3_model'],
                ['1z42DkzZUFhMpuWFHtMNvD6GApcL1lwG7', 'sg3-model/', 'stylegan3-r-ffhqu-1024x1024', '.pkl', 'sg3_model'],
                ['1BOln2JzcatBT6LTqbsdmrwVb8GJzvhNa', 'sg3-model/', 'stylegan3-r-metfaces-1024x1024', '.pkl', 'sg3_model'],
                ['1lh8nIxnX-xmBuu1QQokfFPBvZQPXEo0e', 'sg3-model/', 'stylegan3-r-metfacesu-1024x1024', '.pkl', 'sg3_model'],
                ['18ZAuZj9fWwbHx07RB8COJsepnOYtyHli', 'sg3-model/', 'stylegan3-t-afhqv2-512x512', '.pkl', 'sg3_model'],
                ['14OyRIEfpvhKkHooMpCKnzM3cDkOTXr6p', 'sg3-model/', 'stylegan3-t-ffhq-1024x1024', '.pkl', 'sg3_model'],
                ['1Yb5Cvf2DQ57-hX37gc4dq_Mo2UFMetnw', 'sg3-model/', 'stylegan3-t-ffhqu-256x256', '.pkl', 'sg3_model'],
                ['1XwObqI_egXDiKXoEaCn83utVEzM7Miln', 'sg3-model/', 'stylegan3-t-ffhqu-1024x1024', '.pkl', 'sg3_model'],
                ['1DH6C87Xr5wSG5mPZ8Y9GZgymsgBMTzMP', 'sg3-model/', 'stylegan3-t-metfaces-1024x1024', '.pkl', 'sg3_model'],
                ['11Mn6U-mcJulhSzUetwX1Q03h7EXrZxS_', 'sg3-model/', 'stylegan3-t-metfacesu-1024x1024', '.pkl', 'sg3_model'],
                ['1Ncs7wUsbfSEPJCcxiTLDOYjLT6UY9wT2', 'sg3-model/', 'sg3_alien-sunglases-256_network-snapshot-000074', '.pkl', 'sg3_model'],
                ['1CtKjqv7Te5X3L0KuZLIbzi7fbmpLakYS', 'sg3-model/', 'sg3_Benches-512_network-snapshot-011000', '.pkl', 'sg3_model'],
                ['15LkW8nCsVRrzjjYTVGlUJSnfGDi1RwyI', 'sg3-model/', 'sg3_flowers-256_network-snapshot-000069', '.pkl', 'sg3_model'],
                ['1RcmJNbWy9As2OMVGiVhMFM0qUKCYB1IK', 'sg3-model/', 'sg3_Landscapes_lhq-256-stylegan3-t-25Mimg', '.pkl', 'sg3_model'],
                ['1iO_T0MvNw59MPAueoqUHKzpoh40vyLrZ', 'sg3-model/', 'sg3_mechanical-devices-from-the-future-256_network-snapshot-000029', '.pkl', 'sg3_model'],
                ['1mMZSFynUd_6AIuC8PkDWdeHWY4VyIqYm', 'sg3-model/', 'sg3_scifi-city-256_network-snapshot-000210', '.pkl', 'sg3_model'],
                ['14DpmYfsX3K9JhkS0BV5YtgZ71wJOInsd', 'sg3-model/', 'sg3_scifi-spaceship-256_network-snapshot-000162', '.pkl', 'sg3_model'],
                ['13Q5bDnng7VfqYq6g-t8jVybR-E_Df_q3', 'sg3-model/', 'sg3_wikiart-1024-stylegan3-t-17.2Mimg', '.pkl', 'sg3_model'],
                ['10Q6npsBKdRWMb0LxZUzN6FBSNeB4KTA6', 'sg3-model/', 'sg3_yellow-alien-512_network-snapshot-000236', '.pkl', 'sg3_model'],
                ['10l7ADbHmZgjSrrpNzOD8r5grJqwxfRd3', 'sg3-model/', 'stylegan3_sneaksnap', '.pkl', 'sg3_model']
                 ]
      download_file_from_google_drive(file_id=files_path[sg3_models][0], dest_path=files_path[sg3_models][1]+files_path[sg3_models][2]+files_path[sg3_models][3])
      files_path[0][1]=files_path[sg3_models][1]+files_path[sg3_models][2]+files_path[sg3_models][3]


#    os.system('pip install scipy')
#    !pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
#  !pip install torch
#  !pip install torch==1.7.1
#  %pip install ninja
#  import pickle
    import copy
    import os
  #from time import perf_counter

  #import click
    import imageio
    import numpy as np
    import PIL.Image
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, 'stylegan3-nvlabs-pytorch')
#    sys.path.insert(0, '/content/stylegan3-Expl0dingCat-pytorch')

    import dnnlib
    import legacy

    if True:
#    if False:
      files_path = [
          ['1ie1vWw1JNsfrZWRtMvhteqzVz4mt4KGa', 'model/sg2-ada_abstract_network-snapshot-000188.pkl',
                 'sg2-ada_abstract_network-snapshot-000188','stylegan2-ada'],
          ['1aUrChOhq5jDEddZK1v_Dp1vYNlHSBL9o', 'model/sg2-ada_2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl', 
                 'sg2-ada_2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664','stylegan2-ada'],
                   ]
      if show_game_cons:
        if FLAGS.game_mode=='1':
          files_path=[files_path[0],files_path[1]]
        if FLAGS.game_mode=='3':
          files_path=[files_path[0],files_path[1],files_path[1],files_path[1],files_path[1]]
      
      for i in range(len(files_path)):
        download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])

    G3ms=[{}]*len(files_path)
    for i in range(len(files_path)):
      network_pkl=files_path[i][1]
      device = torch.device('cuda')
      with dnnlib.util.open_url(network_pkl) as fp:
#      G3m = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        G3ms[i] = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore

    if draw_fps:
      time001a=[{}]*len(files_path)
      time111a=[{}]*len(files_path)
      for i in range(len(files_path)):
        time001a[i]=perf_counter()

    if show_game_cons:

                dim_sg2=G3ms[1].z_dim
#                sg3_latents=np.random.rand((1), G3ms[1].z_dim) 
                vol=1


#               base_latents = sg3_latents#.detach().clone()
#            cons_latents = base_latents
#                cons_latents_flatten = base_latents.reshape(len(base_latents[0]))
                if FLAGS.game_mode=='1':
                  cons_latentsa = [
                    np.random.rand((1), G3ms[1].z_dim),
                    np.random.rand((1), G3ms[1].z_dim),
                  ]
                if FLAGS.game_mode=='3':
                  cons_latentsa = [
                    np.random.rand((1), G3ms[1].z_dim),
                    np.random.rand((1), G3ms[1].z_dim),
                    np.random.rand((1), G3ms[1].z_dim),
                    np.random.rand((1), G3ms[1].z_dim),
                    np.random.rand((1), G3ms[1].z_dim),
                  ]
    
    
    import mne
    from mne import io
    from mne.datasets import sample
    from mne.minimum_norm import read_inverse_operator, compute_source_psd

#    from mne.connectivity import spectral_connectivity, seed_target_indices

    import pandas as pd
    import numpy as np      
    

#    from tqdm.auto import tqdm
#    from torch import autocast
#    from diffusers import (
#        StableDiffusionPipeline, AutoencoderKL,
#        UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
#    )
#    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    
    
    # Pipeline
#    scheduler = LMSDiscreteScheduler(
#        beta_start=0.00085, beta_end=0.012,
#        beta_schedule='scaled_linear', num_train_timesteps=1000
#    )

#    scheduler1 = LMSDiscreteScheduler(
#        beta_start=0.00085, beta_end=0.012,
#        beta_schedule='scaled_linear', num_train_timesteps=1000
#    )
  
  if show_game_cons:
    canvas5a=[{}]*len(files_path)
    screen5a=[{}]*len(files_path)
    for i in range(len(files_path)):
#      canvas5a[i] = np.zeros((128,128))
      canvas5a[i] = np.zeros((512,512))
#      canvas5a = np.zeros((1024,1024))
      screen5a[i] = pf.screen(canvas5a[i], 'game_cons'+str(i))
      
  #import os
  #import logging
  #import pandas as pd

  if show_stable_diffusion_cons:
  
   if False:
     import cv2
     import numpy as np 
     import imageio
     import PIL.Image     

     drawing = False # true if mouse is pressed
     pt1_x , pt1_y = None , None
  
     def line_drawing(event,x,y,flags,param):
       global pt1_x,pt1_y,drawing

       if event==cv2.EVENT_LBUTTONDOWN:
           drawing=True
           pt1_x,pt1_y=x,y

       elif event==cv2.EVENT_MOUSEMOVE:
           if drawing==True:
#               cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=30)
#               cv2.line(img,(pt1_x,pt1_y),(x,y),color=(127,127,127),thickness=30)
               cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=30)
               pt1_x,pt1_y=x,y
       elif event==cv2.EVENT_LBUTTONUP:
           drawing=False
#           cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=30)        
#           cv2.line(img,(pt1_x,pt1_y),(x,y),color=(127,127,127),thickness=30)        
           cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=30)        

     img = np.zeros((512,512,3), np.uint8)
#     img = np.ones((512,512,3), np.uint8)
     img=img*255
     cv2.namedWindow('test draw')
     cv2.setMouseCallback('test draw',line_drawing)

   if True:

    import requests
    import torch
    torch.cuda.empty_cache()
    #import google.colab.output
    from torch import autocast
    from torch.nn import functional as F
    from torchvision import transforms
    from diffusers import (
        StableDiffusionPipeline, AutoencoderKL,
        UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
    )
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
    from tqdm.auto import tqdm
    from huggingface_hub import notebook_login
    from PIL import Image, ImageDraw

    device = 'cuda'

    #google.colab.output.enable_custom_widget_manager()
    notebook_login()  

   if False:

    # Default Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', 
        revision='fp16',
        tourch_dtype=torch.float16, 
        use_auth_token=huggingface_hub_token
    )
    pipe = pipe.to(device)

   if False:

    prompt = clip_prompt
    with autocast(device):
        image = pipe(prompt)['images'][0]
    image

   if False:
    
    def image_grid(images, rows, cols):
        assert len(images) == rows * cols

        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
    
   if False:
    
    nrows, ncols = 1, 3
    prompts = [clip_prompt] * nrows * ncols
    with autocast(device):
        images = pipe(prompts)['sample']
    image_grid(images, rows=nrows, cols=ncols)
    
   if True:
    
    # Custom Pipeline
    vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=huggingface_hub_token
    )
    vae = vae.to(device)  # 1GB

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = text_encoder.to(device)  # 1.5 GB with VAE

    unet = UNet2DConditionModel.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=huggingface_hub_token
    )
    unet = unet.to(device)  # 4.8 GB with VAE and CLIP text
    
   if False:
    
    ## VAE
    car_img = Image.open('/content/villa.png')
    #car_img = Image.open(requests.get('https://i.ibb.co/qmcCRQJ/ferrari.png', stream=True).raw)
    car_img = car_img.resize((512, 512))
    car_img
    
   if True:
    
    dict(vae.config)
    
   if True:
    
    def preprocess(pil_image):
        pil_image = pil_image.convert("RGB")
        processing_pipe = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        tensor = processing_pipe(pil_image)
        tensor = tensor.reshape(1, 3, 512, 512)
        return tensor
    
   if True:
    
    def encode_vae(img):
        img_tensor = preprocess(img)
        with torch.no_grad():
            diag_gaussian_distrib_obj = vae.encode(img_tensor.to(device), return_dict=False)
            img_latent = diag_gaussian_distrib_obj[0].sample().detach().cpu()
            img_latent *= 0.18215
        return img_latent

   if False:
    car_latent = encode_vae(car_img)
    car_latent.shape
    
   if True:
    
    def decode_latents(latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            images = vae.decode(latents)['sample']

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype('uint8')
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

   if False:
    images = decode_latents(car_latent.to(device))
    images[0]
    
   if True:
    
    # CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
   if False:
    
    image = Image.open('/content/villa.png')
    #url = 'https://i.ibb.co/qmcCRQJ/ferrari.png'
    #image = Image.open(requests.get(url, stream=True).raw)

    description_candidates = [
        clip_prompt, 
        'villa',
        'anime',
    ]

    inputs = clip_processor(text=description_candidates, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    print(logits_per_image)
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    
   if True:
    
    # UNet
    dict(unet.config)
    
   if True:
    
    # Pipeline
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000
    )

    scheduler1 = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000
    )
    
   if True:
    
    def get_text_embeds(prompt):
        text_input = tokenizer(
            prompt, padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
        uncond_input = tokenizer(
            [''] * len(prompt), padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    prompt = clip_prompt
    test_embeds = get_text_embeds([prompt])
    print(test_embeds)
    print(test_embeds.shape)

   if False:

    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())    
    del tokenizer
    del text_encoder
    del clip_model
    del clip_processor
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())    
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())    


#   if True:
#    test_embeds = torch.randn(2, 77, 768).to(device)
    
   if True:

    unet_num_inference_steps=int(FLAGS.unet_num_inference_steps)
    unet_height=int(FLAGS.unet_height)
    unet_width=int(FLAGS.unet_width)
    if FLAGS.unet_latents is None:
      unet_latents = torch.randn((
                test_embeds.shape[0] // 2,
                unet.in_channels,
                unet_height // 8,
                unet_width // 8
            ))    
#      with open('unet_latents', 'w') as file:
#        file.write(unet_latents)
    else:
      if os.path.isfile(FLAGS.unet_latents):
        with open(FLAGS.unet_latents, 'r') as file:
          unet_latents = file.read().replace('\n', '')
      else:
       unet_latents=FLAGS.unet_latents
       
    unet_guidance_scale=float(FLAGS.unet_guidance_scale)
    
    print(unet_latents)
    print(unet_latents.shape)
          
    def generate_latents(
        text_embeddings,
#        height=128, 
#        width=128,
#        height=256, 
#        width=256,
#        height=384, 
#        width=384,
        height=512, 
        width=512,
#        num_inference_steps=5,
#        num_inference_steps=10,
#        num_inference_steps=25,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None
    ):
        if latents is None:
            latents = torch.randn((
                text_embeddings.shape[0] // 2,
                unet.in_channels,
                height // 8,
                width // 8
            ))
            
        latents = latents.to(device)

        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.sigmas[0]

        with autocast('cuda'):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
            
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, i, latents)['prev_sample']

        return latents

   if False:
    test_latents = generate_latents(test_embeds)
    print(test_latents)
    print(test_latents.shape)

   if True:

    def decode_latents(latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            images = vae.decode(latents)['sample']

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype('uint8')
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

   if False:
    images = decode_latents(test_latents)
    images[0]
    images[0].save('mygraph.png', format='png')
#    image = np.asarray(images[0])
#    screen3.update(image)
    
    
   if True:
    
    def generate(
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
    
        text_embeds = get_text_embeds(prompts)
        latents = generate_latents(
            text_embeds, 
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        images = decode_latents(latents)
        return images
        
   if False:
        
    res = generate('villa by the sea in florence on a sunny day')
    res[0]
    

  if not ((FLAGS.from_bdf is None) and (FLAGS.from_edf is None)):

    if not (FLAGS.from_bdf is None):
      raw = mne.io.read_raw_bdf(FLAGS.from_bdf, eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)
    if not (FLAGS.from_edf is None):
      raw = mne.io.read_raw_edf(FLAGS.from_edf, eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)

    print(raw.info)
    print(raw.info['ch_names'])
    if True:
       # Clean channel names to be able to use a standard 1005 montage
       new_names = dict(
           (ch_name,
            ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
           for ch_name in raw.ch_names)
       raw.rename_channels(new_names)
       print(raw.info['ch_names'])

    sfreq = raw.info['sfreq']  # the sampling frequency
    sample_rate=sfreq

    raw_pick = raw.pick(ch_names_pick)
    
    eeg_channels=ch_names

  if (FLAGS.from_bdf is None) and (FLAGS.from_edf is None) :

    import pyedflib
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d-%H.%M.%S")
    output_path=FLAGS.output_path
  
    if FLAGS.output==None:
      dst=output_path+input_name+"-"+dt_string+".bdf"
    else:
      dst=FLAGS.output
  
  
    pmax=300000
    dmax = 8388607
    dmin = -8388608
#  if not pmax:
#      pmax = max(abs(signals.min()), signals.max())
    pmin = -pmax
  
    dimension="uV"
    data_key="eeg"
    rate=512

    n_channels = len(eeg_channels)
    file_type = 3  # BDF+
    bdf = pyedflib.EdfWriter(dst, n_channels=n_channels, file_type=file_type)

    headers = []
    print(ch_names)
    for channel in ch_names:
        headers.append(
            {
                "label": str(channel),
                "dimension": dimension,
                "sample_rate": rate,
                "physical_min": pmin,
                "physical_max": pmax,
                "digital_min": dmin,
                "digital_max": dmax,
                "transducer": "",
                "prefilter": "",
            }
        )
    bdf.setSignalHeaders(headers)
    bdf.setStartdatetime(now)
  #try:
  #      events = store.select(events_key)
  #except:
  #      logger.warning("Events key not found.")
  #      events = False
  #if events is not False:
  #      for event in events.itertuples():
  #          onset = (event.Index - start).total_seconds()
  #          duration = 0
  #          description = event.label
  #          bdf.writeAnnotation(onset, duration, description)  # meta is lost
#  bdf.close()

  if show_stable_diffusion_cons:
   latents = unet_latents.detach().clone()
   latents = latents.to(device)
   num_inference_steps=int(FLAGS.num_inference_steps)
   scheduler.set_timesteps(num_inference_steps)
   latents = latents * scheduler.sigmas[0]
   latentsa=[{}]*num_inference_steps
#for i in range(len(latentsa)):
#  latentsa[i]=empty
#print(scheduler.timesteps)
#print(tqdm(enumerate(scheduler.timesteps)))
   unet_latents=unet_latents.to(device) 
   i_tqdm=0
   latentsa[i_tqdm]=latents.detach().clone()

  raw_buf = None

  if True:
    def add_data(brain, array, fmin=None, fmid=None, fmax=None,
                 thresh=None, center=None, transparent=False, colormap="auto",
                 alpha=1, vertices=None, smoothing_steps=None, time=None,
                 time_label="auto", colorbar=True,
                 hemi=None, remove_existing=None, time_label_size=None,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 clim=None, src=None, volume_options=0.4, colorbar_kwargs=None,
                 verbose=None):
        """Display data from a numpy array on the surface or volume.
        This provides a similar interface to
        :meth:`surfer.Brain.add_overlay`, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).
        .. note:: ``fmin`` sets the low end of the colormap, and is separate
                  from thresh (this is a different convention from
                  :meth:`surfer.Brain.add_overlay`).
        Parameters
        ----------
        array : numpy array, shape (n_vertices[, 3][, n_times])
            Data array. For the data to be understood as vector-valued
            (3 values per vertex corresponding to X/Y/Z surface RAS),
            then ``array`` must be have all 3 dimensions.
            If vectors with no time dimension are desired, consider using a
            singleton (e.g., ``np.newaxis``) to create a "time" dimension
            and pass ``time_label=None`` (vector values are not supported).
        %(fmin_fmid_fmax)s
        %(thresh)s
        %(center)s
        %(transparent)s
        colormap : str, list of color, or array
            Name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            Alpha level to control opacity of the overlay.
        vertices : numpy array
            Vertices for which the data is defined (needed if
            ``len(data) < nvtx``).
        smoothing_steps : int or None
            Number of smoothing steps (smoothing is used if len(data) < nvtx)
            The value 'nearest' can be used too. None (default) will use as
            many as necessary to fill the surface.
        time : numpy array
            Time points in the data array (if data is 2D or 3D).
        %(time_label)s
        colorbar : bool
            Whether to add a colorbar to the figure. Can also be a tuple
            to give the (row, col) index of where to put the colorbar.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Not supported yet.
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14).
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        scale_factor : float | None (default)
            The scale factor to use when displaying glyphs for vector-valued
            data.
        vector_alpha : float | None
            Alpha level to control opacity of the arrows. Only used for
            vector-valued data. If None (default), ``alpha`` is used.
        clim : dict
            Original clim arguments.
        %(src_volume_options)s
        colorbar_kwargs : dict | None
            Options to pass to :meth:`pyvista.Plotter.add_scalar_bar`
            (e.g., ``dict(title_font_size=10)``).
        %(verbose)s
        Notes
        -----
        If the data is defined for a subset of vertices (specified
        by the "vertices" parameter), a smoothing method is used to interpolate
        the data onto the high resolution surface. If the data is defined for
        subsampled version of the surface, smoothing_steps can be set to None,
        in which case only as many smoothing steps are applied until the whole
        surface is filled with non-zeros.
        Due to a VTK alpha rendering bug, ``vector_alpha`` is
        clamped to be strictly < 1.
        """
#        _validate_type(transparent, bool, 'transparent')
#        _validate_type(vector_alpha, ('numeric', None), 'vector_alpha')
#        _validate_type(scale_factor, ('numeric', None), 'scale_factor')

        # those parameters are not supported yet, only None is allowed
#        _check_option('thresh', thresh, [None])
#        _check_option('remove_existing', remove_existing, [None])
#        _validate_type(time_label_size, (None, 'numeric'), 'time_label_size')
        if time_label_size is not None:
            time_label_size = float(time_label_size)
            if time_label_size < 0:
                raise ValueError('time_label_size must be positive, got '
                                 f'{time_label_size}')

        hemi = brain._check_hemi(hemi, extras=['vol'])
        stc, array, vertices = brain._check_stc(hemi, array, vertices)
        array = np.asarray(array)
        vector_alpha = alpha if vector_alpha is None else vector_alpha
        brain._data['vector_alpha'] = vector_alpha
        brain._data['scale_factor'] = scale_factor

        # Create time array and add label if > 1D
        if array.ndim <= 1:
            time_idx = 0
        else:
            # check time array
            if time is None:
                time = np.arange(array.shape[-1])
            else:
                time = np.asarray(time)
                if time.shape != (array.shape[-1],):
                    raise ValueError('time has shape %s, but need shape %s '
                                     '(array.shape[-1])' %
                                     (time.shape, (array.shape[-1],)))
            brain._data["time"] = time

            if brain._n_times is None:
                brain._times = time
            elif len(time) != brain._n_times:
                raise ValueError("New n_times is different from previous "
                                 "n_times")
            elif not np.array_equal(time, brain._times):
                raise ValueError("Not all time values are consistent with "
                                 "previously set times.")

            # initial time
            if initial_time is None:
                time_idx = 0
            else:
                time_idx = brain._to_time_index(initial_time)

        # time label
        time_label = None
#        time_label, _ = _handle_time(time_label, 's', time)
        y_txt = 0.05 + 0.1 * bool(colorbar)

        if array.ndim == 3:
            if array.shape[1] != 3:
                raise ValueError('If array has 3 dimensions, array.shape[1] '
                                 'must equal 3, got %s' % (array.shape[1],))
#        fmin, fmid, fmax = _update_limits(
#            fmin, fmid, fmax, center, array
#        )
        if colormap == 'auto':
            colormap = 'mne' if center is not None else 'hot'

        if smoothing_steps is None:
            smoothing_steps = 7
        elif smoothing_steps == 'nearest':
            smoothing_steps = -1
        elif isinstance(smoothing_steps, int):
            if smoothing_steps < 0:
                raise ValueError('Expected value of `smoothing_steps` is'
                                 ' positive but {} was given.'.format(
                                     smoothing_steps))
        else:
            raise TypeError('Expected type of `smoothing_steps` is int or'
                            ' NoneType but {} was given.'.format(
                                type(smoothing_steps)))

        brain._data['stc'] = stc
        brain._data['src'] = src
        brain._data['smoothing_steps'] = smoothing_steps
        brain._data['clim'] = clim
        brain._data['time'] = time
        brain._data['initial_time'] = initial_time
        brain._data['time_label'] = time_label
        brain._data['initial_time_idx'] = time_idx
        brain._data['time_idx'] = time_idx
        brain._data['transparent'] = transparent
        # data specific for a hemi
        brain._data[hemi] = dict()
        brain._data[hemi]['glyph_dataset'] = None
        brain._data[hemi]['glyph_mapper'] = None
        brain._data[hemi]['glyph_actor'] = None
        brain._data[hemi]['array'] = array
        brain._data[hemi]['vertices'] = vertices
        brain._data['alpha'] = alpha
        brain._data['colormap'] = colormap
        brain._data['center'] = center
        brain._data['fmin'] = fmin
        brain._data['fmid'] = fmid
        brain._data['fmax'] = fmax
        brain.update_lut()

        # 1) add the surfaces first
        actor = None
        for _ in brain._iter_views(hemi):
            if hemi in ('lh', 'rh'):
                actor = brain._layered_meshes[hemi]._actor
            else:
                src_vol = src[2:] if src.kind == 'mixed' else src
                actor, _ = brain._add_volume_data(hemi, src_vol, volume_options)
        assert actor is not None  # should have added one
        brain._add_actor('data', actor)

        # 2) update time and smoothing properties
        # set_data_smoothing calls "set_time_point" for us, which will set
        # _current_time
        brain.set_time_interpolation(brain.time_interpolation)
        brain.set_data_smoothing(brain._data['smoothing_steps'])

        # 3) add the other actors
        if colorbar is True:
            # bottom left by default
            colorbar = (brain._subplot_shape[0] - 1, 0)
        for ri, ci, v in brain._iter_views(hemi):
            # Add the time label to the bottommost view
            do = (ri, ci) == colorbar
            if not brain._time_label_added and time_label is not None and do:
                time_actor = brain._renderer.text2d(
                    x_window=0.95, y_window=y_txt,
                    color=brain._fg_color,
                    size=time_label_size,
                    text=time_label(brain._current_time),
                    justification='right'
                )
                brain._data['time_actor'] = time_actor
                brain._time_label_added = True
            if colorbar and brain._scalar_bar is None and do:
                kwargs = dict(source=actor, n_labels=8, color=brain._fg_color,
                              bgcolor=brain._brain_color[:3])
                kwargs.update(colorbar_kwargs or {})
                brain._scalar_bar = brain._renderer.scalarbar(**kwargs)
#            brain._renderer.set_camera(
#                update=False, reset_camera=False, **views_dicts[hemi][v])

        # 4) update the scalar bar and opacity
        brain.update_lut(alpha=alpha)

    def clear_glyphs(brain):
        """Clear the picking glyphs."""
        if not brain.time_viewer:
            return
        for sphere in list(brain._spheres):  # will remove itself, so copy
            brain._remove_vertex_glyph(sphere, render=False)
        assert sum(len(v) for v in brain.picked_points.values()) == 0
        assert len(brain.pick_table) == 0
        assert len(brain._spheres) == 0
        for hemi in brain._hemis:
            for label_id in list(brain.picked_patches[hemi]):
                brain._remove_label_glyph(hemi, label_id)
        assert sum(len(v) for v in brain.picked_patches.values()) == 0
        if brain.rms is not None:
            brain.rms.remove()
            brain.rms = None
        brain._renderer._update()

    def _configure_vertex_time_course(brain):
#        if not brain.show_traces:
#            return
        if brain.mpl_canvas is None:
            brain._configure_mplcanvas()
        else:
            clear_glyphs(brain)
#            brain.clear_glyphs()

        # plot RMS of the activation
        y = np.concatenate(list(v[0] for v in brain.act_data_smooth.values()
                                if v[0] is not None))
        rms = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
        del y

        brain.rms, = brain.mpl_canvas.axes.plot(
            brain._data['time'], rms,
            lw=3, label='RMS', zorder=3, color=brain._fg_color,
            alpha=0.5, ls=':')

        # now plot the time line
        brain.plot_time_line(update=False)

        # then the picked points
        for idx, hemi in enumerate(['lh', 'rh', 'vol']):
            act_data = brain.act_data_smooth.get(hemi, [None])[0]
            if act_data is None:
                continue
            hemi_data = brain._data[hemi]
            vertices = hemi_data['vertices']

            # simulate a picked renderer
            if brain._hemi in ('both', 'rh') or hemi == 'vol':
                idx = 0
            brain.picked_renderer = brain._renderer._all_renderers[idx]

            # initialize the default point
            if brain._data['initial_time'] is not None:
                # pick at that time
                use_data = act_data[
                    :, [np.round(brain._data['time_idx']).astype(int)]]
            else:
                use_data = act_data
            ind = np.unravel_index(np.argmax(np.abs(use_data), axis=None),
                                   use_data.shape)
            if hemi == 'vol':
                mesh = hemi_data['grid']
            else:
                mesh = brain._layered_meshes[hemi]._polydata
            vertex_id = vertices[ind[0]]
            brain._add_vertex_glyph(hemi, mesh, vertex_id, update=False)


    def _configure_dock_trace_widget(brain, name):
        if not brain.show_traces:
            return
        # do not show trace mode for volumes
#        if (brain._data.get('src', None) is not None and
#                brain._data['src'].kind == 'volume'):
#            brain._configure_vertex_time_course()
#            return

#        layout = brain._renderer._dock_add_group_box(name)
#        weakself = weakref.ref(brain)

        # setup candidate annots
#        def _set_annot(annot, weakself=weakself):
#            brain = weakself()
#            if brain is None:
#                return
        if True:
#            brain.clear_glyphs()
#            brain.remove_labels()
#            brain.remove_annotations()
#            brain.annot = annot

#            if annot == 'None':
            if True:
                brain.traces_mode = 'vertex'
                _configure_vertex_time_course(brain)
#                brain._configure_vertex_time_course()
            else:
                brain.traces_mode = 'label'
                brain._configure_label_time_course()
            brain._renderer._update()

        # setup label extraction parameters
#        def _set_label_mode(mode, weakself=weakself):
#            brain = weakself()
#            if brain is None:
#                return
#            if brain.traces_mode != 'label':
#                return
        if True:
#            glyphs = copy.deepcopy(brain.picked_patches)
#            brain.label_extract_mode = mode
#            brain.clear_glyphs()
#            for hemi in brain._hemis:
#                for label_id in glyphs[hemi]:
#                    label = brain._annotation_labels[hemi][label_id]
#                    vertex_id = label.vertices[0]
#                    brain._add_label_glyph(hemi, None, vertex_id)
            brain.mpl_canvas.axes.relim()
            brain.mpl_canvas.axes.autoscale_view()
            brain.mpl_canvas.update_plot()
            brain._renderer._update()

#        from ...source_estimate import _get_allowed_label_modes
#        from ...label import _read_annot_cands
        dir_name = op.join(brain._subjects_dir, brain._subject, 'label')
#        cands = _read_annot_cands(dir_name, raise_error=False)
#        cands = cands + ['None']
#        brain.annot = cands[0]
        stc = brain._data["stc"]
#        modes = _get_allowed_label_modes(stc)
#        if brain._data["src"] is None:
#            modes = [m for m in modes if m not in
#                     brain.default_label_extract_modes["src"]]
#        brain.label_extract_mode = modes[-1]
#        if brain.traces_mode == 'vertex':
#            _set_annot('None')
#        else:
#            _set_annot(brain.annot)
#        brain.widgets["annotation"] = brain._renderer._dock_add_combo_box(
#            name="Annotation",
#            value=brain.annot,
#            rng=cands,
#            callback=_set_annot,
#            layout=layout,
#        )
#        brain.widgets["extract_mode"] = brain._renderer._dock_add_combo_box(
#            name="Extract mode",
#            value=brain.label_extract_mode,
#            rng=modes,
#            callback=_set_label_mode,
#            layout=layout,
#        )


    def _configure_dock(brain):
#        brain._renderer._dock_initialize()
 #       brain._configure_dock_playback_widget(name="Playback")
 #       brain._configure_dock_orientation_widget(name="Orientation")
 #       brain._configure_dock_colormap_widget(name="Color Limits")
        _configure_dock_trace_widget(brain,name="Trace")
#        brain._configure_dock_trace_widget(name="Trace")

        # Smoothing widget
#        brain.callbacks["smoothing"] = SmartCallBack(
#            callback=brain.set_data_smoothing,
#        )
#        brain.widgets["smoothing"] = brain._renderer._dock_add_spin_box(
#            name="Smoothing",
#            value=brain._data['smoothing_steps'],
#            rng=brain.default_smoothing_range,
#            callback=brain.callbacks["smoothing"],
#            double=False
#        )
#        brain.callbacks["smoothing"].widget = \
#            brain.widgets["smoothing"]

#        brain._renderer._dock_finalize()

    def setup_time_viewer(brain, time_viewer=True, show_traces=True):
        """Configure the time viewer parameters.
        Parameters
        ----------
        time_viewer : bool
            If True, enable widgets interaction. Defaults to True.
        show_traces : bool
            If True, enable visualization of time traces. Defaults to True.
        Notes
        -----
        The keyboard shortcuts are the following:
        '?': Display help window
        'i': Toggle interface
        's': Apply auto-scaling
        'r': Restore original clim
        'c': Clear all traces
        'n': Shift the time forward by the playback speed
        'b': Shift the time backward by the playback speed
        'Space': Start/Pause playback
        'Up': Decrease camera elevation angle
        'Down': Increase camera elevation angle
        'Left': Decrease camera azimuth angle
        'Right': Increase camera azimuth angle
        """
#        from ..backends._utils import _qt_app_exec
#        if brain.time_viewer:
#            return
        if not brain._data:
            raise ValueError("No data to visualize. See ``add_data``.")
#        brain.time_viewer = time_viewer
#        brain.orientation = list(_lh_views_dict.keys())
        brain.default_smoothing_range = [-1, 15]

        # Default configuration
#        brain.playback = False
#        brain.visibility = False
        brain.refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        brain.default_scaling_range = [0.2, 2.0]
        brain.default_playback_speed_range = [0.01, 1]
        brain.default_playback_speed_value = 0.01
        brain.default_status_bar_msg = "Press ? for help"
        brain.default_label_extract_modes = {
            "stc": ["mean", "max"],
            "src": ["mean_flip", "pca_flip", "auto"],
        }
        brain.default_trace_modes = ('vertex', 'label')
#        brain.annot = None
#        brain.label_extract_mode = None
#        all_keys = ('lh', 'rh', 'vol')
#        brain.act_data_smooth = {key: (None, None) for key in all_keys}
#        brain.color_list = _get_color_list()
        # remove grey for better contrast on the brain
#        brain.color_list.remove("#7f7f7f")
#        brain.color_cycle = _ReuseCycle(brain.color_list)
#        brain.mpl_canvas = None
#        brain.help_canvas = None
#        brain.rms = None
#        brain.picked_patches = {key: list() for key in all_keys}
#        brain.picked_points = {key: list() for key in all_keys}
#        brain.pick_table = dict()
#        brain._spheres = list()
#        brain._mouse_no_mvt = -1
#        brain.callbacks = dict()
#        brain.widgets = dict()
#        brain.keys = ('fmin', 'fmid', 'fmax')

        # Derived parameters:
#        brain.playback_speed = brain.default_playback_speed_value
#        _validate_type(show_traces, (bool, str, 'numeric'), 'show_traces')
#        brain.interactor_fraction = 0.25
        if isinstance(show_traces, str):
            brain.show_traces = True
            brain.separate_canvas = False
            brain.traces_mode = 'vertex'
            if show_traces == 'separate':
                brain.separate_canvas = True
            elif show_traces == 'label':
                brain.traces_mode = 'label'
            else:
                assert show_traces == 'vertex'  # guaranteed above
        else:
            if isinstance(show_traces, bool):
                brain.show_traces = show_traces
            else:
                show_traces = float(show_traces)
                if not 0 < show_traces < 1:
                    raise ValueError(
                        'show traces, if numeric, must be between 0 and 1, '
                        f'got {show_traces}')
                brain.show_traces = True
                brain.interactor_fraction = show_traces
            brain.traces_mode = 'vertex'
            brain.separate_canvas = False
        del show_traces

        brain._configure_time_label()
        brain._configure_scalar_bar()
#        brain._configure_shortcuts()
        brain._configure_picking()
#        brain._configure_tool_bar()
        _configure_dock(brain)
#        brain._configure_dock()
#        brain._configure_menu()
#        brain._configure_status_bar()
        brain._configure_playback()
#        brain._configure_help()
        # show everything at the end
#        brain.toggle_interface()
#        brain._renderer.show()

        # sizes could change, update views
#        for hemi in ('lh', 'rh'):
#            for ri, ci, v in brain._iter_views(hemi):
#                brain.show_view(view=v, row=ri, col=ci)
#        brain._renderer._process_events()

#        brain._renderer._update()
        # finally, show the MplCanvas
#        if brain.show_traces:
#            brain.mpl_canvas.show()
#        if brain._block:
#            _qt_app_exec(brain._renderer.figure.store["app"])


#with autocast('cuda'):
  while True:
   if (FLAGS.from_bdf is None) and (FLAGS.from_edf is None) :

    if raw is None:
      len_raw=0
    else:
      len_raw=len(raw)
    while len_raw<int(sample_rate*duration*2):

     while board.get_board_data_count() > int((sample_rate)/fps): 
     
      if show_inverse:
       if not (brain is None):
        brain.show_view()
     
#    while board.get_board_data_count() > int((sample_rate*5*1/bands[0][0])/fps): 
#    while board.get_board_data_count() > 0: 
# because stream.read_available seems to max out, leading us to not read enough with one read
      data = board.get_board_data()
            #eeg_data.append(data[eeg_channels,:].T)
      eeg_data = data[eeg_channels, :]

      signals = eeg_data
#      print(signals.shape)

      eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE
#      eeg_data.append(data[eeg_channels,:])#.T)
      
      if buf is not None:
        bufs=[{}]*2
        bufs[0] = buf
        bufs[1] = signals
        #raw_picks=[{}]*2
        #raw_picks[0] = raws[0].pick(ch_names_pick)
        #raw_picks[1] = raws[1].pick(ch_names_pick)
        bufs_for_hstack=[{}]*2
        bufs_for_hstack[0] = bufs[0][:][:]
        bufs_for_hstack[1] = bufs[1][:][:]
      else:
        bufs=[{}]*1
        bufs[0] = signals
        #raw_picks=[{}]*1
        #raw_picks[0] = raws[0].pick(ch_names_pick)
        bufs_for_hstack=[{}]*1
        bufs_for_hstack[0] = bufs[0][:][:]

      bufs_hstack = np.hstack(bufs_for_hstack)
#      print(raws_hstack)
#      print(len(raws_hstack))
#      raws_hstack_cut = raws_hstack[:,:]

      buf = bufs_hstack

      if len(bufs_hstack[0])>=int(sample_rate):
        bufs_hstack_cut = bufs_hstack[:,:sample_rate]
      
        buf = bufs_hstack[:,sample_rate:]
        #chech if correct edges
        #print(len(bufs_hstack[0]), len(bufs_hstack_cut[0]), len(buf[0]))

#      if len(bufs_hstack_cut[0])>=int(sample_rate):

#        print(bufs_hstack_cut)
#        print(bufs_hstack_cut.shape)
#        print(bdf)
 
        bdf.writeSamples(bufs_hstack_cut)
#      bdf.blockWriteDigitalSamples(signals)
#      bdf.blockWritePhysicalSamples(signals)


  # Creating MNE objects from brainflow data arrays
      ch_types = ['eeg'] * len(eeg_channels)
#            ch_names = [str(x) for x in range(len(eeg_channels))]
  #ch_names = BoardShim.get_eeg_names(BoardIds.FREEEEG32_BOARD.value)
      sfreq = BoardShim.get_sampling_rate(BoardIds.FREEEEG32_BOARD.value)
      info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
      #eeg = np.concatenate(eeg_data, 0)
      if raw_buf is not None:
        raws=[{}]*2
        raws[0] = raw_buf
        raws[1] = mne.io.RawArray(eeg_data, info, verbose='ERROR')
        raw_picks=[{}]*2
        raw_picks[0] = raws[0].pick(ch_names_pick)
        raw_picks[1] = raws[1].pick(ch_names_pick)
        raws_for_hstack=[{}]*2
        raws_for_hstack[0] = raw_picks[0][:][0]
        raws_for_hstack[1] = raw_picks[1][:][0]
      else:
        raws=[{}]*1
        raws[0] = mne.io.RawArray(eeg_data, info, verbose=50)
        raw_picks=[{}]*1
        raw_picks[0] = raws[0].pick(ch_names_pick)
        raws_for_hstack=[{}]*1
        raws_for_hstack[0] = raw_picks[0][:][0]

      raws_hstack = np.hstack(raws_for_hstack)
#      print(raws_hstack)
#      print(len(raws_hstack))
#      raws_hstack_cut = raws_hstack[:,:]
      raws_hstack_cut = raws_hstack[:,-int(sample_rate*duration*2):]
#      print(raws_hstack_cut)
#      print(len(raws_hstack_cut))

      ch_types_pick = ['eeg'] * len(ch_names_pick)
      info_pick = mne.create_info(ch_names=ch_names_pick, sfreq=sfreq, ch_types=ch_types_pick)
      raw = mne.io.RawArray(raws_hstack_cut, info_pick, verbose='ERROR')
      raw_buf = raw
      len_raw=len(raw)

  # its time to plot something!

#   if True:
#    if True:
	   
   if raw is not None:
    if len(raw)>=int(sample_rate*duration*2):
    
#
     if True:

        datas=[]
 #       for band in range(len(bands)):
# datas.append(raw)
        datas.append(raw.pick(ch_names_pick))

#        if not (FLAGS.from_bdf_file is None):
        raw = None
 
        epochs = []
#        for method in range(len(methods)):
#         for band in range(len(bands)):
        # epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=0.1, preload=False))
        if show_inverse:
          datas[0].set_montage(mon)
          datas[0].set_eeg_reference(projection=True)
          
        epochs.append(mne.make_fixed_length_epochs(datas[0], 
                                            duration=duration, preload=True, overlap=overlap, verbose='ERROR'))
#          epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=5*1/8, preload=False, overlap=5*1/8-0.1))

     if show_inverse:
      if False:
       
#       print(raw1)
       raw1=datas[0]
       print(raw1)
       raw1.set_montage(mon)

       raw1.set_eeg_reference(projection=True)
       #events = mne.find_events(raw)
       #epochs = mne.Epochs(raw, events)
       epochs = mne.make_fixed_length_epochs(raw1, 
                          duration=duration, preload=True, overlap=overlap)#, verbose='ERROR')
       cov = mne.compute_covariance(epochs, tmax=0., n_jobs=10)
#     cov = mne.compute_covariance(epochs, tmax=0.)
       evoked = epochs['1'].average()  # trigger 1 in auditory/left
       evoked.plot_joint()
   
       inv = mne.minimum_norm.make_inverse_operator(
           evoked.info, fwd, cov, verbose=True)
       stc = mne.minimum_norm.apply_inverse(evoked, inv)
#       if not brain is None:

       data_path = mne.datasets.sample.data_path()
       subjects_dir = data_path / 'subjects'
        
       brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1, figure=1, 
           time_viewer=(brain is None))
       
       if False:

           from pynput import mouse

           def on_move(x, y):
               print('Pointer moved to {0}'.format(
                   (x, y)))

           def on_click(x, y, button, pressed):
               print('{0} at {1}'.format(
                   'Pressed' if pressed else 'Released',
                   (x, y)))
               if not pressed:
                   # Stop listener
                   return False

           def on_scroll(x, y, dx, dy):
               print('Scrolled {0} at {1}'.format(
                   'down' if dy < 0 else 'up',
                   (x, y)))

           # Collect events until released
           with mouse.Listener(
                   on_move=on_move,
                   on_click=on_click,
                   on_scroll=on_scroll) as listener:
               listener.join()

           # ...or, in a non-blocking fashion:
           listener = mouse.Listener(
               on_move=on_move,
               on_click=on_click,
               on_scroll=on_scroll)
           listener.start()
       
#       while True:
       brain.show_view()











#     if not show_inverse:
     if True:

      part_len=1
      n_generate=len(epochs[0].events)-2
      n_parts = n_generate//part_len
      if n_generate%part_len>0:
          n_parts=n_parts+1
      n_parts_now = 0
      
      if len(datas[0])==int(sample_rate*duration*2):
        n_parts=1
        part_len=1

      for j in range(n_parts): # display separate audio for each break
       n_parts_now = n_parts_now + 1
       if n_parts_now > 100:#n_parts_one_time:
        break
       for i in range(part_len): # display separate audio for each break
        ji = j * part_len + i
        
#        if (i==0) and (n_generate-ji<part_len):
#            psd_array=np.random.rand((n_generate-ji), dim) 


        if show_inverse:
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmin=0.0, tmax=0.1, n_jobs=10)
            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmax=0., n_jobs=10, verbose=False)
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmin=0.0, tmax=0.1, n_jobs=10)
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmax=0., n_jobs=10)
#     cov = mne.compute_covariance(epochs, tmax=0.)
            evoked = epochs[0][ji].average()  # trigger 1 in auditory/left
            evoked.plot_joint()
   
            inv = mne.minimum_norm.make_inverse_operator(
                  evoked.info, fwd, cov, verbose=True)
            stc = mne.minimum_norm.apply_inverse(evoked, inv)
#       if not brain is None:

            data_path = mne.datasets.sample.data_path()
            subjects_dir = data_path / 'subjects'

#            if not (brain is None):
#                brain.close()        
#                brain.toggle_interface(False)
#            toggle_interface
            if (brain is None):

             if True:
             
             
#    def plot(self, subject=None, surface='inflated', hemi='lh',
#             colormap='auto', time_label='auto', smoothing_steps=10,
#             transparent=True, alpha=1.0, time_viewer='auto',
#             subjects_dir=None,
#             figure=None, views='auto', colorbar=True, clim='auto',
#             cortex="classic", size=800, background="black",
#             foreground=None, initial_time=None, time_unit='s',
#             backend='auto', spacing='oct6', title=None, show_traces='auto',
#             src=None, volume_options=1., view_layout='vertical',
#             add_data_kwargs=None, brain_kwargs=None, verbose=None):
#        brain = plot_source_estimates(
#            self, subject, surface=surface, hemi=hemi, colormap=colormap,
#            time_label=time_label, smoothing_steps=smoothing_steps,
#            transparent=transparent, alpha=alpha, time_viewer=time_viewer,
#            subjects_dir=subjects_dir, figure=figure, views=views,
#            colorbar=colorbar, clim=clim, cortex=cortex, size=size,
#            background=background, foreground=foreground,
#            initial_time=initial_time, time_unit=time_unit, backend=backend,
#            spacing=spacing, title=title, show_traces=show_traces,
#            src=src, volume_options=volume_options, view_layout=view_layout,
#            add_data_kwargs=add_data_kwargs, brain_kwargs=brain_kwargs,
#            verbose=verbose)
#        return brain             
             
#def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
#                          colormap='auto', time_label='auto',
#                          smoothing_steps=10, transparent=True, alpha=1.0,
#                          time_viewer='auto', subjects_dir=None, figure=None,
#                          views='auto', colorbar=True, clim='auto',
#                          cortex="classic", size=800, background="black",
#                          foreground=None, initial_time=None,
#                          time_unit='s', backend='auto', spacing='oct6',
#                          title=None, show_traces='auto',
#                          src=None, volume_options=1., view_layout='vertical',
#                          add_data_kwargs=None, brain_kwargs=None,
#                          verbose=None):             

              brain = stc.plot(
                  hemi='both', 
                  src=inv['src'], 
                  views='coronal',
                  initial_time=0.1, 
                  subjects_dir=subjects_dir,
                  brain_kwargs=dict(silhouette=True), 
                  alpha=0.25,
                  smoothing_steps='nearest',
#                  smoothing_steps=7,
                  show_traces=True, verbose=False)
#                  show_traces=False)
            
#              brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.0, figure=1,
#                 show_traces=False)#, 

              #brain.add_data(array=stc, initial_time=0.0, time=brain._times)
              #brain.apply_auto_scaling()        

#                  time_viewer=(brain is None))
              #brain_data_frame = stc.to_data_frame()
#              time = np.arange(stc.shape[-1])
#              print('time: ',time)
             if False:
              import os
              import numpy as np

              from surfer import Brain
              from surfer.io import read_stc

              print(__doc__)
              
              # define subject, surface and hemisphere(s) to plot:

              subject_id, surf = 'fsaverage', 'inflated'
              hemi = 'lh'

              # create Brain object for visualization
              brain = Brain(subject_id, hemi, surf, size=(400, 400), background='w',
                            interaction='terrain', cortex='bone', units='m')

              # label for time annotation in milliseconds


              def time_label(t):
                  return 'time=%0.2f ms' % (t * 1e3)


              # Read MNE dSPM inverse solution and plot

              for hemi in ['lh']:  # , 'rh']:
#                  stc_fname = os.path.join('example_data', 'meg_source_estimate-' +
#                                           hemi + '.stc')
#                  stc = read_stc(stc_fname)
              
                  # data and vertices for which the data is defined
                  data = stc['data']
                  vertices = stc['vertices']

                  # time points (in seconds)
                  time = np.linspace(stc['tmin'], stc['tmin'] + data.shape[1] * stc['tstep'],
                                     data.shape[1], endpoint=False)

                  # colormap to use
                  colormap = 'hot'

                  # add data and set the initial time displayed to 100 ms,
                  # plotted using the nearest relevant colors
                  brain.add_data(data, colormap=colormap, vertices=vertices,
                                 smoothing_steps='nearest', time=time, time_label=time_label,
                                 hemi=hemi, initial_time=0.1, verbose=False)

              # scale colormap
              brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True,
                                        verbose=False)
              
            else:
             if False:
              for hemi in ['lh']:  # , 'rh']:
#                  stc_fname = os.path.join('example_data', 'meg_source_estimate-' +
#                                           hemi + '.stc')
#                  stc = read_stc(stc_fname)
              
                  # data and vertices for which the data is defined
                  data = stc['data']
                  vertices = stc['vertices']

                  # time points (in seconds)
                  time = np.linspace(stc['tmin'], stc['tmin'] + data.shape[1] * stc['tstep'],
                                     data.shape[1], endpoint=False)

                  # colormap to use
                  colormap = 'hot'

                  # add data and set the initial time displayed to 100 ms,
                  # plotted using the nearest relevant colors
                  brain.add_data(data, colormap=colormap, vertices=vertices,
                                 smoothing_steps='nearest', time=time, time_label=time_label,
                                 hemi=hemi, initial_time=0.1, verbose=False)
              
             if True:
              
#              kwargs = dict(
#                  array=stc.rh_data, hemi='rh', vertices=stc.rh_vertno, fmin=stc.data.min(),
#                  fmax=stc.data.max(), smoothing_steps='nearest', time=brain._times)

#              brain_camera_position=brain._renderer.camera_position

              kwargs = dict(
#                  array=stc, 
                  colormap='plasma', 
                  fmin=stc.data.min(),
                  alpha=0.25,
                  src=inv['src'], 
#                  initial_time=0.1, 
#                  align=False,
#                  focalpoint=brain_camera_position,
#                  fmax=stc.data.max(), smoothing_steps=7, time=brain._times)
                  fmax=stc.data.max(), 
                  fmid=stc.data.min()+(stc.data.max()-stc.data.min())/2, 
                  smoothing_steps='nearest', 
                  time=brain._times, verbose=False)

              # name: works
              add_data(brain, stc.lh_data, hemi='lh', vertices=stc.lh_vertno, **kwargs)
              add_data(brain, stc.rh_data, hemi='rh', vertices=stc.rh_vertno, **kwargs)
#              brain.add_data(stc.lh_data, hemi='lh', vertices=stc.lh_vertno, **kwargs)
#              brain.add_data(stc.rh_data, hemi='rh', vertices=stc.rh_vertno, **kwargs)
#              brain.add_data(colormap='plasma', **kwargs)

              # object: works (not documented)
#              list_of_colors = ['cyan', 'red', 'green']
#              my_cmap = LinearSegmentedColormap.from_list('foo', list_of_colors, N=12)
              #brain.add_data(colormap=my_cmap, **kwargs)            
              #brain.add_data(array=stc, initial_time=0.0, time=brain._times)
              
              #brain.add_data(array=stc, initial_time=0.0, time=brain._times)
              brain.apply_auto_scaling()        
              #print(brain_data_frame)
              #brain_data_frame = stc.to_data_frame()
              #print(brain_data_frame)


#              brain.show_view(view='coronal')
              
              #    kwargs = {
              #        "array": stc,
              #        "colormap": colormap,
              #        "smoothing_steps": smoothing_steps,
              #        "time": times, "time_label": time_label,
              #        "alpha": overlay_alpha,
              #        "colorbar": colorbar,
              #        "vector_alpha": vector_alpha,
              #        "scale_factor": scale_factor,
              #        "initial_time": initial_time,
              #        "transparent": transparent,
              #        "center": center,
              #        "fmin": scale_pts[0],
              #        "fmid": scale_pts[1],
              #        "fmax": scale_pts[2],
              #        "clim": clim,
              #        "src": src,
              #        "volume_options": volume_options,
              #        "verbose": None,
              #        }              
#              time = np.arange(stc.shape[-1])
#              print('time: ',time)
#              stc.time_as_index(time)
#              print('stc.time_as_index(time): ',stc.time_as_index(time))
#              print('brain._times: ',brain._times)
#              print('np.array_equal(time, brain._times): ',np.array_equal(time, brain._times))

#              print('brain._n_times: ',brain._n_times)

              #brain.add_data(array=stc, initial_time=0.1, time=brain._times)
              
              #brain.setup_time_viewer(time_viewer=False, show_traces=False)
              #brain.setup_time_viewer()
#              brain.show()

#            if not (brain is None):
#                brain.close()
#              brain.remove_data()
              brain.plot_time_line()
#              brain.toggle_interface(False)
#              brain.time_viewer=False
#              brain.setup_time_viewer(time_viewer=False, show_traces=False)
              setup_time_viewer(brain)
              #brain.toggle_interface()
#              print(brain.time_viewer)
#              print(brain.mpl_canvas)
#              brain.mpl_canvas.show()
#              brain.mpl_canvas.hide()
#              brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1, figure=1)#, 

#            brain=brain1
              #brain.show_view()


        if show_circle_cons or show_spectrum_cons or sound_cons or show_stable_diffusion_cons or show_stylegan3_cons or show_game_cons:
#        if not show_inverse:

          eeg_step=ji
          #print (f'EEG step: {(eeg_step/3):.1f} s')
          tmin, tmax = 0+(eeg_step/fps), duration+(eeg_step/fps)  # use the first 120s of data

        #ji=0 
        #eeg_step=ji
#        tmin, tmax = 0+(eeg_step/fps), 2+(eeg_step/fps)  # use the first 120s of data
        #tmin, tmax = 0, duration
#        sfreq = raw.info['sfreq']  # the sampling frequency
          for band in range(len(bands)):
           for method in range(len(methods)):
          #fmin=8.
          #fmax=13.
            fmin=bands[band][0]
            fmax=bands[band][1]

          #print(epochs[band][ji:ji+10])
#          print(epochs[band][ji:ji+10].get_data().shape)

#          np_array=np.asarray(epochs[band][ji:ji+10])
#          np_array=epochs[band][ji:ji+10].get_data()
#          freqs=int((bands[band][1]-bands[band][0])/2)
          
#          print(np_array.shape)

          
#          if band == 0:
            mne.set_log_level('CRITICAL')
            con = spectral_connectivity_epochs(
#          con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#            epochs[0], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            epochs[0][ji:ji+4], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            np_array, method=methods[method], n_epochs_used=len(np_array), mode='multitaper', sfreq=sfreq, freqs=freqs,
#            n_nodes=len(epochs[band][0].get_data()), faverage=True, mt_adaptive=True, n_jobs=n_jobs, verbose=50)
              epochs[band][ji:ji+10], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
              fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=cuda_jobs, verbose='CRITICAL')
#          cons=np.roll(cons,1,axis=0)
            cons=np.roll(cons,1,axis=0)
            conmat = con.get_data(output='dense')[:, :, 0]
#          print(conmat.shape)
#          cons[1:,:] = cons[:len(cons),:]
            cons[0]=conmat[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
#          cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          
#          if print_freq_once and not print_freq_once_printed:
#             print(freqs)
#             print_freq_once_printed = True
          #con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
          #  epochs[band][ji,ji+1], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
          #  fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
#          psds_shift1=int(round(method*len(bands)+band)*(len(ch_names)*(len(ch_names)-1)/2))
#          ji1=0
#          for j1 in range(0,len(ch_names)): # display separate audio for each break
#            for i1 in range(0,j1): # display separate audio for each break
#              psds[ji1+psds_shift1]=(con[j1][i1][0]-0.5)*1
#              ji1 = ji1+1
          cons = cons[-int(len(cons[0])):]
        #psds, freqs = mne.time_frequency.psd_welch(raw, picks=picks,
        #                 tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
        #                 n_fft=n_fft)
        #logger.disabled = False

        #print(freqs)
        #print(psds)
        
#        psd_array[i]=psds
        ##z_samples = psds

        #w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        #w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        ##z_avg = np.mean(z_samples, axis=0)      # [1, 1, C]
        #z_avg = np.mean(z_samples, axis=0, keepdims=True)      # [1, 1, C]
        ##psd_array[i]=z_avg
        #psd_array[i]=z_avg
        #print(z_avg)
        #z_std = (np.sum((z_samples - z_avg) ** 2) / z_avg_samples) ** 0.5

        #psd_array[i]=psds
        #psds_transpose=np.transpose(psds)
        #plt.plot(freqs,psds_transpose)
        #plt.xlabel('Frequency (Hz)')
        #plt.ylabel('PSD (dB)')
        #plt.title('Power Spectrum (PSD)')
        #plt.show()
#        if (i==part_len-1) or (ji==n_generate-1) :
#        fig = plt.figure()
#        if True:
        if show_circle_cons:
#          con_res = dict()
#          for method, c in zip(methods, con):
#            con_res[method] = c[:, :, 0]
#            con_res[method] = c[:, :]
         for ii, method in enumerate(methods):
#            fig,_ = plot_connectivity_circle(con_res[method], label_names, n_lines=300, 

#            plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
#                                             title=method, show = False)
            #px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            #fig = plt.figure(figsize=(576*px, 576*px))
#            fig,_ = 

#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
#                                             title=method, show = False, vmin=0, vmax=1)#, fig=fig)
            con_sort=np.sort(np.abs(conmat).ravel())[::-1]
#            con_sort=np.sort(np.abs(con).ravel())[::-1]
            n_lines=np.argmax(con_sort<vmin)
               
#            print(freqs)
            fig,ax = plot_connectivity_circle(conmat, label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names,# n_lines=300, 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2), 
                title=input_fname_name+'_circle_'+methods[0]+'_'+f'{bands[0][0]:.1f}'+'-'+f'{bands[0][len(bands[0])-1]:.1f}'+'hz_'+'vmin'+str(vmin), 
#                title=input_fname_name+'_circle_'+methods[0]+'_'+f'{freqs[0][0]:.1f}'+'-'+f'{freqs[0][len(freqs[0])-1]:.1f}'+'hz_'+'vmin'+str(vmin), 
#               title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin'+str(vmin)+str(len(epochs[0].events)-2)+'\n'+str(ji), 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2)+'_'+str(ji), 
                                             show = False, vmin=vmin, vmax=1, fontsize_names=8)#, fig=fig)
#                                             show = False, vmin=0, vmax=1, fontsize_names=8)#16)#, fig=fig)
            #ax.set_theta_offset(np.deg2rad(360/len(label_names)))
#            fig = plot_sensors_connectivity(raw.info, con[:, :, 0], picks=label_names, cbar_label=method)
          #plot_conmat_file = os.path.abspath('circle_' + fname + '.eps')
          #fig.savefig(plot_conmat_file, facecolor='black')

          #plt.close()
          #plt.close(fig)
          #fig.clf()
          
          #del fig

          #if ji%100==0 :
          #  gc.collect()

         if False:
#        if True:

          #plt.close(fig)
          ##fig1.close()
          #del fig

          #plt.rcParams["figure.figsize"] = [7.50, 3.50]
          #plt.rcParams["figure.autolayout"] = True

          #plt.figure()
          #plt.plot([1, 2])

          img_buf = io.BytesIO()
          #plt.savefig(img_buf, format='png')

          #fig.savefig(img_buf, facecolor='black', format='png')
          #fig.savefig('/content/out/img_buf.png', facecolor='black', format='png')

          #fig.clf()
          #plt.close()
          #ax.cla()

          #import matplotlib.transforms
          #plt.savefig(img_buf, facecolor='black', format='png', 
          #            bbox_inches=matplotlib.transforms.Bbox([[100, 100], [100, 100]]))
          #fig.set_size_inches(8.2, 8.11)

#          plt.savefig(img_buf, facecolor='black', format='png', bbox_inches='tight')
          plt.savefig(img_buf, facecolor='black', format='png')#, bbox_inches='tight')
          #plt.savefig('/content/out/img_buf.png', facecolor='black', format='png')

          #plt.close()
          plt.close(fig)
          #fig.clf()

          # Clear the current axes.
          #plt.cla() 
          # Clear the current figure.
          #plt.clf() 
          # Closes all the figure windows.
          #plt.close('all')   
          #plt.close(fig)
          #gc.collect()
          del fig
          #del ax

          #if ji%100==0 :
          #  gc.collect()

          img_buf.seek(0)

          im1 = Image.open(img_buf)

#          size = 432
          size = 592
          #im3 = im1.resize((576, 576), Image.ANTIALIAS)
          left=348-size/2
          top=404-size/2
 
          im2 = im1.crop((left, top, left+size, top+size))
#          im2 = im1.crop((35, 70, 35+size, 70+size))
          img_buf1 = io.BytesIO()
          im2.save(img_buf1, format='png')
          img_buf1.seek(0)

          im = imageio.imread(img_buf1)
          #out.append_data(im)
          img_buf.close()
          img_buf1.close()

#          im = imageio.imread(img_buf)
#          img_buf.close()

          #print(im.shape)
          #(412, 399, 4)
#          im_arr = np.array(im)
          #print(im_arr.shape)
#          #im_arr.reshape(416, 416, 4)
#          im_arr_rot = np.rot90(im_arr)
          #im_arr_rot.resize((400, 416))

          out.append_data(im)
#          out.append_data(im_arr_rot)
#          imageio.imwrite('/content/out/img_buf.png', im_arr_rot, format='png')
          imageio.imwrite('/content/out/img_buf.png', im, format='png')


         if True:

            fig.canvas.draw()

            #image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = np.frombuffer(fig.canvas.tostring_rgb(),'u1')  
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            size1=16*4
            size = 592+size1
#            size = 608
            #im3 = im1.resize((576, 576), Image.ANTIALIAS)
            left=348-int(size/2)+int(size1/2)
            top=404-int(size/2)+int(size1/16)

            image_crop=image[top:top+size,left:left+size]   
            #im2 = im1.crop((left, top, left+size, top+size))

            if FLAGS.rotate:
              image_rot90 = np.rot90(image_crop)
              image=image_rot90
#              screen.update(image_rot90)
            else:
              image=image_crop
#            image_rot90 = np.rot90(image)

#            screen.update(image)
#              screen.update(image_crop)

            image = image[:,:,::-1]
            screen.update(image)

            plt.close(fig)
            del fig


        if False:
            fig = raw.plot_psd(average=False, show=False)  
#            plt.show(block=False)

#        if False:
        if show_spectrum_cons or sound_cons:
#fig.tight_layout(pad=0)
#fig.canvas.draw()



#          cons.append(con.flatten('F'))
#          cons.append(con[np.tril_indices(len(con[0]))])
#          np.append(cons,con[np.tril_indices(len(con[0]))])
          
#          print(np.tril_indices(len(con[0]),k=-1))

#          cons[cons_index]=con[np.tril_indices(len(con[0]),k=-1)].flatten('F')
          #print(con[np.tril_indices(len(con[0]),k=-1)].flatten('F'))
          #print(np.tril_indices(len(con[0]),k=-1))
          #print(con[cohs_tril_indices].flatten('F'))
          #print((cohs_tril_indices[0],cohs_tril_indices[1]))
          #from scipy.ndimage.interpolation import shift
          #shift(cons, -1)#, cval=np.NaN)
##          cons=np.roll(cons,1,axis=0)
#          cons[1:,:] = cons[:len(cons),:]
##          cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          #cons[cons_dur-cons_index-1]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          #cons_index_current=cons_index
          #cons_index=cons_index+1
          #if cons_index>=cons_dur:
          #  cons_index=0
          #cons = cons[-int(len(cons[0])):]

          if show_spectrum_cons:
#          if False:
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig = plt.figure(figsize=(800*px, 800*px))

#          plt.imshow(cons, extent=[0,4.2,0,int(32*(32-1)/2)], cmap='jet',
#             vmin=-100, vmax=0, origin='lower', aspect='auto')
            plt.imshow(cons.T, cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)
#            plt.imshow(cons.T[:,::-1], cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)
            plt.colorbar()
#          plt.show()
            plt.close()
          #fig.canvas.draw()

#        if False:
#        if True:
            fig.canvas.draw()

            #image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = np.frombuffer(fig.canvas.tostring_rgb(),'u1')  
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            size1=16*4
            size = 592+size1
#            size = 608
            #im3 = im1.resize((576, 576), Image.ANTIALIAS)
            left=348-int(size/2)+int(size1/2)
            top=404-int(size/2)+int(size1/16)

            image_crop=image[top:top+size,left:left+size]   
            #im2 = im1.crop((left, top, left+size, top+size))

#            image_rot90 = np.rot90(image_crop)
#            image_rot90 = np.rot90(image)

#            screen.update(image)
            image = image_crop
            image = image[:,:,::-1]
            screen2.update(image)
#            screen.update(image_rot90)

            plt.close(fig)
            del fig
 
        if sound_cons:
          #import stft

#          spectrum = stft.stft(y, 128)
#          back_y = stft.istft(spectrum, 128)

#          y, sr = librosa.load(librosa.example('brahms'), duration=5.0)
#          y, sr = librosa.load(librosa.example('brahms'), offset=cons_index/10, duration=5.0)

#          y, sample_rate = load('/content/out/1.wav', duration=5.0)
#          spectrum = stft(y)
#          spectrum_abs=np.abs(spectrum)
#          spectrum_db=librosa.amplitude_to_db(spectrum,ref=np.max)
#          back_y = istft(spectrum)

          if False:

            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig = plt.figure(figsize=(800*px, 800*px))

#          cons[cons_index]=spectrum_db[-int(len(cons[0])):]
            cons_index_current=cons_index
            cons_index=cons_index+1
            if cons_index>=cons_len:
              cons_index=0

            plt.imshow(spectrum_db, cmap='jet', origin='lower', aspect='auto')
#          plt.imshow(cons, cmap='jet', origin='lower', aspect='auto')
            plt.colorbar()
#          plt.show()
            plt.close()

 
#          for spectrum_db_range in range(spectrum_db):
#           spectrum_db=cons[cons_index] 

#          spectrum_db=np.abs(cons.T[:,::-1])
          spectrum_db=np.abs(cons.T)
          spectrum_db_l=spectrum_db[:int(len(spectrum_db)/2)]
          spectrum_db_r=spectrum_db[-int(len(spectrum_db)/2):]
          spectrum_db_r=spectrum_db_r[::-1]
#          spectrum_db_s=[spectrum_db_l,spectrum_db_r]
          spectrum=librosa.db_to_amplitude(spectrum_db)
          if sound_cons_swap:
            spectrum_r=librosa.db_to_amplitude(spectrum_db_l)
            spectrum_l=librosa.db_to_amplitude(spectrum_db_r)
          else:
            spectrum_l=librosa.db_to_amplitude(spectrum_db_l)
            spectrum_r=librosa.db_to_amplitude(spectrum_db_r)
#          back_y = stft.istft(spectrum, 128)
          back_y = istft(spectrum)*audio_volume_mult
          back_y_l = istft(spectrum_l)
          back_y_r = istft(spectrum_r)
          back_y_s = np.asarray([back_y_l,back_y_r]).T*audio_volume_mult

#          sf.write('/content/out/stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24')
#          sf.write('/content/out/file_trim_5s.wav', y, sr, 'PCM_24')
#          sf.write('/content/out/file_trim_5s_back.wav', back_y, sr, 'PCM_24')
#          sr=48000
#          sr=44100
#          sr=22050
          #sr=11025
#          sr=int(48000/10)
#          sr=int(48000/20)
#          sr=cons_len
          sr=4000
          sf.write(sound_cons_buffer_path+'cons_back.wav', back_y, sr, 'PCM_24')
          sf.write(sound_cons_buffer_path+'cons_back_s.wav', back_y_s, sr, 'PCM_24')
          filename=sound_cons_buffer_path+'cons_back_s.wav'
#          device=
          #print(sd.query_devices())
#          try:
          if False:
#          if True:
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data, fs)#, device=device)
          if True:
            data=back_y_s
            fs=audio_cons_fs
            sd.play(data, fs)#, device=device)

            #mydata = sd.rec(int(data),fs,channels=2, blocking=True)
            #sf.write(filename, data, fs)


            #status = sd.wait()
#          except KeyboardInterrupt:
#            parser.exit('\nInterrupted by user')
#          except Exception as e:
#            parser.exit(type(e).__name__ + ': ' + str(e))
#          if status:
#            parser.exit('Error during playback: ' + str(status))          

          #from librosa import output
          #librosa.output.write_wav('/content/out/file_trim_5s.wav', y, s_r)
#          librosa.output.write_wav('/content/out/file_trim_5s_back.wav', back_y, sample_rate)
             

#            fig.show()
#            fig.show(0)

#            draw() 
#print 'continuing computation'
#            show()

        if False:    
         if show_stable_diffusion_cons:
          cv2.imshow('test draw',img)
          if cv2.waitKey(1) & 0xFF == 27:
              break
          
          if False:    
#          print(img)
#          print(img.shape)
          #img.reshape(64, 64, 4)
            image_pil=PIL.Image.fromarray(img, 'RGB')
            image_pil_resize=image_pil.resize((64,64),PIL.Image.Resampling.LANCZOS)
            image_asarray=np.asarray(image_pil_resize)
#          print(image_asarray.shape)
            image_resize=np.resize(image_asarray,(64, 64, 4))
            image_resize_rearranged = np.transpose(image_resize, axes=[2, 0, 1])
#          print(image_resize_rearranged.shape)
            image_resize_rearranged.reshape(1, 4, 64, 64)
#          print(image_resize_rearranged.shape)
#          image_resize_rearranged=image_resize_rearranged/255
            img_latents=torch.from_numpy(image_resize_rearranged)

            test_draw_latents=img_latents
          
          if False:    
            test_draw_latents=encode_vae(Image.fromarray(img))
#            unet_img=decode_latents(unet_latents)
#            unet_draw_latents=encode_vae(Image.fromarray(unet_img))
          #unet_img=decode_latents(unet_latents)
          #unet_img=decode_latents(unet_latents)
          
#          unet_and_test_draw_latents=test_draw_latents*unet_draw_latents
          if False:    
            unet_and_test_draw_latents=test_draw_latents*unet_latents


        if show_stable_diffusion_cons:
          t_tqdm=scheduler.timesteps[i_tqdm]
#  i_tqdm1=i_tqdm
#  t_tqdm1=t_tqdm

          if True:    
#          if False:    
##            cons=np.roll(cons,1,axis=0)
#            cons[1:,:] = cons[:len(cons),:]
##            cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
#            print('_')
#            car_latent = encode_vae(car_img)
#            car_latent.shape

#            base_latents = test_latents.detach().clone()
#            base_latents = latents.detach().clone()

#            if True:    
            if False:    
#              text_embeddings1=test_embeds.detach().clone()
#              height1=512
#              width1=512
#              num_inference_steps1=50
#              guidance_scale1=7.5
#              if i_tqdm==0:
#                latents1=latents.detach().clone()
#              if i_tqdm<num_inference_steps-1:
#                i_tqdm1=-1
#              if i_tqdm==num_inference_steps-1:
#              print((latentsa[i_tqdm]))
#              print((latentsa[i_tqdm] is None))
#              if not (latentsa[i_tqdm] is empty):
              base_latents=latentsa[i_tqdm].detach().clone()
              #base_latents.to('cpu')
            if True:    
             base_latents = unet_latents.detach().clone()
#            base_latents = unet_and_test_draw_latents.detach().clone()
            
#            cons_latents = base_latents
            cons_latents_flatten = base_latents.reshape(len(base_latents)*len(base_latents[0])*len(base_latents[0][0])*len(base_latents[0][0][0]))
            for cons_index in range(int(len(cons_latents_flatten)/len(cons[0]))+1):
              for con_index in range(len(cons[0])):
               if con_index + cons_index*len(cons[0]) < len(cons_latents_flatten):
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1-random.randint(0, 10)/200
                cons_latents_flatten[con_index + cons_index*len(cons[0])] = ((cons[cons_index%len(cons)][con_index]+apply_to_latents)/1+0.0001)/((1+apply_to_latents)/1+0.0001)
            cons_latents = cons_latents_flatten.reshape(len(base_latents),len(base_latents[0]),len(base_latents[0][0]),len(base_latents[0][0][0]))

          if True:    
#          if False:    

#            base_embeds = torch.randn(2, 77, 768).to(device)
            base_embeds = test_embeds.detach().clone()
            #base_embeds = car_embeds.detach().clone()
            cons_embeds_flatten = base_embeds.reshape(2*77*768)
#            print(int(len(cons_latent_flatten)/len(cons[0])))
            for cons_index in range(int(len(cons_embeds_flatten)/len(cons[0]))+1):
              for con_index in range(len(cons[0])):
               if con_index + cons_index*len(cons[0]) < len(cons_embeds_flatten):
#                cons_latent_flatten[con_index + cons_index*len(cons[0])] = (cons[cons_index][con_index])*1
#                cons_embeds_flatten[con_index + cons_index*len(cons[0])] = 1-random.randint(0, 10)/20
#                cons_embeds_flatten[con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index])/1
#                cons_embeds_flatten[con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)/2.5
                cons_embeds_flatten[con_index + cons_index*len(cons[0])] = ((cons[cons_index%len(cons)][con_index]+apply_to_embeds)/1+0.0001)/((1+apply_to_embeds)/1+0.0001)
#                cons_embeds_flatten[con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)/10
#                cons_latent_flatten[con_index + cons_index*len(cons[0])] = (cons[0][con_index])*100
#                cons_latent_flatten[con_index + cons_index*len(cons[0])] = (cons[0][con_index]-0.5)*3
            cons_embeds = cons_embeds_flatten.reshape(2, 77, 768)
            
          if True:    
#          if False:    

            if to_sum_latents is None:
#            if to_sum_embeds is None:
#                base_latents = test_latents.detach().clone()
#            cons_img=Image.fromarray(cons)
#            cons_img_resize=cons_img.resize((400, 416))
#            cons_latent = encode_vae(cons_img_resize)
                #print(car_latent)
#            print(cons_latent)
            
#                unet_latents
#                cons_latents
#                unet_latents.to(device)
#                cons_latents.to(device)
                
                to_sum_latents = unet_latents/cons_latents
#                to_sum_latents = unet_latents.to(device)/cons_latents.to(device)
#                to_sum_embeds = test_embeds-cons_embeds
#                if False:    
                if True:    
                  to_sum_embeds = test_embeds/cons_embeds
#                to_sum_embeds = car_embeds/cons_embeds
            
#            cons_latents.to(device)
#            cons_latents
#            sum_latents = cons_latents.to(device)*to_sum_latents.to(device)
            sum_latents = cons_latents*to_sum_latents
#            sum_latents = latents.to(device)
#            sum_embeds = cons_embeds+to_sum_embeds
#            if False:    
            if True:    
              sum_embeds = cons_embeds*to_sum_embeds
#            sum_embeds = test_embeds
            #sum_embeds = cons_embeds
##            sum_embeds = test_embeds
#            sum_latent = car_latent
##            test_latents = generate_latents(sum_embeds)
#            test_latents = generate_latents(test_embeds)

           
#            latents = sum_latents

          if True:    

#            if True:    
            if False:

#              text_embeddings1=sum_embeds.detach().clone()
              text_embeddings=test_embeds
#              text_embeddings1=test_embeds.detach().clone()
#              latents1=sum_latents
#              latents1=latentsa[i_tqdm].detach().clone()
              latents=latentsa[i_tqdm].detach().clone()
#              height1=512
#              width1=512
#              num_inference_steps1=50
              guidance_scale1=unet_guidance_scale
#              if i_tqdm==0:
#                latents1=latents.detach().clone()
#              if i_tqdm<num_inference_steps-1:
#                i_tqdm1=-1
#              if i_tqdm==num_inference_steps-1:
#              print((latentsa[i_tqdm]))
#              print((latentsa[i_tqdm] is None))
#              if not (latentsa[i_tqdm] is empty):
#              latents1=latentsa[i_tqdm].detach().clone()
#              latents1=unet_latents.detach().clone()
#              latents1=None
#              if latents1 is None:
#                  latents1 = torch.randn((
#                      text_embeddings1.shape[0] // 2,
#                      unet.in_channels,
#                      height1 // 8,
#                      width1 // 8
#                  ))
            
#              latents1 = latents1.to(device)

#              scheduler1.set_timesteps(num_inference_steps1)
#              latents1 = latents1 * scheduler1.sigmas[0]

              if True:    
               with autocast('cuda'):
#                  for i1, t1 in tqdm(enumerate(scheduler1.timesteps)):
                      i1=i_tqdm
                      t1=t_tqdm
#                      scheduler1=scheduler
                      latent_model_input = torch.cat([latents] * 2)
                      sigma = scheduler.sigmas[i1]
                      latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
#                      print(i1,t1,sigma1)

                      with torch.no_grad():
                          noise_pred = unet(latent_model_input, t1, encoder_hidden_states=text_embeddings)['sample']
            
                      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                      noise_pred = noise_pred_uncond + guidance_scale1 * (noise_pred_text - noise_pred_uncond)

                      latents = scheduler.step(noise_pred, i1, latents)['prev_sample']
                      
#                      latents=latents1.detach().clone()
#                      scheduler=scheduler1

                      if True:    

                        i_tqdm=i_tqdm+1
                        if i_tqdm<num_inference_steps:
#                          if (latentsa[i_tqdm] is empty):
#                            latentsa[i_tqdm]=latents.detach().clone()
                            latentsa[i_tqdm]=latents.detach().clone()
                        if i_tqdm==num_inference_steps:
#                          i_tqdm=random.randint(0,num_inference_steps-1)
                          i_tqdm=1
#                          i_tqdm=0
#                          for i in range(i_tqdm+1,len(latentsa)):
#                            latentsa[i]=empty
                          

                        test_latents = latents
#                        test_latents = latents.detach().clone()
#                       test_latents = latents1.detach().clone()

                      if False:    

                        images = decode_latents(test_latents.to(device))
                        images[0].save(output_path+'stable-diffusion-'+dt_string+'-'+FLAGS.clip_prompt+'.png', format='png')
                        images[0].save('mygraph.png', format='png')
                        image = np.asarray(images[0])
                        image = image[:,:,::-1]
                        screen3.update(image)

#              test_latents = latents1.detach().clone()
#              test_latents = latents.detach().clone()

#            if True:    
            if False:
            
              latent_model_input = torch.cat([latents] * 2)
              sigma = scheduler.sigmas[i_tqdm]
              latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

              with torch.no_grad():
                  noise_pred = unet(latent_model_input, t_tqdm, encoder_hidden_states=test_embeds.clone())['sample']
            
              noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
              noise_pred = noise_pred_uncond + unet_guidance_scale * (noise_pred_text - noise_pred_uncond)

              latents = scheduler.step(noise_pred, i_tqdm, latents)['prev_sample']
              test_latents = latents.clone()

#            sum_embeds = test_embeds
#            sum_latents = unet_latents

            if True:    
#            if False:

              test_latents = generate_latents(
                  sum_embeds,
#                  test_embeds,
                  height=unet_height, 
                  width=unet_width,
                  num_inference_steps=unet_num_inference_steps,
#                  latents=unet_latents,
#                  latents=cons_latents,
                  latents=sum_latents,
                  guidance_scale=unet_guidance_scale)

            images = decode_latents(test_latents.to(device))
            #images = decode_latents(car_latent.to(device))
            #print(images[0])
            images[0].save(output_path+'stable-diffusion-'+dt_string+'-'+FLAGS.clip_prompt+'.png', format='png')
            images[0].save('mygraph.png', format='png')
#            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
#            fig = plt.figure(figsize=(800*px, 800*px))
#            plt.imshow(images[0])
#            plt.close()
#            fig.canvas.draw()
#            image = np.frombuffer(fig.canvas.tostring_rgb(),'u1')  
#            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = np.asarray(images[0])
            image = image[:,:,::-1]
            screen3.update(image)
            
            #def on_move(x, y):
            #    print ("Mouse moved to ({0}, {1})".format(x, y))
            #    to_sum_latent = car_latent - cons_latent
            #    listener.stop()    

            #def on_click(x, y, button, pressed):
            #    if pressed:
            #        print ('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
            #        to_sum_latent = car_latent - cons_latent
            #    listener.stop()    

            #def on_scroll(x, y, dx, dy):
            #    print ('Mouse scrolled at ({0}, {1})({2}, {3})'.format(x, y, dx, dy))
            #    to_sum_latent = car_latent - cons_latent
            #    listener.stop()    

            #with Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
            #    listener.join()
    
            
        
#            try:  # used try so that if user pressed other than the given key error will not be shown
#              if keyboard.is_pressed(' '):  # if key ' ' is pressed 
#                  to_sum_latent = car_latent - cons_latent
#              if keyboard.is_pressed('q'):  # if key 'q' is pressed 
#                  break  # finishing the loop
#            except:
#              break  # if user pressed a key other than the given key the loop will break



        if show_stylegan3_cons:

#                dim_sg2=512
                sg3_latents=np.random.rand((1), G3ms[0].z_dim) 
                vol=1

                base_latents = sg3_latents#.detach().clone()
#            cons_latents = base_latents
                cons_latents_flatten = base_latents.reshape(len(base_latents[0]))
                for cons_index in range(int(len(cons_latents_flatten)/len(cons[0]))+1):
                  for con_index in range(len(cons[0])):
                    if con_index + cons_index*len(cons[0]) < len(cons_latents_flatten):
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1-random.randint(0, 10)/200
                      cons_latents_flatten[con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)
#                      cons_latents_flatten[con_index + cons_index*len(cons[0])] = ((cons[cons_index%len(cons)][con_index]+apply_to_latents)/1+0.0001)/((1+apply_to_latents)/1+0.0001)
                cons_latents = cons_latents_flatten.reshape(1,len(base_latents[0]))

 #               device = torch.device('cuda')
 
#        if hasattr(G.synthesis, 'input'):
#            m = make_transform(translate, rotate)
#            m = np.linalg.inv(m)
#            G.synthesis.input.transform.copy_(torch.from_numpy(m))

#                z = psd_array_sg2 * vol  
#                seed=1
                z = torch.from_numpy(cons_latents).to(device)
#                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G3m.z_dim)).to(device)
                truncation_psi=1
#                truncation_psi=0.5
#                noise_mode='const'
#                noise_mode='random'
                noise_mode='none'
                label = torch.zeros([1, G3ms[0].c_dim], device=device)
                #if G3m.c_dim != 0:
                #    label[:, class_idx] = 1
                
                img = G3ms[0](z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
                
                images=[img[0].cpu().numpy() ]   
 
#                z_samples = psd_array_sg2 * vol
#                w_samples = G3m.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
#                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
#                w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
#                w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
#                ws3m = (w_opt).repeat([1, G3m.mapping.num_ws, 1])
 
#                synth_images = G3m.synthesis(ws3m, noise_mode='const')
#                synth_images = (synth_images + 1) * (255/2)
#                synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                #out.append_data(synth_images)
#                images=[synth_images]   
                
                if True:
                
                  xsize=1024
                  ysize=1024
#                  xsize=512
#                  ysize=512

                  image_pil=PIL.Image.fromarray(images[0], 'RGB')
                  #if generate&gen_sg2_shawwn:
                  #  display(image_pil)
                  #print(image_pil)
                  image_asarray=np.asarray(image_pil)
                  #print(image_asarray)
#                  time1111=perf_counter()
                  #print (f'1111: {(time1111-time000):.1f}s')
                  #global video_out
                  #video_out.append_data(image_asarray)
#                  time1112=perf_counter()
                  #print (f'1112: {(time1112-time000):.1f}s')
                  img=image_pil.resize((xsize,ysize),PIL.Image.Resampling.LANCZOS)
                  #print(img)
#                  time1113=perf_counter()
                  #print (f'1113: {(time1113-time000):.1f}s')
#                  buffer = BytesIO()
#                  if generate&gen_jpeg:
#                    img.save(buffer,format="JPEG")                  #Enregistre l'image dans le buffer
#                  if generate&gen_png:
#                  img.save(buffer,format="PNG")                  #Enregistre l'image dans le buffer
                  #img.save('/content/gdrive/MyDrive/EEG-GAN-audio-video/out/'+
                  #          f'{(time100*1000):9.0f}'+'/'+f'{(time000*1000):9.0f}'+'.png',format="PNG")
                        
#                  buffer.seek(0)
 #                 time1114=perf_counter()
                  #print (f'1114: {(time1114-time000):.1f}s')
#                  myimage = buffer.getvalue()   


                  if draw_fps:

                    time111=perf_counter()
                    draw_fps=f'fps: {1/(time111-time001):3.2f}'
                    #print (f'fps: {1/(time111-time001):.1f}s')
                    #print (f'111-001: {(time111-time001):.1f}s')
                  
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), draw_fps, font=font, fill='rgb(0, 0, 0)', stroke_fill='rgb(255, 255, 255)', stroke_width=1)
                    img = draw._image
                    time001=time111


                  image = np.asarray(img)
                  image = image[:,:,::-1]
                  screen4.update(image)

        if show_game_cons:

            if True:
                for cons_index in range(int(len(cons_latentsa[0][0])/len(cons[0]))+1):
                  for con_index in range(len(cons[0])):
                    if con_index + cons_index*len(cons[0]) < len(cons_latentsa[0][0]):
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1-random.randint(0, 10)/200
#                      if FLAGS.game_mode=='1':
                      cons_latentsa[0][0][con_index + cons_index*len(cons[0])] = -(cons[cons_index%len(cons)][con_index]-0.5)
                      cons_latentsa[1][0][con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)
                      if FLAGS.game_mode=='3':
                        if con_index<len(cons[0])/3:
                          cons_latentsa[2][0][con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)
                          cons_latentsa[3][0][con_index + cons_index*len(cons[0])] = 0.
                          cons_latentsa[4][0][con_index + cons_index*len(cons[0])] = 0.
                        elif con_index<len(cons[0])*2/3:
                          cons_latentsa[2][0][con_index + cons_index*len(cons[0])] = 0.
                          cons_latentsa[3][0][con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)
                          cons_latentsa[4][0][con_index + cons_index*len(cons[0])] = 0.
                        else:
                          cons_latentsa[2][0][con_index + cons_index*len(cons[0])] = 0.
                          cons_latentsa[3][0][con_index + cons_index*len(cons[0])] = 0.
                          cons_latentsa[4][0][con_index + cons_index*len(cons[0])] = (cons[cons_index%len(cons)][con_index]-0.5)
#                      cons_latents_flatten[con_index + cons_index*len(cons[0])] = ((cons[cons_index%len(cons)][con_index]+apply_to_latents)/1+0.0001)/((1+apply_to_latents)/1+0.0001)



                game_out=0
                #print('con:',con[0])
                #print('game_in')
                #print('game_last_possible_cards:',game_last_possible_cards)
                for j1 in range(0,dim_sg2):  
                  #print('j1:',j1)
                  game_last_possible_cards[game_cur_possible_cards][j1]=cons_latentsa[1][0][j1]+0.5
                game_cur_possible_cards=game_cur_possible_cards+1
                
                if game_cur_possible_cards>0:
                  for i1 in range(game_cur_possible_cards):
                    np.copyto(game_compare_with_possible_cards[i1],game_last_possible_cards[i1])
                  for i2 in range(game_num_user_cards):
                    np.copyto(game_compare_with_possible_cards[game_cur_possible_cards],game_user_cards[i2])
                    game_user_stddev_compare_with_possible_cards[i2]=np.std(game_compare_with_possible_cards[:game_cur_possible_cards+1], axis=0)
                    for j2 in range(game_num_enemy_cards):
                      for j1 in range(0,dim_sg2):
                        game_user_attack_enemy_cards_possible[i2][j2][j1]=((game_user_cards[i2][j1]/(game_enemy_cards[j2][j1]+1))*(1))/game_num_enemy_cards
                        game_user_attack_enemy_cards[i2][j2][j1]=((game_user_cards[i2][j1]/(game_enemy_cards[j2][j1]+1))*(1-game_user_stddev_compare_with_possible_cards[i2][j1]))/game_num_enemy_cards
                
                if game_cur_possible_cards==game_max_possible_cards:
                  game_cur_possible_cards=0
                game_num_possible_cards=game_num_possible_cards+1
                if game_num_possible_cards>game_max_possible_cards:
                  game_num_possible_cards=game_max_possible_cards
                #print('game_cur_possible_cards:',game_cur_possible_cards)

                if game_cur_possible_cards==0:
                  for i1 in range(game_max_possible_cards):
                    np.copyto(game_compare_with_possible_cards[i1],game_last_possible_cards[i1])
                  if game&game_user_attack_enemy:
                    if game_text:
                      print('game_user_attack_enemy')
                    game_out = game_out | game_out_user_attack
                    for i2 in range(game_num_user_cards):
                      np.copyto(game_compare_with_possible_cards[game_num_possible_cards],game_user_cards[i2])
                      game_stddev_compare_with_possible_cards=np.std(game_compare_with_possible_cards, axis=0)
                      #print(np.array2string(((game_stddev_compare_with_possible_cards)*10).astype(int),separator='',max_line_width=130))
                      #print('game_stddev_last_possible_cards:',game_stddev_last_possible_cards)
                      #print('max(game_stddev_last_possible_cards):',max(game_stddev_last_possible_cards))
                      #print('game_stddev_compare_with_possible_cards:',game_stddev_compare_with_possible_cards)
                      #if np.max(game_stddev_compare_with_possible_cards)<0.2:
                      for j2 in range(game_num_enemy_cards):
                        #print(i2,j2)
                        for j1 in range(0,dim_sg2):
                          game_user_attack_enemy_cards[i2][j2][j1]=((game_user_cards[i2][j1]/(game_enemy_cards[j2][j1]+1))*(1-game_stddev_compare_with_possible_cards[j1]))/game_num_enemy_cards
                          game_enemy_cards_life[j2][j1]=game_enemy_cards_life[j2][j1]-game_user_attack_enemy_cards[i2][j2][j1]
                            #if game_enemy_cards_life[j2][j1]<0:
                            #  game_enemy_cards_life[j2][j1]=0
                            #coh[j1]=game_enemy_cards_life[j2][j1]
                    #print('np.max(game_enemy_cards_life):',np.max(game_enemy_cards_life))
                  #print('game_enemy_cards_life:',game_enemy_cards_life)
                  #print('np.sum(game_enemy_cards_life,axis=0):',np.sum(game_enemy_cards_life,axis=0))
                  if np.max(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0))<=0:
                  #if np.max(game_enemy_cards_life)<=0:
                    game = game_enemy_add_card
                    if game_num_enemy_cards>0:
                      game_killed_enemy_cards=game_killed_enemy_cards+1
                      #game_killed_enemy_cards=game_killed_enemy_cards+game_num_enemy_cards
                      game_num_enemy_cards=0
                      if game_text:
                        print('game_killed_enemy_cards:',game_killed_enemy_cards)
                      game_out = game_out | game_out_enemy_killed
                      if game_killed_enemy_cards%game_boss_enemy_cards==(game_boss_enemy_cards-1):
                        game = game | game_user_add_card
                      if game_killed_enemy_cards%game_boss_enemy_cards==0:
                        if game_text:
                          print('user_cards_life_restored')
                        game_out = game_out | game_out_user_restored
                        for i1 in range(game_num_user_cards):
                          for j1 in range(0,dim_sg2):  
                            game_user_cards_life[i1][j1]=game_user_cards[i1][j1]

                  if game&game_enemy_attack_user:
                    if game_text:
                      print('game_enemy_attack_user')
                    game_out = game_out | game_out_user_attack
                    for j2 in range(game_num_user_cards):
                      np.copyto(game_compare_with_possible_cards[game_num_possible_cards],game_user_cards[j2])
                      game_stddev_compare_with_possible_cards=np.std(game_compare_with_possible_cards, axis=0)
                      #print('game_stddev_last_possible_cards:',game_stddev_last_possible_cards)
                      #print('max(game_stddev_last_possible_cards):',max(game_stddev_last_possible_cards))
                      #print('game_stddev_compare_with_possible_cards:',game_stddev_compare_with_possible_cards)
                      #if np.max(game_stddev_compare_with_possible_cards)>=0.2:
                      for i2 in range(game_num_enemy_cards):
                        #print(j2,i2)
                        for j1 in range(0,dim_sg2):
                          game_enemy_attack_user_cards[j2][i2][j1]=((game_enemy_cards[i2][j1]/(game_user_cards[j2][j1]+1))/(1-game_stddev_compare_with_possible_cards[j1]))/game_num_user_cards
                          game_user_cards_life[j2][j1]=game_user_cards_life[j2][j1]-game_enemy_attack_user_cards[j2][i2][j1]
                  if np.max(np.sum(game_user_cards_life[:game_num_user_cards],axis=0))<=0:
                  #if np.max(game_user_cards_life)<=0:
                    if game_num_user_cards>0:
                      game_killed_user_cards=game_killed_user_cards+game_num_user_cards
                      if game_text:
                        print('game_killed_user_cards:',game_killed_user_cards)
                      game_out = game_out | game_out_user_killed
                    game_num_user_cards=0
                    game_num_enemy_cards=0
                    game = game_user_add_card | game_enemy_add_card
                    game_killed_user_cards=0
                    game_killed_enemy_cards=0
                  if game&game_user_add_card:
                    if game_num_user_cards<game_max_user_cards:
                    #if game_num_possible_cards==game_max_possible_cards:
                      #print('game_last_possible_cards:',game_last_possible_cards)
                      game_stddev_last_possible_cards=np.std(game_last_possible_cards, axis=0)
                      #print('game_stddev_last_possible_cards:',game_stddev_last_possible_cards)
                      #print('max(game_stddev_last_possible_cards):',max(game_stddev_last_possible_cards))
                      if np.max(game_stddev_last_possible_cards)<game_stddev_add_user_card:
                        for j1 in range(0,dim_sg2):  
                          #con[j1]=con[j1]
                          #game_user_cards[game_num_user_cards][j1]=np.avg(game_last_possible_cards[:game_num_possible_cards][j1])
                          game_user_cards[game_num_user_cards][j1]=0
                          #for j2 in range(game_num_possible_cards):
                          #  game_user_cards[game_num_user_cards][j1]=game_user_cards[game_num_user_cards][j1]+game_last_possible_cards[j2][j1]
                          game_user_cards[game_num_user_cards][j1]=game_user_cards[game_num_user_cards][j1]+game_last_possible_cards[game_num_possible_cards-1][j1]
                          #game_user_cards[game_num_user_cards][j1]=game_user_cards[game_num_user_cards][j1]/game_num_possible_cards
                          game_user_cards_life[game_num_user_cards][j1]=game_user_cards[game_num_user_cards][j1]
                          #game_user_cards[game_num_user_cards][j1]=psds_sg2[j1]+0.5
                          #game_user_cards_life[game_num_user_cards][j1]=psds_sg2[j1]+0.5
                        #print('game_user_cards:',game_user_cards)
                        game_num_user_cards=game_num_user_cards+1
                        if game_text:
                          print('game_num_user_cards+1')
                        game_out = game_out | game_out_user_add
                  if game&game_enemy_add_card:
                    if game_num_user_cards>0:
                      for i1 in range(game_killed_enemy_cards//game_boss_enemy_cards+1):
                        if game_num_enemy_cards<game_max_enemy_cards:
                          for j1 in range(0,dim_sg2):  
                            #con[j1]=1-con[j1]
                            if game_killed_enemy_cards%game_boss_enemy_cards==(game_boss_enemy_cards-1):
                              #game_enemy_cards[game_num_enemy_cards][j1]=1-(1-np.avg(game_user_cards[:game_num_user_cards][j1]))*2/3
                              game_enemy_cards[game_num_enemy_cards][j1]=0
                              #for j2 in range(game_num_user_cards):
                              #  game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]+1-(1-game_user_cards[j2][j1])*(1-1/game_easy)
                              game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]+1-(1-game_user_cards[game_num_user_cards-1][j1])*(1-1/game_easy)
                              #game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]/game_num_user_cards
                              game_enemy_cards_life[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]
                              #game_enemy_cards[game_num_enemy_cards][j1]=1-(1-(psds_sg2[j1]+0.5))*2/3
                              #game_enemy_cards_life[game_num_enemy_cards][j1]=1-(1-(psds_sg2[j1]+0.5))*2/3
                            else:
                              #game_enemy_cards[game_num_enemy_cards][j1]=np.avg(game_user_cards[:game_num_user_cards][j1])*2/3
                              game_enemy_cards[game_num_enemy_cards][j1]=0
                              #for j2 in range(game_num_user_cards):
                              #  game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]+game_user_cards[j2][j1]*1/game_easy
                              game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]+game_user_cards[game_num_user_cards-1][j1]*1/game_easy
                              #game_enemy_cards[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]/game_num_user_cards
                              game_enemy_cards_life[game_num_enemy_cards][j1]=game_enemy_cards[game_num_enemy_cards][j1]
                              #game_enemy_cards[game_num_enemy_cards][j1]=(psds_sg2[j1]+0.5)/3
                              #game_enemy_cards_life[game_num_enemy_cards][j1]=(psds_sg2[j1]+0.5)/3
                          #print('game_enemy_cards:',game_enemy_cards)
                          game_num_enemy_cards=game_num_enemy_cards+1
                          game = game_user_attack_enemy | game_enemy_attack_user
                          if game_text:
                            print('game_num_enemy_cards+1')
                          game_out = game_out | game_out_enemy_add

              
                sum_user_pos_life=0
                sum_enemy_pos_life=0
                if (game_num_user_cards>0) and (game_num_enemy_cards>0):
                  user_cards_array=(np.sum(game_user_cards_life[:game_num_user_cards],axis=0)+np.abs(np.sum(game_user_cards_life[:game_num_user_cards],axis=0)))/2
                  user_life_array=(np.sum(game_user_cards[:game_num_user_cards],axis=0)+np.abs(np.sum(game_user_cards[:game_num_user_cards],axis=0)))/2
                  enemy_cards_array=(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)+np.abs(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)))/2
                  enemy_life_array=(np.sum(game_enemy_cards[:game_num_enemy_cards],axis=0)+np.abs(np.sum(game_enemy_cards[:game_num_enemy_cards],axis=0)))/2
                  sum_user_pos_life=np.sum(user_life_array)/dim_sg2
                  sum_enemy_pos_life=np.sum(enemy_life_array)/dim_sg2
                max_std_possible_s = f'{(np.max(np.std(game_last_possible_cards, axis=0))):7.2f}'
                out_text_game=max_std_possible_s+' max_std_possible, '
                average_user_s='nan'
                if game_num_user_cards>0:
                  average_user = f'{(np.average(game_user_cards_life[:game_num_user_cards])):7.2f}'
                out_text_game=out_text_game+average_user_s+" average_user, "
                average_enemy_s='nan'
                if game_num_enemy_cards>0:
                  average_enemy_s = f'{(np.average(game_enemy_cards_life[:game_num_enemy_cards])):7.2f}'
                out_text_game=out_text_game+average_enemy_s+' average_enemy, '
                out_text_game=out_text_game+f'{sum_user_pos_life:7.2f}'+" sum_pos_user, "
                out_text_game=out_text_game+f'{sum_enemy_pos_life:7.2f}'+' sum_pos_enemy, '
                #out_text_game=out_text_game+f'{(np.sum(np.sum(game_user_cards_life[:game_num_user_cards],axis=0)+np.abs(np.sum(game_user_cards_life[:game_num_user_cards],axis=0)))/2)/dim_sg2:7.2f}'+" sum_pos_user, "
                #out_text_game=out_text_game+f'{(np.sum(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)+np.abs(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)))/2)/dim_sg2:7.2f}'+' sum_pos_enemy, '
                #out_text_game=out_text_game+f'{(np.sum(game_user_cards_life[:game_num_user_cards]+np.abs(game_user_cards_life[:game_num_user_cards]),axis=0)/2)/dim_sg2:7.2f}'+" sum_pos_user, "
                #out_text_game=out_text_game+f'{(np.sum(game_enemy_cards_life[:game_num_enemy_cards]+np.abs(game_enemy_cards_life[:game_num_enemy_cards]),axis=0)/2)/dim_sg2:7.2f}'+' sum_pos_enemy, '
                #out_text_game=out_text_game+f'{(np.sum(game_user_cards_life[:game_num_user_cards]+np.abs(game_user_cards_life[:game_num_user_cards]))/2)/dim_sg2:7.2f}'+" sum_pos_user, "
                #out_text_game=out_text_game+f'{(np.sum(game_enemy_cards_life[:game_num_enemy_cards]+np.abs(game_enemy_cards_life[:game_num_enemy_cards]))/2)/dim_sg2:7.2f}'+' sum_pos_enemy, '
                out_text_game=out_text_game+f'{(game_num_user_cards):1.0f}'+" user, "
                out_text_game=out_text_game+f'{(game_num_enemy_cards):1.0f}'+' enemy'
                #if debug:
                if game_text:
                  print(out_text_game)
                #print('game_out')
                #if game_num_user_cards>0:
                #  print(np.array2string(((((np.sum(game_user_cards_life[:game_num_user_cards],axis=0)+np.abs(np.sum(game_user_cards_life[:game_num_user_cards],axis=0)))/2)/game_num_user_cards)*10).astype(int),separator='',max_line_width=130))
                #if game_num_enemy_cards>0:
                #  print(np.array2string(((((np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)+np.abs(np.sum(game_enemy_cards_life[:game_num_enemy_cards],axis=0)))/2)/game_num_enemy_cards)*10).astype(int),separator='',max_line_width=130))
                #print(np.array2string((((game_user_cards_life[:game_num_user_cards]+np.abs(game_user_cards_life[:game_num_user_cards]))/2)*10).astype(int),separator='',max_line_width=130))
                #print(np.array2string((((game_enemy_cards_life[:game_num_enemy_cards]+np.abs(game_enemy_cards_life[:game_num_enemy_cards]))/2)*10).astype(int),separator='',max_line_width=130))
                #print(np.array2string(((psds_sg2+0.5)*10).astype(int),separator='',max_line_width=130))




                #if game_num_enemy_cards>0:
                  #psd_array_sg2[0] = game_enemy_cards[0]-0.5




 #               device = torch.device('cuda')
 
#        if hasattr(G.synthesis, 'input'):
#            m = make_transform(translate, rotate)
#            m = np.linalg.inv(m)
#            G.synthesis.input.transform.copy_(torch.from_numpy(m))

            if True:
              for G3m_index in range(len(G3ms)):

#                z = psd_array_sg2 * vol  
#                seed=1
                z = torch.from_numpy(cons_latentsa[G3m_index]).to(device)
#                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G3m.z_dim)).to(device)
                truncation_psi=1
#                truncation_psi=0.5
#                noise_mode='const'
#                noise_mode='random'
                noise_mode='none'
                label = torch.zeros([1, G3ms[1].c_dim], device=device)
                #if G3m.c_dim != 0:
                #    label[:, class_idx] = 1
                
                img = G3ms[G3m_index](z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
                
                images=[img[0].cpu().numpy() ]   
 
#                z_samples = psd_array_sg2 * vol
#                w_samples = G3m.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
#                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
#                w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
#                w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
#                ws3m = (w_opt).repeat([1, G3m.mapping.num_ws, 1])
 
#                synth_images = G3m.synthesis(ws3m, noise_mode='const')
#                synth_images = (synth_images + 1) * (255/2)
#                synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                #out.append_data(synth_images)
#                images=[synth_images]   
                
                if True:
                
#                  xsize=128
#                  ysize=128
#                  xsize=1024
#                  ysize=1024
                  xsize=512
                  ysize=512

                  image_pil=PIL.Image.fromarray(images[0], 'RGB')
                  #if generate&gen_sg2_shawwn:
                  #  display(image_pil)
                  #print(image_pil)
                  image_asarray=np.asarray(image_pil)
                  #print(image_asarray)
#                  time1111=perf_counter()
                  #print (f'1111: {(time1111-time000):.1f}s')
                  #global video_out
                  #video_out.append_data(image_asarray)
#                  time1112=perf_counter()
                  #print (f'1112: {(time1112-time000):.1f}s')
                  img=image_pil.resize((xsize,ysize),PIL.Image.Resampling.LANCZOS)
                  #print(img)
#                  time1113=perf_counter()
                  #print (f'1113: {(time1113-time000):.1f}s')
#                  buffer = BytesIO()
#                  if generate&gen_jpeg:
#                    img.save(buffer,format="JPEG")                  #Enregistre l'image dans le buffer
#                  if generate&gen_png:
#                  img.save(buffer,format="PNG")                  #Enregistre l'image dans le buffer
                  #img.save('/content/gdrive/MyDrive/EEG-GAN-audio-video/out/'+
                  #          f'{(time100*1000):9.0f}'+'/'+f'{(time000*1000):9.0f}'+'.png',format="PNG")
                        
#                  buffer.seek(0)
#                  myimage = buffer.getvalue()   

                  if draw_fps:

                    time111a[G3m_index]=perf_counter()
                    draw_fps=f'fps: {1/(time111a[G3m_index]-time001a[G3m_index]):3.2f}'
                    #print (f'fps: {1/(time111-time001):.1f}s')
                    #print (f'111-001: {(time111-time001):.1f}s')
                  
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), draw_fps, font=font, fill='rgb(0, 0, 0)', stroke_fill='rgb(255, 255, 255)', stroke_width=1)
                    img = draw._image
                    time001a[G3m_index]=time111a[G3m_index]


                  image = np.asarray(img)
                  image = image[:,:,::-1]
                  screen5a[G3m_index].update(image)




  bdf.close()

if False:    
  cv2.destroyAllWindows()
