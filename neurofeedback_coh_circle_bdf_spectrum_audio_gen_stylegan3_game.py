#!pip install matplotlib brainflow mne==0.23.3 librosa sounddevice absl-py pyformulas pyedflib
#!pip install diffusers transformers scipy ftfy "ipywidgets>=7,<8"

#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from matplotlib.pyplot import draw, figure, show

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.viz import circular_layout, plot_connectivity_circle

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
flags.DEFINE_list('ch_names', ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz'], 'ch_names')
flags.DEFINE_list('ch_names_pick', ['Cz','Fz','FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2'], 'ch_names')
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
flags.DEFINE_boolean('sound_cons_swap', True, 'sound_cons_swap')
flags.DEFINE_string('sound_cons_buffer_path', '', 'sound_cons_buffer_path')
flags.DEFINE_boolean('rotate', True, 'rotate')
flags.DEFINE_boolean('show_stable_diffusion_cons', False, 'show_stable_diffusion_cons')
flags.DEFINE_string('huggingface_hub_token', './huggingface_hub_token', 'huggingface_hub_token: token or file with token')
flags.DEFINE_string('unet_height', '512', 'unet_height')
flags.DEFINE_string('unet_width', '512', 'unet_width')
flags.DEFINE_string('unet_num_inference_steps', '5', 'unet_num_inference_steps')
flags.DEFINE_string('unet_latents', None, 'unet_latents: if None, load random')
flags.DEFINE_string('unet_guidance_scale', '7.5', 'unet_guidance_scale')
flags.DEFINE_string('apply_to_latents', '0.5', 'apply_to_latents: closer to zero means apply more')
flags.DEFINE_string('apply_to_embeds', '1', 'apply_to_embeds: closer to zero means apply more')
flags.DEFINE_string('clip_prompt', 'villa by the sea in florence on a sunny day', 'clip_prompt')
#flags.DEFINE_boolean('show_stylegan3_cons', False, 'show_stylegan3_cons')
flags.DEFINE_boolean('show_stylegan3_cons', True, 'show_stylegan3_cons')
#flags.DEFINE_boolean('show_game_cons', True, 'show_game_cons')
flags.DEFINE_boolean('show_game_cons', False, 'show_game_cons')
flags.DEFINE_string('n_jobs', '32', 'n_jobs')
flags.DEFINE_boolean('draw_fps', True, 'draw_fps')
#flags.DEFINE_string('n_jobs', '8', 'n_jobs')
#flags.DEFINE_string('n_parts_one_time', None, 'n_parts_one_time')
#flags.DEFINE_string('part_len', None, 'part_len')
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

if True:
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


  import matplotlib.pyplot as plt
  import numpy as np
  import time
 
        #fig = plt.figure()

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
  if show_game_cons:
    canvas5 = np.zeros((1024,1024))
#    canvas5 = np.zeros((800,800))
    screen5 = pf.screen(canvas5, 'game_cons')
  

  to_sum_embeds = None

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


  if draw_fps:

    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont

    font_fname = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    font_size = 10
    font = ImageFont.truetype(font_fname, font_size)
  
    from time import perf_counter
    time000=perf_counter()
    time001=perf_counter()

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
                 'sg2-ada_2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664','stylegan2-ada']       
                   ]
      for i in range(len(files_path)):
        download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])

    G3ms=[{}]*len(files_path)
    for i in range(len(files_path)):
      network_pkl=files_path[i][1]
      device = torch.device('cuda')
      with dnnlib.util.open_url(network_pkl) as fp:
#      G3m = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        G3ms[i] = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore
    
    
    import mne
    from mne import io
    from mne.datasets import sample
    from mne.minimum_norm import read_inverse_operator, compute_source_psd

    from mne.connectivity import spectral_connectivity, seed_target_indices

    import pandas as pd
    import numpy as np      
    
    
    
    
      
  #import os
  #import logging
  #import pandas as pd

  if show_stable_diffusion_cons:
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
  for channel in ch_names_pick:
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

  while True:

    while board.get_board_data_count() > int((sample_rate)/fps): 
#    while board.get_board_data_count() > int((sample_rate*5*1/bands[0][0])/fps): 
#    while board.get_board_data_count() > 0: 
# because stream.read_available seems to max out, leading us to not read enough with one read
      data = board.get_board_data()
            #eeg_data.append(data[eeg_channels,:].T)
      eeg_data = data[eeg_channels, :]

      signals = eeg_data

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

#      print(raws_hstack_cut)
     
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
      if raw is not None:
        raws=[{}]*2
        raws[0] = raw
        raws[1] = mne.io.RawArray(eeg_data, info, verbose=50)
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

      ch_types_pick = ['eeg'] * len(ch_names_pick)
      info_pick = mne.create_info(ch_names=ch_names_pick, sfreq=sfreq, ch_types=ch_types_pick)
      raw = mne.io.RawArray(raws_hstack_cut, info_pick, verbose=50)


    if raw is not None:
     if len(raws_hstack_cut[0])>=int(sample_rate*duration*2):

  # its time to plot something!


      if True:

        datas=[]
 #       for band in range(len(bands)):
# datas.append(raw)
        datas.append(raw.pick(ch_names_pick))
 
        epochs = []
#        for method in range(len(methods)):
#         for band in range(len(bands)):
        # epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=0.1, preload=False))
        epochs.append(mne.make_fixed_length_epochs(datas[0], 
                                            duration=duration, preload=True, overlap=overlap, verbose=50))
#          epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=5*1/8, preload=False, overlap=5*1/8-0.1))

        #ji=0 
        #eeg_step=ji
#        tmin, tmax = 0+(eeg_step/fps), 2+(eeg_step/fps)  # use the first 120s of data
        tmin, tmax = 0, duration
        sfreq = raw.info['sfreq']  # the sampling frequency
        for band in range(len(bands)):
         for method in range(len(methods)):
          #fmin=8.
          #fmax=13.
          fmin=bands[band][0]
          fmax=bands[band][1]
#          if band == 0:
          con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs[0], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            epochs[0][ji:ji+4], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            epochs[band][ji:ji+10], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
            fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=n_jobs, verbose=50)
          cons=np.roll(cons,1,axis=0)
#          cons[1:,:] = cons[:len(cons),:]
          cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          
          if print_freq_once and not print_freq_once_printed:
             print(freqs)
             print_freq_once_printed = True
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
            con_sort=np.sort(np.abs(con).ravel())[::-1]
            n_lines=np.argmax(con_sort<vmin)
               
#            print(freqs)
            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=n_lines, 
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

        if show_stable_diffusion_cons:
##            cons=np.roll(cons,1,axis=0)
#            cons[1:,:] = cons[:len(cons),:]
##            cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
#            print('_')
#            car_latent = encode_vae(car_img)
#            car_latent.shape

#            base_latents = test_latents.detach().clone()
            base_latents = unet_latents.detach().clone()
#            cons_latents = base_latents
            cons_latents_flatten = base_latents.reshape(len(base_latents)*len(base_latents[0])*len(base_latents[0][0])*len(base_latents[0][0][0]))
            for cons_index in range(int(len(cons_latents_flatten)/len(cons[0]))+1):
              for con_index in range(len(cons[0])):
               if con_index + cons_index*len(cons[0]) < len(cons_latents_flatten):
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1
#                cons_latents_flatten[con_index + cons_index*len(cons[0])] = 1-random.randint(0, 10)/200
                cons_latents_flatten[con_index + cons_index*len(cons[0])] = ((cons[cons_index%len(cons)][con_index]+apply_to_latents)/1+0.0001)/((1+apply_to_latents)/1+0.0001)
            cons_latents = cons_latents_flatten.reshape(len(base_latents),len(base_latents[0]),len(base_latents[0][0]),len(base_latents[0][0][0]))


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
            

            if to_sum_embeds is None:
#                base_latents = test_latents.detach().clone()
#            cons_img=Image.fromarray(cons)
#            cons_img_resize=cons_img.resize((400, 416))
#            cons_latent = encode_vae(cons_img_resize)
                #print(car_latent)
#            print(cons_latent)
            
                to_sum_latents = unet_latents/cons_latents
#                to_sum_embeds = test_embeds-cons_embeds
                to_sum_embeds = test_embeds/cons_embeds
#                to_sum_embeds = car_embeds/cons_embeds
            
            sum_latents = cons_latents*to_sum_latents
#            sum_embeds = cons_embeds+to_sum_embeds
            sum_embeds = cons_embeds*to_sum_embeds
            #sum_embeds = cons_embeds
##            sum_embeds = test_embeds
#            sum_latent = car_latent
##            test_latents = generate_latents(sum_embeds)
#            test_latents = generate_latents(test_embeds)

            test_latents = generate_latents(
                sum_embeds,
#                test_embeds,
                height=unet_height, 
                width=unet_width,
                num_inference_steps=unet_num_inference_steps,
#                latents=unet_latents,
#                latents=cons_latents,
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


                dim_sg2=G3ms[1].z_dim
                sg3_latents=np.random.rand((1), G3ms[1].z_dim) 
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



                game_out=0
                #print('con:',con[0])
                #print('game_in')
                #print('game_last_possible_cards:',game_last_possible_cards)
                for j1 in range(0,dim_sg2):  
                  #print('j1:',j1)
                  game_last_possible_cards[game_cur_possible_cards][j1]=cons_latents[0][j1]+0.5
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
                out_text_game=f'{(np.max(np.std(game_last_possible_cards, axis=0))):7.2f}'+' max_std_possible, '
                out_text_game=out_text_game+f'{(np.average(game_user_cards_life[:game_num_user_cards])):7.2f}'+" average_user, "
                out_text_game=out_text_game+f'{(np.average(game_enemy_cards_life[:game_num_enemy_cards])):7.2f}'+' average_enemy, '
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

#                z = psd_array_sg2 * vol  
#                seed=1
                z = torch.from_numpy(cons_latents).to(device)
#                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G3m.z_dim)).to(device)
                truncation_psi=1
#                truncation_psi=0.5
#                noise_mode='const'
#                noise_mode='random'
                noise_mode='none'
                label = torch.zeros([1, G3ms[1].c_dim], device=device)
                #if G3m.c_dim != 0:
                #    label[:, class_idx] = 1
                
                img = G3ms[1](z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
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
                  screen5.update(image)



  bdf.close()

