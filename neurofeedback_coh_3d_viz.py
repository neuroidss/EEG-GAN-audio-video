#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


from matplotlib.pyplot import draw, figure, show


import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, IpProtocolType
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import mne
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.viz import circular_layout, plot_connectivity_circle

debug = False

if True:
  params = BrainFlowInputParams()
  if debug:
    board_id = -1 # synthetic
    sample_rate = 512
  else:
    board_id = BoardIds.FREEEEG32_BOARD.value
    params.serial_port = '/dev/ttyACM0'
#            params.serial_port = '/dev/ttyS11'
    sample_rate = 512
    eeg_channels = BoardShim.get_eeg_channels(board_id)
#        if num_channels is not None:
#            eeg_channels = eeg_channels[:num_channels]

  board = BoardShim(board_id, params)
        #global board
  board.release_all_sessions()

  board.prepare_session()
  board.start_stream()

  import pyformulas as pf 
  import matplotlib.pyplot as plt
  import numpy as np
  import time
 
        #fig = plt.figure()
         
  canvas = np.zeros((800,800))
#  canvas = np.zeros((480,640))
  screen = pf.screen(canvas, 'Sinusoid')

  raw = None
  input_fname_name='neurofeedback'
  vmin=0#0.7
  vmin=0.7
  fps=10
#        bands = [[30.,45.]]
#bands = [[4.,7.],[8.,12.],[13.,29.],[30.,45.]]
  bands = [[8.,12.]]
  methods = ['coh']
#biosemi32
  ch_names = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz']
#  ch_names_pick = ['Cz','Fz','FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2']
#  ch_names_pick = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','PO3','O1','Oz','Pz','Cz','Fz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2']
#  ch_names_pick = ['F7','FC5','T7','CP5','P7','O1','PO3','P3','CP1','C3','FC1','F3','AF3','FP1','Fz','Cz','Pz','Oz','O2','PO4','P4','CP2','C4','FC2','F4','AF4','FP2','F8','FC6','T8','CP6','P8']
#  ch_names_pick = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5']
#Bernard's 19ch headset
#  ch_names = ["O2","T6","T4","F8","Fp2","F4","C4","P4","ch9","ch10","ch11","ch12","Pz","ch14","ch15","ch16","Fz","ch18","ch19","ch20","ch21","ch22","ch23","ch24","Fp1","F3","C3","P3","O1","T5","T3","F7"]
  ch_names_pick = ['FP1','AF3','F7','F3','FC5','T7','C3','CP5','P7','P3','PO3','O1','Oz','CP1','FC1','Fz','Cz','FC2','CP2','Pz','O2','PO4','P4','P8','CP6','C4','T8','FC6','F4','F8','AF4','FP2']
#  ch_names_pick = ['Fz','Fp1','F7','F3','T3','C3','T5','P3','O1','Pz','O2','P4','T6','C4','T4','F4','F8','Fp2']

  label_names = ch_names
  label_names = ch_names_pick

#  cons=[]
#  for cons_index in range(int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)):
#    cons.append(np.zeros(int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)))
  cons_len=int(len(ch_names_pick)*(len(ch_names_pick)-1)/2)
#  cons_dur=fps
  fs_mult=3
  audio_volume_mult=200
  cons_dur=fs_mult#fps
  show_cons=False
  sound_cons_swap=False
#  sound_cons_swap=True
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

  while True:

    while board.get_board_data_count() > int((sample_rate)/fps): 
#    while board.get_board_data_count() > int((sample_rate*5*1/bands[0][0])/fps): 
#    while board.get_board_data_count() > 0: 
# because stream.read_available seems to max out, leading us to not read enough with one read
      data = board.get_board_data()
            #eeg_data.append(data[eeg_channels,:].T)
      eeg_data = data[eeg_channels, :]
      eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE
#      eeg_data.append(data[eeg_channels,:])#.T)

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
      raws_hstack_cut = raws_hstack[:,-int(sample_rate*5*1/bands[0][0]):]
#      print(raws_hstack_cut)

      ch_types_pick = ['eeg'] * len(ch_names_pick)
      info_pick = mne.create_info(ch_names=ch_names_pick, sfreq=sfreq, ch_types=ch_types_pick)
      raw = mne.io.RawArray(raws_hstack_cut, info_pick, verbose=50)







    if raw is not None:
     if len(raws_hstack_cut[0])>=int(sample_rate*5*1/bands[0][0]):

  # its time to plot something!


      if True:

        datas=[]
        for band in range(len(bands)):
# datas.append(raw)
          datas.append(raw.pick(ch_names_pick))
 
        epochs = []
        for band in range(len(bands)):
        # epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=0.1, preload=False))
          epochs.append(mne.make_fixed_length_epochs(datas[band], 
                                            duration=5*1/bands[band][0], preload=False, overlap=5*1/bands[band][0]-0.1, verbose=50))
#          epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=5*1/8, preload=False, overlap=5*1/8-0.1))

        ji=0 
        eeg_step=ji
        tmin, tmax = 0+(eeg_step/fps), 2+(eeg_step/fps)  # use the first 120s of data
        sfreq = raw.info['sfreq']  # the sampling frequency
        for method in range(len(methods)):
         for band in range(len(bands)):
          #fmin=8.
          #fmax=13.
          fmin=bands[band][0]
          fmax=bands[band][1]
          if band == 0:
           con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs[band][ji:ji+1], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            epochs[band][ji:ji+10], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
            fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=4, verbose=50)
          #con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
          #  epochs[band][ji,ji+1], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
          #  fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
#          psds_shift1=int(round(method*len(bands)+band)*(len(ch_names)*(len(ch_names)-1)/2))
#          ji1=0
#          for j1 in range(0,len(ch_names)): # display separate audio for each break
#            for i1 in range(0,j1): # display separate audio for each break
#              psds[ji1+psds_shift1]=(con[j1][i1][0]-0.5)*1
#              ji1 = ji1+1
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
        if False:
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
            if vmin>0:
              con_sort=np.sort(np.abs(con).ravel())[::-1]
              n_lines=np.argmax(con_sort<0.7)
              vmin_text='vmin0.7_'
            else:
              n_lines=None
              vmin_text=''
            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names,# n_lines=300, 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2), 
                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+vmin_text+str(len(epochs[0].events)-2)+'\n'+str(ji), 
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

        if False:
            fig = raw.plot_psd(average=False, show=False)  
#            plt.show(block=False)

#        if False:
        if True:
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
          cons=np.roll(cons,1,axis=0)
#          cons[1:,:] = cons[:len(cons),:]
          cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          #cons[cons_dur-cons_index-1]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
          #cons_index_current=cons_index
          #cons_index=cons_index+1
          #if cons_index>=cons_dur:
          #  cons_index=0
          #cons = cons[-int(len(cons[0])):]

          if show_cons:
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
            screen.update(image_crop)
#            screen.update(image_rot90)

            plt.close(fig)
            del fig
 
        if True:
          #import stft

#          spectrum = stft.stft(y, 128)
#          back_y = stft.istft(spectrum, 128)

          import librosa
          from librosa import load
#          y, sr = librosa.load(librosa.example('brahms'), duration=5.0)
#          y, sr = librosa.load(librosa.example('brahms'), offset=cons_index/10, duration=5.0)

          from librosa.core import stft, istft
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
          back_y = istft(spectrum)
          back_y_l = istft(spectrum_l)
          back_y_r = istft(spectrum_r)
          back_y_s = np.asarray([back_y_l,back_y_r]).T

          import numpy as np
          import soundfile as sf
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
          sf.write('/content/out/cons_back.wav', back_y, sr, 'PCM_24')
          sf.write('/content/out/cons_back_s.wav', back_y_s, sr, 'PCM_24')
          import sounddevice as sd
          filename='/content/out/cons_back_s.wav'
#          device=
          #print(sd.query_devices())
#          try:
          if False:
#          if True:
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data, fs)#, device=device)
          if True:
            data=back_y_s*audio_volume_mult
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


#for k in range(100):
#  fig, ax = plt.subplots(figsize=(37.33, 21))
#  fig.savefig(f'figure{k}.png')
#  plt.close(fig)
ch_names_sg2 = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz']
ch_locations_sg2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
bands = [[8.,12.]]
methods = ['coh']
duration=5*1/8
overlap=0
fps_sg2=1
#if generate_wavegan:
fps_wg=1#hz/(32768*2)
fps_sg2=fps_wg*4
#fps_sg2=fps_wg
fps_hm=fps_wg

if 2*1/fps_wg>duration:
  duration=2*1/fps_wg
  overlap=0

#if generate&gen_wavegan:
#  dim_wg = 100
#if generate&gen_stylegan2:
#  dim_sg2 = 512
#if generate&gen_sg2_shawwn:
#  dim_sg2 = 1024

#stepSize = 1/pow(2,24)
#vref = 2.50 #2.5V voltage ref +/- 250nV
#gain = 8

#vscale = (vref/gain)*stepSize #volts per step.
#uVperStep = 1000000 * ((vref/gain)*stepSize) #uV per step.
#scalar = 1/(1000000 / ((vref/gain)*stepSize)) #steps per uV.
def download_file_from_google_drive(file_id,dest_path):
  import os.path
#  if not os.path.isfile(dest_path):  
#    !mkdir -p $(dirname {dest_path})
#    !wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='{file_id} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
#    !wget --load-cookies cookies.txt -O {dest_path} 'https://docs.google.com/uc?export=download&id='{file_id}'&confirm='$(<confirm.txt)

#!mkdir /content/eeg
#!pip install --upgrade gdown

#!pip install googledrivedownloader
#from google_drive_downloader import GoogleDriveDownloader as gdd
files_path=[]
#if generate&gen_drums:
files_path = [['1Nfzi6yT83SBZxtgIYVtYp7C7g_Sq9OdS', '/content/eeg/record-[2019.11.13-22.23.59].csv'],
              ['1LtMfr9GduR3semMVgh_6JoUSKbCG8XbH', '/content/eeg/record-[2020.06.28-14.26.09].csv']]
for i in range(len(files_path)):
#  gdd.download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])
  download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])

files_path = [['1nIiilGVq8XXU7bb1UC5GDtmGbovuil9C', '/content/eeg/5min_experienced_meditator_unfiltered_signals.bdf', '5min_experienced_meditator_unfiltered_signals']]

for i in range(len(files_path)):
#  gdd.download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])
  download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])

files_path = [['1BphDQFJZ0aIMiZbkQ55_MzNxOrdq0Jcb', '/content/eeg/01-07-2022_15-34.bdf', '01-07-2022_15-34'],
              ['1HlDuAO8n_hmH1PWlT46T1Waqyy5yz1LV', '/content/eeg/01-07-2022_18-32.bdf', '01-07-2022_18-32'],
              ['1u_8ANTzYB9jP7wCnErivF1ClM1rXixSg', '/content/eeg/01-07-2022_16-52.bdf', '01-07-2022_16-52'],
              ['1_cgoBcHAf3VRIorYafC-vHLAGpv8Papz', '/content/eeg/01-07-2022_17-39.bdf', '01-07-2022_17-39'],
              ['1JMLtO_sQtZtdQHQ2RCZ0RbRj3cvk0G_2', '/content/eeg/01-07-2022_14-05.bdf', '01-07-2022_14-05'],
              ['1ag-1yK-HSxNRh_5wnP9h-kOkVyX3YqSs', '/content/eeg/01-07-2022_18-32_signals.csv']]

for i in range(len(files_path)):
#  gdd.download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])
  download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])

files_path = [['16j4PwPoudiCVB_HF9m0fCYIt3uUZR3kA', '/content/eeg/20211130-162023-sync1.bdf', '20211130-162023-sync1', '20211130-162023-syncs'],
              ['16kSyskI7qOZRitvNIzd4iy6irqhgLwsI', '/content/eeg/20211130-162023-sync2.bdf', '20211130-162023-sync2', '20211130-162023-syncs'],
              ['16k88jpP7uZKKDT-WhZKcvGsnPclDA2TF', '/content/eeg/20211130-162023-sync3.bdf', '20211130-162023-sync3', '20211130-162023-syncs'],
              ['16licSohQPcbLnJmMD88In5PbiiLjOB_N', '/content/eeg/20211130-162023-sync4.bdf', '20211130-162023-sync4', '20211130-162023-syncs'],
              ['16nHtHn1BBkzb9oTl4EhJmPHPEB4Q8CVl', '/content/eeg/20211130-221138-sync1.bdf', '20211130-221138-sync1', '20211130-221138-syncs'],
              ['16m5Kp0WEdz3GixvQ7-jPVLdQUhXmIbA6', '/content/eeg/20211130-221138-sync2.bdf', '20211130-221138-sync2', '20211130-221138-syncs'],
              ['16mJVZV1XT-RULgcSKUSROOQnQr6CHYCK', '/content/eeg/20211130-221138-sync3.bdf', '20211130-221138-sync3', '20211130-221138-syncs'],
              ['16nh5R44VIRDFbYmFk7sDQeMXEAA1veY4', '/content/eeg/20211130-221138-sync4.bdf', '20211130-221138-sync4', '20211130-221138-syncs'],
              ['16oygYh5ZerFWmNHNMaFqZxUNbC0yj-oW', '/content/eeg/20211130-231131-sync1.bdf', '20211130-231131-sync1', '20211130-231131-syncs'],
              ['16q-Rl-L6NcTxdKco347Q63eK8Z3e5h-F', '/content/eeg/20211130-231131-sync2.bdf', '20211130-231131-sync2', '20211130-231131-syncs'],
              ['16o9MNod2FHz3tplGv14a8vxNR8EVGfxx', '/content/eeg/20211130-231131-sync3.bdf', '20211130-231131-sync3', '20211130-231131-syncs'],
              ['16oszc_L44xtZNOgBuEJtgSjwl1Pob825', '/content/eeg/20211130-231131-sync4.bdf', '20211130-231131-sync4', '20211130-231131-syncs']]

for i in range(len(files_path)):
#  gdd.download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])
  download_file_from_google_drive(file_id=files_path[i][0], dest_path=files_path[i][1])
#!pip install mne==0.23.3

import mne
from mne import io

if True:
    ch_names = ['Fz','Fp1','F7','F3','T3','C3','T5','P3','O1','Pz','O2','P4','T6','C4','T4','F4','F8','Fp2']
#    ch_names = ['Fp1','F7','F3','T3','C3','T5','P3','O1','Pz','O2','P4','T6','C4','T4','F4','F8','Fp2','Fz']
#    ch_names = ['F4','F8','Fp2','Fz','Fp1','F7','F3','T3','C3','T5','P3','O1','Pz','O2','P4','T6','C4','T4']
#    ch_names = ['Fp1','F7','F3','T3','C3','T5','P3','Pz','O1','O2','P4','T6','C4','T4','F4','F8','Fp2','Fz']
#    ch_names = ['FP1','F7','F3','T3','C3','T5','P3','PZ','O1','O2','P4','T6','C4','T4','F4','F8','FP2','FZ']
#    ch_names = ['O2','T6','T4','F8','FP2','F4','C4','P4','PZ','FZ','FP1','F3','C3','P3','O1','T5','T3','F7']

    sfreq = 512 
    ch_types=['eeg']*len(ch_names)
    info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types=ch_types)
    misc_ch_names = ['','']

if False:
    input_fname=files_path[0][1]
    input_fname_name=files_path[0][2]
    raw = io.read_raw_bdf(input_fname, eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)

if True:
    files_path_from=0
    input_fname_name=files_path[files_path_from*4+0][3]

    input_fnames=[{}]*4
    input_fnames[0]=files_path[files_path_from*4+0][1]
    input_fnames[1]=files_path[files_path_from*4+1][1]
    input_fnames[2]=files_path[files_path_from*4+2][1]
    input_fnames[3]=files_path[files_path_from*4+3][1]
    input_fname_names=[{}]*4
    input_fname_names[0]=files_path[files_path_from*4+0][2]
    input_fname_names[1]=files_path[files_path_from*4+1][2]
    input_fname_names[2]=files_path[files_path_from*4+2][2]
    input_fname_names[3]=files_path[files_path_from*4+3][2]
    raws=[{}]*4
    raws[0] = io.read_raw_bdf(input_fnames[0], eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)
    raws[1] = io.read_raw_bdf(input_fnames[1], eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)
    raws[2] = io.read_raw_bdf(input_fnames[2], eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)
    raws[3] = io.read_raw_bdf(input_fnames[3], eog=None, misc=None, stim_channel='auto', 
                          exclude=(), preload=False, verbose=True)

    raw_picks=[{}]*4
    raw_picks[0] = raws[0].pick(ch_names)
    raw_picks[1] = raws[1].pick(ch_names)
    raw_picks[2] = raws[2].pick(ch_names)
    raw_picks[3] = raws[3].pick(ch_names)
#    raw_picks[0].ch_names

#!pip install mne==0.23.3
#!pip install pandas
#!pip install matplotlib

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd

from mne.connectivity import spectral_connectivity, seed_target_indices

import pandas as pd
import numpy as np

raws_for_hstack=[{}]*4
raws_for_hstack[0] = raw_picks[0][:][0]
raws_for_hstack[1] = raw_picks[1][:][0]
raws_for_hstack[2] = raw_picks[2][:][0]
raws_for_hstack[3] = raw_picks[3][:][0]

raws_vstack = np.vstack(raws_for_hstack)
len(raws[0][:][0])
len(raws_vstack)
raws_vstack
ch_names_syncs=[{}]*4
syncs=[{}]*4
syncs[0]=['sync1']*len(ch_names)
syncs[1]=['sync2']*len(ch_names)
syncs[2]=['sync3']*len(ch_names)
syncs[3]=['sync4']*len(ch_names)

ch_names_syncs[0] = [syncs[0][i] + '_' + ch_names[i] for i in range(len(ch_names))]
ch_names_syncs[1] = [syncs[1][i] + '_' + ch_names[i] for i in range(len(ch_names))]
ch_names_syncs[2] = [syncs[2][i] + '_' + ch_names[i] for i in range(len(ch_names))]
ch_names_syncs[3] = [syncs[3][i] + '_' + ch_names[i] for i in range(len(ch_names))]
ch_names_sync=ch_names_syncs[0]+ch_names_syncs[1]+ch_names_syncs[2]+ch_names_syncs[3]
#ch_names_sync=ch_names_syncs[0]+ch_names_syncs[1]+ch_names_syncs[2]
#ch_names_sync=ch_names_syncs[0]+ch_names_syncs[1]
ch_names_sync
ch_names = ch_names_sync
#raws_vstack = raws_for_hstack[0]
#raws_for_hstack=[{}]*3
#raws_for_hstack[0] = raw_picks[0][:][0]
#raws_for_hstack[0] = raw_picks[1][:][0]
#raws_for_hstack[1] = raw_picks[1][:][0]
#raws_for_hstack[1] = raw_picks[2][:][0]
#raws_for_hstack[2] = raw_picks[2][:][0]
#raws_for_hstack[2] = raw_picks[3][:][0]
#raws_for_hstack[3] = raw_picks[3][:][0]
#raws_vstack = np.vstack(raws_for_hstack)
sfreq = 512 

ch_types=['eeg']*len(ch_names)
info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types=ch_types)
#info = mne.create_info(sfreq = sfreq)
raw_sync = mne.io.RawArray(raws_vstack, info)

#raw.get_data
raw = raw_sync
#raw.info

#!pip install Pillow
import PIL.Image 
#!pip install tqdm
from tqdm import tqdm
#!pip install imageio==2.4.1
#!pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0
#!mkdir '/content/out'

if False:
    key = 0
    idx = 0

    data_path = '/content/eeg'
    #raw_fname = data_path + '/record-[2019.11.13-22.23.59].gdf'
    #raw = mne.io.read_raw_gdf(raw_fname, preload=True)

    path = data_path

    ch_names = ['FP1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','FP2','Fz','Cz']
    data = pd.read_csv(path + '/record-[2019.11.13-22.23.59].csv', skiprows=0, usecols=ch_names, header=0, delimiter=';') 
    
    #ch_names = ['FP1','F3','P3','O1','O2','P4','F4','FP2']
    #data = pd.read_csv(path + '/record-[2020.06.28-14.26.09].csv', skiprows=0, usecols=ch_names, header=0, delimiter=';') 
    
    #data = pd.read_csv(path + '/record-[2020.06.28-00.36.11].csv', skiprows=0, usecols=ch_names, header=0, delimiter=';') 
    #data = pd.read_csv(path + '/record-[2020.06.29-19.49.23].csv', skiprows=0, usecols=ch_names, header=0, delimiter=';') 
    
    #print(data)
    data_transpose=np.transpose(data)

    sfreq = 512 
    ch_types=['eeg']*len(ch_names)
    info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types=ch_types)
    #info = mne.create_info(sfreq = sfreq)
    raw = mne.io.RawArray(data_transpose, info)
    #raw.plot()
    
    input_fname=files_path[0][1]
    mne.io.read_raw_bdf(input_fname, eog=None, misc=None, stim_channel='auto', exclude=(), infer_types=False, include=None, preload=False, verbose=None)

    # Setup for reading the raw data
    #raw = io.read_raw_fif(raw_fname, verbose=False)
    #events = mne.find_events(raw, stim_channel='STI 014')
    #inverse_operator = read_inverse_operator(fname_inv)
    #raw.info['bads'] = ['MEG 2443', 'EEG 053']

    # picks MEG gradiometers
    #picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    picks = ch_names

#raw[:][0]
#raw1 = mne.io.RawArray(raw[:][0], info)
#raw1.info
#raw#
#raw.ch_names
#print(raw)
raw.pick(ch_names)

#!pip install imageio==2.9
#!pip install imageio==2.4.1
#!pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0

raw = raw_sync

#import imageio
#fps=10
#out = imageio.get_writer('/content/out/output.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')

#for img in imgs:
#  out.append_data(np.asarray(img))
#out.close()
bands = [[8.,12.]]
#bands = [[4.,7.],[8.,12.],[13.,29.]]
#bands = [[8.,12.],[8.,12.],[8.,12.]]
#bands_name = ['theta','alpha','beta']
datas=[]
for band in range(len(bands)):
# datas.append(raw)
 datas.append(raw.pick(ch_names_pick))
 
# datas.append(raw.filter(l_freq=bands[band][0], h_freq=bands[band][1],method='iir'))
#theta_data = raw.filter(l_freq=4, h_freq=7,method='iir')
#alpha_data = raw.filter(l_freq=8, h_freq=12,method='iir')
#beta_data = raw.filter(l_freq=13, h_freq=29,method='iir')
methods = ['coh']
#methods = ['ciplv']
#methods = ['wpli']
#methods = ['coh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli']
epochs = []
for band in range(len(bands)):
# epochs.append(mne.make_fixed_length_epochs(datas[band], 
#                                            duration=0.1, preload=False))
 epochs.append(mne.make_fixed_length_epochs(datas[band], 
                                            duration=5*1/8, preload=False, overlap=5*1/8-0.1))
# epochs.append(mne.make_fixed_length_epochs(datas[band], duration=1.25, preload=False, overlap=1.15))
#epochs = [mne.make_fixed_length_epochs(theta_data, duration=0.1, preload=False),
#          mne.make_fixed_length_epochs(alpha_data, duration=0.1, preload=False),
#          mne.make_fixed_length_epochs(beta_data, duration=0.1, preload=False)]
#epochs = mne.make_fixed_length_epochs(alpha_data, duration=0.1, preload=False)
#epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
#epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)
#epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=False)
#event_related_plot = epochs.plot_image(picks=['FP1'])

#len(epochs[0].events)
#raw[:][0]

#fmin=8.
#fmax=13.
fmin=bands[0][0]
fmax=bands[0][1]+1.
          
sfreq = raw.info['sfreq']  # the sampling frequency
#con_methods = ['coh', 'pli', 'ciplv']
con_methods = ['coh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli']
#con_methods = ['coh', 'plv', 'ciplv', 'ppc', 'pli', 'pli2_unbiased', 'wpli', 'wpli2_debiased']
#con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#    epochs, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
#    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=10)
#con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#    epochs[1][2000:2010], method=con_methods, mode='multitaper', sfreq=sfreq, 
#    faverage=True, mt_adaptive=True, n_jobs=1)
#con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#    epochs[0][2000,2001], method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
#    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epochs[0][2000:2010], method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

#epochs[0][2000:2010]
#print(epochs[0][2000:2010])

con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c[:, :, 0]

#sfreq
#con_res

from mne.viz import circular_layout, plot_connectivity_circle
#import matplotlib.pyplot as plt

label_names = ch_names

#plot_connectivity_circle(con_res['pli'], label_names, 
#                                                  title='All-to-All Connectivity (PLI)')
#fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')
no_names = [''] * len(label_names)
for ii, method in enumerate(con_methods):
    plot_connectivity_circle(con_res[method], label_names, n_lines=300,
                             title=method)
    #plot_connectivity_circle(con_res[method], label_names, n_lines=300,
    #                         title=method, padding=0, fontsize_colorbar=6,
    #                         fig=fig, subplot=(1, 8, ii + 1))

    #plot_connectivity_circle(con_res[method], no_names, n_lines=300,
    #                         title=method, padding=0, fontsize_colorbar=6,
    #                         fig=fig, subplot=(1, 6, ii + 1))
plt.show()

#con_res['coh']

import io
from PIL import Image

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('agg')
#matplotlib.interactive(False)
#import gc

import imageio

#!pip install mne-connectivity

from mne_connectivity.viz import plot_sensors_connectivity

n_parts_one_time = 3
#matplotlib.use('agg')
#fig = plt.figure()

fps=10
#out = imageio.get_writer('/content/out/output.mp4', fps=fps)
#out = imageio.get_writer('/content/out/'+input_fname_name+'_circle_'+methods[0]+'_'+
#                         str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2)+'.mp4', fps=fps)
#out = imageio.get_writer('/content/out/output.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')

# Generate random breaks and display audio

# CHANGE THIS to change number of examples generated
#n_generate = 30
#n_generate = 150
#n_generate = 300
#n_generate = 305
#n_generate = 390

# Sample latent vectors
#seed = 666 # change this seed to generate different set of breaks
#np.random.seed(seed)
#_z = (np.random.rand(n_generate, dim) * 2.) - 1.


hz=44100
#hz=39936
#hz=int(32768*2*(600/240))
#hz=int(32768*2*(480/240))
#hz=int(32768*2*(360/240))
#hz=int(32768*2*(300/240))
#hz=int(32768*2*(265/240))
#hz=int(32768*2*(250/240))
#hz=int(32768*2*(240/240)*1.6666666)
#hz=int(32768*2*(240/240))
#hz=int(32768*2*(120/240))
#fps=hz/(32768*2)
#fps=10
#fps=0.5
#fps=44100/(32768*2)
#fps=1
#fps=1/3
#fps=1.5

#n_generate=int((307-2)*fps)
#n_generate=int((160-2)*fps)
#n_generate=int((1598-2)*fps)
#n_generate=int((1648-2)*fps)
n_generate=len(epochs[0].events)-2

#n_generate=int((10-2)*fps)
#n_generate=int((1607-2)*fps)
#n_generate=150
part_len = 100
#part_len = 1000
#part_len = 275
dim = 512


n_parts = n_generate//part_len
if n_generate%part_len>0:
    n_parts=n_parts+1

vol=0.1

#psd_array=np.random.rand(part_len, dim) 

#out = imageio.get_writer('./output.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')

#Gs_kwargs = dnnlib.EasyDict()
#Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
#Gs_kwargs.randomize_noise = False

imgs = []
imgs1 = []

def plot_func(con,methods,label_names,out):
        #plt.ioff()
        if True:
#          con_res = dict()
#          for method, c in zip(methods, con):
#            con_res[method] = c[:, :, 0]
#            con_res[method] = c[:, :]
          #for ii, method in enumerate(methods):
#          if False:
          if True:
#            fig,_ = plot_connectivity_circle(con_res[method], label_names, n_lines=300, 

#            plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
#                                             title=method, show = False)
            #fig = plt.figure()
            #fig,_ = 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
#                                             title=method, show = False, vmin=0, vmax=1)#, fig=fig)
            #fig, ax = plt.subplots(figsize=(37.33, 21)) 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
#                                             title=methods[0], show = False, vmin=0, vmax=1, fontsize_names=16)#, fig=fig)
            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=300, 
                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2), 
                                             show = False, vmin=0, vmax=1, fontsize_names=16)#, fig=fig)
            #ax.set_theta_zero_location("W")
            
            #ax.get_th
            #ax.set_rotation()
            #print(ax.get_xticks())
#            ax.set_theta_zero_location((90+360/len(label_names)))
            #print(ax.get_xticklabels())
            #print(ax.get_xticks())
            #xticks=ax.get_xticks()
            #xticks.
            #ax.set_rotation(0)

            #rtick_locs = range(len(label_names))
            #ax.set_rgrids(rtick_locs)
            #ax.set_rticks(rtick_locs)
            
            #set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kwargs)

            #thetatick_locs = range(len(label_names))
#            thetatick_locs = np.linspace(0.,45.,4)
            #thetatick_labels = [u'%i\u00b0'%np.round(x) for x in thetatick_locs]
            #ax.set_thetagrids(thetatick_locs, thetatick_labels, fontsize=16)

            #set_thetagrids(self, angles, labels=None, fmt=None, **kwargs)[source]
            
            #ax.set_xticks(np.linspace(0, 2*np.pi, len(label_names), endpoint=False))
            #ax.set_xticklabels(range(len(label_names)))  

#            ax.set_theta_offset(np.deg2rad(90+360/len(label_names)))
            ax.set_theta_offset(np.deg2rad(360/len(label_names)))
#            ax.set_thetagrids(range(len(label_names)), 
#                  labels=range(len(label_names)), fontsize=12)

            #ax.set_rgrids(range(len(label_names)), labels=range(len(label_names)), fontsize=12, angle=180)

            
            #ax.set_xticklabels(ax.get_xticks(), rotation = np.deg2rad(90+360/len(label_names)))
            #for tick in ax.get_xticklabels():
            #  tick.set_rotation(45)
#            for label, angle in zip(ax.get_xticklabels(), angles):
            #labels = []
            #for label in zip(ax.get_xticklabels()):
            #  x,y = label.get_position()
            #  lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
            #                ha=label.get_ha(), va=label.get_va())
            #  #lab.set_rotation(angle)
            #  lab.set_rotation(10)
            #  labels.append(lab)
            ##ax.set_xticklabels([])
            #ax.set_xticklabels(labels)
            #plt.show()  

#            fig = plot_sensors_connectivity(raw.info, con[:, :, 0], picks=label_names, cbar_label=method)
          #plot_conmat_file = os.path.abspath('circle_' + fname + '.eps')
          #fig.savefig(plot_conmat_file, facecolor='black')

          #plt.close()
          #plt.close(fig)
          #fig.clf()
          
          #del fig

          #if ji%100==0 :
          #  gc.collect()

        #if False:

          #plt.close(fig)
          ##fig1.close()
          #del fig

          #plt.rcParams["figure.figsize"] = [7.50, 3.50]
          #plt.rcParams["figure.autolayout"] = True

          #plt.figure()
          #plt.plot([1, 2])

          img_buf = io.BytesIO()
          #plt.savefig(img_buf, format='png')

          fig.savefig(img_buf, facecolor='black', format='png')
          fig.savefig('/content/out/img_buf.png', facecolor='black', format='png')

          #fig.clear()
          #fig.clf()
          #plt.close()
          #ax.cla()

          #plt.savefig(img_buf, facecolor='black', format='png')
          #plt.savefig('/content/out/img_buf.png', facecolor='black', format='png')

          #plt.close()
          #plt.close(fig)
          #fig.clf()

          # Clear the current axes.
          #plt.cla() 
          # Clear the current figure.
          #plt.clf() 
          # Closes all the figure windows.
          #plt.close('all')   
          #plt.close(fig)
          #gc.collect()
          #del fig

          #if ji%100==0 :
          #  gc.collect()

          img_buf.seek(0)

          im1 = Image.open(img_buf)
          im1.crop(0,0,576,576)
          img_buf1 = io.BytesIO()
          im1.write(img_buf1)
          img_buf1.seek(0)

          im = imageio.imread(img_buf1)
          out.append_data(im)
          img_buf.close()
          img_buf1.close()

          plt.close('all')   

n_parts_now = 0

#z_avg_samples=n_generate
#for i in range(n_generate): # display separate audio for each break
for j in range(n_parts): # display separate audio for each break
  out_file = '/content/out/parts_out/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'_'+str(j)+'.mp4'
  out_file_parts = '/content/out/parts/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'_'+str(j)+'.mp4'

  import os.path
  if not os.path.isfile(out_file_parts):  
    n_parts_now = n_parts_now + 1
    if n_parts_now > n_parts_one_time:
      break

#    if os.path.isfile(out_file):  
#      !rm {out_file}
    out = imageio.get_writer(out_file, fps=fps)

    for i in range(part_len): # display separate audio for each break
        ji = j * part_len + i
        
#        if (i==0) and (n_generate-ji<part_len):
#            psd_array=np.random.rand((n_generate-ji), dim) 


        eeg_step=ji
        #print (f'EEG step: {(eeg_step/3):.1f} s')
        tmin, tmax = 0+(eeg_step/fps), 2+(eeg_step/fps)  # use the first 120s of data
        #tmin, tmax = 0+(10*eeg_step/512), 2+(10*eeg_step/512)  # use the first 120s of data
        #fmin, fmax = 0.5, 256  # look at frequencies between 4 and 100Hz
        #fmin, fmax = 0.5, 50  # look at frequencies between 4 and 100Hz
        #fmin, fmax = 8, 12  # look at frequencies between 4 and 100Hz
        #n_fft = 512  # the FFT size (n_fft). Ideally a power of 2
        #n_fft = 1024  # the FFT size (n_fft). Ideally a power of 2
        #n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
        #label = mne.read_label(fname_label)
        
        print(str(ji) + '/' + str(n_generate))
        #logger = logging.getLogger()
        #logger.disabled = True

        sfreq = raw.info['sfreq']  # the sampling frequency
        
#        psds=np.zeros(dim)
        
        for method in range(len(methods)):
         for band in range(len(bands)):
          #fmin=8.
          #fmax=13.
          fmin=bands[band][0]
          fmax=bands[band][1]
          if band == 0:
           con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs[band][ji:ji+1], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
#            epochs[band][ji:ji+10], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
            fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1, verbose=50)
          #con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
          #  epochs[band][ji,ji+1], method=methods[method], mode='multitaper', sfreq=sfreq, fmin=fmin,
          #  fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
#          psds_shift1=int(round(method*len(bands)+band)*(len(ch_names)*(len(ch_names)-1)/2))
#          ji1=0
#          for j1 in range(0,len(ch_names)): # display separate audio for each break
#            for i1 in range(0,j1): # display separate audio for each break
#              psds[ji1+psds_shift1]=(con[j1][i1][0]-0.5)*1
#              ji1 = ji1+1
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
        if True:
#        if False:
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
            n_lines=np.argmax(con_sort<0.7)
            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names,# n_lines=300, 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2), 
                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'\n'+str(ji), 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2)+'_'+str(ji), 
                                             show = False, vmin=0.7, vmax=1, fontsize_names=8)#, fig=fig)
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

#        if False:
        if True:

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
#          out.append_data(im)
          #del img_buf
          #del im

          #if ji == 100:
          #  out.close()
          #  break

          #im = img_buf.getvalue()

          #im = Image.open(img_buf)
#          im.copy()
          #im.show(title="My Image")
          
          #out.append_data(np.asarray(im))

          #im.close()

          #img_buf.close()

          #  _z = psd_array * vol
          #  images = Gs.run(_z, None, **Gs_kwargs) # [minibatch, height, width, channel]
          #  for image in images:
              #imgs1.append(image)
#              out.append_data(np.asarray(PIL.Image.fromarray(image, 'RGB')))

              #imgs.append(PIL.Image.fromarray(image, 'RGB'))
            #out.append(PIL.Image.fromarray(images[0], 'RGB'))
            #_G_z = sess.run(G_z, {z: _z})[:,:,0]
            #if j==0:
            #    _G_z_full=_G_z
            #else:
            #    _G_z_full=np.append(_G_z_full,_G_z)

        #if True:
        #  plot_func(con,methods,label_names,out)
          #if ji%100==0 :
          #  gc.collect()
          if (ji==n_generate-1) :
                break
    out.close()
#    !mv {out_file} {out_file_parts}


#print(psd_array)
#print(psd_array.shape)
#print(psd_array.ndim)
#_z = psd_array / 5.
#_z = psd_array / 10.
#_z = (psd_array * 2.) - 1.
#_G_z = sess.run(G_z, {z: _z})[:,:,0]

# display(Audio(_G_z.flatten(), rate=39936)) # display all in one audio

#for i in range(n_generate): # display separate audio for each break
  #print(i)
  #display(Audio(np.tile(_G_z[i][1:_G_z[i].ndim/2], 2), rate=39936)) # change rate for different tempo
  #display(Audio(np.tile(_G_z[i][1:32768], 2), rate=32768)) # change rate for different tempo
  #display(Audio(np.tile(_G_z[i], 1), rate=32768)) # change rate for different tempo
#out.close()

#!mkdir '/content/out/parts'
out_file_parts_txt = '/content/out/parts/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'.txt'
#!rm {out_file_parts_txt}
out_file_mp4 = '/content/out/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'.mp4'

out_file_sh = '/content/out/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'.sh'

if os.path.isfile(out_file_parts_txt): 
  os.remove(out_file_parts_txt)

with open(out_file_parts_txt, 'a') as the_file:
 for j in range(n_parts): # display separate audio for each break
  out_file_part = '/content/out/parts/'+input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin0.7_'+str(len(epochs[0].events)-2)+'_'+str(j)+'.mp4'
  the_file.write("file '")
  the_file.write(out_file_part)
  the_file.write("'\n")
#  import os.path
#  if not os.path.isfile(out_file):  
#    out = imageio.get_writer(out_file, fps=fps)
#!rm {out_file_mp4}

if os.path.isfile(out_file_sh): 
  os.remove(out_file_sh)

with open(out_file_sh, 'a') as the_file:
  the_file.write("ffmpeg -f concat -safe 0 -i ")
  the_file.write(out_file_parts_txt)
  the_file.write(" -c copy ")
  the_file.write(out_file_mp4)
#!ffmpeg -f concat -safe 0 -i {out_file_parts_txt} -c copy {out_file_mp4}
#!ffmpeg -vfilters "rotate=90" -i {out_file_mp4} {out_file_mp4}

