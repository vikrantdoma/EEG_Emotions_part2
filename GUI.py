from tkinter import * 
from tkinter.ttk import *
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
import time
from functools import reduce
import math as m
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from keras.models import load_model
from tkinter import messagebox
import pathlib
from datetime import datetime
import webbrowser
global SampleRate 
SampleRate = 128



def make_data_pipeline(file_names,labels,image_size,frame_duration,overlap):
    '''
    IN: 
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)
    
    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''

    Fs = 128.0   #sampling rate
    frame_length = Fs * frame_duration
    
    print('Generating training data...')
    
    
    for i, file in enumerate(file_names):
        print ('Processing session: ',file, '. (',i+1,' of ',len(file_names),')')
        
        data=pd.read_csv(file)
        data=data.drop(['Unnamed: 0'], axis=1)
        df = pd.DataFrame(data)
        #print(df)
        
        X_0 = make_frames(df,frame_duration)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),14*3)
        
        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3) 
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        print(labels[0])
        if i == 0:
            X = images
            y = np.ones(len(images))*labels[0]
        else:
            X = np.concatenate((X,images),axis = 0)
            y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)
        
    print(X)
    return X,np.array(y)




def gen_images(locs, features, n_gridpoints, normalize=True, pca=False, std_mult=0.1, n_components=2):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
    n_samples = features.shape[0]

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

     # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        ##print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]


def get_fft(snippet):
    Fs = 128.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet)/Fs
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    #Added in: (To remove bias.)
    Y[0] = 0
    return frq,abs(Y)

def theta_alpha_beta_averages(f,Y):
    theta_range = (4,7)
    alpha_range = (8,12)
    beta_range = (13,38)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta

def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 128.0
    frame_length = Fs*frame_duration
    frames = []
    frames_ar_alpha= []
    frames_ar_theta= []
    frames_ar_beta= []
    steps = np.arange(0,len(df),frame_length)
    for i,_ in enumerate(steps):
        frame = []
        ar_alpha= []
        ar_theta= []
        ar_beta= []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                snippet = np.array(df.ix[steps[i-1]:steps[i]-1,int(channel)])
                #print(i, channel)
                f,Y =  get_fft(snippet)
                theta, alpha, beta = theta_alpha_beta_averages(f,Y)
                #print theta, alpha, beta
                ar_alpha.append([alpha])
                ar_theta.append([theta])
                ar_beta.append([beta])
                frame.append([theta, alpha, beta])
            
        frames.append(frame)
              
        #print("________________Frame_no {"+str(i)+"}________________")
        #plt.figure(figsize=(50, 6))
        #plt.plot(ar_theta,color='green',label='theta')
        #plt.plot(ar_alpha,color='blue',label='alpha')
        #plt.plot(ar_beta,color='red', label='beta' )
        #plt.legend()
        #plt.xlabel('sampled time',fontsize=35)
        #plt.ylabel('Hz',fontsize=35)
        #plt.title('EEG channel '+str(d[i]), fontsize=35)
        #plt.xlabel('Sensors')
        #plt.ylabel('AVG (freq)');
        
            
    #print(len(frames))
    #plt.show(block=False)
    return np.array(frames)

def make_new_frames(file_names,image_size,frame_duration,overlap):
    '''
    IN: 
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)
    
    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''

    Fs = 128.0   #sampling rate
    frame_length = Fs * frame_duration
    
    print('Generating training data...')
    
    
    for i, file in enumerate(file_names):
        print ('Processing session: ',file, '. (',i+1,' of ',len(file_names),')')
        
        data=pd.read_csv(file)
        data=data.drop(['Unnamed: 0'], axis=1)
        df = pd.DataFrame(data)
        #print(df)
        
        X_0 = make_frames(df,frame_duration)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),14*3)
        
        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3) 
        if i == 0:
            X = images
        else:
            X = np.concatenate((X,images),axis = 0)

    return X



########################################################################################################################







def import_csv_data():
	global v
	global df
	global csv_file_path 
	csv_file_path = askopenfilename()
	print(csv_file_path)
	v.set(csv_file_path)
	df = pd.read_csv(csv_file_path,low_memory=False)
	df=df.drop(['Unnamed: 0'], axis=1)


def crop(start,end):
	global df
	df=df[SampleRate*start.get():SampleRate*end.get()].reset_index()
	df=df.drop(['index'], axis=1)
	global startend 
	startend= end.get()-start.get()-1

def process():
	global df
	global startend
	
	print(df.info())
	X = make_frames(df,1)
	global locs_2d
	locs_2d = [(-2.0,4.0), #af3
		           (2.0,4.0), #af4
		           (-1.0,3.0),#f3
		           (1.0,3.0),#f4
		           (-3.0,3.0),#f7
		           (3.0,3.0),#f8
		           (-2.0,2.0),#fc5
		           (2.0,2.0),#fc6
		           (-2.0,-2.0),#p7
		           (2.0,-2.0),#p8
		           (-4.0,1.0),#t7
		           (4.0,1.0),#t8
		           (-1.0,-3.0),#o1
		           (1.0,-3.0)] #o2
	X = make_frames(df,1)
	X_1 = X.reshape(startend,14*3)

	now = datetime.now()
	newpath = "images"
	if not os.path.exists(newpath):
    		os.makedirs(newpath)
	os.system('xdg-open "%s"' % newpath)


	images = gen_images(np.array(locs_2d),X_1, 32, normalize=False)
	for i in range(0,startend):
	    plt.figure(figsize=(50, 6))
	    #plt.imsavefig('images_puj1/puj_test'+str(i)+'.png',dpi=100)
	    #img=plt.imshow((images[i].T * 255).astype(np.uint8))
	    plt.imsave('/home/vikmachine/EEG-Demo/images/images'+str(i)+'.png', (images[i].T * 255).astype(np.uint8),dpi=100)



def model_select():
	global file_path
	file_path = askopenfilename()
	print(file_path)
	filename.set(file_path)
	model = load_model(file_path)

def callback(url,i1):
	global linknum
	linknum=i1
	webbrowser.open_new(url)



def predict():
	global df
	global startend


	X = make_frames(df,1)
	global locs_2d
	locs_2d = [(-2.0,4.0), #af3
		           (2.0,4.0), #af4
		           (-1.0,3.0),#f3
		           (1.0,3.0),#f4
		           (-3.0,3.0),#f7
		           (3.0,3.0),#f8
		           (-2.0,2.0),#fc5
		           (2.0,2.0),#fc6
		           (-2.0,-2.0),#p7
		           (2.0,-2.0),#p8
		           (-4.0,1.0),#t7
		           (4.0,1.0),#t8
		           (-1.0,-3.0),#o1
		           (1.0,-3.0)] #o2
	X = make_frames(df,1)
	X_1 = X.reshape(startend,14*3)

	now = datetime.now()

	newpath = "images"#+str(now)
	if not os.path.exists(newpath):
    		os.makedirs(newpath)
	os.system('xdg-open "%s"' % newpath)

	images = gen_images(np.array(locs_2d),X_1, 32, normalize=False)
	for i in range(0,startend):
	    plt.figure(figsize=(50, 6))
	    #plt.imsavefig('images_puj1/puj_test'+str(i)+'.png',dpi=100)
	    #img=plt.imshow((images[i].T * 255).astype(np.uint8))
	    plt.imsave('/home/vikmachine/EEG-Demo/images/images'+str(i)+'.png', (images[i].T * 255).astype(np.uint8),dpi=100)



	global csv_file_path
	global file_path
	file_names=[]
	file_names.append(csv_file_path)
	
	image_size = 28
	frame_duration = 2.0
	overlap = 0.5
	Xnew = make_new_frames(file_names,image_size,frame_duration,overlap)
	print("\n")
	print(csv_file_path)	
	print(Xnew)
	print("\n")


	Xnew=Xnew.reshape(1,31,28,28,3)

	global fv
	if (fv.get()==1):
		
		file_path="/home/vikmachine/EEG-Demo/lstm_model_valence.h5"
		filename.set(file_path)
		model = load_model(file_path)
		yhat = model.predict_classes(Xnew)
		#yhat=1
		if yhat == 1:
			mssg="high valency"
			recomend(fv.get())

		else :
			mssg="low valency"
			tk.messagebox.showinfo("IT works","\n\n The predicted class for input is : ["+mssg+"]")
			

	if (fv.get()==2):
		file_path="/home/vikmachine/EEG-Demo/lstm_model_arousal.h5"
		filename.set(file_path)
		model = load_model(file_path)
		yhat = model.predict_classes(Xnew)
		#yhat=1
		if yhat == 1:
			mssg="high Arousal"
			recomend(fv.get())

		else :
			mssg="low Arousal"
			tk.messagebox.showinfo("IT works"," \n\n The predicted class for input is : ["+mssg+"]")		

	if (fv.get()==3):
		file_path="/home/vikmachine/EEG-Demo/lstm_model_dominance.h5"
		filename.set(file_path)
		model = load_model(file_path)
		yhat = model.predict_classes(Xnew)
		#yhat=1
		if yhat == 1:
			mssg="high dominance"
			recomend(fv.get())

		else :
			mssg="low dominance"
			tk.messagebox.showinfo("IT works"," \n\n The predicted class for input is : ["+mssg+"]")
		recomend(fv.get())

	if (fv.get()==4):
		file_path="/home/vikmachine/EEG-Demo/lstm_model_liking.h5"
		filename.set(file_path)
		model = load_model(file_path)
		yhat = model.predict_classes(Xnew)
		#yhat=1
		if yhat == 1:
			mssg="high dominance"
			recomend(fv.get())

		else :
			mssg="low dominance"
			tk.messagebox.showinfo("IT works"," \n\n The predicted class for input is : ["+mssg+"]")

		


#first button press list videos
def initial_watch():

	dataframe = pd.read_excel('/home/vikmachine/video_list.xls')

	dataframe=dataframe[['Online_id', 'Title','Youtube_link', 'Experiment_id', 'AVG_Valence', 'STD_Valence', 'Q1_Valence', 'Q2_Valence',
	       'Q3_Valence', 'AVG_Arousal', 'STD_Arousal', 'Q1_Arousal', 'Q2_Arousal',
	       'Q3_Arousal', 'AVG_Dominance', 'STD_Dominance', 'Q1_Dominance',
	       'Q2_Dominance', 'Q3_Dominance']]

	global value
	names=['Jungle Drum',
	'Scotty Doesnt Know',
	'Say Hey (I Love You)',
	'Blame It On The Boogie',
	'Miniature Birds',
	'First Day Of My Life',
	'Im Yours',
	'Butterfly Nets',
	'Normal',
	'How To Fight Loneliness',
	'Darkest Things',
	'Goodbye My Lover',
	'Goodbye My Almost Lover',
	'The Weight Of My Words',
	'The One I Once Was',
	'The Beautiful People',
	'Bastard Set Of Dreams',
	'Love Shack',
	'Song 2',
	'First Date',
	'Satisfaction',
	'Moon Safari',
	'What A Wonderful World',
	'Me Gustas Tu',
	'Fuck You',
	'Love Story',
	'I Want To Break Free',
	'Rain',
	'Breathe Me',
	'Hurt',
	'May It Be (Saving Private Ryan)',
	'A Hardcore State Of Mind',
	'Gloomy Sunday',
	'Procrastination On The Empty Vessel',
	'Refuse Resist',
	'Scorched Earth Erotica',
	'Carving A Giant',
	'My Funeral',
	'Bombtrack',
	'My Apocalypse'
	 ]

	links=['http://www.youtube.com/watch?v=iZ9vkd7Rp-g',
	 'http://www.youtube.com/watch?v=51ncDQYxsm8',
	 'http://www.youtube.com/watch?v=eoaTl7IcFs8',
	 'http://www.youtube.com/watch?v=nb1u7wMKywM',
	 'http://www.youtube.com/watch?v=_iEnN9ip1Qk',
	 'http://www.youtube.com/watch?v=zwFS69nA-1w',
	 'http://www.youtube.com/watch?v=EkHTsc9PU2A',
	 'http://www.youtube.com/watch?v=B8eI64H1Cqk',
	 'http://www.youtube.com/watch?v=A-BSL5Av89w',
	 'http://www.youtube.com/watch?v=zLDPhPrr5Ig',
	 'http://www.youtube.com/watch?v=ijLKoqN5_EY',
	 'http://www.youtube.com/watch?v=wVyggTKDcOE',
	 'http://www.youtube.com/watch?v=EDEEzS7OV2k',
	 'http://www.youtube.com/watch?v=G-k19OCq7vE',
	 'http://www.youtube.com/watch?v=O0yoxveh7Tg',
	 'http://www.youtube.com/watch?v=Ypkv0HeUvTc',
	 'http://www.youtube.com/watch?v=HdH8NYsDwWk',
	 'http://www.youtube.com/watch?v=leohcvmf8kM',
	 'http://www.youtube.com/watch?v=WlAHZURxRjY',
	 'http://www.youtube.com/watch?v=vVy9Lgpg1m8',
	 'http://www.youtube.com/watch?v=eoRiVwFP02s',
	 'http://www.youtube.com/watch?v=kxWFyvTg6mc',
	 'http://www.youtube.com/watch?v=3orLNBS2ZbU',
	 'http://www.youtube.com/watch?v=mzgjiPBCsss',
	 'http://www.youtube.com/watch?v=S0zMHf7J15g',
	 'http://www.youtube.com/watch?v=8xg3vE8Ie_E',
	 'http://www.youtube.com/watch?v=EVYgRPfC9nQ',
	 'http://www.youtube.com/watch?v=15kWlTrpt5k',
	 'http://www.youtube.com/watch?v=ghPcYqn0p4Y',
	 'http://www.youtube.com/watch?v=zHqZmtAD0lQ',
	 'http://www.youtube.com/watch?v=xxvw5vrJxos',
	 'http://www.youtube.com/watch?v=6fxI1JsXv1A',
	 'http://www.youtube.com/watch?v=KzWVWY5QUzg',
	 'http://www.youtube.com/watch?v=Y9eXzmVzDR8',
	 'http://www.youtube.com/watch?v=AJQ4sg3VCsk',
	 'http://www.youtube.com/watch?v=ScD62dww0Fw',
	 'http://www.youtube.com/watch?v=aFn26ntmSsg',
	 'http://www.youtube.com/watch?v=TEVodXzNmPM',
	 'http://www.youtube.com/watch?v=Tu1wAP2Baco',
	 'http://www.youtube.com/watch?v=mZM-d2qD15E']

	popup = tk.Toplevel(root)

	i1 = 0
	tk.Label(popup, text="Select a video to watch:")

	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[0],0))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[1],1))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[2],2))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[3],3))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[4],4))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[5],5))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[6],6))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[7],7))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[8],8))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[9],9))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[10],10))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[11],11))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[12],12))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[13],13))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[14],14))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[15],15))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[16],16))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[17],17))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[18],18))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[19],19))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[20],20))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[21],21))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[22],22))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[23],23))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[24],24))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[25],25))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[26],26))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[27],27))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[28],28))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[29],29))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[30],30))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[31],31))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[32],32))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[33],33))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[34],34))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[35],35))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[36],36))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[37],37))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[38],38))
	i1=i1+1
	link = tk.Label(popup, text=names[i1], fg="blue")
	link.grid(row=i1, column=1)
	link.bind("<Button-1>", lambda e: callback(links[39],39))


def recomend(value):

	print(value)
	global linknum
	print (linknum)
	popup1 = tk.Toplevel(root)
	df = pd.read_excel('/home/vikmachine/EEG-Demo/video_list.xls')

	print("Column headings:")
	print(df.columns)
	df=df[['Online_id', 'Title','Youtube_link', 'Experiment_id', 'AVG_Valence', 'STD_Valence', 'Q1_Valence', 'Q2_Valence',
	       'Q3_Valence', 'AVG_Arousal', 'STD_Arousal', 'Q1_Arousal', 'Q2_Arousal',
	       'Q3_Arousal', 'AVG_Dominance', 'STD_Dominance', 'Q1_Dominance',
	       'Q2_Dominance', 'Q3_Dominance']]
	df=df.dropna(subset=['Experiment_id'])
	df1=df[['Experiment_id','Title','Youtube_link', 'AVG_Valence',  'AVG_Arousal',  'AVG_Dominance', ]]
	df1=df1.sort_values(by=['Experiment_id'])
	if (value==1):
		
		newval = df1['Experiment_id'].values[linknum+1]
		avg = df.loc[df1['Experiment_id'] == linknum+1, 'AVG_Valence']
		print("\n")
		print(avg)
		print("\n")
		df1=df1.sort_values(by=['AVG_Valence'])

		Index_lower = df1[df1['AVG_Valence']>=float(avg)-0.2].index.tolist()
		Index_upper = df1[df1['AVG_Valence']<=float(avg)+0.7].index.tolist() 
		st3 = [value for value in Index_lower if value in Index_upper]
		new_names=df1.Title.loc[st3].values.tolist()

		link = tk.Label(popup1, text="{The predicted class for input is : [High Valence] \n Your video recomendations are as follows\n")
		link.grid(row=0, column=1)
		i=1
		print("\n")
		print(new_names)
		for name in new_names:
		    lb = tk.Label(popup1, text=name)
		    lb.grid(row=i, column=1)
		    i += 1

	elif(value==2):

		newval = df1['Experiment_id'].values[linknum+1]
		avg = df.loc[df1['Experiment_id'] == linknum+1, 'AVG_Arousal']
		print("\n")
		print(avg)
		print("\n")
		df1=df1.sort_values(by=['AVG_Arousal'])

		Index_lower = df1[df1['AVG_Arousal']>=float(avg)-0.2].index.tolist()
		Index_upper = df1[df1['AVG_Arousal']<=float(avg)+0.7].index.tolist() 
		st3 = [value for value in Index_lower if value in Index_upper]
		new_names=df1.Title.loc[st3].values.tolist()

		link = tk.Label(popup1, text="{The predicted class for input is : [High Arousal] \n Your video recomendations are as follows\n")
		link.grid(row=0, column=1)
		i=1
		print("\n")
		print(new_names)
		for name in new_names:
		    lb = tk.Label(popup1, text=name)
		    lb.grid(row=i, column=1)
		    i += 1

	elif(value==3):

		newval = df1['Experiment_id'].values[linknum+1]
		avg = df.loc[df1['Experiment_id'] == linknum+1, 'AVG_Dominance']
		print("\n")
		print(avg)
		print("\n")
		df1=df1.sort_values(by=['AVG_Dominance'])

		Index_lower = df1[df1['AVG_Dominance']>=float(avg)-0.2].index.tolist()
		Index_upper = df1[df1['AVG_Dominance']<=float(avg)+0.7].index.tolist() 
		st3 = [value for value in Index_lower if value in Index_upper]
		new_names=df1.Title.loc[st3].values.tolist()

		link = tk.Label(popup1, text="{The predicted class for input is : [High Dominance] \n Your video recomendations are as follows\n")
		link.grid(row=0, column=1)
		i=1
		print("\n")
		print(new_names)
		for name in new_names:
		    lb = tk.Label(popup1, text=name)
		    lb.grid(row=i, column=1)
		    i += 1

	elif (value==4):
		newval = df1['Experiment_id'].values[linknum+1]
		avg = df.loc[df1['Experiment_id'] == linknum+1, 'AVG_Valence']
		print("\n")
		print(avg)
		print("\n")
		df1=df1.sort_values(by=['AVG_Valence'])

		Index_lower = df1[df1['AVG_Valence']>=float(avg)-0.2].index.tolist()
		Index_upper = df1[df1['AVG_Valence']<=float(avg)+0.7].index.tolist() 
		st3 = [value for value in Index_lower if value in Index_upper]
		new_names=df1.Title.loc[st3].values.tolist()

		link = tk.Label(popup1, text="{The predicted class for input is : [High liking] \n Your video recomendations are as follows\n")
		link.grid(row=0, column=1)
		i=1
		print("\n")
		print(new_names)
		for name in new_names:
		    lb = tk.Label(popup1, text=name)
		    lb.grid(row=i, column=1)
		    i += 1




#GUI for main window
root = tk.Tk()

tk.Button(root, text='Select Video',command=initial_watch).grid(row=0,column=0)

tk.Label(root, text='EEG File Path').grid(row=1, column=0)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v, width=75).grid(row=1,columnspan=3, column=1)
tk.Button(root, text='Browse Recording',command=import_csv_data).grid(row=1, column=4)


start = tk.IntVar()
tk.Label(root, text="""EEG-Time from""").grid(row=2, column=0)
entry = tk.Entry(root, textvariable=start).grid(row=2, column=1)

tk.Label(root, text="""Till""").grid(row=2, column=2)

end = tk.IntVar()
entry = tk.Entry(root, textvariable=end).grid(row=2, column=3)
tk.Button(root, text='crop Recording',command= lambda: crop(start,end)).grid(row=2, column=4)
tk.Button(root, text='create images',command=process).grid(row=3, column=4)
tk.Label(root, text="""_____________________________________________________________________________________________________________________""").grid(row=4, column=0, columnspan=5)


filename = tk.StringVar()
tk.Label(root, text="""Choose a prediction model:""").grid(row=5, column=2)

fv = tk.IntVar()
tk.Radiobutton(root, text="Valency",  variable=fv,  value=1).grid(row=6, column=2)
tk.Radiobutton(root, text="Arrousal",   variable=fv,  value=2).grid(row=7, column=2)
tk.Radiobutton(root, text="Dominance",  variable=fv,  value=3).grid(row=8, column=2)
tk.Radiobutton(root, text="Liking",   variable=fv,  value=4).grid(row=9, column=2)



#tk.Button(root, text='choose model',command=model_select).grid(row=2, column=2)
#entry = tk.Entry(root, textvariable=filename).grid(row=2, column=2)


tk.Button(root, text='Predict EEG signal',command=predict).grid(row=11, column=2)

root.mainloop()



