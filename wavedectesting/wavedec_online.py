#Dated: Feb 11 2022
#Author: Sanika S. Khadkikar
#Pennsylvania State University

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import h5py
import argparse
from watpy.coredb.coredb import *
import os
import scipy.signal

#Global constants
time_con_f = 4.975e-6


# Creating the parser
parser = argparse.ArgumentParser()

# Adding arguments
#parser.add_argument('--mode', type=str, required=True, help='Specific mode in which to run this code - Online or Offline')
parser.add_argument('--wavdecm', type=str, required=True, help='Method to be used to perform the wavelet decomposition -- cwt or dwt')
parser.add_argument('--dirname', type=str, required=True, help = 'Name of the directory which will house our results')
parser.add_argument('--cat', type=str, required=True, help = 'Name of the category for which we need reconstructions like a specific EoS (eos) or Mass ratio (mass_ratio)')
parser.add_argument('--specs', required = True, help = 'Value of the parameter for the aforementioned category')
parser.add_argument('--runno', type=str, required=False, help = 'Run to pick from the dpath directory. Default is --runno=R01', default='R01')
parser.add_argument('--gwplot', type=bool, required=False, help = 'Plot the gravitational wave being analysed as a sanity check and save the image', default=False)
parser.add_argument('--scale_min', type=float, required=False, help='Minimum value to be used for the scale array to be fed to the Continuous Wavelet Transform', default=1)
parser.add_argument('--scale_max', type=float, required=False, help='Maximum value to be used for the scale array to be fed to the Continuous Wavelet Transform', default=150)
parser.add_argument('--dscale', type=float, required=False, help='Spacing between the scales', default=1)
parser.add_argument('--sampling_rate', type=int, required=False, help='Resample data uniformly with this given sampling rate')
parser.add_argument('--polikar_diag', type=bool, required=False, help='Making the Robi Polikar wavelet tutorial plot', default=False)
parser.add_argument('--lowbound', type=float, required=False, help='The lower cut off for keeping specific coefficients', default=0.05)


# Parse these arguments
args = parser.parse_args()
os.makedirs('./'+args.dirname, exist_ok=True)

#Watpy stuff
db_path = './CoRe_DB/'
cdb = CoRe_db(db_path)
idb = cdb.idb

if args.cat=='eos':
	key = 'id_eos'
	print('You have chosen to sort by the EoS named', args.specs)
elif args.cat=='mass_ratio':
	key = 'id_mass_ratio'
	print('You have chosen to sort by the mass ratio valued', args.specs)
else:
	print('Invalid keyword input for the specs argument')
	
val = str(args.specs)
mdl_id_eos_DD2 = [i for i in idb.index if i.data[key]== val]
massA_list = []
massB_list = []
lambdat_list = []
eos_list = []


for md in mdl_id_eos_DD2:
    for k, v in md.data.items():
        if k == 'id_mass_starA':
        	massA_list.append(v)
        elif k == 'id_mass_starB':
        	massB_list.append(v)
        elif k == 'id_kappa2T':
        	lambdat_list.append(v)
        elif k == 'id_eos':
        	eos_list.append(v)
        	
   
          

 
dbkeys_id_eos_DD2 = [md.data['database_key'] for md in mdl_id_eos_DD2]
print('The simulation directories according to your specifications are: ', dbkeys_id_eos_DD2)
cdb.sync(dbkeys=dbkeys_id_eos_DD2, verbose=False, lfs=True, prot='https')
sim = cdb.sim
sim_keys = dbkeys_id_eos_DD2
energy_scaling = []
num_samples = [] 
euclidean_overlap = []

if args.wavdecm=='cwt': 
	for i in range(len(sim_keys)):
		massA = massA_list[i]
		massB = massB_list[i]
		lambdat = lambdat_list[i]
		eoss = eos_list[i]
		data = sim[sim_keys[i]]
		name = str(sim_keys[i])
		data_run = data.run['R01']
		data_run.data.write_strain_to_txt()
		data_run.clean_txt()
		gwf = data_run.data.read('rh_22')
		#Defining the gravitational waveform(gwf)
		hplus = gwf[:,1]
		hcross = -gwf[:,2]
		time = gwf[:,8]*time_con_f                         #converting to milliseconds
		
		
		if args.sampling_rate != 0 and args.sampling_rate != None:
			#Resampling the signal
			sampling_rate = args.sampling_rate
			no_samp = int(sampling_rate * time[-1])
			resampled_hplus = scipy.signal.resample(hplus, no_samp)
			resampled_hcross = scipy.signal.resample(hcross, no_samp)
			resamp_time = np.arange(0,no_samp/sampling_rate,1/sampling_rate)
			env = np.sqrt(resampled_hplus**2 + resampled_hcross**2)
			
			#Cutting inspiral off
			env_max = np.argmax(env)

			for l in range(env_max, no_samp):
	    			if env[l] < env[l+1]:
	        			cut_point = l
	        			break
	        
			postmerger = resampled_hplus[cut_point:]
			pm_time = resamp_time[cut_point:]     
		

			#Plotting the gwf
			if args.gwplot:
				fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
				ax[0].plot(resamp_time, resampled_hplus, label = r'$h_+$ - Real Strain of the resampled signal')
				ax[0].plot(resamp_time, env, label = 'Magnitude of Strain of resampled signal')
				ax[0].legend()
				ax[1].plot(pm_time, postmerger)
				ax[0].set_xlabel('Time (s)')
				ax[0].set_ylabel('Strain')
				ax[0].set_title(name + ' ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ ' + str(eoss)+ r' binary with $\tilde{\Lambda}$ equal to '+ str(lambdat)+ ' at ' + str(sampling_rate) + 'Hz')
				ax[1].set_xlabel('Time (s)')
				ax[1].set_ylabel('Strain')
				ax[1].set_title(name + ' Binary neutron star postmerger signal for ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ binary  at ' + str(sampling_rate) + 'Hz')
				plt.savefig('./'+args.dirname +'/gwfplotfor'+ name +'.png')
				#plt.show()


			#Defining sampling period and frequency
			sam_p = 1 / sampling_rate

			#Defining scale for the wavelet analysis
			scales = np.arange(args.scale_min, args.scale_max, args.dscale)
		
			#CWT on the gwf using the Morlet wavelet
			coefs, freqs = pywt.cwt(postmerger, scales, 'morl', sampling_period = sam_p)
			print(freqs)

			#Normalising the coefficient matrix using the Frobenius norm
			norm_mat = (np.abs(coefs))/(np.linalg.norm(coefs))
			new_mat = norm_mat
			
			
			#Keep only top n percentage of values
			maxval = np.max(new_mat)
			counter = 0
			wavs = 0
			imp_scales = []
			
			#Keeping top coefficients
			locap = 0.005
			percent = 90
			
			for sc in range(1,150):
				print(sc)
				diff = new_mat[sc, :] - locap               #Array broadcasting
				no_zeros = len(scales) - len(np.nonzero(diff))		
				vacant = (no_zeros * 100) / len(new_mat[sc, :])
				print(vacant)
				if vacant > percent:
					new_mat = np.delete(new_mat, sc - counter, 0)    #Delete array which is more than X% vacant
					counter = counter + 1
				else:
					print(sc, new_mat.shape)
					imp_scales.append(sc)
					wavs = wavs + 1

			
			print(str(wavs) + 'number of wavelets used')

			#Reconstructing the signal
			reconstructed = np.zeros_like(postmerger)
			for n in range(len(pm_time)):
				reconstructed[n] = np.sum(coefs[:,n]/scales**0.5)
				# Use the reduced matrix instead of the complete matrix
				#reconstructed[n] = np.sum(new_mat[:,n]/scales**0.5)
				
				
			energy_scaling.append(np.max(reconstructed)/np.max(postmerger))
			num_samples.append(no_samp)
			
			reconstructed = reconstructed/np.max(reconstructed)*np.max(postmerger)

			#reconstructed += og.mean()
			
			#Euclidean Overlap
			int_pm_sig = np.trapz(postmerger**2, dx=1/sampling_rate)
			int_rec_sig = np.trapz(reconstructed**2, dx=1/sampling_rate)
			int_both = np.trapz(postmerger*reconstructed, dx=1/sampling_rate)
			euc_ov = int_both/(np.sqrt(int_pm_sig*int_rec_sig))
			euclidean_overlap.append(euc_ov)
			print('The Euclidean overlap for ' + name + ' is ', euc_ov)
			
					
			

			#Plotting the wavelet transform coefficients
			print(counter)
			X = pm_time
			Y = imp_scales
			X, Y = np.meshgrid(X, Y)
			Z1 = norm_mat
			Z2 = new_mat
			fig,ax=plt.subplots(3,1, figsize=[10,10])
			cp = ax[0].contourf(X, Y, Z1, cmap = cm.inferno)
			fig.colorbar(cp, ax=ax[0]) # Add a colorbar to a plot
			ax[0].set_title('Continuous Wavelet transform (all coefficients) - Contour Plot for '+ name)
			ax[0].set_xlabel('Time (s)')
			ax[0].set_ylabel('Scale')
			cp = ax[1].contourf(X, Y, Z2, cmap = cm.inferno)
			fig.colorbar(cp, ax=ax[1]) # Add a colorbar to a plot
			ax[1].set_title('Continuous Wavelet transform (reduced coefficients) - Contour Plot for '+ name)
			ax[1].set_xlabel('Time (s)')
			ax[1].set_ylabel('Scale')
			ax[2].set_xlabel('Time (s)')
			ax[2].set_ylabel('Strain')
			ax[2].set_title('Post reconstruction comparison')
			ax[2].plot(pm_time, postmerger, label='Original Signal')
			ax[2].plot(pm_time, reconstructed,'*', label='Reconstructed Signal')
			ax[2].legend()
			plt.savefig('./'+args.dirname+'/gwave_trans_for' + name + '.png', bbox_inches="tight")
			#plt.show()
			#print('Analysed ' + name + ' successfully')
			
		else:	
			#Cutting inspiral off
			env = np.sqrt(hplus**2 + hcross**2)
			n = len(hplus)
			env_max = np.argmax(env)

			for l in range(env_max, n):
	    			if env[l] < env[l+1]:
	        			cut_point = l
	        			break
	        
			postmerger = hplus[cut_point:]
			pm_time = time[cut_point:]     
		
			print()

			#Plotting the gwf
			if args.gwplot:
				fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
				ax[0].plot(time, hplus, label = r'$h_+$ - Real Strain')
				ax[0].plot(time, env, label = 'Magnitude of Strain')
				#ax[0].plot(resamp_time, resampled, label = 'Resampled signal')
				ax[0].legend()
				ax[1].plot(pm_time, postmerger)
				ax[0].set_xlabel('Time (s)')
				ax[0].set_ylabel('Strain')
				ax[0].set_title(name + ' Binary neutron star merger signal for ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ binary')
				ax[1].set_xlabel('Time (s)')
				ax[1].set_ylabel('Strain')
				ax[1].set_title(name + ' Binary neutron star postmerger signal for ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ binary')
				plt.savefig('./'+args.dirname +'/gwfplotfor'+ name +'.png')
				#plt.show()
				plt.clf()


			#Defining sampling period and frequency
			sam_p = (pm_time[-1] - pm_time[0])/len(pm_time)
			sam_f = 1/sam_p
			print(sam_f)

			#Defining scale for the wavelet analysis
			scales = np.arange(args.scale_min, args.scale_max, args.dscale)
		
			#CWT on the gwf using the Morlet wavelet
			coefs, freqs = pywt.cwt(postmerger, scales, 'morl', sampling_period = sam_p)

			#Normalising the coefficient matrix using the Frobenius norm
			norm_mat = (np.abs(coefs))/(np.linalg.norm(coefs))

			#Reconstructing the signal
			reconstructed = np.zeros_like(postmerger)
			for n in range(len(pm_time)):
				reconstructed[n] = np.sum(coefs[:,n]/scales**0.5)
			#print(np.max(reconstructed)/np.max(postmerger))
			reconstructed = reconstructed/np.max(reconstructed)*np.max(postmerger)
		
			#print(n)

			#reconstructed += og.mean()

			#Plotting the wavelet transform coefficients
			X = pm_time
			Y = scales
			X, Y = np.meshgrid(X, Y)
			Z = norm_mat
			fig,ax=plt.subplots(2,1, figsize=[10,10])
			cp = ax[0].contourf(X, Y, Z, cmap = cm.inferno)
			fig.colorbar(cp, ax=ax[0]) # Add a colorbar to a plot
			ax[0].set_title('Continuous Wavelet transform - Contour Plot for '+ name)
			ax[0].set_xlabel('Time (s)')
			ax[0].set_ylabel('Scale')
			ax[1].set_xlabel('Time (s)')
			ax[1].set_ylabel('Strain')
			ax[1].set_title('Post reconstruction comparison')
			ax[1].plot(pm_time, postmerger, label='Original Signal')
			ax[1].plot(pm_time, reconstructed,'*', label='Reconstructed Signal')
			ax[1].legend()
			plt.savefig('./'+args.dirname+'/gwave_trans_for' + name + '.png', bbox_inches="tight")
			#plt.show()
			plt.clf()
			#print('Analysed ' + name + ' successfully')

elif args.wavdecm=='dwt':
		for i in range(len(sim_keys)):
			massA = massA_list[i]
			massB = massB_list[i]
			lambdat = lambdat_list[i]
			eoss = eos_list[i]
			data = sim[sim_keys[i]]
			name = str(sim_keys[i])
			data_run = data.run['R01']               #Hard coded run number
			data_run.data.write_strain_to_txt()
			data_run.clean_txt()
			gwf = data_run.data.read('rh_22')
			#Defining the gravitational waveform(gwf)
			hplus = gwf[:,1]
			hcross = -gwf[:,2]
			time = gwf[:,8]*time_con_f                         #converting to milliseconds
			
			
			if args.sampling_rate != 0:
				#Resampling the signal
				sampling_rate = args.sampling_rate
				no_samp = int(sampling_rate * time[-1])
				resampled_hplus = scipy.signal.resample(hplus, no_samp)
				resampled_hcross = scipy.signal.resample(hcross, no_samp)
				resamp_time = np.arange(0,no_samp/sampling_rate,1/sampling_rate)
				time_step = 1/sampling_rate

				
				#Cutting inspiral off
				env = np.sqrt(resampled_hplus**2 + resampled_hcross**2)
				env_max = np.argmax(env)

				for l in range(env_max, no_samp):
		    			if env[l] < env[l+1]:
		        			cut_point = l
		        			break
		        
				postmerger = resampled_hplus[cut_point:]             #only real data
				pm_time = resamp_time[cut_point:]

				#Padding signal with zeros to regularise the number of levels
				no_pads = 2048
				n = len(postmerger)
				postmerger = np.append(postmerger, np.zeros(no_pads-n))
				pm_time = np.append(pm_time, np.arange(n/sampling_rate, no_pads/sampling_rate, 1/sampling_rate))


				#print(len(hplus), len(resampled_hplus))

				#Making sure the signal has even number of samples for minimizing discrepancy (a little hard coding)

				#if len(postmerger)%2 != 0:
				#	postmerger = postmerger[:-1]
				#	pm_time = pm_time[:-1]


				#Plotting the gwf
				if args.gwplot:
					fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
					ax[0].plot(resamp_time, resampled_hplus, label = r'$h_+$ - Real Strain of the resampled signal')
					ax[0].plot(resamp_time, env, label = 'Magnitude of Strain of resampled signal')
					ax[0].legend()
					ax[1].plot(pm_time, postmerger)
					ax[0].set_xlabel('Time (s)')
					ax[0].set_ylabel('Strain')
					ax[0].set_title(name + ' ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ ' + str(eoss)+ r' binary with $\tilde{\Lambda}$ equal to '+ str(lambdat)+ ' at ' + str(sampling_rate) + 'Hz')
					ax[1].set_xlabel('Time (s)')
					ax[1].set_ylabel('Strain')
					ax[1].set_title(name + ' Binary neutron star postmerger signal for ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ binary  at ' + str(sampling_rate) + 'Hz')
					plt.savefig('./'+args.dirname +'/gwfplotfor'+ name +'.png')
					#plt.show()


				#Defining sampling period and frequency
				sam_p = 1 / sampling_rate

				#Creating a wavelet object
				waveletobj = pywt.Wavelet('db14')

			
				#DWT on the gwf using the Morlet wavelet
				coefs = pywt.wavedec(postmerger, waveletobj, level=4)
				print(len(postmerger), len(coefs))

				fig = plt.figure(figsize=[15,7])
				fig.tight_layout()

				plt.subplots_adjust(wspace= 0.15, hspace= 0.15)

				sub3 = fig.add_subplot(6,2,(1,2))
				sub3.plot(pm_time, postmerger)
				sub3.set_title('Original GW postmerger signal')

				int_pm_sig = np.trapz(postmerger**2)

				for itr in range(len(coefs)):
					ctr = sampling_rate/2
					sub1 = fig.add_subplot(6,2,3+2*itr)
					sub1.plot(coefs[itr])
					if itr==0:
						sub1.set_title('Level' + str(len(coefs)-itr) + ' ' + '0Hz to ' + str(ctr/2**(len(coefs)-itr-1)) + 'Hz')
					else:
						sub1.set_title('Level' + str(len(coefs)-itr) + ' ' + str(ctr/2**(len(coefs)-itr)) + 'Hz to ' + str(ctr/2**(len(coefs)- itr-1)) + 'Hz')
					coefcp = [np.zeros_like(x) for x in coefs]
					coefcp[itr] = coefs[itr].copy()
					sig = pywt.waverec(coefcp, waveletobj)
					sigcp = sig.copy()
					sigcp = scipy.signal.resample(sigcp, len(postmerger))
					int_rec_sig = np.trapz(sigcp**2)
					int_both = np.trapz(postmerger*sigcp)
					euc_ov = int_both/(np.sqrt(int_pm_sig*int_rec_sig))
					sub2 = fig.add_subplot(6,2,4+2*itr)
					sub2.plot(sig)
					sub2.set_title('Euclidean overlap = ' + str(euc_ov))

				

				#ax[0,1].set_title('Level wise reconstruction')
				fig.suptitle(name + ' BNSPM signal ' + str(massA) + '-' + str(massB) + r' $M_{sol}$ binary  at ' + str(sampling_rate) + 'Hz')
				plt.tight_layout()   


				plt.savefig(args.dirname + '/' + name + '_levelwise.png')

				#Reconstructing the signal using complete information
				sig1 = pywt.waverec(coefs, waveletobj)
				rec =  coefs

				#Robi Polikar figure 4.2

				
				n = 0
				netl = 0
				for arr in coefs :
				    netl = netl + len(arr)

				flat_coef = np.zeros([netl])

				for arr in coefs:
					flat_coef[n:n+len(arr)] = arr
					n += len(arr)

					if args.polikar_diag==True:
						x = np.arange(1,netl+1,1)
						fig, ax = plt.subplots(2,1,figsize=[15,7])
						ax[0].plot(pm_time, postmerger)
						ax[0].set_title('GW signal being used')
						ax[0].set_xlabel('Time')
						ax[0].set_ylabel('Strain magnitude')

						ax[1].plot(x, flat_coef)
						ax[1].set_title('Coefficient amplitude vs DWT samples (mode-smooth padding)')
						ax[1].set_xlabel('DWT coefficients')
						ax[1].set_ylabel('Coefficient Amplitude')

						

						plt.savefig('./'+args.dirname +'/coefamps'+ name +'.png')
						plt.close()

				#Reconstructing signal using lesser coefficients
				# counter = 0
				# for carr in rec:
				#     for i in range(len(carr)):
				#         if np.abs(carr[i]) < args.lowbound: 
				#             carr[i] = 0
				#             counter += 1
				# print(str(counter*100/netl) + '% of coefficients have been successfully ignored')

				# sig2 = pywt.waverec(rec, waveletobj)

				# fig, ax = plt.subplots(2,1,figsize=[15,7])
				# ax[0].plot(pm_time, postmerger, label = 'Original Signal')
				# ax[0].plot(pm_time, sig1, '.', label = 'Reconstruction using all coefficients')
				# int_pm_sig = np.trapz(postmerger**2)
				# int_rec_sig = np.trapz(sig1**2)
				# int_both = np.trapz(postmerger*sig1)
				# euc_ov = int_both/(np.sqrt(int_pm_sig*int_rec_sig))
				# ax[0].set_title('Reconstruction of ' + name + ' using all coefficients with a Euclidean overlap of ' + "{:.2f}".format(euc_ov*100) + '%')
				
				# ax[1].plot(pm_time, postmerger, label = 'Original Signal')
				# ax[1].plot(pm_time, sig2, '.', label = 'Reconstruction using top coefficients')
				# int_rec_sig = np.trapz(sig2**2)
				# int_both = np.trapz(postmerger*sig2)
				# euc_ov = int_both/(np.sqrt(int_pm_sig*int_rec_sig))
				# ax[1].set_title('Reconstruction of ' + name + ' using top coefficients with a Euclidean overlap of ' + "{:.2f}".format(euc_ov*100) + '% ignoring ' + "{:.2f}".format(counter*100/netl) + '% of coefficients')
				# plt.legend()
				# plt.savefig('./'+args.dirname +'/frecons'+ name +'.png')
				# plt.close()
				# #plt.plot(pm_time, postmerger, label = 'original')

else:
	print('Invalid wavelet decomposition method. Please correctly specify either DWT or CWT')

plt.figure()
plt.plot(num_samples, energy_scaling)
plt.xlabel('Number of samples')
plt.ylabel('Scaling factor between reconstructed and the original signal')
plt.savefig('ratios.png')

plt.figure()
plt.plot(num_samples, euclidean_overlap)
plt.xlabel('Number of samples')
plt.ylabel('Euclidean Overlap')
plt.savefig('euc_ov.png')