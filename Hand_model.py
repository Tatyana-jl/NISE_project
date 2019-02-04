import numpy as np
import pandas as pd
from collections import *

class Hand_model:

	def __init__(self):
		self.fingers = ['index','middle','ring','little']
		'''
		Angles for fingers: 
		    - MCP_fe: metacarpophalangeal flexion/extension 
		    - MCP_aa: metacarpophalangeal abduction/adduction
		    - PIP: Proximal-interphalangeal 

		Angles for thumb:
		    - TMC_fe: trapeziometacarpal flexion/extension
		    - TMC_aa: trapeziometacarpal abduction/adduction
		    - MCP_fe: metacarpophalangeal flexion/extension
		'''
		self.angles = {'MCP_fe','MCP_aa','PIP'}
		self.angles_thumb = {'TMC_fe','TMC_aa','MCP_fe'}

	def create_posture(self):
		# Initialize angles to 0 (Rest position of the hand)
		default_params = defaultdict(dict)

		for finger in self.fingers:
		    for angle in self.angles:
		        default_params[finger][angle] = 0
		# Thumb
		for angle in self.angles_thumb:
		    default_params['thumb'][angle]=0

		# Create a dataframe from default_parameters
		default_posture = pd.DataFrame.from_dict(default_params)

		return default_posture

	def create_samples(self,letter,n_samples,variance):
		# Convert to array and eliminate nan values
	    array = letter.as_matrix().ravel()
	    array = array[~np.isnan(array)]
	    
	    # Create samples and add gausian noise
	    data = np.tile(array, (n_samples,1))
	    noise = np.random.normal(0, variance, data.shape)
	    samples = data+noise
	    
	    return samples

