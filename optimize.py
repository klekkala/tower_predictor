import numpy as np
from scipy.optimize import minimize



def minimize(x, weight):
  x0 = np.array([0.0, 0.0])
  cost = weight(0)*point(x) + weight(1)*terrain(x) + weight(2)*habitat(x) + weight(3)*windspeed(x) + weight(4)*landcost(x)
  res = minimize(cost, x0, method='lm', options={'xtol': 1e-8, 'disp': True})
  return res.x




def point(x):
	return 100 - norm(x)



def terrain(x):
	return 100


def habitat(x):
	return 100

def windspeed(x):
	return 100


def landcost(x):
	return 100
