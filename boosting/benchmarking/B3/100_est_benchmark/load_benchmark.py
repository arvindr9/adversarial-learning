import csv
import numpy as np
from numpy import genfromtxt

total_boosting_time = genfromtxt('total_boosting_time_10.csv', delimiter=',')
fitting_time = genfromtxt('fitting_time_10.csv', delimiter = ',')
adv_generation_time = genfromtxt('adv_generation_time_10.csv', delimiter = ',')
concatenate_time = genfromtxt('concatenate_time_10.csv', delimiter = ',')
per_image_advGen_time = genfromtxt('per_image_advGen_time_10.csv', delimiter = ',')

f = open('../benchmark', 'w')

f.write('Average total boosting time: {}\n'.format(np.mean(total_boosting_time)))
f.write('Average training time : {}\n'.format(np.mean(fitting_time)))
f.write('Average total adversarial image generation time : {}\n'.format(np.mean(adv_generation_time)))
f.write('Average concatenation time :{}\n'.format(np.mean(concatenate_time)))
f.write('Average per adversarial image generation time of one image : {}\n'.format(np.mean(per_image_advGen_time)))