

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from hypercube_data import Cube_Read

from hypercube_data import cube
#from __future__ import print_function


from hypercube_data import Cube_Read
from hypercube_data import cube

import csv
import os
import pandas as pd

from matplotlib.lines import Line2D

###########################################################
#Datei einlesen


test_path = ['D:\Arbeit\ÖphStud\Ösphagus/2017_09_28_14_38_29',
    'D:/Arbeit/ÖphStud/Ösphagus/2018_08_03_11_36_55',
             'D:/Arbeit/ÖphStud/Ösphagus/2018_08_03_11_37_47',
             'D:/Arbeit/ÖphStud/Ösphagus/2018_08_03_11_38_29',
             'D:/Arbeit/ÖphStud/Ösphagus/2018_03_22_11_49_23', ]
for path in test_path:
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith(".dat"):
            with open(os.path.join(path, file), newline='')  as filex:
                filename=filex.name

                #rediction = model.predict(learn_data)


                spectrum_data, pixely = Cube_Read(filename,wavearea=100, Firstnm=1,Lastnm=100).cube_matrix()
                spectrum_data1, pixely = Cube_Read(filename,wavearea=100, Firstnm=1,Lastnm=100).cube_matrix()

                #abc = prediction.reshape((640, pixely))
                for x in range(640):
                    for y in range(pixely):
                        # 640,480,100
                        A=np.average(spectrum_data[x, y, 2:14])
                        C=np.average(spectrum_data[x, y, 30:42])
                        B=-np.log(A/C)
                        B=B- 0.1
                        B=B/ (1.6)
                        if np.average(spectrum_data[x, y,:])<0.1 or A>0.7 or B<0:
                            spectrum_data1[x, y, 1] = 3

                colors = np.zeros((640, pixely, 4))
                for i in range(pixely):
                    for j in range(640):


                        if spectrum_data1[j, i,1] == 3:
                            colors[j, i, :] = [0, 0, 0, 1]  # black background

                plot = cube(
                    filename,wavearea=100, Firstnm=0,Lastnm=99).cube_plot()
                colors = np.rot90(colors)
                plt.imshow(colors)



