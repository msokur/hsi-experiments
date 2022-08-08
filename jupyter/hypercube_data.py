import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn import preprocessing
# Example:Plot image
#from hypercube_data import cube
# im = cube(
#                        filename, wavearea=100, Firstnm=0, Lastnm=100).RGB_Image()
#                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#Example: get sepctrum data
#from hypercube_data import Cube_Read

#spectrum_data, pixel = Cube_Read(fileDat_Name, wavearea=100,
#                                                                                 Firstnm=0,
#                                                                                 Lastnm=100).cube_matrix()


class Cube_Read(object):
    def __init__(self, file_address, wavearea, Firstnm,Lastnm):
        self.file_address = file_address
        self.wavearea = wavearea
        self.Firstnm=Firstnm
        self.Lastnm=Lastnm
    
    def read_cube_dimension(file_adress):


        dim = np.fromfile(file_adress, dtype='>i4', count=3)

        return dim
        
    def read_cube(self):
        data_type = np.dtype ('float32').newbyteorder ('>')
        x = np.fromfile (self.file_address, dtype=data_type)
    
        spectrum = x[3:] #eliminating the 3 leading zeros 
        
        #eliminate negative values
        spectrum[spectrum < 0.00001] = 0.00001
        #eliminate huge values (assume 10 looking at the graphs)
        spectrum[spectrum>10] = 10
                
        return spectrum
    
    def cube_matrix(self):
        spectrum_data = self.read_cube()
        #reorganise the data as the origibal HSI cube
        pixelXSize=int(np.size(spectrum_data)/100)
        pixelYSize=int(round(pixelXSize/640))
        data = spectrum_data.reshape((640,pixelYSize,100))
        #data = np.vstack([np.hstack(cell) for cell in spectrum_data])



        return np.rot90(data[..., self.Firstnm:self.Lastnm+1]), pixelYSize
    
    def cube_matrix_learn(self):
        spectrum_data = self.read_cube()
        #reorganise spectrum for classification test
        x = int(np.size(spectrum_data) / 100)
        data = spectrum_data.reshape((x,100))
        return data,x
    
    def cube_SG_learn(self):
        spectrum_data = self.read_cube()
        pixelXSize = int(np.size(spectrum_data) / 100)
        data = spectrum_data.reshape((pixelXSize,100))
        data_SG = savgol_filter(data, 9, 2, mode='nearest', axis=1)
        return data_SG
    
    def cube_SNV_learn(self):
        spectrum_data = self.cube_SG_learn()
        data_SNV = preprocessing.scale(spectrum_data, axis=1)
        return data_SNV
    
    def cube_snv_matrix(self):
        spectrum,pixely = self.cube_matrix_learn()
        mean = np.mean(spectrum, axis=0)
        print(mean.size)
        std = np.std(spectrum, axis=0)
        pixelXSize = int(np.size(spectrum) / 100)
        #data = np.zeros((307200,100))
        #data = np.zeros((307200, 80))
        data = np.zeros((pixelXSize, 100))
        print(mean.shape)
        for n in range(1,100):
            for i in range(pixelXSize):
                data[i,n] = (spectrum[i,n]-mean[n])/std[n]
        data=data[:,self.Firstnm:self.Lastnm]
        return data,pixely
    

    
class cube(object):
    def __init__(self,address, wavearea, Firstnm,Lastnm):
        self.address = address
        self.wavearea = wavearea
        self.Firstnm = Firstnm
        self.Lastnm = Lastnm
    
    #https://stackoverflow.com/questions/1627376/how-do-i-extract-a-ieee-be-binary-file-embedded-in-a-zipfile
    
    def cube_plot(self):
        from skimage import exposure
        #the plot is done following the guidance from Tivita TM
        #Dokumentation RGB-Image1.4.vi
        initial_cube,y = Cube_Read(self.address,self.wavearea,self.Firstnm,self.Lastnm).cube_matrix()
        
        #pixel rgb values
        RGB_values = np.zeros((y,640,3), dtype=np.dtype ('float32'))

        #for blue pixel take the 530-560nm
        new_array_blue = initial_cube[:,:,6:13].transpose()
        RGB_values[:,:,2] = (new_array_blue.mean(axis=(0)))*1.5
        #for the green pixel take 540-590nm
        new_array_green = initial_cube[:,:,8:19].transpose()
        RGB_values[:,:,1] = (new_array_green.mean(axis=(0)))*1.5
        #for the red pixel take 585-725nm
        new_array_red = initial_cube[:,:,17:46].transpose()
        RGB_values[:,:,0] = (new_array_red.mean(axis=(0)))*1.5

        #for normalisation of pixels to be between (0,1)
        R_min = np.min(RGB_values[:,:,0])
        R_max = np.max(RGB_values[:,:,0])
    
        G_min = np.min(RGB_values[:,:,1])
        G_max = np.max(RGB_values[:,:,1])
        
        B_min = np.min(RGB_values[:,:,2])
        B_max = np.max(RGB_values[:,:,2])
        
        #scaled RGB values
        scaled_RGB = np.zeros((y,640,3), dtype=np.dtype ('float32'))
        scaled_RGB[:,:,2] = (RGB_values[:,:,2]-B_min)/(B_max-B_min)
        scaled_RGB[:,:,1] = (RGB_values[:,:,1]-G_min)/(G_max-G_min)
        scaled_RGB[:,:,0] = (RGB_values[:,:,0]-R_min)/(R_max-R_min)
        
        #add the gamma-factor
        gamma_corrected = exposure.adjust_gamma(scaled_RGB, 0.5)

        #plot the rgb image
        imgplot = plt.figure()
        plt.grid(False)
       # import cv2
        #imgplot=cv2.imshow('Test image',np.flipud(gamma_corrected))
        imgplot = plt.imshow(np.flipud(gamma_corrected))#((out * 255).astype(np.uint8))
        #plt.show()
        return imgplot