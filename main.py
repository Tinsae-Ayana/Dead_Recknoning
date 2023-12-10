import yaml
import scipy.integrate as integrate
from scipy.signal import detrend
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# read yaml file and return the data as python dictionary
def read_yaml(filepath) :  
    print('reading yaml file...')
    with open(filepath, 'r') as file:
     data_list = list(yaml.safe_load_all(file))
    return data_list

# extract the time stamp of the measurement
def extract_time(dic) : 
    print(dic['header']['stamp']['sec'] + (dic['header']['stamp']['nanosec'] / 1000000000))
    return dic['header']['stamp']['sec'] + (dic['header']['stamp']['nanosec'] / 1000000000)

# extract the accleration for each axis
def extract_acc(acc,axis,bias, sf) :
    # print('this is for :' + axis,acc['linear_acceleration'][axis])
    return (acc['linear_acceleration'][axis] - bias) / (1 + sf)

# estimate velocity
def est_vel(acc, timepoints) :
    print('estimating velocity by integrating acc...')
    vel = np.zeros_like(acc)
    # method-1
    # for i in range(len(acc)):
    #     vel[i] = integrate.simps(acc[0:i+1],timepoints[0:i+1])
    # method-2
    # for i in range(len(vel)) :
    #      vel[i] = np.trapz(acc[0:i+1],timepoints[0:i+1],axis=None)
    vel = integrate.cumtrapz(acc,x=timepoints,initial=0)
    return vel

# estimate displacement
def est_position(vel ,timepoints) :
    print('estimating position by integrating vel...')
    position = np.zeros_like(vel)
    # method-1
    # for i in range(len(vel)) :
    #     position[i] = integrate.simps(vel[0:i+1],timepoints[0:i+1])
    # method-2
    # for i in range(len(vel)) :
    #      position[i] = np.trapz(vel[0:i+1],timepoints[0:i+1],axis=None)
    position  = integrate.cumtrapz(vel,timepoints,initial=0)
    return position

# plot 2d data
def plot_2D(xdata,ydata, xlabel, ylabel,title):
    plt.plot(xdata,ydata,linestyle ='-', color ='g')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.show()

# plot 3d data
def plot_3D(xdata, ydata, zdata):
    pass

if __name__ == "__main__" :

    # calibartion datas
    bias_x = 0.00273906
    bias_y = 0.01719892
    bias_z = -0.002825
    sf_x   = 0.00068967
    sf_y   = 0.0003230
    sf_z   = 0.00147083
    file_path = 'imu_data.yaml'
    data = read_yaml(filepath=file_path)
    print('extracting time points....')
    time_points = list(map(lambda x: extract_time(x),data))
    # get the acceleration
    print('extracting acceleration data...')
    acc_x =  detrend(list(map(lambda x: extract_acc(x,'x', bias_x, sf_x),data))) # get acceleration in x direction
    acc_y =  detrend(list(map(lambda x: extract_acc(x,'y', bias_y, sf_y),data)))  # get acceleration in y direction
    # acc_z = list(map(lambda x: extract_acc(x,'z', bias_z, sf_z),data))  # get accelration in z direction
    # get the velocity
    vel_x = detrend(est_vel(acc_x,time_points))
    vel_y = detrend(est_vel(acc_y,time_points))
    # vel_z = est_vel(acc_y,time_points)
    # # get the position
    pos_x = est_position(vel_x,time_points)
    pos_y = est_position(vel_y,time_points)
    # pos_z = est_position(vel_z,time_points)

    plot_2D(pos_x,pos_y,xlabel='x-position', ylabel='y-position', title='position')
    # plot_2D(time_points,vel_x,xlabel='time', ylabel='x-position', title='x-position')
    # plot_2D(time_points,vel_y,xlabel='time', ylabel='y-position', title='xy-plane')


                
