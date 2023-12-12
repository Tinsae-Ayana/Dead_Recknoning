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
    return dic['header']['stamp']['sec'] + (dic['header']['stamp']['nanosec'] / 1000000000) # change the nano to second by dividing 10^9

# extract the accleration for each axis
def extract_acc(acc,axis,bias, sf) :
    return acc['linear_acceleration'][axis]

# estimate velocity
def est_vel(acc, timepoints) :
    print('estimating velocity by integrating acc...')
    vel = np.zeros_like(acc)
    vel = integrate.cumtrapz(acc,x=timepoints,initial=0)
    return vel

# estimate displacement
def est_position(vel ,timepoints) :
    print('estimating position by integrating vel...')
    position = np.zeros_like(vel)
    position  = integrate.cumtrapz(vel,timepoints,initial=0)
    return position

# plot 2d data
def plot_2D(xdata,ydata, xlabel, ylabel,title):
    plt.plot(xdata,ydata,linestyle ='-', color ='g')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.savefig('{}.png'.format(title))
    plt.show()

# plot 3d data
def plot_3D(xdata, ydata, zdata):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(xdata,ydata,zdata,label='Trajectory')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.savefig('Trajectory_plot.png')
    plt.show()

if __name__ == "__main__" :
    # calibartion datas
    bias_x = 0.00273906
    bias_y = 0.01719892
    bias_z = -0.002825
    sf_x   = 0.00068967
    sf_y   = 0.0003230
    sf_z   = 0.00147083
    gravity = 9.8
    file_path = 'imu_data.yaml'
    data = read_yaml(filepath=file_path)
    print('extracting time points....')
    time_points = list(map(lambda x: extract_time(x),data))
    # get the acceleration
    print('extracting acceleration data...')
    acc_x = (list(map(lambda x: extract_acc(x,'x', bias_x, sf_x),data)))  # get acceleration in x direction
    acc_y = (list(map(lambda x: extract_acc(x,'y', bias_y, sf_y),data)))  # get acceleration in y direction
    acc_z = (list(map(lambda x: extract_acc(x,'z', bias_z, sf_z),data)))  # get accelration in z direction
    # get the velocity
    vel_x = (est_vel(acc_x,time_points))
    vel_y = (est_vel(acc_y,time_points))
    vel_z = (est_vel(acc_y,time_points))
    # get the position
    pos_x = est_position(vel_x,time_points)
    pos_y = est_position(vel_y,time_points)
    pos_z = est_position(vel_z,time_points)

    plot_2D(pos_x,pos_y,xlabel='x-position', ylabel='y-position', title='position-xy-plane')
       
