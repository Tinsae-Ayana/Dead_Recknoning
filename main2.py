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

# extract time
def extract_time(dic) : 
    return dic['header']['stamp']['sec'] + (dic['header']['stamp']['nanosec'] / 1000000000) #change the nano to second by dividing 10^9

# extract quternions from yaml file
def extract_quternions(dic) :
   orientation = dic['orientation']
   return orientation

# change the quternion to rotation matrix
def quaternion_to_matrix(qtrn) :
   q0 = qtrn['w']
   q1 = qtrn['x']
   q2 = qtrn['y']
   q3 = qtrn['z']
   # First row of the rotation matrix
   r00 = 2 * (q0 * q0 + q1 * q1) - 1
   r01 = 2 * (q1 * q2 - q0 * q3)
   r02 = 2 * (q1 * q3 + q0 * q2)
    
   # Second row of the rotation matrix
   r10 = 2 * (q1 * q2 + q0 * q3)
   r11 = 2 * (q0 * q0 + q2 * q2) - 1
   r12 = 2 * (q2 * q3 - q0 * q1)
    
   # Third row of the rotation matrix
   r20 = 2 * (q1 * q3 - q0 * q2)
   r21 = 2 * (q2 * q3 + q0 * q1)
   r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
   # 3x3 rotation matrix
   rot_matrix = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])          
   return rot_matrix

# extract the accleration for each axis
def extract_acc(dic,bias, sf) :
    qtrn = extract_quternions(dic)
    rotation_matrix = quaternion_to_matrix(qtrn=qtrn)
    print("rotation matrix: ",rotation_matrix.shape)
    raw_data = dic['linear_acceleration']
    acc = np.zeros(3 ,dtype=np.float64)
    acc[0] = raw_data['x']
    acc[1] = raw_data['y']
    acc[2] = raw_data['z']
    # apply scale factor and bias before transforming it
    for i in range(3):
        acc[i] = (acc[i] - bias[i]) / (1 + sf[i])
    acc_tranformed = rotation_matrix @ acc.reshape((-1,1)) # 3,1 column vector
    return acc_tranformed

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
    plt.scatter(xdata[0], ydata[0], color='red', marker='o', label='Start')
    plt.scatter(xdata[-1], ydata[-1], color='green', marker='x', label='End')
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


def main() :
    # calibartion datas
    bias_x =0 #0.00273906
    bias_y =0 #0.01719892
    bias_z =0 #-0.002825
    bias = [bias_x, bias_y, bias_z]
    sf_x   = 0 #0.00068967
    sf_y   = 0 #0.0003230
    sf_z   = 0 #0.00147083
    sf = [sf_x, sf_y, sf_z]
    gravity = 9.8
    file_path = 'imu_data.yaml'
    data = read_yaml(filepath=file_path)
    print('extracting time points....')
    time_points = list(map(lambda x: extract_time(x),data))
    # get the acceleration
    print('extracting acceleration data...') # this is goint to be changed
    acclrtn = list(map(lambda a : extract_acc(a, bias=bias,sf=sf), data))
    acc_x =  (list(map( lambda d : d[0][0], acclrtn))) # get acceleration in x direction
    acc_y =  (list(map( lambda d : d[1][0], acclrtn)))  # get acceleration in y direction
    acc_z =  detrend(list(map( lambda d : d[2][0], acclrtn)))  # get accelration in z direction
    # get the velocity
    vel_x = (est_vel(acc_x,time_points))
    vel_y = (est_vel(acc_y,time_points))
    vel_z = (est_vel(acc_y,time_points))
    # get the position
    pos_x = est_position(vel_x,time_points)
    pos_y = est_position(vel_y,time_points)
    pos_z = est_position(vel_z,time_points)
    # plot
    plot_2D(pos_x,pos_y,xlabel='x-position', ylabel='y-position', title='position-xy-plane')
    # plot_2D(time_points,vel_x,xlabel='time', ylabel='vel-x', title='velocity-x')
    # plot_2D(time_points,vel_y,xlabel='time', ylabel='vel-y', title='velocity-y')
    # plot_3D(pos_x, pos_y, pos_z)
    # plot_2D(time_points,acc_x,xlabel='time', ylabel='vel-y', title='velocity-y')

main()