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

# extract angular velocity
def extract_angvel(data) :
    raw_data = data['angular_velocity']
    angular_vel = np.zeros(3, dtype=np.float64)
    angular_vel[0] = raw_data['x']
    angular_vel[1] = raw_data['y']
    angular_vel[2] = raw_data['z']
    return angular_vel

# compute transfromation matrix
def skew_matrix(dic):
    angular_vel = extract_angvel(dic)
    skew = np.array([[0, -angular_vel[2], angular_vel[1]],
                     [angular_vel[2], 0,  -angular_vel[0]],
                     [-angular_vel[1],    angular_vel[0], 0]
                     ])
    return skew
    
# extract time
def extract_time(dic) : 
    return dic['header']['stamp']['sec'] + (dic['header']['stamp']['nanosec'] / 1000000000) #change the nano to second by dividing 10^9

# extract acceleration in the givin axis
def extract_acc_axis(dic,axis,bias, sf) :
    return (dic['linear_acceleration'][axis] - bias) / (1 + sf)
   
# extract acceleration using velocity vector
def extract(data, timepoints, bias, sf):
    I  = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    R_K = I.copy()
    acc_x = detrend(list(map(lambda x: extract_acc_axis(x,'x', bias = bias[0], sf=sf[0]), data)))
    acc_y = detrend(list(map(lambda x: extract_acc_axis(x,'y', bias = bias[1], sf=sf[1]), data)))
    acc_z = detrend(list(map(lambda x: extract_acc_axis(x,'z', bias = bias[2], sf=sf[2]), data)))
    acc_x_trn = np.zeros(len(acc_x))
    acc_x_trn[0] = acc_x[0]
    acc_y_trn = np.zeros(len(acc_y))
    acc_y_trn[0] = acc_y[0]
    acc_z_trn = np.zeros(len(acc_z))
    acc_z_trn[0] = acc_z[0]
    for i in range(1,len(timepoints)):
        deltaT = timepoints[i]-timepoints[i-1]
        ang_vel = extract_angvel(data=data[i])
        theta_x = ang_vel[0] * deltaT
        theta_y = ang_vel[1] * deltaT
        theta_z = ang_vel[2] * deltaT
        theta   = np.sqrt(theta_x**2 + theta_y**2 + theta_z**2) * deltaT**2
        skew_mtrx = skew_matrix(data[i]) 
        print('Skew matrix : ',skew_matrix)
        S = skew_mtrx * deltaT
        s = np.sin(theta)/theta
        c = (1- np.cos(theta)) / theta ** 2
        R_K1 = R_K @ (I + S * s + c* S  @ S )
        R_K = R_K1
        acc_trn = R_K1 @ np.array([[acc_x[i]],[acc_y[i]],[acc_z[i]]])
        print("type of acc_trn", acc_trn)
        acc_x_trn[i] = acc_trn[0, 0]
        acc_y_trn[i] = acc_trn[1, 0]
        acc_z_trn[i] = acc_trn[2, 0]
    return {'acc_x' : acc_x_trn, 'acc_y' : acc_y_trn, 'acc_z' : acc_z_trn} # acceleration tranformed to initial frame

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
    acclrtn = extract(data=data,bias=bias,sf=sf, timepoints=time_points)
    acc_x =  (acclrtn['acc_x']) # get acceleration in x direction
    acc_y =  (acclrtn['acc_y'])  # get acceleration in y direction
    acc_z =  (acclrtn['acc_z'])  # get accelration in z direction
    # get the velocity
    vel_x = detrend(est_vel(acc_x,time_points))
    vel_y = detrend(est_vel(acc_y,time_points)) 
    vel_z = detrend(est_vel(acc_z,time_points))
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