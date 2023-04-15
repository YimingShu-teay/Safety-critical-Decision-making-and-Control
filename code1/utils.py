import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.ticker import MaxNLocator
import imageio
import os

MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param
GOAL_DIS_X = 1.0  # goal distance
GOAL_DIS_Y = 1.0  # goal distance
STOP_SPEED = 2.0  # stop speed
MAX_TIME = 500.0  # max simulation time
TARGET_SPEED = 15.0  # [m/s] target speed
N_IND_SEARCH = 20  # Search index number
DESIRED_Y = -5.0
DESIRED_X = 1000.0

DT = 0.1  # [s] time tick
# Vehicle parameters
LENGTH = 4.4  # [m]
WIDTH = 1.9  # [m]
BACKTOWHEEL = 0.9  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 1.9  # [m]
def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    
    outline = np.matrix([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                         [WIDTH / 2, WIDTH / 2,  -WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.matrix([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                          [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                      [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
                      [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T * Rot2).T
    fl_wheel = (fl_wheel.T * Rot2).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T * Rot1).T
    fl_wheel = (fl_wheel.T * Rot1).T

    outline = (outline.T * Rot1).T
    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(x, y, "*")
    
    
def check_goal(ugv, desired_x, desired_y):
    
    dx = desired_x - ugv.x
    dy = desired_y - ugv.y

    if (dx <= GOAL_DIS_X):
        isgoal_x = True
    else:
        isgoal_x = False
        
    if (dy <= GOAL_DIS_Y):
        isgoal_y = True
    else:
        isgoal_y = False

    if isgoal_x and isgoal_y :
        return True

    return False

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def png_count(addr):
    path =  addr
    files = os.listdir(path)   
    num_png = -1     
    for file in files:
        if file.endswith(".png"):
            num_png = num_png + 1
    return num_png

def animation_generation(addr,now_time):
    pic_num = png_count(addr)
    with imageio.get_writer(uri=addr+'\\{}.gif'.format(now_time), mode='I', fps=15) as writer:
        for i in range(pic_num):
            writer.append_data(imageio.imread((addr + "\\{}.png").format(i)))


def Fig_delete(addr):
    path = addr
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"):             
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def draw_fig(dirpath,delta_list,variable_name,unit_name):
    length = len(delta_list)
    t_list = np.zeros(length)
    for i in range(length):
        t_list[i] = 0.1*i
    fig, ax = plt.subplots(figsize=(17,6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    font = {'family': 'Times New Roman',

            'size': 13,
            }
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(t_list,delta_list,c="orange",linewidth =4.0)
    plt.tick_params(width=0.5, labelsize=20)
    plt.xlabel('Time(s)',fontdict=font,fontsize =26)
    if variable_name == "delta":
        plt.ylabel('$\{}$ ({})'.format(variable_name,unit_name),fontdict=font,fontsize =24)
    elif variable_name == "V":
        plt.ylabel('${}$ ({})'.format(variable_name,unit_name),fontdict=font,fontsize =24)
    else:
        plt.ylabel('{}'.format(variable_name),fontdict=font,fontsize =24)
    legend = plt.legend(["Normal Scenario"])
    plt.legend(["Normal Scenario"],loc="upper right")
    plt.rcParams.update({'font.size':21})
    plt.savefig(dirpath+"\{}.png".format(variable_name),dpi=600)
    plt.savefig(dirpath+"\{}.svg".format(variable_name),dpi=600)
    plt.show()

def initial_excel(dir):
    df_k = pd.DataFrame({"500":[],"800":[]})
    df_k.to_excel(dir)

def save_to_excel(dir,Time_dict,label):
    value_list = []
    for key,value in Time_dict.items():
        value_list.append(value)
    df = pd.read_excel(dir) 
    df = df[["500","800"]]
    if label == "500":
       df.loc[len(df),"500"] = value_list[-1]
    elif label == "800":
       df.loc[len(df)-1,"800"] = value_list[-1]
    df.to_excel(dir)
    
    
    