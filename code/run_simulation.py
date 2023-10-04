import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
from IDM_general import *
from KinematicModel import *
from utils import *
import time
import random

initial_position = [random.uniform(10, 20),random.uniform(4.8, 5.8),random.uniform(6, 8),0.0,
                    random.uniform(70, 100),random.uniform(4.8, 5.8),random.uniform(6, 8),0.0,
                    random.uniform(120, 160),random.uniform(4.8, 5.8),random.uniform(6, 8),0.0,
                    random.uniform(10, 40),random.uniform(8.7, 9.2),random.uniform(6, 8),0.0,
                    random.uniform(70, 100),random.uniform(8.7, 9.2),random.uniform(6, 8),0.0,
                    random.uniform(120, 160),random.uniform(8.7, 9.2),random.uniform(6, 8),0.0,
                    random.uniform(20, 40),random.uniform(1.5, 2.1),random.uniform(6, 8),0.0,
                    random.uniform(70, 100),random.uniform(1.5, 2.1),random.uniform(6, 8),0.0,
                    random.uniform(120, 160),random.uniform(1.5, 2.1),random.uniform(6, 8),0.0,
                    45.0,random.uniform(4.9, 5.7),random.uniform(6, 10)]
initial_V = {"U1":random.uniform(8, 18),"U2":random.uniform(8, 18),"U3":random.uniform(8, 18),"D1":random.uniform(8, 18),"D2":random.uniform(8, 18),"D3":random.uniform(8, 18),"E0":random.uniform(8, 18),"E1":random.uniform(8, 18),"E2":random.uniform(8, 18)}

#Faster scenario in the paper
initial_position1 = [16.805938962745547, 5.418193557944047, 6.912696629611112, 0.0,
                    71.62082605729698, 4.885082983540965, 6.70014080869617, 0.0,
                    146.53669655956725, 4.995793235473276, 7.305458298290304, 0.0,
                    36.48594100014558, 8.806220102817132, 6.412325242782501, 0.0,
                    70.38475751235215, 9.031535578854887, 6.842290410034565, 0.0,
                    148.86617401537106, 9.045716005374517, 7.643467012437377, 0.0,
                    37.457162684331536, 1.9245666872045581, 7.865530516203213, 0.0,
                    73.00281098424256, 1.560144862874428, 6.9763560370788165, 0.0,
                    153.85384534469543, 1.650001315061247, 6.530688488027288, 0.0,
                    45.0, 5.1266228940539085, 6.456184279880635]

initial_V1 = {'U1': 16.62809574679761, 'U2': 16.07967289012602, 'U3': 8.283122042898329, 'D1': 12.7985450321384, 'D2': 16.681644180564525, 'D3': 12.073704403430613, 'E0': 15.995906883908326, 'E1': 8.572012707291984, 'E2': 8.40821945511576}

#Dangerous Scenario in paper
initial_position2 = [18.0,5.6,7.0,0.0,
                        100,5.2,7.0,0.0,
                        130,5.0,7.0,0.0,
                        35.0,9.0,7.0,0.0,
                        75.0,9.0,7.0,0.0,
                        120.0,9.0,7.0,0.0,
                        34.0,1.8,7.0,0.0,
                        70.0,2.0,7.0,0.0,
                        140.0,1.6,7.0,0.0,
                        60.0,5.3,10.0]
initial_V2 = {"U1":12.0,"U2":10.0,"U3":9.0,"D1":9.0,"D2":12.0,"D3":9.0,"E0":8.5,"E1":11.5,"E2":12.0}

def let_us_run(LAS,position,initial_V0):
    dirpath = os.path.abspath(os.path.dirname(__file__))
    info_dict_500 = {}
    info_dict_800 = {}
    params={}
    params["dt"] = 0.1
    params["W"] =  1.9  #车宽
    params["L"] = 4.4
    params["L_diag"] = np.sqrt(params["W"]**2+params["L"]**2)
    params["u_lim"] = [3.0,math.radians(7.0)]  
    params["v_lim"] = [35.0,4.0]
    params["du_lim"] = [-3.0,math.radians(6.0)]
    params["gamma_L"] = 0.8
    params["gamma_F"] = 0.8
    params["Position"] = position
    params["Y_Line"] = [3.5,7.0]
    params["rx"] = 10.0
    params["v_d"] = 30.0
    params["S0"] = 2.0
    params["T_head"] = 1.5
    params["a_max"] = 5.0
    params["b"] = 1.67
    params["V0"] = initial_V0
    params["goal_x"] = 2000.0
    params["epsilon_y"] = 0.22
    params["lambda"] = 0.7
    params["epsilon"] = 1.1
    params["ROI"] = params["lambda"]*params["L_diag"] + params["epsilon"]
    params["d_safe"] = np.array([[params["rx"],params["rx"]],[params["epsilon_y"]*params["L_diag"]+params["W"]/2,params["epsilon_y"]*params["L_diag"]+params["W"]/2]])
    params["direction"] = ["Stay","Change_up","Change_down"]
    params["Bmax"] = 15.0 # After formula conversion delta_l = 0.41
    params["Bmax_l"] = 8.0 # After formula conversion delta_f = 0.77
    params["d0"] = 10.0
    params["Hp"] = 5
    params["LAS"] = LAS
    params["gamma_ycbf"] = 0.8
    params["alpha_l"] = -0.8
    # params["al"] = 0.3
    # params["Vl"] = 18.0

    N = 1000
    R = np.diag([8.0,0.1]) 
    Q = np.diag([0.01,0.01]) #对U的权重
    Qf = 1.0#对slack的权重
    P = np.diag([0.1,0.1])
    S = np.diag([0.0,5.0])

    T = 20
    NX = 4
    NU = 2
    dt = 0.1
    MAX_ITER = 3
    DU_TH = 0.2
    x_area = 50.0
    y_area = 28.0
    Line_center_dict = {"U":8.75,"D":1.75,"E":5.25}
    IDM_model = IDM(params["Position"],params["dt"],params["L"])

    time_n = 0.0

    x = [IDM_model.dyn.x]
    y = [IDM_model.dyn.y]
    theta = [IDM_model.dyn.yaw]
    v = [IDM_model.dyn.v]
    t = [0.0]
    delta = [0.0]
    a = [0.0]

    x_u1 = [IDM_model.U1_X]
    y_u1 = [IDM_model.U1_Y]
    x_u2 = [IDM_model.U2_X]
    y_u2 = [IDM_model.U2_Y]
    x_u3 = [IDM_model.U3_X]
    y_u3 = [IDM_model.U3_Y]

    x_d1 = [IDM_model.D1_X]
    y_d1 = [IDM_model.D1_Y]
    x_d2 = [IDM_model.D2_X]
    y_d2 = [IDM_model.D2_Y]
    x_d3 = [IDM_model.D3_X]
    y_d3 = [IDM_model.D3_Y]

    x_e1 = [IDM_model.E0_X]
    y_e1 = [IDM_model.E0_Y]
    x_e2 = [IDM_model.E1_X]
    y_e2 = [IDM_model.E1_Y]
    x_e3 = [IDM_model.E2_X]
    y_e3 = [IDM_model.E2_Y]

    final_signal_list = [0]
    odelta, oa = None, None
    bar = Bar(max=N-1)
    fig = plt.figure()
    # fig = plt.figure(figsize=(15,1))
    # camera = Camera(fig)
    # plt.ioff()
    for i in range(N-2):
        # fig = plt.figure()
        bar.next()
        x0 = [IDM_model.dyn.x, IDM_model.dyn.y, IDM_model.dyn.v,IDM_model.dyn.yaw] 
        oa, odelta, ox, oy, oyaw, ov, Final_signal = IDM_model.Solve_step_mpc(params["u_lim"], x0, oa, odelta,T,NX,NU,Q,R,P,S,params["du_lim"],MAX_ITER,DU_TH,params["v_lim"],dt,params["Y_Line"][1],params["Y_Line"][0],params["goal_x"],Line_center_dict["E"],Line_center_dict["U"],Line_center_dict["D"],params["d_safe"][0],params["d_safe"][1],params["L"],params["v_d"],params["gamma_L"],params["gamma_F"],S,params["ROI"],params["Bmax"],params["d0"],params["Hp"],params["LAS"],params["gamma_ycbf"],params["alpha_l"],params["alpha_l"],params["Bmax_l"],params["rx"])
        
        if odelta is not None:
            di, ai = odelta[0], oa[0]
        
        #用于测试前车减速场景
        # if i>95:
        #     params["V0"] = {"U1":12.0,"U2":14.0,"U3":11.0,"D1":12.0,"D2":16.0,"D3":12.0,"E0":12.0,"E1":10.0,"E2":17.0}
        # if i>110:
        #     params["V0"] = {"U1":12.0,"U2":14.0,"U3":11.0,"D1":12.0,"D2":16.0,"D3":12.0,"E0":12.0,"E1":7.0,"E2":17.0}
        # if i>130:
        #     params["V0"] = {"U1":12.0,"U2":14.0,"U3":11.0,"D1":12.0,"D2":16.0,"D3":12.0,"E0":12.0,"E1":5.5,"E2":17.0}
        # if i>136:
        #     params["V0"] = {"U1":12.0,"U2":14.0,"U3":11.0,"D1":12.0,"D2":16.0,"D3":12.0,"E0":12.0,"E1":4.0,"E2":17.0}
        
        IDM_model.dyn.update_state(ai, di, params["u_lim"][1])
        ude = IDM_model.Judge_Location(params["Y_Line"][1],params["Y_Line"][0])
        if ude == "U":
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_E(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_D(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
        elif ude == "D":
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_E(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_U(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
        if ude == "E":
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_U(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_D(params["L"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            
        time_n = time_n + params["dt"]
        
        if IDM_model.dyn.x >= 498 and IDM_model.dyn.x <= 502:
            info_dict_500["time_{}".format(time_n)] = time_n
        if IDM_model.dyn.x >= 798 and IDM_model.dyn.x <= 802:
            info_dict_800["time_{}".format(time_n)] = time_n
            break

        x.append(IDM_model.dyn.x)
        y.append(IDM_model.dyn.y)
        theta.append(IDM_model.dyn.yaw)
        v.append(IDM_model.dyn.v)
        t.append(time_n)
        delta.append(di)
        a.append(ai)

        x_u1.append(IDM_model.U1_X)
        y_u1.append(IDM_model.U1_Y)
        x_u2.append(IDM_model.U2_X)
        y_u2.append(IDM_model.U2_Y)
        x_u3.append(IDM_model.U3_X)
        y_u3.append(IDM_model.U3_Y)

        x_d1.append(IDM_model.D1_X)
        y_d1.append(IDM_model.D1_Y)
        x_d2.append(IDM_model.D2_X)
        y_d2.append(IDM_model.D2_Y)
        x_d3.append(IDM_model.D3_X)
        y_d3.append(IDM_model.D3_Y)

        x_e1.append(IDM_model.E0_X)
        y_e1.append(IDM_model.E0_Y)
        x_e2.append(IDM_model.E1_X)
        y_e2.append(IDM_model.E1_Y)
        x_e3.append(IDM_model.E2_X)
        y_e3.append(IDM_model.E2_Y)
        
        final_signal_list.append(Final_signal)
        # if check_goal(IDM_model.dyn,params["goal_x"],10.0):
        #     print("Goal")
        #     break
        plt.cla()
        if LAS == 10000.0:
            plt.text(IDM_model.dyn.x + 5.0, 11.2, "No_LAS".format(LAS),c='orange',fontsize=8,style='oblique')
        else:
            plt.text(IDM_model.dyn.x + 5.0, 11.2, "LAS={}m".format(LAS),c='orange',fontsize=8,style='oblique')
        plt.text(IDM_model.dyn.x-13.6, 11.2, "Time={}s".format(round(t[i],1)),c='orange',fontsize=6,style='oblique',bbox=dict(boxstyle='round,pad=0.3', fc='yellow', ec='k',lw=1 ,alpha=0.5))
        plt.text(IDM_model.U1_X-13.6, 8.75+0.5, "V={} m/s".format(round(IDM_model.U1_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.U2_X-13.6, 8.75+0.5, "V={} m/s".format(round(IDM_model.U2_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.U3_X-13.6, 8.75+0.5, "V={} m/s".format(round(IDM_model.U3_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D1_X-13.6, 1.75+0.5, "V={} m/s".format(round(IDM_model.D1_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D2_X-13.6, 1.75+0.5, "V={} m/s".format(round(IDM_model.D2_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D3_X-13.6, 1.75+0.5, "V={} m/s".format(round(IDM_model.D3_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E0_X-13.6, 5.25+0.5, "V={} m/s".format(round(IDM_model.E0_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E1_X-13.6, 5.25+0.5, "V={} m/s".format(round(IDM_model.E1_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E2_X-13.6, 5.25+0.5, "V={} m/s".format(round(IDM_model.E2_V,1)),c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.U1_X-7.6, 8.75-1.5, "L1",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.U2_X-7.6, 8.75-1.5, "L2",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.U3_X-7.6, 8.75-1.5, "L3",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D1_X-7.6, 1.75-1.5, "R1",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D2_X-7.6, 1.75-1.5, "R2",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.D3_X-7.6, 1.75-1.5, "R3",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E0_X-7.6, 5.25-1.5, "C1",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E1_X-7.6, 5.25-1.5, "C2",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.E2_X-7.6, 5.25-1.5, "C3",c='k',fontsize=6,style='oblique')
        plt.text(IDM_model.dyn.x-6.6, IDM_model.dyn.y-1.5, "Ego",c='r',fontsize=6,style='oblique')
        plt.text(IDM_model.dyn.x-13.6, IDM_model.dyn.y+0.5, "V={} m/s".format(round(IDM_model.dyn.v,1)),c='r',fontsize=6,style='oblique')
        plt.plot([params["Y_Line"][0] for i in range(2000)],color='b')
        plt.plot([params["Y_Line"][1] for i in range(2000)],color='b')
        plt.plot([10.5 for i in range(2000)],color='b')
        plt.plot([0 for i in range(2000)],color='b')
        plt.plot([Line_center_dict["U"] for i in range(2000)],linestyle='dashdot',color='b')
        plt.plot([Line_center_dict["D"] for i in range(2000)],linestyle='dashdot',color='b')
        plt.plot([Line_center_dict["E"] for i in range(2000)],linestyle='dashdot',color='b')
        plt.plot(ox, oy, linestyle='dashed',color='r')
        plot_car(IDM_model.dyn.x-2.2, IDM_model.dyn.y, IDM_model.dyn.yaw, steer=odelta[0],truckcolor='orange')
        plot_car(IDM_model.U1_X-2.2, IDM_model.U1_Y, 0, 0,truckcolor='g')
        plot_car(IDM_model.U2_X-2.2, IDM_model.U2_Y, 0, 0,truckcolor='g')
        plot_car(IDM_model.U3_X-2.2, IDM_model.U3_Y, 0, 0,truckcolor='g')
        plot_car(IDM_model.D1_X-2.2, IDM_model.D1_Y, 0, 0,truckcolor='g')
        plot_car(IDM_model.D2_X-2.2, IDM_model.D2_Y, 0, 0,truckcolor='g')
        plot_car(IDM_model.D3_X-2.2, IDM_model.D3_Y, 0, 0,truckcolor="g")
        plot_car(IDM_model.E0_X-2.2, IDM_model.E0_Y, 0, 0,truckcolor="g")
        plot_car(IDM_model.E1_X-2.2, IDM_model.E1_Y, 0, 0,truckcolor="g")
        plot_car(IDM_model.E2_X-2.2, IDM_model.E2_Y, 0, 0,truckcolor="g")
        plt.xlim((IDM_model.dyn.x - x_area, IDM_model.dyn.x + x_area))
        plt.ylim((IDM_model.dyn.y - y_area, IDM_model.dyn.y + y_area))
        plt.pause(0.0001)  #可以每一帧展示图片

        plt.xticks([])
        if not os.path.exists(dirpath+"\\figsave"):
            os.mkdir(dirpath+"\\figsave")
        plt.ioff()
        plt.savefig((dirpath+"\\figsave\{}".format(i)),dpi=600)  
        # plt.savefig((dirpath+"\\figsave3\{}.svg".format(i)),dpi=600)  
    # plt.ioff()
    
    if LAS == 10000.0 and bool(info_dict_500):
        save_to_excel(dirpath+"\\excel_files\\LAS_None_500.xlsx".format(int(LAS)),info_dict_500,"500")
    else:
        save_to_excel(dirpath+"\\excel_files\\LAS_{}_500.xlsx".format(int(LAS)),info_dict_500,"500")
        
    if LAS == 10000.0 and bool(info_dict_800):
        save_to_excel(dirpath+"\\excel_files\\LAS_None_800.xlsx".format(int(LAS)),info_dict_800,"800")
    else:
        save_to_excel(dirpath+"\\excel_files\\LAS_{}_800.xlsx".format(int(LAS)),info_dict_800,"800")

    time_now = int(time.time())
    animation_generation(dirpath+"\\figsave",time_now)   #可以生成video
    Fig_delete(dirpath)

    # draw_fig(dirpath+"\\figsave",delta,"delta","rad")
    # draw_fig(dirpath+"\\figsave",v,"V","m/s")
    # draw_fig(dirpath+"\\figsave",final_signal_list,"Final Signal","")
    print(position,initial_V0)
# let_us_run(50.0,initial_position,initial_V)
let_us_run(60.0,initial_position,initial_V)
# let_us_run(70.0,initial_position,initial_V)
# let_us_run(80.0,initial_position,initial_V)
# let_us_run(10000.0,initial_position,initial_V)
