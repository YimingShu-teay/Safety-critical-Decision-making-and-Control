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
# initial_position = [18.0,5.6,7.0,0.0,
#                         100,5.2,7.0,0.0,
#                         130,5.0,7.0,0.0,
#                         35.0,9.0,7.0,0.0,
#                         75.0,9.0,7.0,0.0,
#                         120.0,9.0,7.0,0.0,
#                         34.0,1.8,7.0,0.0,
#                         70.0,2.0,7.0,0.0,
#                         140.0,1.6,7.0,0.0,
#                         60.0,5.3,10.0]
# initial_V1 = {"U1":12.0,"U2":10.0,"U3":9.0,"D1":11.0,"D2":12.0,"D3":9.0,"E0":8.5,"E1":11.5,"E2":12.0}
# initial_V2 = {"U1":9.4,"U2":10.2,"U3":8.9,"D1":13.0,"D2":10.1,"D3":9.3,"E0":11.3,"E1":9.6,"E2":7.8}

def let_us_run(LAS,position,initial_V0):
    dirpath = os.path.abspath(os.path.dirname(__file__))
    info_dict = {}
    params={}
    params["dt"] = 0.1
    params["L"] =  1.9  #车宽
    params["Length"] = 4.4
    params["L_k"] = np.sqrt(params["L"]**2+params["Length"]**2)
    params["u_lim"] = [3.0,math.radians(7.0)]  #最大加速度,最大前轮转向角限制
    params["v_lim"] = [35.0,4.0]
    params["du_lim"] = [-3.0,math.radians(6.0),3.0]
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
    # params["V0"] = {"U1":12.0,"U2":10.0,"U3":9.0,"D1":11.0,"D2":12.0,"D3":9.0,"E0":8.5,"E1":11.5,"E2":12.0}
    # params["V0"] = {"U1":9.4,"U2":10.2,"U3":8.9,"D1":13.0,"D2":10.1,"D3":9.3,"E0":11.3,"E1":9.6,"E2":7.8}
    params["V0"] = initial_V0
    params["goal_x"] = 2000.0
    params["epsilon"] = 0.22
    params["lambda"] = 0.7
    params["epsilon_region"] = 1.0
    params["ROI"] = params["lambda"]*5.0 + params["epsilon_region"]
    params["d_safe"] = np.array([[params["rx"],params["rx"]],[params["epsilon"]*params["L_k"]+params["L"]/2,params["epsilon"]*params["L_k"]+params["L"]/2]])
    params["direction"] = ["Stay","Change_up","Change_down"]
    params["Bc"] = 15.0
    params["Bc_l"] = 8.0
    params["d0"] = 10.0
    params["H_pred"] = 5
    params["LAS"] = LAS
    params["gamma_ycbf"] = 0.8
    params["sigma"] = -0.8
    params["al"] = 0.3
    params["Vl"] = 18.0

    N = 1000
    Q = np.diag([8.0,0.1]) 
    R = np.diag([0.01,0.01]) #对U的权重
    Qf = 1.0#对slack的权重
    Rd = np.diag([0.1,0.1])
    Rt = np.diag([0.0,5.0])

    T = 20
    NX = 4
    NU = 2
    dt = 0.1
    MAX_ITER = 3
    DU_TH = 0.2
    x_area = 50.0
    y_area = 28.0
    Line_center_dict = {"U":8.75,"D":1.75,"E":5.25}
    IDM_model = IDM(params["Position"],params["dt"],params["Length"])

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
    plt.ion()
    # fig = plt.figure(figsize=(15,1))
    # camera = Camera(fig)
    for i in range(N-2):
        bar.next()
        x0 = [IDM_model.dyn.x, IDM_model.dyn.y, IDM_model.dyn.v,IDM_model.dyn.yaw] 
        oa, odelta, ox, oy, oyaw, ov, Final_signal = IDM_model.Solve_step_mpc(params["u_lim"], x0, oa, odelta,T,NX,NU,R,Q,Rd,Qf,params["du_lim"],MAX_ITER,DU_TH,params["v_lim"],dt,params["Y_Line"][1],params["Y_Line"][0],params["goal_x"],Line_center_dict["E"],Line_center_dict["U"],Line_center_dict["D"],params["d_safe"][0],params["d_safe"][1],params["Length"],params["v_d"],params["gamma_L"],params["gamma_F"],Rt,params["ROI"],params["Bc"],params["d0"],params["H_pred"],params["LAS"],params["gamma_ycbf"],params["sigma"],params["al"],params["Vl"],params["Bc_l"],params["rx"])
        
        if odelta is not None:
            di, ai = odelta[0], oa[0]
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
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_E(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_D(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
        elif ude == "D":
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_E(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_U(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
        if ude == "E":
            IDM_model.update_state_onlane(params["Y_Line"][1],params["Y_Line"][0],params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_U(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
            IDM_model.update_state_D(params["Length"],params["S0"] ,params["T_head"],params["a_max"],params["b"],params["V0"])
        time_n = time_n + params["dt"]
        if IDM_model.dyn.x >= 498 and IDM_model.dyn.x <= 502:
            info_dict["time_{}".format(time_n)] = time_n
            if LAS == 10000.0:
                save_to_excel(dirpath+"\\excel_files\\No_LAS.xlsx".format(int(LAS)),info_dict,"500")
            else:
                save_to_excel(dirpath+"\\excel_files\\LAS_{}.xlsx".format(int(LAS)),info_dict,"500")
        if IDM_model.dyn.x >= 798 and IDM_model.dyn.x <= 802:
            info_dict["time_{}".format(time_n)] = time_n
            if LAS == 10000.0:
                save_to_excel(dirpath+"\\excel_files\\No_LAS.xlsx".format(int(LAS)),info_dict,"800")
            else:
                save_to_excel(dirpath+"\\excel_files\\LAS_{}.xlsx".format(int(LAS)),info_dict,"800")
            break
        # if IDM_model.dyn.x >= 1000:
        #     info_dict["time_1000_{}".format(time)] = time
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
        plot_car(IDM_model.dyn.x, IDM_model.dyn.y, IDM_model.dyn.yaw, steer=odelta[0])
        plt.cla()
        if LAS == 10000.0:
            plt.text(IDM_model.dyn.x-13.6, 18.0, "No_LAS".format(LAS),c='orange',fontsize=12,style='oblique')
        else:
            plt.text(IDM_model.dyn.x-13.6, 18.0, "LAS={}m".format(LAS),c='orange',fontsize=12,style='oblique')
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
        plt.pause(0.0001)

        plt.xticks([])
        if not os.path.exists(dirpath+"\\figsave"):
            os.mkdir(dirpath+"\\figsave")
        plt.savefig((dirpath+"\\figsave\{}".format(i)),dpi=600)    
        # plt.show()
    plt.ioff()

    time_now = int(time.time())
    # animation_generation(dirpath+"\\figsave",time_now)
    Fig_delete(dirpath)

    # draw_fig(dirpath+"\\figsave",delta,"delta","rad")
    # draw_fig(dirpath+"\\figsave",v,"V","m/s")
    # draw_fig(dirpath+"\\figsave",final_signal_list,"Final Signal","")
    
let_us_run(50.0,initial_position,initial_V)
# let_us_run(60.0,initial_position,initial_V)
# let_us_run(70.0,initial_position,initial_V)
# let_us_run(80.0,initial_position,initial_V)
# let_us_run(10000.0,initial_position,initial_V)
