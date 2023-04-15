import numpy as np
import matplotlib.pyplot as plt
import math
from KinematicModel import *
import cvxpy
from utils import *
#一共需要10辆车。
class IDM:
    def __init__(self,Position,dt,L):
        
        self.E0_X,self.E0_Y,self.E0_V,self.E0_a = Position[0],Position[1],Position[2],Position[3]
        self.E1_X,self.E1_Y,self.E1_V,self.E1_a = Position[4],Position[5],Position[6],Position[7]
        self.E2_X,self.E2_Y,self.E2_V,self.E2_a = Position[8],Position[9],Position[10],Position[11]
        
        self.U1_X,self.U1_Y,self.U1_V,self.U1_a = Position[12],Position[13],Position[14],Position[15]
        self.U2_X,self.U2_Y,self.U2_V,self.U2_a = Position[16],Position[17],Position[18],Position[19]
        self.U3_X,self.U3_Y,self.U3_V,self.U3_a = Position[20],Position[21],Position[22],Position[23]
        
        self.D1_X,self.D1_Y,self.D1_V,self.D1_a = Position[24],Position[25],Position[26],Position[27]
        self.D2_X,self.D2_Y,self.D2_V,self.D2_a = Position[28],Position[29],Position[30],Position[31]
        self.D3_X,self.D3_Y,self.D3_V,self.D3_a = Position[32],Position[33],Position[34],Position[35]
        
        self.dt = dt
        
        self.dyn = KinematicModel(self.dt,L,Position[36],Position[37],v=Position[38])
 
    
    #判断当前位于哪条车道上
    def Judge_Location(self,U_Line,D_Line):
        if self.dyn.y <= U_Line and self.dyn.y >= D_Line:
            UDE_Signal = "E"
        elif self.dyn.y >= U_Line:
            UDE_Signal = "U"
        elif self.dyn.y <= D_Line:
            UDE_Signal = "D"
        return UDE_Signal
    
    def get_id_dict(self,E_center_y,U_center_y,D_center_y):
        id_dict = {"U1":[self.U1_X,self.U1_Y,self.U1_V,self.U1_a],"U2":[self.U2_X,self.U2_Y,self.U2_V,self.U2_a],"U3":[self.U3_X,self.U3_Y,self.U3_V,self.U3_a],
                    "D1":[self.D1_X,self.D1_Y,self.D1_V,self.D1_a],"D2":[self.D2_X,self.D2_Y,self.D2_V,self.D2_a],"D3":[self.D3_X,self.D3_Y,self.D3_V,self.D3_a],
                    "E0":[self.E0_X,self.E0_Y,self.E0_V,self.E0_a],"E1":[self.E1_X,self.E1_Y,self.E1_V,self.E1_a],"E2":[self.E2_X,self.E2_Y,self.E2_V,self.E2_a],
                    "Unone":[2000.0,U_center_y,33.0,5.0],"Dnone":[2000.0,D_center_y,33.0,5.0],"Enone":[2000.0,E_center_y,33.0,5.0],
                    "UNoone":[0.0,U_center_y,1.0,-5.0],"DNoone":[0.0,D_center_y,1.0,-5.0],"ENoone":[0.0,D_center_y,1.0,-5.0]}
        return id_dict    
    
    def pre_calc_single_lane(self,LF,LF_dict,distance):    
        L_signal = np.where(distance > 0, distance,  np.inf).argmin()
        L_id = LF[L_signal]
        Lead_X = LF_dict[L_signal]
        if np.all(distance < 0):
            L_signal = None
            L_id = None
            Lead_X = None
 
        F_signal = np.where(distance < 0, distance, -np.inf).argmax()
        F_id = LF[F_signal]
        Follow_X = LF_dict[F_signal]
        if np.all(distance > 0):
            F_signal = None
            F_id = None
            Follow_X = None
        return L_id, Lead_X, distance[L_signal], F_id, Follow_X, distance[F_signal]
                      
    #判断在三条道上的leader 和 follower
    def Clac_LandLF(self):
        Distance_e = np.array([self.E0_X,self.E1_X,self.E2_X])-self.dyn.x
        Distance_u = np.array([self.U1_X,self.U2_X,self.U3_X])-self.dyn.x
        Distance_d = np.array([self.D1_X,self.D2_X,self.D3_X])-self.dyn.x
        LF_u = {0:"U1",1:"U2",2:"U3"}
        LF_dict_u = {0:self.U1_X,1:self.U2_X,2:self.U3_X}
        LF_d = {0:"D1",1:"D2",2:"D3"}
        LF_dict_d = {0:self.D1_X,1:self.D2_X,2:self.D3_X}
        LF_e = {0:"E0",1:"E1",2:"E2"}
        LF_dict_e = {0:self.E0_X,1:self.E1_X,2:self.E2_X}
        L_id_u, L_X_u, D_L_u, F_id_u, F_X_u, D_F_u = self.pre_calc_single_lane(LF_u,LF_dict_u,Distance_u)
        L_id_d, L_X_d, D_L_d, F_id_d, F_X_d, D_F_d = self.pre_calc_single_lane(LF_d,LF_dict_d,Distance_d)
        L_id_e, L_X_e, D_L_e, F_id_e, F_X_e, D_F_e = self.pre_calc_single_lane(LF_e,LF_dict_e,Distance_e)
        if L_id_u is None:
            L_id_u = "Unone"
            D_L_u = 2000.0 - self.dyn.x
        if L_id_d is None:
            L_id_d = "Dnone"
            D_L_d = 2000.0 - self.dyn.x
        if L_id_e is None:
            L_id_e = "Enone"
            D_L_e = 2000.0 - self.dyn.x
        if F_id_u is None:
            F_id_u = "UNoone"
            D_F_u = self.dyn.x - 2000.0
        if F_id_d is None:
            F_id_d = "DNoone"
            D_F_d = self.dyn.x - 2000.0
        if F_id_e is None:
            F_id_e = "ENoone"
            D_F_e = self.dyn.x - 2000.0
        return L_id_u,  D_L_u, F_id_u,  D_F_u, L_id_d, D_L_d, F_id_d ,D_F_d, L_id_e, D_L_e, F_id_e, D_F_e

    
###########################################################################################################################
    #周围状态更新策略:IDM 后车让行模式，后车不让行模式
    def update_XV_U(self):
        self.U3_X = self.U3_X + self.U3_V*self.dt
        self.U2_X = self.U2_X + self.U2_V*self.dt
        self.U1_X = self.U1_X + self.U1_V*self.dt
        self.U3_V = self.U3_V + self.U3_a*self.dt
        self.U2_V = self.U2_V + self.U2_a*self.dt
        self.U1_V = self.U1_V + self.U1_a*self.dt
    
    def update_XV_D(self):
        self.D3_X = self.D3_X + self.D3_V*self.dt
        self.D2_X = self.D2_X + self.D2_V*self.dt
        self.D1_X = self.D1_X + self.D1_V*self.dt
        self.D3_V = self.D3_V + self.D3_a*self.dt
        self.D2_V = self.D2_V + self.D2_a*self.dt
        self.D1_V = self.D1_V + self.D1_a*self.dt
    
    def update_XV_E(self):
        self.E2_X = self.E2_X + self.E2_V*self.dt
        self.E1_X = self.E1_X + self.E1_V*self.dt
        self.E0_X = self.E0_X + self.E0_V*self.dt
        self.E2_V = self.E2_V + self.E2_a*self.dt
        self.E1_V = self.E1_V + self.E1_a*self.dt
        self.E0_V = self.E0_V + self.E0_a*self.dt
  

    def update_state_onlane(self,U_Line,D_Line,Length,S0,T_head,a_max,b,V0):
        Lane = self.Judge_Location(U_Line,D_Line)
        L_id_u,  D_L_u, F_id_u,  D_F_u, L_id_d, D_L_d, F_id_d ,D_F_d, L_id_e, D_L_e, F_id_e, D_F_e = self.Clac_LandLF()
        if Lane == "U": 
            if F_id_u == "UNoone":
                self.update_state_U(Length,S0,T_head,a_max,b,V0)
                
            if F_id_u == 'U1':
                self.S_23 = self.U3_X - self.U2_X - Length
                self.S_23_d = S0 + T_head*self.U2_V + self.U2_V*(self.U2_V-self.U3_V)/(2*math.sqrt(a_max*b))
                self.S_1e = self.dyn.x- self.U1_X -Length
                self.S_1e_d = S0 + T_head*self.U1_V + self.U1_V*(self.U1_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                
                self.update_XV_U()
                
                self.U3_a = a_max*(1-round((self.U3_V/V0["U3"]),1)**4)
                self.U2_a = a_max*(1-round((self.U2_V/V0["U2"]),1)**4-(self.S_23_d/self.S_23)**2)
                self.U1_a = a_max*(1-round((self.U1_V/V0["U1"]),1)**4-(self.S_1e_d/self.S_1e)**2)
                
            if F_id_u == "U2":
                self.S_2e = self.dyn.x - self.U2_X - Length
                self.S_2e_d = S0 + T_head*self.U2_V + self.U2_V*(self.U2_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_12 = self.U2_X - self.U1_X -Length
                self.S_12_d =S0 + T_head*self.U1_V + self.U1_V*(self.U1_V-self.U2_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_U()
                
                self.U3_a = a_max*(1-round((self.U3_V/V0["U3"]),1)**4)
                self.U2_a = a_max*(1-round((self.U2_V/V0["U2"]),1)**4-(self.S_2e_d/self.S_2e)**2)
                self.U1_a = a_max*(1-round((self.U1_V/V0["U1"]),1)**4-(self.S_12_d/self.S_12)**2)
                
                
            if F_id_u == "U3":
                self.S_3e = self.dyn.x - self.U3_X - Length
                self.S_3e_d = S0 + T_head*self.U3_V + self.U3_V*(self.U3_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_23 = self.U3_X - self.U2_X - Length
                self.S_23_d = S0 + T_head*self.U2_V + self.U2_V*(self.U2_V-self.U3_V)/(2*math.sqrt(a_max*b))
                self.S_12 = self.U2_X - self.U1_X -Length
                self.S_12_d = S0 + T_head*self.U1_V + self.U1_V*(self.U1_V-self.U2_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_U()
                
                self.U3_a = a_max*(1-round((self.U3_V/V0["U3"]),1)**4-(self.S_3e_d/self.S_3e)**2)
                self.U2_a = a_max*(1-round((self.U2_V/V0["U2"]),1)**4-(self.S_23_d/self.S_23)**2)
                self.U1_a = a_max*(1-round((self.U1_V/V0["U1"]),1)**4-(self.S_12_d/self.S_12)**2)
                             
        if Lane == "D": 
            if F_id_d == "DNoone":
                self.update_state_D(Length,S0,T_head,a_max,b,V0)
                
            if F_id_d == 'D1':
                self.S_23 = self.D3_X - self.D2_X - Length
                self.S_23_d = S0 + T_head*self.D2_V + self.D2_V*(self.D2_V-self.D3_V)/(2*math.sqrt(a_max*b))
                self.S_1e = self.dyn.x - self.D1_X -Length
                self.S_1e_d = S0 + T_head*self.D1_V + self.D1_V*(self.D1_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                
                self.update_XV_D()
                
                self.D3_a = a_max*(1-round((self.D3_V/V0["D3"]),1)**4)
                self.D2_a = a_max*(1-round((self.D2_V/V0["D2"]),1)**4-(self.S_23_d/self.S_23)**2)
                self.D1_a = a_max*(1-round((self.D1_V/V0["D1"]),1)**4-(self.S_1e_d/self.S_1e)**2)
                
            if F_id_d == "D2":
                self.S_2e = self.dyn.x - self.D2_X - Length
                self.S_2e_d = S0 + T_head*self.D2_V + self.D2_V*(self.D2_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_12 = self.D2_X - self.D1_X -Length
                self.S_12_d =S0 + T_head*self.D1_V + self.D1_V*(self.D1_V-self.D2_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_D()
                
                self.D3_a = a_max*(1-round((self.D3_V/V0["D3"]),1)**4)
                self.D2_a = a_max*(1-round((self.D2_V/V0["D2"]),1)**4-(self.S_2e_d/self.S_2e)**2)
                self.D1_a = a_max*(1-round((self.D1_V/V0["D1"]),1)**4-(self.S_12_d/self.S_12)**2)
                
                
            if F_id_d == "D3":
                self.S_3e = self.dyn.x - self.D3_X - Length
                self.S_3e_d = S0 + T_head*self.D3_V + self.U3_V*(self.D3_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_23 = self.D3_X - self.D2_X - Length
                self.S_23_d = S0 + T_head*self.D2_V + self.D2_V*(self.D2_V-self.D3_V)/(2*math.sqrt(a_max*b))
                self.S_12 = self.D2_X - self.D1_X -Length
                self.S_12_d = S0 + T_head*self.D1_V + self.D1_V*(self.D1_V-self.D2_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_D()
                
                self.D3_a = a_max*(1-round((self.D3_V/V0["D3"]),1)**4-(self.S_3e_d/self.S_3e)**2)
                self.D2_a = a_max*(1-round((self.D2_V/V0["D2"]),1)**4-(self.S_23_d/self.S_23)**2)
                self.D1_a = a_max*(1-round((self.D1_V/V0["D1"]),1)**4-(self.S_12_d/self.S_12)**2)
                
        if Lane == "E":
            if F_id_e == "ENoone":
                self.update_state_E(Length,S0,T_head,a_max,b,V0)
                
            if F_id_e == "E0":
                self.S_12 = self.E2_X - self.E1_X - Length
                self.S_12_d = S0 + T_head*self.E1_V + self.E1_V*(self.E1_V-self.E2_V)/(2*math.sqrt(a_max*b))
                self.S_0e = self.dyn.x - self.E0_X - Length
                self.S_0e_d = S0 + T_head*self.E0_V + self.E0_V*(self.E0_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                                
                self.update_XV_E()
                
                self.E2_a = a_max*(1-round((self.E2_V/V0["E2"]),1)**4)
                self.E1_a = a_max*(1-round((self.E1_V/V0["E1"]),1)**4-(self.S_12_d/self.S_12)**2)
                self.E0_a = a_max*(1-round((self.E0_V/V0["E0"]),1)**4-(self.S_0e_d/self.S_0e)**2)
                
            if F_id_e == "E1":
                self.S_1e = self.dyn.x - self.E1_X - Length
                self.S_1e_d = S0 + T_head*self.E1_V + self.E1_V*(self.E1_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_01 = self.E1_X - self.E0_X - Length
                self.S_01_d = S0 + T_head*self.E0_V + self.E0_V*(self.E0_V-self.E1_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_E()
                
                self.E2_a = a_max*(1-round((self.E2_V/V0["E2"]),1)**4)
                self.E1_a = a_max*(1-round((self.E1_V/V0["E1"]),1)**4-(self.S_1e_d/self.S_1e)**2)
                self.E0_a = a_max*(1-round((self.E0_V/V0["E0"]),1)**4-(self.S_01_d/self.S_01)**2)
            
            if F_id_e == "E2":
                self.S_2e = self.dyn.x - self.E2_X - Length
                self.S_2e_d = S0 + T_head*self.E2_V + self.E2_V*(self.E2_V-self.dyn.v)/(2*math.sqrt(a_max*b))
                self.S_12 = self.E2_X - self.E1_X - Length
                self.S_12_d = S0 + T_head*self.E1_V + self.E1_V*(self.E1_V-self.E2_V)/(2*math.sqrt(a_max*b))
                self.S_01 = self.E1_X - self.E0_X - Length
                self.S_01_d = S0 + T_head*self.E0_V + self.E0_V*(self.E0_V-self.E1_V)/(2*math.sqrt(a_max*b))
                
                self.update_XV_E()
                
                self.E2_a = a_max*(1-round((self.E2_V/V0["E2"]),1)**4-(self.S_2e_d/self.S_2e)**2)         
                self.E1_a = a_max*(1-round((self.E1_V/V0["E1"]),1)**4-(self.S_12_d/self.S_12)**2)
                self.E0_a = a_max*(1-round((self.E0_V/V0["E0"]),1)**4-(self.S_01_d/self.S_01)**2)
    
    def update_state_U(self,Length,S0,T_head,a_max,b,V0):
        self.S_23 = self.U3_X - self.U2_X - Length
        self.S_23_d = S0 + T_head*self.U2_V + self.U2_V*(self.U2_V-self.U3_V)/(2*math.sqrt(a_max*b))
        self.S_12 = self.U2_X - self.U1_X -Length
        self.S_12_d = S0 + T_head*self.U1_V + self.U1_V*(self.U1_V-self.U2_V)/(2*math.sqrt(a_max*b))
        
        self.update_XV_U()
        
        self.U3_a = a_max*(1-round((self.U3_V/V0["U3"]),1)**4)
        self.U2_a = a_max*(1-round((self.U2_V/V0["U2"]),1)**4-(self.S_23_d/self.S_23)**2)
        self.U1_a = a_max*(1-round((self.U1_V/V0["U1"]),1)**4-(self.S_12_d/self.S_12)**2)
        
    def update_state_D(self,Length,S0,T_head,a_max,b,V0):
        self.S_23 = self.D3_X - self.D2_X - Length
        self.S_23_d = S0 + T_head*self.D2_V + self.D2_V*(self.D2_V-self.D3_V)/(2*math.sqrt(a_max*b))
        self.S_12 = self.D2_X - self.D1_X -Length
        self.S_12_d = S0 + T_head*self.D1_V + self.D1_V*(self.D1_V-self.D2_V)/(2*math.sqrt(a_max*b))
        
        self.update_XV_D()
        
        self.D3_a = a_max*(1-round((self.D3_V/V0["D3"]),1)**4)
        self.D2_a = a_max*(1-round((self.D2_V/V0["D2"]),1)**4-(self.S_23_d/self.S_23)**2)
        self.D1_a = a_max*(1-round((self.D1_V/V0["D1"]),1)**4-(self.S_12_d/self.S_12)**2)

    def update_state_E(self,Length,S0,T_head,a_max,b,V0):
        self.S_01 = self.E1_X - self.E0_X -Length
        self.S_01_d = S0 + T_head*self.E0_V + self.E0_V*(self.E0_V-self.E1_V)/(2*math.sqrt(a_max*b))
        self.S_12 = self.E2_X - self.E1_X -Length
        self.S_12_d = S0 + T_head*self.E1_V + self.E1_V*(self.E1_V-self.E2_V)/(2*math.sqrt(a_max*b))
        
        self.update_XV_E()
    
        self.E2_a = a_max*(1-round((self.E2_V/V0["E2"]),1)**4)
        self.E1_a = a_max*(1-round((self.E1_V/V0["E1"]),1)**4-(self.S_12_d/self.S_12)**2)
        self.E0_a = a_max*(1-round((self.E0_V/V0["E0"]),1)**4-(self.S_01_d/self.S_01)**2)
        
 #############################################################################################################################       
    #Control Design 
    def predict_motion(self,x0, oa, od, NX,T,dt,L,u_lim):
        xbar = np.zeros((NX, T + 1))
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
        ugv =  KinematicModel(dt,L,x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        if oa is not None:
            for (ai, di, i) in zip(oa, od, range(1, T + 1)):
                ugv.update_state(ai, di,u_lim[1])
                xbar[0, i] = ugv.x
                xbar[1, i] = ugv.y
                xbar[2, i] = ugv.v
                xbar[3, i] = ugv.yaw
        elif oa is None:
            oa = [0.0] * T
            od = [0.0] * T
            for (ai, di, i) in zip(oa, od, range(1, T + 1)):
                ugv.update_state(ai, di,u_lim[1])
                xbar[0, i] = ugv.x
                xbar[1, i] = ugv.y
                xbar[2, i] = ugv.v
                xbar[3, i] = ugv.yaw            
        return xbar

    def calc_destination(self,desired_x,desired_y,NX,T,TARGET_SPEED):
        xref = np.zeros((NX-1, T + 1))
        dref = np.zeros((1, T + 1))
        for i in range(T + 1):
            xref[0, i] = desired_x
            xref[1, i] = desired_y
            xref[2, i] = TARGET_SPEED
            dref[0, i] = 0.0
        return xref,dref
    
    def Solve_MPC(self,desired_x,desired_y,NX,NU,T,R,Q,Rd,Qf,ai,di,x0,du_lim,u_lim,v_lim,dt,d_safe_x,d_safe_y,L,U_Line,D_Line,TARGET_SPEED,gamma_L,gamma_F,Rt,ROI,E_center_y,U_center_y,D_center_y,gamma_ycbf,sigma):
        xref,dref = self.calc_destination(desired_x,desired_y,NX,T,TARGET_SPEED)
        xbar = self.predict_motion(x0, ai, di, NX,T,dt,L,u_lim)
        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))
        d1 = cvxpy.Variable((1, T))
        d2 = cvxpy.Variable((1, T))
        d3 = cvxpy.Variable((1, T))
        constraints_onlane_x, constraints_beside_y,constaints_y_signal = self.get_constraint_feature(U_Line,D_Line,ROI,T,dt,x,E_center_y,U_center_y,D_center_y,sigma)
        cost = 0.0
        constraints = []
        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                cost += cvxpy.quad_form(xref[1:, t] - x[1:3, t], Q)
                # cost += cvxpy.multiply(xref[0, t] - x[1, t], Qf)
                # cost += cvxpy.quad_form(xref[1:, t] - x[2:, t], Q)
           
            A, B, C = self.dyn.get_state_space(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [u[1, t + 1] - u[1, t]-du_lim[1] * dt<= 0]
                constraints += [u[0, t + 1] - u[0, t]-du_lim[0] * dt>= 0]
                constraints += [u[1, t + 1] - u[1, t]+du_lim[1] * dt>= 0]
            if constaints_y_signal != 0:
                constraints += [constraints_beside_y[:,t+1][0]-d_safe_y[0]-(constraints_beside_y[:,t][0]-d_safe_y[0])+gamma_ycbf*(constraints_beside_y[:,t][0]-d_safe_y[0]) >= 0]#-d3[0,t]
                # constraints += [constraints_beside_y[:,t+1][0]-d_safe_y[0]>=0]
 
            # cost += cvxpy.multiply(d1[0,t],Qf)
            # cost += cvxpy.multiply(d2[0,t],Qf)
            # cost += cvxpy.multiply(d3[0,t],Qf)
            # constraints += [d1[0,t]<=3.0]
            # constraints += [d1[0,t]>=0]
            # constraints += [d2[0,t]<=1.0]
            # constraints += [d2[0,t]>=0]
            # constraints += [d3[0,t]<=0.08]
            # constraints += [d3[0,t]>=0]
            constraints += [x[1,t+1]<= U_Line + 2.8]
            constraints += [x[1,t+1]>= D_Line - 2.8]

            constraints += [constraints_onlane_x[:,t+1][0]-d_safe_x[0]-(constraints_onlane_x[:,t][0]-d_safe_x[0])+gamma_L*(constraints_onlane_x[:,t][0]-d_safe_x[0])>=0]#-d1[0,t]
            constraints += [constraints_onlane_x[:,t+1][1]-d_safe_x[1]-(constraints_onlane_x[:,t][1]-d_safe_x[1])+gamma_F*(constraints_onlane_x[:,t][1]-d_safe_x[1])>=0]#-d2[0,t]
        # cost += cvxpy.multiply(xref[0, T] - x[1, T], Qf)
        # cost += cvxpy.quad_form(xref[1:, T] - x[2:, T], Q)
        # cost += cvxpy.multiply(x[3,t], Rt)#terminal cost
        cost += cvxpy.quad_form(x[2:,t], Rt)
        cost += cvxpy.quad_form(xref[1:, T] - x[1:3, T], Q)
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= v_lim[0]]
        constraints += [x[2, :] >= v_lim[1]]
        # constraints += [cvxpy.abs(u[0, :]) <= u_lim[0]]
        constraints += [u[0, :] <= u_lim[0]]
        constraints += [u[0, :] >= -8.0]
        constraints += [cvxpy.abs(u[1, :]) <= u_lim[1]]
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)
        
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov
    
    def Iterative_linear_mpc(self,x0, oa, od,T,MAX_ITER,DU_TH,NX,NU,R,Q,Rd,Qf,du_lim,u_lim,v_lim,dt,desired_x,desired_y,d_safe_x,d_safe_y,L,U_Line,D_Line,TARGET_SPEED,gamma_L,gamma_F,Rt,ROI,E_center_y,U_center_y,D_center_y,gamma_ycbf,sigma):
        if oa is None or od is None:
            oa = [0.0] * T
            od = [0.0] * T

        for i in range(MAX_ITER):
            if od is not None:
                poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.Solve_MPC(desired_x,desired_y,NX,NU,T,R,Q,Rd,Qf,oa,od,x0,du_lim,u_lim,v_lim,dt,d_safe_x,d_safe_y,L,U_Line,D_Line,TARGET_SPEED,gamma_L,gamma_F,Rt,ROI,E_center_y,U_center_y,D_center_y,gamma_ycbf,sigma)
            if oa is not None:
                du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
                if du <= DU_TH:
                    break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov
        
####################################################################################################################################   

    # get constraints feature
    def get_feature_pre(self,id_dict,T,dt,ROI,object_list_v,L_id,F_id,beside_delta,beside_id,x,sigma):
        constraints_onlane_x = np.array([[id_dict[L_id][0]-x[0,0]-(1+sigma)*x[2,0],x[0,0]-id_dict[F_id][0]]]) #本车道的delta_X,都大于D_safe
        for i in range(T):
            constraints_onlane_x = np.concatenate((constraints_onlane_x,[[id_dict[L_id][0]+object_list_v[0]*(i+1)*dt-(x[0,i+1]+x[2,i+1]*dt)-(1+sigma)*x[2,i+1],(x[0,i+1]+x[2,i+1]*dt)-(id_dict[F_id][0]+object_list_v[1]*(i+1)*dt)]]))
        need_constraints_id = []
        constraints_beside_y = []
        y_all_list = []
        for i in range(len(beside_delta)):
            if beside_delta[i] <= ROI:
                print(beside_delta[i])
                need_constraints_id.append(beside_id[i])
        if len(need_constraints_id) == 0:
            constraints_beside_y = np.zeros((T,1))
            constaints_y_signal = 0
        else:
            constaints_y_signal = 1

            for i in range(len(need_constraints_id)):
                y_all_list.append(id_dict[need_constraints_id[i]][1])
            for i in range(len(y_all_list)):
                yi = y_all_list[i]
                yi_list = [yi]
                for k in range(T):
                    if yi > self.dyn.y:
                        yi_list.append(yi-x[1,i+1])
                    elif yi < self.dyn.y:
                        yi_list.append(x[1,i+1]-yi)
                constraints_beside_y.append(yi_list)
            constraints_beside_y = np.array(constraints_beside_y)
            # for i in range(len(need_constraints_id)):
            #     y_all_list.append(id_dict[need_constraints_id[i]][1])
            #     constraints_beside_y.append(id_dict[need_constraints_id[i]][1] - x[1,0])
            # constraints_beside_y = np.array([constraints_beside_y])
        #     for i in range(T):
        #         print(constraints_beside_y.shape,np.array([[y_all_list-x[1,i+1]]]).shape,x[1,i+1])
        #         constraints_beside_y = np.concatenate((constraints_beside_y,np.array([[y_all_list-x[1,i+1]]])))

        return constraints_onlane_x,constraints_beside_y,constaints_y_signal
    
    
    def get_constraint_feature(self,U_Line,D_Line,ROI,T,dt,x,E_center_y,U_center_y,D_center_y,sigma):
        ude = self.Judge_Location(U_Line,D_Line)
        L_id_u,  D_L_u, F_id_u,  D_F_u, L_id_d, D_L_d, F_id_d ,D_F_d, L_id_e, D_L_e, F_id_e, D_F_e = self.Clac_LandLF()
        id_dict = self.get_id_dict(E_center_y,U_center_y,D_center_y)
            
        if ude == "U":
            print("constraints in U!")
            object_list_v = [id_dict[L_id_u][2],id_dict[F_id_u][2],id_dict[L_id_e][2],id_dict[F_id_e][2]]#本车道V，旁车道V
            beside_delta = [abs(D_L_e),abs(D_F_e)]
            beside_id = [L_id_e,F_id_e]                     
            constraints_onlane_x,constraints_beside_y,constaints_y_signal = self.get_feature_pre(id_dict,T,dt,ROI,object_list_v,L_id_u,F_id_u,beside_delta,beside_id,x,sigma)
                
        if ude == "D":
            print("constraints in D!")
            object_list_v = [id_dict[L_id_d][2],id_dict[F_id_d][2],id_dict[L_id_e][2],id_dict[F_id_e][2]]#本车道V，旁车道V
            beside_delta = [abs(D_L_e),abs(D_F_e)]
            beside_id = [L_id_e,F_id_e]                           
            constraints_onlane_x,constraints_beside_y,constaints_y_signal = self.get_feature_pre(id_dict,T,dt,ROI,object_list_v,L_id_d,F_id_d,beside_delta,beside_id,x,sigma)
                
        if ude == "E":
            print("constraints in E!")
            object_list_v = [id_dict[L_id_e][2],id_dict[F_id_e][2],id_dict[L_id_u][2],id_dict[F_id_u][2],id_dict[L_id_d][2],id_dict[F_id_d][2]]#本车道V，旁车道V 
            beside_delta = [abs(D_L_u),abs(D_F_u),abs(D_L_d),abs(D_F_d)]
            beside_id = [L_id_u,F_id_u,L_id_d,F_id_d]
            constraints_onlane_x,constraints_beside_y,constaints_y_signal = self.get_feature_pre(id_dict,T,dt,ROI,object_list_v,L_id_e,F_id_e,beside_delta,beside_id,x,sigma)    
        return constraints_onlane_x.T,constraints_beside_y,constaints_y_signal

################################################################################################################################    
    #Decision Making  Finite State Machine
    # 计算threshold
    def Clac_threshold_F(self,v_l,Bc,d0):
        th = 4*v_l**2/(3*np.sqrt(3)*Bc) + d0
        return th
    
    def Clac_threshold_L(self,v_l,Bc_l,d0):
        th = 4*v_l**2/(3*np.sqrt(3)*Bc_l) + d0
        return th
    
    def threshold_demand_checking_l(self,rx,al):
        th_checking = (1 + al)*self.dyn.v + rx 
        return th_checking

    def threshold_demand_checking_f(self,rx):
        th_checking = rx 
        return th_checking

    # def distance_equation_L(self,al,Vl,L_id,id_dict,epsilon):
    #     info = id_dict[L_id]
    #     X_L,V_L = info[0],info[2]
    #     deltaX_L = X_L - self.dyn.x + V_L*(Vl-self.dyn.v)/al - (Vl**2-self.dyn.v**2)/(2*al) - (1 + epsilon)*self.dyn.v
    #     return deltaX_L
    
    # def distance_equation_F(self,al,Vl,F_id,id_dict,epsilon):
    #     info = id_dict[F_id]
    #     X_F,V_F = info[0],info[2]
    #     deltaX_F = self.dyn.x - X_F - V_F*(Vl-self.dyn.v)/al + (Vl**2-self.dyn.v**2)/(2*al) - (1 + epsilon)*V_F
    #     return deltaX_F
    
    def Clac_X_d(self,id_dict,id,H_pred,dt,a_ego):
        info = id_dict[id]
        X,V,a = info[0],info[2],info[3]
        X_list,V_list = np.array([X]),np.array([V])
        Ego_X_list,Ego_V_list = np.array([self.dyn.x]),np.array([self.dyn.v])
        for i in range(H_pred):
            L_Xmid = X_list[i] + V_list[i]*dt + dt**2/2*a
            L_Vmid = V_list[i] + a*dt
            X_list = np.append(X_list,L_Xmid)
            V_list = np.append(V_list,L_Vmid)
            if a_ego is None:
                a_ego = [0.0]
            Ego_Xmid = Ego_X_list[i] + Ego_V_list[i]*dt + dt**2/2*a_ego[0]
            Ego_Vmid = Ego_V_list[i] + a_ego[0]*dt
            Ego_X_list =np.append(Ego_X_list,Ego_Xmid)
            Ego_V_list = np.append(Ego_V_list,Ego_Vmid)
        X_diff_list = abs(X_list - Ego_X_list)
        X_diff_min = np.min(X_diff_list)
        return X_diff_min, V

        
    def Give_signal(self,H_pred,U_Line,D_Line,dt,a_ego,Bc,d0,E_center_y,U_center_y,D_center_y,change_distance,al,Vl,epsilon,Bc_l,rx):
        L_id_u,  D_L_u, F_id_u,  D_F_u, L_id_d, D_L_d, F_id_d ,D_F_d, L_id_e, D_L_e, F_id_e, D_F_e = self.Clac_LandLF()
        print("D_L_u=",D_L_u,"D_F_u=",D_F_u,"D_L_d=",D_L_d,"D_F_d=",D_F_d,"D_L_e=",D_L_e,"D_F_e=",D_F_e)
        print("Dfollower=",F_id_d)
        id_dict = self.get_id_dict(E_center_y,U_center_y,D_center_y)
        UDE_Signal = self.Judge_Location(U_Line,D_Line)
        th_checking_l = self.threshold_demand_checking_l(rx,al)
        th_checking_f = self.threshold_demand_checking_f(rx)
        
        if UDE_Signal == "U":
            Velocity_signal_L = 0
            Safe_signal_L = 0
            if L_id_u != "Unone":
                if L_id_e != "Enone":
                    X_diffLE_min, Le_V = self.Clac_X_d(id_dict,L_id_e,H_pred,dt,a_ego)
                    X_diffLU_min, Lu_V = self.Clac_X_d(id_dict,L_id_u,H_pred,dt,a_ego)
                    X_diffFE_min, Fe_V = self.Clac_X_d(id_dict,F_id_e,H_pred,dt,a_ego)
                    # threshold_F = self.distance_equation_F(al,Vl,F_id_e,id_dict,epsilon)
                    # threshold_L = self.distance_equation_L(al,Vl,L_id_e,id_dict,epsilon)
                    threshold_F = self.Clac_threshold_F(Fe_V,Bc,d0)
                    threshold_L = self.Clac_threshold_L(Le_V,Bc_l,d0)
                    Lu_V = id_dict[L_id_u][2]
                    if X_diffLE_min >= threshold_L and X_diffLE_min >= th_checking_l and X_diffFE_min >= threshold_F and X_diffFE_min >= th_checking_f:
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0
                    if Lu_V >= Le_V:
                        Velocity_signal_R = 0
                    else:    
                        Velocity_signal_R = 1        
                    if X_diffLE_min > change_distance:
                        Velocity_signal_R = 1    
                    if X_diffLU_min > change_distance:
                        Velocity_signal_R = 0
                        
                if L_id_e == "Enone":
                    Velocity_signal_R = 1
                    X_diffFE_min, Fe_V = self.Clac_X_d(id_dict,F_id_e,H_pred,dt,a_ego)
                    # threshold_F = self.distance_equation_F(al,Vl,F_id_e,id_dict,epsilon)
                    threshold_F = self.Clac_threshold_F(Fe_V,Bc,d0)
                    if X_diffFE_min >= threshold_F and X_diffFE_min >= th_checking_f: 
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0                  
            if L_id_u == "Unone":
                Safe_signal_R = 0
                Velocity_signal_R = 0
                                
        if UDE_Signal == "D":
            Velocity_signal_R = 0
            Safe_signal_R = 0
            if L_id_d != "Dnone":
                if L_id_e != "Enone":
                    X_diffLE_min, Le_V = self.Clac_X_d(id_dict,L_id_e,H_pred,dt,a_ego)
                    X_diffLD_min, Ld_V = self.Clac_X_d(id_dict,L_id_d,H_pred,dt,a_ego)
                    X_diffFE_min, Fe_V = self.Clac_X_d(id_dict,F_id_e,H_pred,dt,a_ego)
                    # threshold_F = self.distance_equation_F(al,Vl,F_id_e,id_dict,epsilon)
                    # threshold_L = self.distance_equation_L(al,Vl,L_id_e,id_dict,epsilon)
                    threshold_F = self.Clac_threshold_F(Fe_V,Bc,d0)
                    threshold_L = self.Clac_threshold_L(Le_V,Bc_l,d0)
                    Ld_V = id_dict[L_id_d][2]               
                    if X_diffLE_min >= threshold_L and X_diffLE_min >= th_checking_l and X_diffFE_min >= threshold_F and X_diffFE_min >= th_checking_f:
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0      
                    if Ld_V >= Le_V:
                        Velocity_signal_L = 0
                    elif Ld_V < Le_V:    
                        Velocity_signal_L = 1    
                    if X_diffLE_min > change_distance:
                        Velocity_signal_L = 1  
                    if X_diffLD_min > change_distance:
                        Velocity_signal_L = 0                      
                if L_id_e == "Enone":
                    Velocity_signal_L = 1    
                    X_diffFE_min, Fe_V = self.Clac_X_d(id_dict,F_id_e,H_pred,dt,a_ego)
                    # threshold_F = self.distance_equation_F(al,Vl,F_id_e,id_dict,epsilon)
                    threshold_F = self.Clac_threshold_F(Fe_V,Bc,d0) 
                    if X_diffFE_min >= threshold_F and  X_diffFE_min >= th_checking_f: 
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0                           
            if L_id_d == "Dnone":
                Safe_signal_L = 0
                Velocity_signal_L = 0

        if UDE_Signal == "E":
            if L_id_e != "Enone":
                X_diffFU_min, Fu_V = self.Clac_X_d(id_dict,F_id_u,H_pred,dt,a_ego)
                X_diffFD_min, Fd_V = self.Clac_X_d(id_dict,F_id_d,H_pred,dt,a_ego)
                # threshold_FU = self.distance_equation_F(al,Vl,F_id_u,id_dict,epsilon)
                # threshold_FD = self.distance_equation_F(al,Vl,F_id_d,id_dict,epsilon)
                threshold_FU = self.Clac_threshold_F(Fu_V,Bc,d0)
                threshold_FD = self.Clac_threshold_F(Fd_V,Bc,d0)
                Le_V = id_dict[L_id_e][2]
                if L_id_u != "Unone" and L_id_d != "Dnone":
                    X_diffLU_min, Lu_V = self.Clac_X_d(id_dict,L_id_u,H_pred,dt,a_ego)
                    X_diffLD_min, Ld_V = self.Clac_X_d(id_dict,L_id_d,H_pred,dt,a_ego)
                    X_diffLE_min, Le_V = self.Clac_X_d(id_dict,L_id_e,H_pred,dt,a_ego)
                    # threshold_LU = self.distance_equation_L(al,Vl,L_id_u,id_dict,epsilon)
                    # threshold_LD = self.distance_equation_L(al,Vl,L_id_d,id_dict,epsilon)
                    threshold_LU = self.Clac_threshold_L(Lu_V,Bc_l,d0)
                    threshold_LD = self.Clac_threshold_L(Ld_V,Bc_l,d0)
                    if X_diffFU_min >= threshold_FU and X_diffFU_min >= th_checking_f and X_diffLU_min >= threshold_LU and X_diffLU_min >= th_checking_l:
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0
                    if X_diffFD_min >= threshold_FD and X_diffFD_min >= th_checking_f and X_diffLD_min >= threshold_LD and X_diffLD_min >=th_checking_l:
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0
                    if Le_V >= Ld_V:
                        Velocity_signal_R = 0
                    elif Le_V < Ld_V:
                        Velocity_signal_R = 1
                    if Le_V >= Lu_V:
                        Velocity_signal_L = 0
                    elif Le_V < Lu_V:
                        Velocity_signal_L = 1
                        
                    if X_diffLU_min > change_distance:
                        Velocity_signal_L = 1
                        
                    if X_diffLD_min > change_distance:
                        Velocity_signal_R = 1 
                         
                    if X_diffLE_min > change_distance:
                        Velocity_signal_L = 0                     
                        Velocity_signal_R = 0    
                                         
                    if Velocity_signal_L and Safe_signal_L and Velocity_signal_R and Safe_signal_R:
                        if Lu_V > Ld_V:
                            Velocity_signal_R = 0
                        elif Lu_V < Ld_V:
                            Velocity_signal_L = 0
                            
                if L_id_u == "Unone" and L_id_d != "Dnone":
                    Velocity_signal_L = 1
                    Velocity_signal_R = 0
                    X_diffLD_min, Ld_V = self.Clac_X_d(id_dict,L_id_d,H_pred,dt,a_ego)
                    # threshold_LD = self.distance_equation_L(al,Vl,L_id_d,id_dict,epsilon)
                    threshold_LD = self.Clac_threshold_L(Ld_V,Bc_l,d0)
                    if X_diffFU_min >= threshold_FU:
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0
                    if X_diffFD_min >= threshold_FD and X_diffFD_min >= th_checking_f and  X_diffLD_min >= threshold_LD and X_diffLD_min >=th_checking_l:
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0

                if L_id_d == "Dnone" and L_id_u != "Unone":
                    Velocity_signal_R = 1
                    Velocity_signal_L = 0
                    X_diffLU_min, Lu_V = self.Clac_X_d(id_dict,L_id_u,H_pred,dt,a_ego)
                    # threshold_LU = self.distance_equation_L(al,Vl,L_id_u,id_dict,epsilon)
                    threshold_LU = self.Clac_threshold_L(Lu_V,Bc_l,d0)
                    if X_diffFD_min >= threshold_FD and X_diffFD_min >= th_checking_f:
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0
                    if X_diffFU_min >= threshold_FU and  X_diffFU_min >= th_checking_f and X_diffLU_min >= threshold_LU and X_diffLU_min >= th_checking_l:
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0
                        
                if L_id_u == "Unone" and L_id_d == "Dnone":
                    Velocity_signal_L = 1
                    Velocity_signal_R = 1
                    if X_diffFU_min >= threshold_FU and X_diffFU_min >= th_checking_f:
                        Safe_signal_L = 1
                    else:
                        Safe_signal_L = 0
                    
                    if X_diffFD_min >= threshold_FD and X_diffFD_min >= th_checking_f:
                        Safe_signal_R = 1
                    else:
                        Safe_signal_R = 0
                        
                    if Velocity_signal_L and Safe_signal_L and Velocity_signal_R and Safe_signal_R:
                        if threshold_FU <= threshold_FD:
                            Safe_signal_R = 1
                            Safe_signal_L = 0
                        elif threshold_FD <= threshold_FU:
                            Safe_signal_L = 1
                            Safe_signal_R = 0                            
                                       
            if L_id_e == "Enone":
                Safe_signal_L = 0
                Safe_signal_R = 0
                Velocity_signal_L = 0
                Velocity_signal_R = 0
        return Safe_signal_L, Velocity_signal_L, Safe_signal_R, Velocity_signal_R, UDE_Signal
    
    #做出并返回direction
    def Decision_Making(self,U_Line,D_Line,Bc,d0,H_pred,dt,a_ego,desired_v,E_center_y,U_center_y,D_center_y,change_distance,al,Vl,epsilon,Bc_l,rx):    
        Safe_signal_L, Velocity_signal_L, Safe_signal_R, Velocity_signal_R, UDE_Signal = self.Give_signal(H_pred,U_Line,D_Line,dt,a_ego,Bc,d0,E_center_y,U_center_y,D_center_y,change_distance,al,Vl,epsilon,Bc_l,rx)       
        print(Safe_signal_L, Velocity_signal_L, Safe_signal_R, Velocity_signal_R, UDE_Signal)
        if UDE_Signal == "U":
           if Safe_signal_R and Velocity_signal_R and self.dyn.v < desired_v:
               Final_signal = -1
               direction = "Change_down"
           else:
               Final_signal = 0
               direction = "Stay"
               
        if UDE_Signal == "D":
           if Safe_signal_L and Velocity_signal_L and self.dyn.v < desired_v:
               Final_signal = 1
               direction = "Change_up"
           else:
               Final_signal = 0
               direction = "Stay"
               
        if UDE_Signal == "E":
            if Safe_signal_L and Velocity_signal_L and self.dyn.v < desired_v:
                Final_signal = 1
                direction = "Change_up"
            elif Safe_signal_R and Velocity_signal_R and self.dyn.v < desired_v:
                Final_signal = -1
                direction = "Change_down" 
            else:
                Final_signal = 0
                direction = "Stay"                   
               
        return direction, Final_signal
    
    def Judge_Goal(self,U_Line,D_Line,goal_x,E_center_y,U_center_y,D_center_y,Bc,d0,H_pred,dt,a_ego,desired_v,change_distance,al,Vl,epsilon,Bc_l,rx):
        lane_now = self.Judge_Location(U_Line,D_Line)
        direction,Final_signal = self.Decision_Making(U_Line,D_Line,Bc,d0,H_pred,dt,a_ego,desired_v,E_center_y,U_center_y,D_center_y,change_distance,al,Vl,epsilon,Bc_l,rx)
        if lane_now == "U":
            if direction == "Stay":
               goal = [goal_x,U_center_y]   
            elif direction == "Change_down":
               goal = [goal_x,E_center_y]
        if lane_now == "D":
            if direction == "Stay":
               goal = [goal_x,D_center_y]
            elif direction == "Change_up":
               goal = [goal_x,E_center_y]      
        if lane_now == "E":
            if direction == "Stay":
               goal = [goal_x,E_center_y]
            elif direction == "Change_up":
               goal = [goal_x,U_center_y]            
            elif direction == "Change_down":
               goal = [goal_x,D_center_y]  
        return goal, Final_signal    

    #single solve step
    def Solve_step_mpc(self,u_lim, x0, oa, od,T,NX,NU,R,Q,Rd,Qf,du_lim,MAX_ITER,DU_TH,v_lim,dt,U_Line,D_Line,goal_x,E_center_y,U_center_y,D_center_y,d_safe_x,d_safe_y,L,TARGET_SPEED,gamma_L,gamma_F,Rt,ROI,Bc,d0,H_pred,change_distance,gamma_ycbf,sigma,al,Vl,Bc_l,rx):
        goal,Final_signal= self.Judge_Goal(U_Line,D_Line,goal_x,E_center_y,U_center_y,D_center_y,Bc,d0,H_pred,dt,oa,TARGET_SPEED,change_distance,al,Vl,sigma,Bc_l,rx)
        oa_, od_, ox, oy, oyaw, ov = self.Iterative_linear_mpc(x0, oa, od,T,MAX_ITER,DU_TH,NX,NU,R,Q,Rd,Qf,du_lim,u_lim,v_lim,dt,goal[0],goal[1],d_safe_x,d_safe_y,L,U_Line,D_Line,TARGET_SPEED,gamma_L,gamma_F,Rt,ROI,E_center_y,U_center_y,D_center_y,gamma_ycbf,sigma)
        return oa_, od_, ox, oy, oyaw, ov, Final_signal
        
