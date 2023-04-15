import numpy as np
import matplotlib.pyplot as plt
import math

class KinematicModel:
    
    def __init__(self, dt, L, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.dt = dt
        self.L = L
        
    def get_state_space(self, v, phi, delta):
    
        A = np.matrix(np.zeros((4, 4)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.dt * np.cos(phi)
        A[0, 3] = - self.dt * v * np.sin(phi)
        A[1, 2] = self.dt * np.sin(phi)
        A[1, 3] = self.dt * v * np.cos(phi)
        A[3, 2] = self.dt * np.tan(delta) / self.L

        B = np.matrix(np.zeros((4, 2)))
        B[2, 0] = self.dt
        B[3, 1] = self.dt * v / (self.L * math.cos(delta) ** 2)

        C = np.zeros(4)
        C[0] = self.dt * v * np.sin(phi) * phi
        C[1] = - self.dt * v * np.cos(phi) * phi
        C[3] = v * delta / (self.L * np.cos(delta) ** 2)

        return A, B, C
    
    def update_state(self, a, delta,MAX_STEER):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        self.x = self.x + self.v * np.cos(self.yaw) * self.dt
        self.y = self.y + self.v * np.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.v / self.L * np.tan(delta) * self.dt
        self.v = self.v + a * self.dt


    

        