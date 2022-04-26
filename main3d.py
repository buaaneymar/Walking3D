import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib
import planner
# import cv2
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


class Biped3D(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        self.ground = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        self.robot = p.loadURDF("/yanshee/robots/yanshee.urdf",  basePosition = [0,0,0],baseOrientation = [0, 0, 0, 1],useFixedBase=False)
        self.joints = self.get_joints()
        self.n_j = len(self.joints)
        self.simu_f = 500 # Simulation frequency, Hz
        p.setTimeStep(1.0/self.simu_f)
        self.motion_f = 2  # Controlled motion frequency, Hz
        self.stance_idx = 0
        self.pre_foot_contact = np.array([1, 0])
        self.foot_contact = np.array([1, 0])
        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.q_d_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.v_mat = np.zeros((self.simu_f * 3))
        self.v_d_mat = np.ones(self.simu_f * 3)
        self.init_plot()
#

#
#
    def run(self):
        for i in range(int(5e3)):
            t = i / self.simu_f
            torque_array = self.controller(t)
            self.q_vec, self.dq_vec = self.step(torque_array)
            if 0 == i % 20:
                self.update_plot()
            time.sleep(1/self.simu_f)
        p.disconnect()

    def step(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[self.q_vec[0], 0, 0.4])
        return self.get_joint_states()

    def get_joints(self):
        all_joints = []
        for j in range(p.getNumJoints(self.robot)):
            # Disable motor in order to use direct torque control.
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
                all_joints.append(j)
                p.setJointMotorControl2(self.robot, j,
                                        controlMode=p.POSITION_CONTROL, force=0)
        joints = all_joints[0:]
        # print("Number of All Joints:", p.getNumJoints(self.robot))
        # print("Number of All Revolute Joints:", joints)
        return joints

    def get_joint_states(self):
        '''
        :return: q_vec: joint angle, dq_vec: joint angular velocity
        '''
        q_vec = np.zeros(self.n_j)
        dq_vec = np.zeros(self.n_j)
        for j in range(self.n_j):
            q_vec[j], dq_vec[j], _, _  = p.getJointState(self.robot, self.joints[j])
        return q_vec, dq_vec

    def set_motor_torque_array(self, torque_array = None):
        '''
        :param torque_array: the torque of [rHAA, rHFE, rKFE]
        :Specifically, H: Hip, K: Knee, AA:abduction/adduction, FE:flexion/extension
        '''
        if torque_array is None:
            torque_array = np.zeros(self.n_j)
        for j in range(len(self.joints)):
            p.setJointMotorControl2(self.robot, self.joints[j], p.TORQUE_CONTROL, force=torque_array[j])

    def controller(self, t, type='joint'):
        if 'joint' == type:
            return self.joint_controller(t)

    def joint_controller1(self, t):
        a = 0.15
        b = 0.15
        phi = 2 * np.pi * self.motion_f * t
        x = a * np.cos(phi)
        y = b * np.sin(phi) - 0.75
        if (y<=-0.75):
            y = -0.75
        q_d_h, q_d_k,q_d_a= self.cal(x, y)
        return q_d_h, q_d_k,q_d_a

    def cal(self, x, y):
        l1 = 0.5
        l2 = 0.5
        l3 = np.sqrt(x ** 2 + y ** 2)
        z = (l3 ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        q_d_k = -np.arccos(z)
        q_d_h = np.arctan(x / y) - q_d_k / 2
        q_d_a = -(q_d_h+q_d_k)
        return q_d_h, q_d_k, q_d_a

    def joint_controller(self, t):
        q_d_b = 0.25*t
        q_d_l_h, q_d_l_k,q_d_l_a = self.joint_controller1(t)
        q_d_r_h, q_d_r_k,q_d_r_a= self.joint_controller1(t - 0.2)
        q_d_r_h = -q_d_r_h
        q_d_r_k = -q_d_r_k
        q_d_r_a = -q_d_r_a
        q_d_vec = np.array([q_d_b, q_d_l_h, q_d_l_k, q_d_l_a, q_d_r_h, q_d_r_k, q_d_r_a])
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec


        dq_d_b = 0.25
        q_d_l_h_pre,q_d_l_k_pre,q_d_l_a_pre = self.joint_controller1(t - 0.2 / self.simu_f)
        q_d_r_h_pre,q_d_r_k_pre,q_d_r_a_pre = self.joint_controller1(t - 0.2 - 0.2 / self.simu_f)
        q_d_r_h_pre = -q_d_r_h_pre
        q_d_r_k_pre = -q_d_r_k_pre
        q_d_r_a_pre = -q_d_r_a_pre
        q_d_vec_pre = np.array([dq_d_b,q_d_l_h_pre,q_d_l_k_pre,q_d_l_a_pre,q_d_r_h_pre,q_d_r_k_pre, q_d_r_a_pre])
        dq_d_vec = (q_d_vec - q_d_vec_pre) / (1 / self.simu_f)
        self.v_mat[:-1] = self.v_mat[1:]
        self.v_mat[-1] = self.dq_vec[0]
        k_vec = np.array([100,50, 20, 20,50, 20, 20])
        b_vec = np.array([0.05, 0.05, 0.01, 0.01,0.05, 0.01, 0.01])
        return self.joint_PD_controller(k_vec, b_vec, self.q_vec, self.dq_vec, q_d_vec, dq_d_vec)

    def joint_PD_controller(self, k_vec, b_vec, q_vec, dq_vec, q_d_vec, dq_d_vec):
        return k_vec*(q_d_vec-q_vec) + b_vec*(dq_d_vec-dq_vec)

    def init_plot(self):
        self.fig = plt.figure(figsize=(5, 9))
        joint_names = ['b', 'l_h', 'l_k','l_a','r_h','r_k','r_a' ]
        self.q_d_lines = []
        self.q_lines = []
        self.v_line = []
        self.v_d_line = []
        for i in range(7):
            plt.subplot(8, 1, i+1)
            q_d_line, = plt.plot(self.q_d_mat[:, i], '-')
            q_line, = plt.plot(self.q_mat[:, i], '--')
            self.q_d_lines.append(q_d_line)
            self.q_lines.append(q_line)
            plt.ylabel('q_{} (rad)'.format(joint_names[i]))
            plt.ylim([-1.5, 1.5])
        plt.subplot(8, 1, 8)
        v_d_line, = plt.plot(self.v_d_mat[:], '-')
        v_line, = plt.plot(self.v_mat[:], '--')
        self.v_d_line.append(v_d_line)
        self.v_line.append(v_line)
        plt.ylim([-0.5, 0.5])
        plt.ylabel('velocity')
        self.fig.legend(['v_d', 'v'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.125), frameon=False)
        plt.xlabel('Simulation steps')
        self.fig.legend(['q_d', 'q'], loc='lower center', ncol=2, bbox_to_anchor=(0.49, 0.97), frameon=False)
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        for i in range(7):
            self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
            self.q_lines[i].set_ydata(self.q_mat[:, i])
        self.v_d_line[0].set_ydata(0.25*self.v_d_mat[:])
        self.v_line[0].set_ydata(self.v_mat[:])

        plt.draw()
        plt.pause(0.001)

if __name__ == '__main__':
    robot = Biped3D()
    robot.run()
