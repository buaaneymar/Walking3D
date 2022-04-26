import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

g = 9.81  # m/s^2
l_thigh = 0.06
l_shin = 0.07


def update_paras(zh):
    # 都相对于stance phase支撑点的坐标系
    w = np.sqrt(g / zh)  # LIPM 常数
    xh_0 = 0  # hip的起始位置
    vh_0 = 1.6 * w * 0.001 # hip的起始速度  需要合理设计，否则轨道能不足以冲过平稳点
    xh_1 = -xh_0  # hip的终止位置
    vh_1 = 0  # hip的终止速度
    xf_0 = 0  # swing phase起始落脚点
    xf_1 = -0.02  # swing phase终止落脚点
    T_sw = 0.4  # 完成stance phase所需的时间
    # 描述一个行走单元所需要的参数
    paras = {'xh_0': xh_0,
             'vh_0': vh_0,
             'xh_1': xh_1,
             'vh_1': vh_1,
             'xf_0': xf_0,
             'xf_1': xf_1,
             'T_sw': T_sw,
             'zh': zh}
    return paras


# def calc_gait_paras(zh):  # 计算步长
#     leg_length = 0.1  # 虚拟腿长
#     half_step_length = np.sqrt(leg_length ** 2 - zh ** 2)  # 虚拟腿足端的位置
#     return half_step_length


def calc_foot_trajectory_control_points(xf_0, xf_1):
    # 贝塞尔曲线的控制点
    control_points = np.asarray([
        [xf_0, 0],
        [-0.008, 0.0008],
        [-0.015, 0.0032],
        [-0.018, 0.0064],
        [xf_1, 0.01]])
    return control_points


# def calc_T_sw(xh_0, vh_0, xh_1, vh_1, zh):
#     # 计算完成stance phase所需的时间
#     w = np.sqrt(g / zh)
#     T_sw = np.log((w * xh_1 + vh_1) / (w * xh_0 + vh_0)) / w
#     return T_sw


def calc_hip_and_ankle_position(t, paras):
    xh_t, vh_t = calc_hip_trajectory(t, paras['xh_0'], paras['vh_0'], paras['zh'])
    xzf_t, v_xzf_t = calc_foot_tajectory(t, paras['T_sw'], paras['xf_0'], paras['xf_1'])
    return xh_t, xzf_t[0], xzf_t[1]


def calc_hip_trajectory(t, xh_0, vh_0, zh):
    # 根据LIPM计算hip的位置
    w = np.sqrt(g / zh)
    xh_t = xh_0 * np.cosh(w * t) + vh_0 * np.sinh(w * t) / w
    vh_t = xh_0 * w * np.sinh(w * t) + vh_0 * np.cosh(w * t)
    return xh_t, vh_t


def calc_foot_tajectory(t, T_sw, xf_0, xf_1):
    # 根据贝塞尔曲线轨迹，计算swing phase的足端位置
    control_points = calc_foot_trajectory_control_points(xf_0, xf_1)
    s = t / T_sw
    xzf_t, v_xzf_t = bezier_curve(s, control_points, T_sw)
    return xzf_t, v_xzf_t


def IK(xh, zh, xf, zf):  # inverse kinematics
    l_hf = np.sqrt((xh - xf) ** 2 + (zh - zf) ** 2)
    q_knee = - np.arccos((l_hf ** 2 - l_thigh ** 2 - l_shin ** 2) / (2 * l_thigh * l_shin))
    print('q_knee',q_knee)
    q_hip = np.arctan2(xf - xh, zh - zf) - q_knee / 2
    print('q_hip',q_hip)
    q_ankle = -q_knee - q_hip
    return np.array([q_hip, q_knee, q_ankle])


def bernstein_poly(i, n, s):  # 伯恩斯坦多项式
    """
     The Bernstein polynomial of n, k as a function of t
    """

    return comb(n, i) * (s ** i) * (1 - s) ** (n - i)


def bezier_curve(s, control_points, T_sw):  # 贝塞尔曲线
    """
        Given a set of control points, return the
        bezier curve defined by the control points.

        points should be a 2d numpy array:
               [ [1,1],
                 [2,3],
                 [4,5],
                 ..
                 [Xn, Yn] ]
        s in [0, 1] is the current phase
        See https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
        :return
    """
    n = control_points.shape[0] - 1
    B_vec = np.zeros((1, n + 1))  # a vector of each bernstein polynomial
    for i in range(n + 1):
        B_vec[0, i] = bernstein_poly(i, n, s)
    x_d = np.matmul(B_vec, control_points).squeeze()  # np.squeeze() 将表示向量的数组转换为秩为1的数组  i.e. (10,1) -> (10,)
    d_control_points = control_points[1:] - control_points[:-1]  # 求差
    v_d = (1 / T_sw) * n * np.matmul(B_vec[:, :-1], d_control_points).squeeze()

    return x_d, v_d


# def update_paras(zh):
#     # 都相对于stance phase支撑点的坐标系
#     w = np.sqrt(g / zh) # LIPM 常数
#     half_step_length = calc_gait_paras(zh)
#     xh_0 = -half_step_length # hip的起始位置
#     vh_0 = 1.5 * w * abs(xh_0) # hip的起始速度  需要合理设计，否则轨道能不足以冲过平稳点
#     xh_1 = -xh_0 # hip的终止位置
#     vh_1 = vh_0 # hip的终止速度
#     xf_0 = 2 * xh_0 # swing phase起始落脚点
#     xf_1 = 1.8 * xh_1 # swing phase终止落脚点
#     T_sw = calc_T_sw(xh_0, vh_0, xh_1, vh_1, zh) # 完成stance phase所需的时间
#     # 描述一个行走单元所需要的参数
#     paras = {'xh_0': xh_0,
#              'vh_0': vh_0,
#              'xh_1': xh_1,
#              'vh_1': vh_1,
#              'xf_0': xf_0,
#              'xf_1': xf_1,
#              'T_sw': T_sw,
#              'zh': zh}
#     return paras
#
#
# def calc_gait_paras(zh): # 计算步长
#     leg_length = 0.1 # 虚拟腿长
#     half_step_length = np.sqrt(leg_length ** 2 - zh ** 2) # 虚拟腿足端的位置
#     return half_step_length
#
#
# def calc_foot_trajectory_control_points(xf_0, xf_1):
#     # 贝塞尔曲线的控制点
#     control_points = np.asarray([
#         [xf_0, 0],
#         [xf_0, 0.02],
#         [0, 0.04],
#         [xf_1, 0.02],
#         [xf_1, 0]])
#     return control_points
#
#
# def calc_T_sw(xh_0, vh_0, xh_1, vh_1, zh):
#     # 计算完成stance phase所需的时间
#     w = np.sqrt(g / zh)
#     T_sw = np.log((w * xh_1 + vh_1) / (w * xh_0 + vh_0)) / w
#     return T_sw
#
#
# def calc_hip_and_ankle_position(t, paras):
#     xh_t, vh_t = calc_hip_trajectory(t, paras['xh_0'], paras['vh_0'], paras['zh'])
#     xzf_t, v_xzf_t = calc_foot_tajectory(t, paras['T_sw'], paras['xf_0'], paras['xf_1'])
#     return xh_t, xzf_t[0], xzf_t[1]
#
#
# def calc_hip_trajectory(t, xh_0, vh_0, zh):
#     # 根据LIPM计算hip的位置
#     w = np.sqrt(g / zh)
#     xh_t = xh_0 * np.cosh(w * t) + vh_0 * np.sinh(w * t) / w
#     vh_t = xh_0 * w * np.sinh(w * t) + vh_0 * np.cosh(w * t)
#     return xh_t, vh_t
#
#
# def calc_foot_tajectory(t, T_sw, xf_0, xf_1):
#     # 根据贝塞尔曲线轨迹，计算swing phase的足端位置
#     control_points = calc_foot_trajectory_control_points(xf_0, xf_1)
#     s = t / T_sw
#     xzf_t, v_xzf_t = bezier_curve(s, control_points, T_sw)
#     return xzf_t, v_xzf_t
#
#
# def IK(xh, zh, xf, zf): # inverse kinematics
#     l_hf = np.sqrt((xh-xf)**2 + (zh - zf)**2)
#     q_knee = - np.arccos((l_hf ** 2 - l_thigh ** 2 - l_shin ** 2) / (2 * l_thigh * l_shin))
#     q_hip = np.arctan2(xf - xh, zh - zf) - q_knee / 2
#     q_ankle = -q_knee - q_hip
#     return np.array([q_hip, q_knee, q_ankle])
#
#
#
#
#
#
# def bernstein_poly(i, n, s): # 伯恩斯坦多项式
#     """
#      The Bernstein polynomial of n, k as a function of t
#     """
#
#     return comb(n, i) * (s ** i) * (1 - s) ** (n - i)
#
#
# def bezier_curve(s, control_points, T_sw): # 贝塞尔曲线
#     """
#         Given a set of control points, return the
#         bezier curve defined by the control points.
#
#         points should be a 2d numpy array:
#                [ [1,1],
#                  [2,3],
#                  [4,5],
#                  ..
#                  [Xn, Yn] ]
#         s in [0, 1] is the current phase
#         See https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
#         :return
#     """
#     n = control_points.shape[0] - 1
#     B_vec = np.zeros((1, n + 1)) # a vector of each bernstein polynomial
#     for i in range(n + 1):
#         B_vec[0, i] = bernstein_poly(i, n, s)
#     x_d = np.matmul(B_vec, control_points).squeeze() # np.squeeze() 将表示向量的数组转换为秩为1的数组  i.e. (10,1) -> (10,)
#     d_control_points = control_points[1:] - control_points[:-1] # 求差
#     v_d = (1 / T_sw) * n * np.matmul(B_vec[:, :-1], d_control_points).squeeze()
#
#     return x_d, v_d
#




def main():
    xx_d_mat = []
    xz_d_mat = []
    vx_d_mat = []
    vz_d_mat = []
    T_sw = 0.4 # s
    control_points = np.asarray([
        [0, 0],
        [-0.008, 0.0004],
        [-0.015, 0.0016],
        [-0.018, 0.0032],
        [-0.02, 0.005]])

    for t in range(0, 40):
        s = t / 100 / T_sw
        x_d, v_d = bezier_curve(s, control_points, T_sw)
        xx_d_mat.append(x_d[0])
        xz_d_mat.append(x_d[1])
        vx_d_mat.append(v_d[0])
        vz_d_mat.append(v_d[1])
    t = [i / 100 / T_sw for i in range(0,40)]
    
    # fig0 = plt.figure()
    plt.plot(xx_d_mat, xz_d_mat)
    plt.plot(control_points[:,0], control_points[:,1])
    plt.axis('equal')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax1.plot(t, xx_d_mat)
    ax2 = fig1.add_subplot(222)
    ax2.plot(t, xz_d_mat)
    ax3 = fig1.add_subplot(223)
    ax3.plot(t, vx_d_mat)
    ax4 = fig1.add_subplot(224)
    ax4.plot(t, vz_d_mat)
    
    plt.show()

if __name__ == '__main__':
    main()
