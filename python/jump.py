#!/usr/bin/env python

import time
import numpy as np

import os
import rospy
import subprocess
import rospkg

import casadi as cs
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions import integrators
from horizon.solvers import solver

from horizon import problem
from horizon.ros import utils as horizon_ros_utils
from horizon.ros.replay_trajectory import *


def rotationMatrixToQuaternion(R):
    q = np.zeros(4)
    q[3] = 0.5 * cs.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2])
    q[0] = 0.5 * (R[2, 1] - R[1, 2]) / (4.0 * q[3])
    q[1] = 0.5 * (R[0, 2] - R[2, 0]) / (4.0 * q[3])
    q[2] = 0.5 * (R[1, 0] - R[0, 1]) / (4.0 * q[3])
    return q


matstorer = mat_storer.matStorerIO()

horizon_ros_utils.roslaunch("g1_horizon", "g1.launch")
time.sleep(3.0)

urdf = rospy.get_param("robot_description", "")
left_foot_frames = rospy.get_param("left_foot_list", [])
right_foot_frames = rospy.get_param("right_foot_list", [])
q_init = rospy.get_param("q_init", [])

q_init = np.array(q_init)

# print(left_foot_frames)
# print(right_foot_frames)

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

forward_kin = kindyn.fk("right_foot_point_contact")
pos = forward_kin(q=q_init)["ee_pos"]
rot = forward_kin(q=q_init)["ee_rot"]

q_init[0:3] -= np.array(pos).flatten()
q_init[3:7] = rotationMatrixToQuaternion(np.array(rot.T))

# Create horizon problem
ns = 50
prb = problem.Problem(ns)

# Creates problem STATE variables
q = prb.createStateVariable("q", kindyn.nq())
qdot = prb.createStateVariable("qdot", kindyn.nv())

# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", kindyn.nv())

left_foot_forces = dict()
for frame in left_foot_frames:
    left_foot_forces[frame] = prb.createInputVariable("f_" + frame, 3)

right_foot_forces = dict()
for frame in right_foot_frames:
    right_foot_forces[frame] = prb.createInputVariable("f_" + frame, 3)

foot_forces = dict()
foot_forces.update(left_foot_forces)
foot_forces.update(right_foot_forces)

dt = prb.createInputVariable("dt", 1)

# Limits
q.setBounds(kindyn.q_min(), kindyn.q_max())
q.setBounds(q_init, q_init, 0)  # starts in homing
q_target = q_init.copy()
q_target[0] = 0.5
q.setBounds(q_target, q_target, ns)  # ends in homing
q.setInitialGuess(q_init)

qdot_init = np.zeros((kindyn.nv(), 1))
qdot.setBounds(-kindyn.velocityLimits(), kindyn.velocityLimits())
qdot.setBounds(qdot_init, qdot_init, 0)  # starts with 0 velocity
qdot.setBounds(qdot_init, qdot_init, ns)  # ends with 0 velocity
qdot.setInitialGuess(qdot_init)

qddot_lim = 10.0 * kindyn.velocityLimits()
qddot.setBounds(-qddot_lim, qddot_lim)
qddot.setInitialGuess(qdot_init)

f_lim = 4000.0 * np.ones(3)

for frame, var in foot_forces.items():
    var.setBounds(-f_lim / 4.0, f_lim / 4.0)
    var.setInitialGuess(kindyn.mass() * np.array([0, 0, 10]) / 8.0)

dt.setBounds(0.02, 0.1)
dt.setInitialGuess(0.05)

variables_dict = {"q": q, "qdot": qdot, "qddot": qddot, "dt": dt}
for frame, var in foot_forces.items():
    variables_dict.update({var.getName(): var})

solution_guess = dict()
if matstorer.load(solution_guess):
    mat_storer.setInitialGuess(variables_dict, solution_guess)

# CONSTRAINTS
# Formulate discrete time dynamics
x = cs.vertcat(q, qdot)
xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
# print(xdot)
prb.setDynamics(xdot)
dae = {"x": x, "p": qddot, "ode": xdot, "quad": 0}
F_integrator = integrators.RK4(dae, opts=None)

qddot_prev = qddot.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
q_prev = q.getVarOffset(-1)
dt_prev = dt.getVarOffset(-1)
x_prev = cs.vertcat(q_prev, qdot_prev)
x_int = F_integrator(x=x_prev, u=qddot_prev, dt=dt_prev)
prb.setDt(dt)
prb.createConstraint("multiple_shooting", x_int["f"] - x, nodes=list(range(1, ns + 1)))

lift_node = 20
touch_down_node = 40

# Inverse Dynamics
tau_lim = kindyn.effortLimits()
tau_lim[0:6] = 0.0
# print(tau_lim)
tau = kin_dyn.InverseDynamics(
    kindyn,
    contact_frames=foot_forces.keys(),
    force_reference_frame=cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED,
).call(q, qdot, qddot, frame_force_mapping=foot_forces, tau_ext=0)
prb.createConstraint(
    "inverse_dynamics",
    tau,
    nodes=list(range(0, ns)),
    bounds=dict(lb=-tau_lim, ub=tau_lim),
)

# Foot constraints
for frame, f in foot_forces.items():
    # CONTACT
    V = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    v = V(q=q, qdot=qdot)["ee_vel_linear"]
    prb.createConstraint(
        f"{frame}_contact_before", v, nodes=list(range(0, lift_node + 1))
    )
    # FRICTION CONE
    mu = 0.8  # friction coefficient
    R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createConstraint(
        f"{frame}_friction_cone_before",
        fc,
        nodes=list(range(0, lift_node)),
        bounds=dict(lb=fc_lb, ub=fc_ub),
    )

    prb.createConstraint(
        f"{frame}_no_force", f, nodes=list(range(lift_node, touch_down_node))
    )

    prb.createConstraint(
        f"{frame}_contact_after", v, nodes=list(range(touch_down_node, ns + 1))
    )
    prb.createConstraint(
        f"{frame}_friction_cone_after",
        fc,
        nodes=list(range(touch_down_node, ns)),
        bounds=dict(lb=fc_lb, ub=fc_ub),
    )
    FK = kindyn.fk(frame)
    p = FK(q=q)["ee_pos"]
    p_init = FK(q=q_init)["ee_pos"]
    # p_init[1] += 0.5
    prb.createConstraint(
        f"{frame}_contact_after_position",
        p[2] - p_init[2],
        nodes=touch_down_node,
    )
    # prb.createConstraint(
    #     f"{frame}_contact_final",
    #     p[2] - p_init[2],
    #     nodes=ns,
    # )

# SET UP COST FUNCTION
MOM = kindyn.computeCentroidalDynamics()
h = MOM(q=q, v=qdot)["h_ang"]
prb.createCost("min_h_ang", 1000.0 * cs.sumsqr(h))

# q_ref = q_init
# q_ref[2] += 0.2
# prb.createCost("postural_xy", 1e5 * cs.sumsqr(q[0:2] - q_init[0:2]))
#
# prb.createCost(
#     "postural_z_before",
#     1e3 * cs.sumsqr(q[2] - q_init[2]),
#     nodes=list(range(0, lift_node)),
# )
# prb.createCost(
#     "jump",
#     6e5 * cs.sumsqr(q[2] - q_ref[2]),
#     nodes=list(range(lift_node, touch_down_node)),
# )
# prb.createCost(
#     "postural_z_after",
#     1e5 * cs.sumsqr(q[2] - q_init[2]),
#     nodes=list(range(touch_down_node, ns + 1)),
# )
# prb.createCost(
#     "postural_quat",
#     1e5 * cs.sumsqr(q[3:7] - q_init[3:7]),
#     nodes=list(range(touch_down_node, ns + 1)),
# )

# prb.createCost("postural", 1e3 * cs.sumsqr(q[7:-1] - q_init[7:-1]))

# prb.createCostFunction("min_qdot", 10.*cs.sumsqr(qdot), nodes=list(range(0, lift_node)))
# prb.createCost("min_qddot", 0.2 * cs.sumsqr(qddot), nodes=list(range(0, ns)))
# prb.createCost("min_f", 0.001 * cs.sumsqr(foot_forces), nodes=list(range(0, ns)))

# PROBLEM SOLVE
opts = {
    "ipopt.tol": 0.0001,
    "ipopt.constr_viol_tol": 0.0001,
    "ipopt.max_iter": 2000,
    "ipopt.linear_solver": "ma57",
}
solver = solver.Solver.make_solver("ipopt", prb, opts)
solver.solve()

solution = solver.getSolutionDict()

q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
left_foot_forces_hist = dict()
for frame in left_foot_forces:
    left_foot_forces_hist[frame] = solution["f_" + frame]
right_foot_forces_hist = dict()
for frame in right_foot_forces:
    right_foot_forces_hist[frame] = solution["f_" + frame]
foot_forces_hist = dict()
foot_forces_hist.update(left_foot_forces_hist)
foot_forces_hist.update(right_foot_forces_hist)

dt_hist = solution["dt"]

matstorer.store(solution)

# print(f"dt_hist: {dt_hist.flatten()}")
# print(f"time before jump: {np.sum(dt_hist.flatten()[0:lift_node])}")
# print(f"time during jump: {np.sum(dt_hist.flatten()[lift_node:touch_down_node])}")
# print(f"time after jump: {np.sum(dt_hist.flatten()[touch_down_node:-1])}")

tau_hist = np.zeros(qddot_hist.shape)
ID = kin_dyn.InverseDynamics(kindyn, left_foot_frames + right_foot_frames)
frame_forces_hist_i = dict()
for i in range(ns):
    for frame, force in foot_forces_hist.items():
        frame_forces_hist_i[frame] = force[:, i]

# resampling
resample_period = 0.001
q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = (
    resampler_trajectory.resample_torques(
        q_hist,
        qdot_hist,
        qddot_hist,
        dt_hist.flatten(),
        resample_period,
        dae,
        foot_forces_hist,
        kindyn,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED,
    )
)
# resampled torques does not contain the effect of the transmission, we need to add it:
time = np.arange(
    0.0,
    np.around(np.sum(dt_hist.flatten()), decimals=3) + resample_period,
    resample_period,
)

# Open rviz
rospack = rospkg.RosPack()
package_path = rospack.get_path("g1_horizon")
config_file_path = os.path.join(package_path, "launch", "g1_horizon.rviz")

if not os.path.exists(config_file_path):
    rospy.logerr("RViz config file not found: {}".format(config_file_path))

command = ["rosrun", "rviz", "rviz", "-d", config_file_path]

try:
    subprocess.Popen(command)
    rospy.loginfo("RViz opened with config: {}".format(config_file_path))
except Exception as e:
    rospy.logerr("Failed to open RViz: {}".format(e))

while not rospy.is_shutdown():
    repl = replay_trajectory(
        resample_period,
        kindyn.joint_names(),
        q_res,
        frame_force_res_mapping,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED,
        kindyn,
    )

    # repl.setSlowDownFactor(25.)
    repl.sleep(1.0)
    repl.replay(is_floating_base=True)
