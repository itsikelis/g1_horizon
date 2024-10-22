#!/usr/bin/env python
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt
import os
import time
from horizon.ros import utils as horizon_ros_utils
import kangaroo

matstorer = mat_storer.matStorerIO()

#INIT
horizon_ros_utils.roslaunch("kangaroo_horizon", "bring_up_id.launch")
time.sleep(3.)

urdf = rospy.get_param("robot_description", "")
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
kng = kangaroo.kangaroo(kindyn)

# Create horizon problem
ns = 50
prb = problem.Problem(ns)

# Creates problem STATE variables
q = prb.createStateVariable("q", kindyn.nq())
qdot = prb.createStateVariable("qdot", kindyn.nv())


# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", kindyn.nv())

left_foot_forces = dict()
for frame in kng.left_foot_frames:
    left_foot_forces[frame] = prb.createInputVariable("f_"+frame, 3)

right_foot_forces = dict()
for frame in kng.right_foot_frames:
    right_foot_forces[frame] = prb.createInputVariable("f_"+frame, 3)

foot_forces = dict()
foot_forces.update(left_foot_forces)
foot_forces.update(right_foot_forces)

left_lambda = prb.createInputVariable("left_lambda", 2)
right_lambda = prb.createInputVariable("right_lambda", 2)

dt = prb.createInputVariable("dt", 1)

#Limits
q.setBounds(kng.q_min, kng.q_max)
q.setBounds(kng.q_init, kng.q_init, 0) # starts in homing
q.setInitialGuess(kng.q_init)

qdot.setBounds(kng.qdot_min, kng.qdot_max)
qdot.setBounds(kng.qdot_init, kng.qdot_init, 0) # starts with 0 velocity
qdot.setBounds(kng.qdot_init, kng.qdot_init, ns) # ends with 0 velocity
qdot.setInitialGuess(kng.qdot_init)

qddot.setBounds(kng.qddot_min, kng.qddot_max)
qddot.setInitialGuess(kng.qddot_init)

for frame, var in left_foot_forces.items():
    var.setBounds(kng.f_min / 4., kng.f_max / 4.)
    var.setInitialGuess(kng.f_init / 6.)

for frame, var in right_foot_forces.items():
    var.setBounds(kng.f_min / 4., kng.f_max / 4.)
    var.setInitialGuess(kng.f_init / 6.)

left_lambda.setBounds([-100., -100.], [100, 100])
left_lambda.setInitialGuess([10., 10.])
right_lambda.setBounds([-100., -100.], [100, 100])
right_lambda.setInitialGuess([10., 10.])

dt.setBounds(0.02, 0.1)
dt.setInitialGuess(0.01)

variables_dict = {"q":q, "qdot":qdot, "qddot":qddot, "dt": dt, "left_lambda": left_lambda, "right_lambda":right_lambda}
for frame, var in right_foot_forces.items():
    variables_dict.update({var.getName(): var})
for frame, var in left_foot_forces.items():
    variables_dict.update({var.getName(): var})
solution_guess = dict()
if matstorer.load(solution_guess):
    mat_storer.setInitialGuess(variables_dict, solution_guess)

# CONSTRAINTS
# Formulate discrete time dynamics
x = cs.vertcat(q, qdot)
xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
prb.setDynamics(xdot)
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 0}
F_integrator = integrators.RK4(dae, opts=None)

qddot_prev = qddot.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
q_prev = q.getVarOffset(-1)
dt_prev = dt.getVarOffset(-1)
x_prev = cs.vertcat(q_prev, qdot_prev)
x_int = F_integrator(x=x_prev, u=qddot_prev, dt=dt_prev)
prb.setDt(dt)
prb.createConstraint("multiple_shooting", x_int["f"] - x, nodes=list(range(1, ns+1)))


lift_node = 20
touch_down_node = 40

# TRANSMISSION
kng.kinematicPositionTransmissionTaskLeftLeg(prb, q, 1e9)
kng.kinematicPositionTransmissionTaskRightLeg(prb, q, 1e9)

kng.kinematicTransmissionConstraintLeftLeg(prb, q, qdot)
kng.kinematicTransmissionConstraintRightLeg(prb, q, qdot)

tau_transmission = kng.computeTransmissionLeftLegTorques(q, left_lambda) + \
                   kng.computeTransmissionRightLegTorques(q, right_lambda)

# Inverse Dynamics
tau_min = -kng.tau_lims
tau_max = kng.tau_lims
tau = kin_dyn.InverseDynamics(kindyn, foot_forces.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(
    q, qdot, qddot, foot_forces, tau_transmission)
prb.createConstraint("inverse_dynamics", tau, nodes=list(range(0, ns)), bounds=dict(lb=tau_min, ub=tau_max))


for frame, f in foot_forces.items():
    # CONTACT
    V = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    v = V(q=q, qdot=qdot)['ee_vel_linear']
    prb.createConstraint(f"{frame}_contact_before", v, nodes=list(range(0, lift_node+1)))
    # FRICTION CONE
    mu = 0.8  # friction coefficient
    R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createConstraint(f"{frame}_friction_cone_before", fc, nodes=list(range(0, lift_node)), bounds=dict(lb=fc_lb, ub=fc_ub))

    prb.createConstraint(f"{frame}_no_force", f, nodes=list(range(lift_node, touch_down_node)))

    prb.createConstraint(f"{frame}_contact_after", v, nodes=list(range(touch_down_node, ns+1)))
    prb.createConstraint(f"{frame}_friction_cone_after", fc, nodes=list(range(touch_down_node, ns)), bounds=dict(lb=fc_lb, ub=fc_ub))
    FK = kindyn.fk(frame)
    p = FK(q=q)['ee_pos']
    p_init = FK(q=kng.q_init)['ee_pos']
    #p_init[1] += 0.5
    prb.createConstraint(f"{frame}_contact_after_position", p[1:3]-p_init[1:3], nodes=touch_down_node)


# SET UP COST FUNCTION
MOM = kindyn.computeCentroidalDynamics()
h = MOM(q=q, v=qdot)['h_ang']
prb.createCost("min_h_ang", 1000.*cs.sumsqr(h))

q_ref = kng.q_init
q_ref[2] += 0.2
prb.createCost("postural_xy", 1e5*cs.sumsqr(q[0:2] - kng.q_init[0:2]))

prb.createCost("postural_z_before", 1e3*cs.sumsqr(q[2] - kng.q_init[2]), nodes=list(range(0, lift_node)))
prb.createCost("jump", 6e5*cs.sumsqr(q[2] - q_ref[2]), nodes=list(range(lift_node, touch_down_node)))
prb.createCost("postural_z_after", 1e5*cs.sumsqr(q[2] - kng.q_init[2]), nodes=list(range(touch_down_node, ns+1)))

prb.createCost("postural_quat", 1e5*cs.sumsqr(q[3:7] - kng.q_init[3:7]), nodes=list(range(touch_down_node, ns+1)))

prb.createCost("postural", 1e3*cs.sumsqr(q[7:-1] - kng.q_init[7:-1]))

#prb.createCostFunction("min_qdot", 10.*cs.sumsqr(qdot), nodes=list(range(0, lift_node)))
prb.createCost("min_qddot", 0.2*cs.sumsqr(qddot), nodes=list(range(0, ns)))
prb.createCost("min_f", 0.001*cs.sumsqr(kangaroo.concat(foot_forces)), nodes=list(range(0, ns)))


# PROBLEM SOLVE
opts = {'ipopt.tol': 0.0001,
        'ipopt.constr_viol_tol': 0.0001,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}
solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()

solution = solver.getSolutionDict()

q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
left_foot_forces_hist = dict()
for frame in left_foot_forces:
    left_foot_forces_hist[frame] = solution["f_"+frame]
right_foot_forces_hist = dict()
for frame in right_foot_forces:
    right_foot_forces_hist[frame] = solution["f_"+frame]
foot_forces_hist = dict()
foot_forces_hist.update(left_foot_forces_hist)
foot_forces_hist.update(right_foot_forces_hist)

left_lambda_hist = solution["left_lambda"]
right_lambda_hist = solution["right_lambda"]

dt_hist = solution["dt"]

matstorer.store(solution)


print(f"dt_hist: {dt_hist.flatten()}")
print(f"time before jump: {np.sum(dt_hist.flatten()[0:lift_node])}")
print(f"time during jump: {np.sum(dt_hist.flatten()[lift_node:touch_down_node])}")
print(f"time after jump: {np.sum(dt_hist.flatten()[touch_down_node:-1])}")


tau_hist = np.zeros(qddot_hist.shape)
ID = kin_dyn.InverseDynamics(kindyn, kng.left_foot_frames + kng.right_foot_frames)
frame_forces_hist_i = dict()
for i in range(ns):
    for frame, force in foot_forces_hist.items():
        frame_forces_hist_i[frame] = force[:, i]

    tau_ext = kng.computeTransmissionLeftLegTorques(q_hist[:, i], left_lambda_hist[:, i]) + \
                       kng.computeTransmissionRightLegTorques(q_hist[:, i], right_lambda_hist[:, i])

    tau_hist[:, i] = ID.call(q_hist[:, i], qdot_hist[:, i], qddot_hist[:, i], frame_forces_hist_i, np.array(tau_ext)).toarray().flatten()

kng.plotTransmissionRelativeError(q_hist)
kng.plotBaseWrench(tau_hist)
kng.plotUnderactuatedJointTorques(tau_hist)
kng.plotBasePosition(q_hist)
kng.plotActuatedJointTorques(tau_hist)
kng.plotBaseVelocities(qdot_hist)
kng.plotBaseAccelerations(qddot_hist)
kng.plotContactForces(left_foot_forces_hist, title_append=" left")
kng.plotContactForces(right_foot_forces_hist, title_append=" right")
kng.plotFramePosition(kng.sole_frames[0], q_hist, kindyn)
kng.plotFramePosition(kng.sole_frames[1], q_hist, kindyn)


# resampling
resample_period = 0.001
q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = resampler_trajectory.resample_torques(
    q_hist, qdot_hist, qddot_hist, dt_hist.flatten(), resample_period, dae, foot_forces_hist, kindyn,
    cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
#resampled torques does not contain the effect of the transmission, we need to add it:
left_lambda_res = resampler_trajectory.resample_input(left_lambda_hist, dt_hist.flatten(), resample_period)
right_lambda_res = resampler_trajectory.resample_input(right_lambda_hist, dt_hist.flatten(), resample_period)
for k in range(0, tau_res.shape[1]):
    tau_left_res = kng.computeTransmissionLeftLegTorques(q_res[:, k].T, left_lambda_res[:, k].T)
    tau_right_res = kng.computeTransmissionRightLegTorques(q_res[:, k].T, right_lambda_res[:, k].T)
    for i in range(0, tau_res.shape[0]):
        tau_res[i, k] = tau_res[i, k] - tau_left_res[i] - tau_right_res[i]

time = np.arange(0.0, np.around(np.sum(dt_hist.flatten()), decimals=3)+resample_period, resample_period)
# kng.plotTransmissionRelativeError(q_res, time.T, title_append=" resampled")
# kng.plotBaseWrench(tau_res, time.T, title_append=" resampled")
# kng.plotUnderactuatedJointTorques(tau_res, time.T, title_append=" resampled")
# kng.plotBasePosition(q_res, time.T, title_append=" resampled")
# kng.plotActuatedJointTorques(tau_res, time.T, title_append=" resampled")


repl = replay_trajectory(resample_period, kng.joint_list, q_res,
                         frame_force_res_mapping, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
#repl.setSlowDownFactor(25.)
repl.sleep(1.)
repl.replay(is_floating_base=True)


