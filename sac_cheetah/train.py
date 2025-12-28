# sac_cheetah/train.py

# observation breakdown (17 dims):")
#   [0]: z-coordinate of torso (height)")
#   [1]: y-rotation angle of torso (pitch)")
#   [2:8]: 6 joint angles")
#   [8]: x-velocity of torso")
#   [9]: z-velocity of torso")
#   [10]: y-angular velocity of torso")
#   [11:17]: 6 joint angular velocities")

# action breakdown
#   six torques
#   Box(-1.0, 1.0, (6,), float32)

# obs: z, p, th_bar | xdot, zdot, p_dot, th_bar_dot
# (x excluded so policy generalises to joints not exact position)
# action: tau_bar
# reward: r = xdot - 0.1 * |action|**2
# no termination, cheetah can flip/fail silently
# truncates at 1000 steps
