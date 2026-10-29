# Learning algorithms

`baselines.py` provides seeded random, stationary, and reactive bullet-avoidance
policies. All use the environment's encoded observation and return action IDs
0-8. The reactive policy predicts linear bullet motion over 12 frames and assumes
fixed low-speed movement at 2 px/frame. It ignores enemies, lasers, acceleration,
and future spawns. This is a comparison baseline, not a trained model.
Holding Z does not ensure damage: this policy never aligns with enemies.
Surviving a spell timeout is not necessarily a successful spell clear.

PPO is not implemented yet. Update-time game control and successful spell ending
must be addressed before unattended training.
