# Policy models

The initial PPO uses SB3's MultiInputPolicy with flattened dictionary features
and separate 128x128 actor/critic MLPs, configured in agent/ppo_backend.py.
There is no custom model implementation in this directory yet.

No custom model is implemented yet. The current environment produces player and
bullet arrays. A state encoder or a later image encoder belongs here; process
memory and keyboard access remain in `env/`.
