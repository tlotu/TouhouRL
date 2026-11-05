"""Offline combined bullet/laser candidates; not connected to PPO or input."""
from .boundary_control import feasible_actions
from .laser_geometry import forecast_static_lasers
from .risk_control import RISK_OPTIONS, canonical_risk_observation, forecast_clearances


def combined_candidates(obs, lasers, hold_frames=4):
    """Apply the existing clearance gate to both hypotheses and physical bounds."""
    state=canonical_risk_observation(obs)
    allowed=feasible_actions(state,hold_frames)
    bullet=forecast_clearances(state)
    player=((float(state['player'][0])+1)*192,(float(state['player'][1])+1)*224)
    laser=forecast_static_lasers(player,lasers)
    gaps=[min(b,l) for b,l in zip(bullet,laser)]
    best=max(gaps[a] for a in allowed)
    emergency=best<RISK_OPTIONS['min_clearance']
    threshold=best if emergency else max(RISK_OPTIONS['min_clearance'],best-RISK_OPTIONS['max_clearance_loss'])
    return {'allowed':allowed,'bullet_clearances':bullet,'laser_clearances':laser,
        'combined_clearances':gaps,'best':best,'threshold':threshold,
        'mode':'emergency' if emergency else 'safe',
        'candidates':[a for a in allowed if gaps[a]>=threshold]}
