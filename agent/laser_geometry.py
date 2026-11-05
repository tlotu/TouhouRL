"""Offline static-laser clearance hypothesis; not a native collision model."""
import math

from env.action_emulator import DIRECTIONS


def point_segment_distance(px, py, ax, ay, bx, by):
    dx,dy=bx-ax,by-ay
    length=dx*dx+dy*dy
    t=max(0.,min(1.,((px-ax)*dx+(py-ay)*dy)/length)) if length else 0.
    return math.hypot(px-(ax+t*dx),py-(ay+t*dy))


def candidate_active(laser, frame):
    """Reference-based hypothesis, not independently verified damage state."""
    phase,timer=laser['phase_raw'],laser['timer_raw']
    if phase == 0:
        return frame >= max(0,laser['warning_frames_raw']-timer) and frame < max(0,laser['warning_frames_raw']-timer)+laser['firing_frames_raw']
    if phase == 1:
        return frame < max(0,laser['firing_frames_raw']-timer)
    if phase == 2:
        return False
    raise ValueError('Unsupported laser phase')


def forecast_static_lasers(player, lasers, horizon=12, player_radius=1.25):
    """Integer-frame capsule clearance, capped at 24; static axes assumed."""
    if not 0 < horizon <= 60 or player_radius < 0:
        raise ValueError('Invalid forecast settings')
    prepared=[]
    for laser in lasers:
        if laser['issues'] or laser['speed'] != 0 or laser['segment_candidate'] is None:
            raise ValueError('Invalid or moving laser')
        if laser['phase_raw'] not in (0,1,2) or laser['timer_raw'] < 0 or min(laser['warning_frames_raw'],laser['firing_frames_raw'],laser['fading_frames_raw']) < 0:
            raise ValueError('Invalid laser phase or timing')
        (ax,ay),(bx,by)=laser['segment_candidate']
        if not all(math.isfinite(v) for v in (ax,ay,bx,by,laser['width'])) or laser['width'] < 0:
            raise ValueError('Invalid laser geometry')
        prepared.append((laser,ax,ay,bx,by))
    results=[]
    for keys in DIRECTIONS:
        dx=int('right' in keys)-int('left' in keys)
        dy=int('down' in keys)-int('up' in keys)
        scale=2/max(1.,math.hypot(dx,dy))
        gap=24.
        for frame in range(horizon+1):
            px,py=player[0]+frame*dx*scale,player[1]+frame*dy*scale
            for laser,ax,ay,bx,by in prepared:
                if candidate_active(laser,frame):
                    gap=min(gap,point_segment_distance(px,py,ax,ay,bx,by)-laser['width']/2-player_radius)
        results.append(gap)
    return results
