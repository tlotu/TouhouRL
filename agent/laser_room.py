"""Offline grid component diagnostics for a static laser partition."""
from collections import deque
import math
import numpy as np


def partition_room(player,lasers,step=4.,radius=1.25):
    """Approximate component with all visible warning/firing beams as walls.

    This ignores arrival deadlines, bullets, moving beams and future births.
    A component is not necessarily reachable before the warning activates.
    """
    if not 0<step<=16 or radius<0 or not all(math.isfinite(v)for v in (*player,step,radius)):
        raise ValueError('Invalid room grid settings')
    xs=np.arange(8+step/2,376,step);ys=np.arange(16+step/2,432,step)
    xx,yy=np.meshgrid(xs,ys);free=np.ones(xx.shape,dtype=bool)
    for laser in lasers:
        if laser['issues'] or laser['speed']!=0 or laser['phase_raw']not in (0,1,2) or laser['segment_candidate'] is None:
            raise ValueError('Unsupported laser geometry')
        (ax,ay),(bx,by)=laser['segment_candidate'];dx,dy=bx-ax,by-ay;length=dx*dx+dy*dy
        if laser['width']<0 or not all(math.isfinite(v)for v in (ax,ay,bx,by,laser['width'])):
            raise ValueError('Invalid laser geometry')
        if laser['phase_raw']==2:continue
        t=np.clip(((xx-ax)*dx+(yy-ay)*dy)/length,0,1)if length else np.zeros_like(xx)
        distance=np.hypot(xx-ax-t*dx,yy-ay-t*dy)
        free &= distance>laser['width']/2+radius
    if not 8<=player[0]<=376 or not 16<=player[1]<=432:
        raise ValueError('Player outside grid field')
    ix=int(np.argmin(abs(xs-player[0])));iy=int(np.argmin(abs(ys-player[1])))
    seed_free=bool(free[iy,ix]);seen=set()
    if seed_free:
        seen.add((iy,ix));queue=deque([(iy,ix)])
        while queue:
            y,x=queue.popleft()
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<len(ys)and 0<=nx<len(xs)and free[ny,nx]and(ny,nx)not in seen:
                    seen.add((ny,nx));queue.append((ny,nx))
    return {'grid_step':step,'seed_free':seed_free,'component_cells':len(seen),
            'area_pixels2':len(seen)*step*step,'free_grid_cells':int(free.sum()),
            'model':'all-visible-warning-and-firing-static-walls-no-bullets'}
