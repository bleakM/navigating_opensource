"""Rank existing safe endpoints using recently measured faces as view hints.

This is an endpoint preference, not a collision model or an identity detector.
Nav2 still checks the full route and the mission still requires fresh faces.
"""
import numpy as np
from vision_geometry import safe_approaches


def ranked_face_approaches(person,robot,grid,resolution,origin,other_faces,max_standoff=None):
    candidates=list(safe_approaches(person,robot,grid,resolution,origin))
    person=np.asarray(person,float);robot=np.asarray(robot,float)
    if max_standoff is not None:
        candidates=[g for g in candidates if np.linalg.norm(np.asarray(g[:2])-person)<=max_standoff]
    others=[np.asarray(p,float) for p in other_faces]
    def cost(goal):
        point=np.asarray(goal[:2]);view=person-point
        length=float(np.linalg.norm(view));penalty=0.
        for other in others:
            # Prefer not to finish immediately beside another observed person.
            proximity=float(np.linalg.norm(point-other))
            penalty+=4.*max(0.,.7-proximity)
            along=float(np.dot(other-point,view)/max(length,1e-6))
            across=abs(float(np.cross(view,other-point)))/max(length,1e-6)
            if 0.<along<length:
                # A head/body may occlude the target's face along this view.
                penalty+=3.*max(0.,1.-across/.45)
        return float(np.linalg.norm(point-robot))+penalty
    return sorted(candidates,key=cost)
