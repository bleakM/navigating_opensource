"""Per-map caller search priors from face observations and measured audio only.

Radii and room scores are search heuristics, not calibrated error bounds or
posterior probabilities. A single array bearing never supplies source range.
"""
import math,time,heapq
import numpy as np
from shapely.geometry import Point,Polygon
from common import read_json,write_json


def angle_error(point,call):
    x,y=np.asarray(point,float)-call['origin']
    delta=math.atan2(y,x)-call['map_azimuth']
    return abs(math.atan2(math.sin(delta),math.cos(delta)))


def bearing_hypotheses(call):
    """Transform spectral and secondary SRP peaks at the audio capture pose.

    Peaks of one recording are alternatives, never independent observations.
    Keep a bounded share for them, including the back of a reverberant array.
    """
    rotation=call['map_azimuth']-call['array_azimuth']
    bands=[{'map_azimuth':call['map_azimuth'],'weight':.6,'band':'full'}]
    for candidate in call.get('frequency_candidates',[]):
        if candidate.get('peak_contrast',0)<.08:continue
        angle=float(candidate['azimuth_rad'])+rotation
        if not math.isfinite(angle):continue
        bands.append({'map_azimuth':math.atan2(math.sin(angle),math.cos(angle)),
                      'weight':.12,'band':candidate.get('band_hz','spectral')})
    for candidate in call.get('angular_candidates',[])[:3]:
        if candidate.get('relative_score',0)<.6:continue
        angle=float(candidate['azimuth_rad'])+rotation
        if not math.isfinite(angle):continue
        if any(abs(math.remainder(angle-b['map_azimuth'],2*math.pi))<math.radians(15) for b in bands):continue
        bands.append({'map_azimuth':math.atan2(math.sin(angle),math.cos(angle)),
                      'weight':.08,'band':'secondary SRP peak'})
    return bands


def arrival_portals(origin,rooms):
    """First openings along observed-room graph, not the source room's door.

    Used as one possible arrival direction alongside direct wall transmission.
    Graph and containment come exclusively from Canopy's observed SLAM map.
    """
    table={r['id']:r for r in rooms};root=None
    for r in rooms:
        if len(r.get('outline',[]))>=3 and Polygon(r['outline']).buffer(.1).covers(Point(origin)):
            root=r['id'];break
    if root is None:return {}
    distance={root:0.};first={};queue=[(0.,root)]
    while queue:
        cost,room_id=heapq.heappop(queue)
        if cost>distance[room_id]:continue
        for door in table[room_id].get('doorways',[]):
            if not isinstance(door,dict) or door.get('to_room') not in table:continue
            target=door['to_room'];point=door['centre']
            start=origin if room_id==root else table[room_id]['centroid']
            step=float(np.linalg.norm(np.asarray(point)-start))+float(np.linalg.norm(np.asarray(point)-table[target]['centroid']))
            if cost+step>=distance.get(target,float('inf')):continue
            distance[target]=cost+step;first[target]=point if room_id==root else first[room_id]
            heapq.heappush(queue,(cost+step,target))
    return first


class CallerMemory:
    def __init__(self,path,scope):
        self.path=path;self.scope=scope
        saved=read_json(path,{})
        self.people=saved.get('people',{}) if saved.get('scope')==scope else {}
        for row in self.people.values():row.pop('audio_history',None)
        self.face_stamps={};self.body_stamps={};self.last_save=0.

    def observe_face(self,identity,position,stamp,now=None):
        now=time.time() if now is None else now
        p=np.asarray(position,float)
        if p.shape!=(2,) or not np.isfinite(p).all() or not math.isfinite(stamp):return False
        row=self.people.setdefault(identity,{})
        old=row.get('visual',{})
        if now<=old.get('seen_wall',-1) or stamp<=self.face_stamps.get(identity,-1):return False
        self.face_stamps[identity]=stamp
        # Never average a moved person with their old location.
        row['visual']={'position':p.tolist(),'stamp':float(stamp),'seen_wall':now,
            'source':'camera face ROI + aligned depth + measurement-time map TF'}
        self.save(now);return True

    def observe_body(self,identity,position,stamp,now=None):
        now=time.time() if now is None else now
        p=np.asarray(position,float)
        if p.shape!=(2,) or not np.isfinite(p).all() or not math.isfinite(stamp):return False
        row=self.people.setdefault(identity,{})
        if now<=row.get('body',{}).get('seen_wall',-1) or stamp<=self.body_stamps.get(identity,-1):return False
        self.body_stamps[identity]=stamp
        row['body']={'position':p.tolist(),'stamp':float(stamp),'seen_wall':now,
                     'source':'multi-frame OSNet body appearance + RGB-D + measurement-time map TF; hint only'}
        self.save(now);return True

    def observe_audio(self,identity,call,now=None):
        now=time.time() if now is None else now
        origin=np.asarray(call.get('origin'),float);bearing=float(call['map_azimuth'])
        if origin.shape!=(2,) or not np.isfinite(origin).all() or not math.isfinite(bearing):return False
        row=self.people.setdefault(identity,{})
        history=[a for a in row.get('audio_history',[]) if 0<=now-a['seen_wall']<=120]
        if history and call['stamp']<=history[-1]['stamp']:return False
        audio={'origin':origin.tolist(),'map_azimuth':bearing,
            'stamp':call['stamp'],'seen_wall':now,'range_estimated':False,
            'half_angle_deg':max(20.,min(70.,float(call.get('half_angle_deg',30.)))),
            'map_hypotheses':call.get('map_hypotheses',[]),
            'source':'SRP-PHAT waveform + measurement-time map TF; doorway arrival may differ from caller bearing'}
        row['audio']=audio;row['audio_history']=(history+[audio])[-6:]
        self.save(now,force=True);return True

    def visual_prior(self,identity,now=None):
        now=time.time() if now is None else now
        person=self.people.get(identity,{})
        row=person.get('visual');body=person.get('body')
        # A fresh confirmed face outranks body appearance; body observations
        # can replace an older hint after a person actually moves.
        use_body=bool(body and (not row or (now-row['seen_wall']>5 and body['seen_wall']>row['seen_wall'])))
        if use_body:row=body
        if not row:return None
        age=now-row['seen_wall']
        if not 0<=age<=(45 if use_body else 120):return None
        return {**row,'age_s':age,'identity_evidence':'body_hint' if use_body else 'face',
            'search_radius_m':min(8.,(1. if use_body else .6)+.15*age),
            'weight':(.55 if use_body else 1.)*math.exp(-age/(20. if use_body else 45.)),
            'uncertainty':'heuristic search radius, not guaranteed position accuracy'}

    def predict(self,identity,call,rooms,now=None,scans=None,robot=None):
        """Only Canopy's observed-map outlines/doors are eligible room regions."""
        now=time.time() if now is None else now
        prior=self.visual_prior(identity,now);candidates=[];scans=scans or {}
        history=[a for a in self.people.get(identity,{}).get('audio_history',[]) if 0<=now-a['seen_wall']<=120]
        if not history:history=[{**call,'seen_wall':now}]
        # Bounded evidence averaging avoids treating nearly identical repeated
        # bearings as independent triangulation or forcing all mass into a room.
        for room in rooms:
            centre=np.asarray(room['centroid'],float)
            own_doors=[np.asarray(p['centre'] if isinstance(p,dict) else p,float) for p in room.get('doorways',[])]
            directions=[centre]+own_doors
            error=min(angle_error(p,call) for p in directions)
            support=[];weights=[]
            for audio in history:
                portals=arrival_portals(audio['origin'],rooms)
                portal=portals.get(room['id'])
                # A room covers an angular interval, not just its centroid.
                direct_points=[centre]+[np.asarray(p,float) for p in room.get('outline',[])][::max(1,len(room.get('outline',[]))//12)]
                sigma=math.radians(max(35.,audio.get('half_angle_deg',30.)))
                bands=audio.get('map_hypotheses') or [{'map_azimuth':audio['map_azimuth'],'weight':1.}]
                band_support=[]
                for band in bands:
                    bearing={**audio,'map_azimuth':band['map_azimuth']}
                    direct_error=min(angle_error(p,bearing) for p in direct_points)
                    portal_error=angle_error(portal,bearing) if portal is not None else min(angle_error(p,bearing) for p in directions)
                    band_support.append(.55*math.exp(-.5*(direct_error/sigma)**2)+.45*math.exp(-.5*(portal_error/sigma)**2))
                support.append(float(np.average(band_support,weights=[b['weight'] for b in bands])))
                weights.append(math.exp(-(now-audio['seen_wall'])/40.))
            audio_support=float(np.average(support,weights=weights))
            score=.25+.75*audio_support
            bonus=0.
            if prior:
                outline=room.get('outline',[])
                distance=float(Polygon(outline).distance(Point(prior['position']))) if len(outline)>=3 else float(np.linalg.norm(centre-prior['position']))
                scale=prior['search_radius_m']
                bonus=prior['weight']*math.exp(-.5*(distance/max(scale,.5))**2)
                score+=.65*bonus
            scan=scans.get(room['id'],{})
            # A camera scan only weakens this viewpoint's evidence. It never
            # establishes that an entire room (occluded furniture etc.) is empty.
            misses=scan.get('count',0)
            freshness=math.exp(-max(0,now-scan.get('wall',now))/90.)
            score*=max(.35,1-.35*misses*freshness)
            travel=float(np.linalg.norm(centre-np.asarray(robot,float))) if robot is not None else 0.
            candidates.append({'id':room['id'],'score':score,'audio_angle_error_deg':math.degrees(error),
                'audio_support':audio_support,'audio_observations':len(history),
                'visual_support':bonus,'viewpoint_misses':misses,'travel_distance_m':travel,
                'centroid':room['centroid'],'doorways':room.get('doorways',[]),
                'outline':room.get('outline',[]),'area_m2':room.get('area',0)})
        total=sum(r['score'] for r in candidates)
        for row in candidates:
            row['search_weight']=.15/max(len(candidates),1)+.85*row['score']/max(total,1e-9)
            row['priority']=row['search_weight']/(1+.04*row['travel_distance_m'])
        candidates.sort(key=lambda r:r['score'],reverse=True)
        eligible=[r for r in candidates if r['area_m2']>=3]
        # The controller already observes its current region before leaving.
        # A large corridor surrounding the receiver should not consume one of
        # the three remote room slots simply because it spans many bearings.
        # Keep the local hypothesis explicit: it has not been ruled out.
        local=[r for r in eligible if len(r['outline'])>=3 and
               Polygon(r['outline']).buffer(.1).covers(Point(call['origin']))]
        local_ids={r['id'] for r in local}
        remote=[r for r in eligible if r['id'] not in local_ids]
        primary=remote[:3] if remote else local[:3]
        reserve=remote[3:]
        ambiguous=len(remote)>3 and remote[3]['score']>=remote[0]['score']*.85
        return {'identity':identity,'frame':'map','visual':prior,
            'audio':{'origin':list(call['origin']),'bearing_rad':call['map_azimuth'],
                'half_angle_deg':call.get('half_angle_deg',30.),'range_estimated':False},
            'room_candidates':candidates,'scores_are_probabilities':False,
            'primary_room_ids':[r['id'] for r in primary],
            'local_region_ids':[r['id'] for r in local],
            'local_region_policy':'inspect current region first; not excluded as a caller location',
            'reserve_room_ids':[r['id'] for r in reserve],
            'candidate_limit':3,'acoustically_ambiguous':ambiguous,
            'primary_is_guaranteed':False,
            'audio_observations':len(history),
            'arrival_hypotheses':'direct / wall-transmitted bearing + first connected opening from receiver room; observed map only',
            'limitation':'Search weights are uncalibrated heuristics. Multiple rooms may be acoustically indistinguishable; no acoustic range estimate.'}

    def save(self,now=None,force=False):
        now=time.time() if now is None else now
        if not force and now-self.last_save<.5:return
        self.last_save=now
        write_json(self.path,{'scope':self.scope,'people':self.people,'saved_wall':now})
