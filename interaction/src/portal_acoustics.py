"""SIMULATOR ONLY: interior walls and explicit, approximate doorway diffraction.

Non-convex image-source reflections reuse pyroomacoustics. Blocked direct sound
uses a visibility path through door portals as an equivalent secondary source.
This is an engineering approximation, not a validated wave/diffraction solver.
No source/portal/path coordinates are published to perception or navigation.
"""
from pathlib import Path
import numpy as np
import pyroomacoustics as pra
import yaml
from scipy.signal import butter, sosfilt
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from acoustics import MICROPHONES


class ApartmentAcoustics:
    def __init__(self, scene='/workspace/g1/scenes/apartment.yaml'):
        self.scene = yaml.safe_load(Path(scene).read_text())
        thickness = self.scene['wall']['thickness']
        solids = []
        for start, end in self.scene['walls']:
            wall = LineString([start,end]).buffer(thickness/2, cap_style=2)
            for door in self.scene['doorways']:
                c = np.array(door['centre'],float)
                if LineString([start,end]).distance(__import__('shapely').geometry.Point(c)) < .01:
                    direction = np.array(end)-np.array(start)
                    direction = direction/np.linalg.norm(direction)
                    gap = LineString([c-direction*door['width']/2,
                                      c+direction*door['width']/2]).buffer(thickness,cap_style=2)
                    wall = wall.difference(gap)
            solids.append(wall)
        self.solids = unary_union(solids)
        low,high = self.scene['outline']
        # GEOS may retain near-identical doorway-intersection vertices; the
        # acoustic engine uses float32 and needs nonzero wall lengths.
        self.free = box(*low,*high).difference(self.solids).simplify(1e-5,preserve_topology=True)
        if self.free.geom_type != 'Polygon' or self.free.interiors:
            raise ValueError('Acoustic baseline requires a connected polygon without holes')
        self.portals = []
        for door in self.scene['doorways']:
            centre = np.array(door['centre'],float)
            self.portals.append(centre)
            # Both sides let the visibility graph follow a corridor through a doorway.
            for delta in ([.35,0],[-.35,0],[0,.35],[0,-.35]):
                p = centre+delta
                if self.free.contains(__import__('shapely').geometry.Point(p)):
                    self.portals.append(p)

    def path(self, source, receiver):
        points = np.array([source[:2],receiver[:2],*self.portals])
        n = len(points)
        graph = np.full((n,n),np.inf)
        np.fill_diagonal(graph,0)
        for i in range(n):
            for j in range(i):
                line = LineString([points[i],points[j]])
                if not line.buffer(.002).intersects(self.solids):
                    graph[i,j] = graph[j,i] = np.linalg.norm(points[i]-points[j])
        distances,previous = shortest_path(graph,directed=False,indices=0,return_predecessors=True)
        if not np.isfinite(distances[1]):
            raise ValueError('No acoustic portal path')
        indices = [1]
        while indices[-1] != 0:
            indices.append(int(previous[indices[-1]]))
        return points[indices[::-1]],float(distances[1])

    def render(self, speech, fs, source, centre, yaw=0., seed=0):
        source,centre = np.asarray(source,float),np.asarray(centre,float)
        route,length = self.path(source,centre)
        blocked = len(route)>2
        secondary = source.copy()
        if blocked:
            secondary[:2] = route[-2]
            secondary[:2] += .015*(centre[:2]-secondary[:2])/max(np.linalg.norm(centre[:2]-secondary[:2]),1e-6)
        c,s = np.cos(yaw),np.sin(yaw)
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        microphones = MICROPHONES@rotation.T+centre
        room = pra.Room.from_corners(np.asarray(self.free.exterior.coords[:-1]).T,
            fs=fs,max_order=2,materials=pra.Material(.55))
        room.extrude(self.scene['wall']['height'],materials=pra.Material(.55))
        room.add_microphone_array(microphones.T)
        signal = np.asarray(speech,float)
        if blocked:
            signal = sosfilt(butter(2,2500,fs=fs,output='sos'),signal)
        room.add_source(secondary,signal=signal)
        room.simulate()
        audio = room.mic_array.signals.T
        if blocked:
            extra = max(0,length-np.linalg.norm(centre[:2]-secondary[:2]))
            audio = np.pad(audio,((round(extra/343*fs),0),(0,0)))
        audio /= max(np.max(np.abs(audio)),1e-9)
        rng = np.random.default_rng(seed)
        audio += rng.normal(0,max(np.sqrt(np.mean(audio**2)),1e-5)/10**(20/20),audio.shape)
        return audio.astype(np.float32),{'direct_blocked':blocked,'portal_count':max(0,len(route)-2),
            'propagation':'nonconvex ISM + approximate portal secondary sources',
            'path_length_m':length}
