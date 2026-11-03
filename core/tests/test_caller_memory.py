"""Memory is sensor-only, scoped per map, decays, and never invents range."""
import math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from caller_memory import CallerMemory,arrival_portals


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.path=Path(self.folder.name)/'memory.json'
        self.memory=CallerMemory(self.path,'map-a')
        self.rooms=[{'id':'east','centroid':[5,2],'area':16,'outline':[[3,0],[7,0],[7,4],[3,4]],'doorways':[[3,1]]},
                    {'id':'north','centroid':[1,6],'area':16,'outline':[[-1,4],[3,4],[3,8],[-1,8]],'doorways':[[1,4]]}]
        self.call={'origin':[0,0],'map_azimuth':0.,'stamp':10.}
    def tearDown(self):self.folder.cleanup()

    def test_sound_only_is_sector_and_room_regions_no_fake_point(self):
        self.memory.observe_audio('person',self.call,now=100)
        p=self.memory.predict('person',self.call,self.rooms,now=101)
        self.assertIsNone(p['visual']);self.assertFalse(p['audio']['range_estimated'])
        self.assertNotIn('position',p['audio']);self.assertEqual(p['room_candidates'][0]['id'],'east')
        self.assertEqual(p['room_candidates'][0]['outline'],self.rooms[0]['outline'])
        self.assertFalse(p['scores_are_probabilities'])

    def test_seen_face_persists_for_same_map_but_other_map_is_empty(self):
        self.memory.observe_face('person',[5,2],4.,now=100)
        self.assertEqual(CallerMemory(self.path,'map-a').visual_prior('person',now=101)['position'],[5.,2.])
        self.assertIsNone(CallerMemory(self.path,'map-b').visual_prior('person',now=101))

    def test_aging_increases_search_radius_then_expires(self):
        self.memory.observe_face('person',[5,2],4.,now=100)
        young=self.memory.visual_prior('person',now=101);old=self.memory.visual_prior('person',now=150)
        self.assertGreater(old['search_radius_m'],young['search_radius_m'])
        self.assertLess(old['weight'],young['weight']);self.assertIsNone(self.memory.visual_prior('person',now=221))

    def test_movement_replaces_old_position_and_invalid_data_is_rejected(self):
        self.memory.observe_face('person',[5,2],4.,now=100)
        self.memory.observe_face('person',[1,6],5.,now=101)
        self.assertEqual(self.memory.visual_prior('person',now=102)['position'],[1.,6.])
        self.assertFalse(self.memory.observe_face('person',[float('nan'),0],6.,now=102))
        self.assertFalse(self.memory.observe_audio('person',{**self.call,'map_azimuth':float('nan')},now=102))

    def test_acoustic_turn_uses_measured_room_doorway_not_truth(self):
        call={**self.call,'map_azimuth':math.atan2(4,1)}
        p=self.memory.predict('unknown',call,self.rooms,now=100)
        self.assertEqual(p['room_candidates'][0]['id'],'north')
        self.assertLess(p['room_candidates'][0]['audio_angle_error_deg'],.001)

    def test_other_rooms_keep_positive_weight_even_behind_array(self):
        rooms=self.rooms+[{'id':'west','centroid':[-5,0],'area':16,'outline':[],'doorways':[]}]
        p=self.memory.predict('person',self.call,rooms,now=100)
        weights={r['id']:r['search_weight'] for r in p['room_candidates']}
        self.assertAlmostEqual(sum(weights.values()),1.)
        self.assertGreater(weights['west'],.1)
        self.assertLess(weights['east'],.8)

    def test_repeated_audio_updates_all_regions_and_deduplicates(self):
        self.assertTrue(self.memory.observe_audio('person',self.call,now=100))
        self.assertFalse(self.memory.observe_audio('person',self.call,now=101))
        second={**self.call,'map_azimuth':math.pi/2,'stamp':12.}
        self.memory.observe_audio('person',second,now=102)
        p=self.memory.predict('person',second,self.rooms,now=103)
        self.assertEqual(p['audio_observations'],2)
        for r in p['room_candidates']:self.assertGreater(r['search_weight'],.2)

    def test_not_seen_weakens_priority_without_declaring_room_empty(self):
        before=self.memory.predict('person',self.call,self.rooms,now=100)
        after=self.memory.predict('person',self.call,self.rooms,now=101,scans={'east':{'count':2,'wall':100}})
        a={r['id']:r['search_weight'] for r in before['room_candidates']}
        b={r['id']:r['search_weight'] for r in after['room_candidates']}
        self.assertLess(b['east'],a['east']);self.assertGreater(b['east'],0)
        self.assertGreater(b['north'],a['north'])

    def test_new_sim_clock_keeps_visual_memory_but_accepts_new_epoch(self):
        self.memory.observe_face('person',[5,2],1000.,now=100)
        other=CallerMemory(self.path,'map-a')
        self.assertTrue(other.observe_face('person',[1,6],2.,now=101))
        self.assertFalse(other.observe_face('person',[5,2],1.,now=102))

    def test_distant_room_uses_receiver_side_opening_along_observed_graph(self):
        rooms=[{'id':'a','centroid':[0,0],'outline':[[-2,-2],[2,-2],[2,2],[-2,2]],
            'doorways':[{'centre':[2,0],'to_room':'b'}]},
            {'id':'b','centroid':[4,0],'doorways':[{'centre':[2,0],'to_room':'a'},{'centre':[6,0],'to_room':'c'}]},
            {'id':'c','centroid':[8,0],'doorways':[{'centre':[6,0],'to_room':'b'}]}]
        self.assertEqual(arrival_portals([0,0],rooms),{'b':[2,0],'c':[2,0]})

    def test_spectral_bearing_disagreement_preserves_alternative_region(self):
        call={**self.call,'map_hypotheses':[{'map_azimuth':0.,'weight':.6},
            {'map_azimuth':math.pi/2,'weight':.4}]}
        a=self.memory.predict('x',self.call,self.rooms,now=100)
        self.memory.observe_audio('x',call,now=100)
        b=self.memory.predict('x',call,self.rooms,now=101)
        weight=lambda p:next(r['search_weight'] for r in p['room_candidates'] if r['id']=='north')
        self.assertGreater(weight(b),weight(a))


if __name__=='__main__':unittest.main()
