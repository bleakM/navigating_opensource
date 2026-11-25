import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mapping_visits import unvisited_regions


class VisitTests(unittest.TestCase):
    def test_seen_through_door_is_not_physical_visit(self):
        rooms=[dict(id='room',area=20,outline=[[0,0],[5,0],[5,4],[0,4]])]
        self.assertEqual(len(unvisited_regions(rooms,[[-1,2],[.1,2],[.2,2]])),1)
        self.assertEqual(len(unvisited_regions(rooms,[[1,2],[1.3,2]])),1)
        self.assertEqual(unvisited_regions(rooms,[[1,2],[1.3,2],[1.6,2]]),[])

    def test_region_relabelling_uses_trajectory_not_cached_ids(self):
        rooms=[dict(id='new-id',area=20,outline=[[0,0],[5,0],[5,4],[0,4]])]
        self.assertEqual(unvisited_regions(rooms,[[1,2],[1.3,2],[1.6,2]]),[])


if __name__=='__main__':unittest.main()
