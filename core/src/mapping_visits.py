"""Visit each observed room once, without photographic surface quotas.

Canopy supplies regions and safe approach poses; Nav2 owns every movement.
Only map TF history is used. Scene geometry/actor truth is never an input.
"""
import math,time
from shapely.geometry import Point,Polygon
from common import DATA,write_json


def unvisited_regions(rooms,trajectory):
    missing=[]
    for room in rooms:
        if room['area']<3 or len(room['outline'])<3:continue
        interior=Polygon(room['outline']).buffer(0).buffer(-.3)
        if interior.is_empty:continue
        if sum(interior.covers(Point(p)) for p in trajectory)<3:missing.append(room)
    return missing


class MappingVisits:
    def __init__(self,node):
        from tf2_ros import Buffer,TransformListener
        from canopy_msgs.msg import RoomArray
        from canopy_msgs.srv import GetApproachPose
        from rtabmap_msgs.msg import MapGraph
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.qos import QoSProfile,DurabilityPolicy
        self.node=node;self.tf=Buffer();self.listener=TransformListener(self.tf,node)
        self.rooms=[];self.trajectory=[];self.graph=None;self.events=[]
        self.client=node.create_client(GetApproachPose,'/canopy/get_approach_pose')
        self.nav=ActionClient(node,NavigateToPose,'/navigate_to_pose')
        node.create_subscription(RoomArray,'/canopy/rooms',self.receive,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_subscription(MapGraph,'/rtabmap/mapGraph',self.receive_graph,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_timer(.5,self.sample)

    def receive(self,msg):
        if msg.header.frame_id!='map':return
        self.rooms=[dict(id=r.id,area=r.area,centroid=[r.centroid.x,r.centroid.y],
            outline=[[p.x,p.y] for p in r.outline.points]) for r in msg.rooms]

    def receive_graph(self,msg):
        if msg.header.frame_id=='map':self.graph=[[p.position.x,p.position.y] for p in msg.poses]

    def sample(self):
        from rclpy.time import Time
        from tf2_ros import TransformException
        try:
            t=self.tf.lookup_transform('map','base_footprint',Time())
            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
            if not 0<=self.node.get_clock().now().nanoseconds/1e9-stamp<1.:return
            p=t.transform.translation;self.trajectory.append([p.x,p.y])
        except TransformException:pass

    def wait(self,future,seconds):
        import rclpy
        end=time.monotonic()+seconds
        while rclpy.ok() and not future.done() and time.monotonic()<end:rclpy.spin_once(self.node,timeout_sec=.1)
        if not future.done():raise TimeoutError('Observed-room ROS operation timed out')
        return future.result()

    def finish(self):
        import rclpy
        from canopy_msgs.srv import GetApproachPose
        from nav2_msgs.action import NavigateToPose
        from action_msgs.msg import GoalStatus
        if not self.rooms:raise RuntimeError('No observed rooms available for final visit check')
        tried=set();unreachable={}
        while True:
            history=self.graph if self.graph is not None else self.trajectory
            missing=[r for r in unvisited_regions(self.rooms,history) if r['id'] not in unreachable]
            write_json(DATA/'mapping-room-visits.json',dict(source='Canopy observed regions + optimized SLAM graph / map TF; no truth',
                missing=[r['id'] for r in missing],unreachable_observed_regions=unreachable,
                events=self.events,samples=len(history)))
            if not missing:return
            robot=self.trajectory[-1] if self.trajectory else [0,0]
            room=min(missing,key=lambda r:math.dist(r['centroid'],robot))
            if room['id'] in tried:raise RuntimeError('Room approach did not enter observed region: '+room['id'])
            tried.add(room['id']);self.node.get_logger().info('One required visit to observed room '+room['id'])
            reply=self.wait(self.client.call_async(GetApproachPose.Request(target=room['id'])),15)
            if not reply.success:
                # Segmentation can label furniture pockets as regions. Canopy's
                # unchanged footprint/travel test decides reachability.
                # This is not an independent audit against the full scene.
                unreachable[room['id']]=reply.message
                continue
            handle=self.wait(self.nav.send_goal_async(NavigateToPose.Goal(pose=reply.pose)),10)
            if not handle.accepted:raise RuntimeError('Nav2 rejected required room visit')
            result_future=handle.get_result_async()
            try:result=self.wait(result_future,100)
            except TimeoutError:
                self.wait(handle.cancel_goal_async(),8);self.wait(result_future,8)
                raise RuntimeError('Required room visit timed out; cancelled safely')
            self.events.append(dict(room=room['id'],status=int(result.status),goal=[reply.pose.pose.position.x,reply.pose.pose.position.y]))
            if result.status!=GoalStatus.STATUS_SUCCEEDED:raise RuntimeError('Nav2 could not reach required observed room')
            end=time.monotonic()+2
            while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self.node,timeout_sec=.1)
