#!/usr/bin/env python3
"""Only enable calls after saved-map registration AND Nav2 lifecycle activation."""
import time,rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile,DurabilityPolicy
from common import DATA,write_json
def main():
 rclpy.init();node=Node('workbench_ready_gate',parameter_overrides=[Parameter('use_sim_time',value=True)])
 clients=[node.create_client(GetState,'/'+name+'/get_state') for name in ['planner_server','controller_server','smoother_server','velocity_smoother','bt_navigator','collision_monitor','behavior_server','navigation_map/costmap']]
 observed={}
 node.create_subscription(OccupancyGrid,'/navigation_map/costmap',
  lambda msg:observed.update(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
   valid=msg.header.frame_id=='map' and msg.info.width>0 and msg.info.height>0 and len(msg.data)==msg.info.width*msg.info.height),
  QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
 end=time.monotonic()+120
 try:
  while rclpy.ok() and time.monotonic()<end:
   okay=all(c.service_is_ready() for c in clients)
   if okay:
    for client in clients:
     future=client.call_async(GetState.Request());deadline=time.monotonic()+3
     while not future.done() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
     okay &= future.done() and future.result() is not None and future.result().current_state.id==3
    age=node.get_clock().now().nanoseconds/1e9-observed.get('stamp',-1e9)
    if okay and observed.get('valid') and 0<=age<2.:
     write_json(DATA/'ready.json',dict(ready=True,phase='ready',source='actual RTAB-Map registration + Nav2 and navigation-map lifecycle active + fresh observed navigation map'));return
   rclpy.spin_once(node,timeout_sec=.25)
  write_json(DATA/'ready.json',dict(ready=False,phase='failed',error='Nav2 or fresh navigation-map activation timed out'))
 except KeyboardInterrupt:pass
 finally:node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
