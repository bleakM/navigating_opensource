#!/usr/bin/env python3
"""Separate simulation/frontend roles: actor poses never enter audio perception."""
import argparse,threading,time,queue
import numpy as np,rclpy,sherpa_onnx
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from g1_interaction_msgs.msg import MicrophoneAudio
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from common import DATA,ROOT,read_json,write_json
from identity import Speaker,match
from room_audio import WorkbenchAcoustics,reliable_direction
from audio_pipeline import write_monitor,REQUEST
from wake_word import WakeWord


class DoubleWake(WakeWord):
    def __init__(self):
        folder=ROOT/'models/kws-zh'
        self.spotter=sherpa_onnx.KeywordSpotter(tokens=str(folder/'tokens.txt'),
            encoder=str(next(folder.glob('encoder*.onnx'))),decoder=str(next(folder.glob('decoder*.onnx'))),joiner=str(next(folder.glob('joiner*.onnx'))),
            keywords_file=str(ROOT/'core/config/keywords.txt'),num_threads=2,keywords_score=1.,keywords_threshold=.25)


class AcousticNode(Node):
    def __init__(self):
        super().__init__('workbench_audio_simulator',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.base=None;self.busy=False
        resource=read_json(DATA/'resource.json')
        self.room=WorkbenchAcoustics(resource['acoustic'])
        self.pub=self.create_publisher(MicrophoneAudio,'/interaction/microphones',3)
        self.create_subscription(Odometry,'/g1_sensor_relay/base_state',lambda m:setattr(self,'base',m),qos_profile_sensor_data)
        self.create_timer(.12,self.poll)

    def poll(self):
        files=sorted((DATA/'audio-inbox').glob('*.npz'))
        if not files or self.base is None:return
        file=files[0];request_id='';monitor=False
        try:
            with np.load(file,allow_pickle=False) as packet:
                audio=packet['audio'];slot=int(packet['slot'])
                request_id=str(packet['request_id']) if 'request_id' in packet else ''
                monitor=bool(packet['monitor']) if 'monitor' in packet else False
            file.unlink()
            state=read_json(DATA/'actors.json',{});actor=next(a for a in state['actors'] if a['slot']==slot)
            base=self.base;p=base.pose.pose.position;q=base.pose.pose.orientation
            yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            # Head-height source belongs to the generator only. No sender label is published.
            source=[actor['x'],actor['y'],actor.get('height',1.6)]
            wave,meta=self.room.render(audio,16000,source,[p.x,p.y,1.35],yaw=yaw,seed=int(time.time())%65536)
            msg=MicrophoneAudio();msg.header.stamp=base.header.stamp;msg.header.frame_id='microphone_array';msg.sample_rate=16000;msg.channel_count=4;msg.samples=wave.reshape(-1).tolist();self.pub.publish(msg)
            stamp=base.header.stamp.sec+base.header.stamp.nanosec/1e9
            record={'generated_at':time.time(),'duration_s':len(audio)/16000,'blockwise_pose':True,'stamp':stamp,**meta}
            write_json(DATA/'acoustic-last.json',record)
            if monitor:write_monitor(DATA,request_id,wave,record)
        except Exception as exc:
            file.unlink(missing_ok=True)
            if monitor and REQUEST.fullmatch(request_id):
                write_json(DATA/'audio-monitor'/(request_id+'.json'),{'status':'failed','message':str(exc),'created_at':time.time()})
            self.get_logger().error(str(exc),throttle_duration_sec=3)


class Frontend(Node):
    def __init__(self):
        super().__init__('workbench_audio_perception',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.speaker=Speaker();self.wake=DoubleWake();self.queue=queue.Queue(maxsize=1);self.stop=threading.Event()
        self.pub=self.create_publisher(String,'/wb/call',10)
        self.create_subscription(MicrophoneAudio,'/interaction/microphones',self.receive,2)
        self.worker=threading.Thread(target=self.process,daemon=True);self.worker.start()

    def receive(self,msg):
        try:self.queue.put_nowait(msg)
        except queue.Full:pass

    def diagnostic(self,reason,message,**extra):
        write_json(DATA/'audio-perception.json',{'at':time.time(),'reason':reason,'message':message,**extra})

    def process(self):
        while not self.stop.is_set():
            try:msg=self.queue.get(timeout=.2)
            except queue.Empty:continue
            try:
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                if not 0<=self.get_clock().now().nanoseconds/1e9-stamp<=3 or msg.channel_count!=4:
                    self.diagnostic('stale_audio','机器人收到的音频已过期或通道无效，请稍后重新呼叫');continue
                audio=np.asarray(msg.samples,np.float32).reshape(-1,4)
                keywords=[]
                for channel in range(4):
                    keywords+=self.wake.detect(audio[:,channel],16000)
                    if '小北小北' in keywords:break
                if '小北小北' not in keywords:
                    self.diagnostic('keyword_not_detected','入口声纹已通过；机器人未识别到完整“小北小北”，请连贯呼叫后停顿');continue
                gallery=read_json(DATA/'voice-gallery.json',{})
                who,score,scores=match(self.speaker.embed(audio[:,0]),gallery,threshold=.55,margin=.07)
                if who is None:
                    self.diagnostic('propagated_speaker_rejected','已听到“小北小北”；传播后声纹未通过，请再次呼叫或缩短距离',best_score=score);continue
                try:direction=reliable_direction(audio,16000)
                except ValueError as exc:
                    self.diagnostic('direction_unreliable',str(exc));continue
                row={'stamp':stamp,'identity':who,'speaker_score':score,'speaker_scores':scores,'keyword':'小北小北',**direction}
                row['age_at_estimate_s']=self.get_clock().now().nanoseconds/1e9-stamp
                row['published']=row['age_at_estimate_s']<=2
                if row['published']:
                    self.pub.publish(String(data=__import__('json').dumps(row)))
                    self.diagnostic('call_published','已听到“小北小北”并识别人声，呼叫已交给小北')
                else:self.diagnostic('inference_too_slow','声音识别完成但已过期，请稍后重新呼叫')
                write_json(DATA/'heard-last.json',row)
            except Exception as exc:
                self.diagnostic('audio_error','机器人语音处理失败：'+str(exc))
                self.get_logger().warning(str(exc),throttle_duration_sec=5)


def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['simulate','perceive']);args=p.parse_args();rclpy.init()
    node=AcousticNode() if args.role=='simulate' else Frontend()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if hasattr(node,'stop'):node.stop.set();node.worker.join(timeout=3)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
