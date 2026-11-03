#!/usr/bin/env python3
"""Local command-line interface; no HTTP service or simulator identity shortcut."""
import argparse,fcntl,json,math,os,re,shutil,subprocess,time
from pathlib import Path
from common import ROOT,DATA,read_json,write_json


def identity(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise ValueError('Identity must be a short ASCII identifier')
    return value


def local_path(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def prepare(manifest,mode,database=None):
    config=json.loads(local_path(manifest).read_text())
    resource=dict(config['resource']);identity(resource['id'])
    for name in ['scene','acoustic']:
        resource[name]=str(local_path(resource[name]).resolve())
        if not Path(resource[name]).is_file():raise ValueError('Missing resource: '+resource[name])
    spawn=resource.get('spawn',[0.,0.])
    if len(spawn)!=2 or not all(math.isfinite(float(v)) for v in spawn):raise ValueError('Invalid initial spawn')
    actors=config.get('actors',[])
    if len(actors)>4:raise ValueError('Maximum four actors')
    ids=set()
    for slot,actor in enumerate(actors):
        name=identity(actor['id'])
        if name in ids:raise ValueError('Duplicate actor identity')
        ids.add(name);actor['slot']=slot
        if not all(math.isfinite(float(actor[k])) for k in ['x','y']):raise ValueError('Invalid actor position')
        for file in ['person-raised.obj','person.png']:
            if not (ROOT/'assets/avatars'/name/file).is_file():raise ValueError(f'Missing avatar: {name}/{file}')
        for gallery in ['voice-gallery.json','face-gallery.json']:
            if name not in read_json(DATA/gallery,{}):raise ValueError(f'Enroll {name} in {gallery} first')
    write_json(DATA/'actors.json',{'actors':actors,'locked_slot':-1})
    db=DATA/('localization.db' if mode=='saved' else 'mapping.db')
    saved=local_path(database or resource.get('saved','')) if mode=='saved' else None
    if saved and (not saved.is_file() or saved.resolve()==db.resolve()):raise ValueError('Choose an existing immutable saved database')
    for suffix in ['','-wal','-shm']:
        Path(str(db)+suffix).unlink(missing_ok=True)
    if saved:
        shutil.copyfile(saved,db);db.chmod(0o600);resource['saved']=str(saved)
    write_json(DATA/'resource.json',resource)
    for name in ['cancel-request','mission-state.json','mission.json','last-mission.json','heard-last.json','audio-perception.json','localization.json','exploration.json']:
        (DATA/name).unlink(missing_ok=True)
    (DATA/'audio-inbox').mkdir(exist_ok=True)
    for file in (DATA/'audio-inbox').glob('*.npz'):file.unlink()
    write_json(DATA/'ready.json',{'ready':False,'phase':'starting'})
    from scene import build_scene
    build_scene(actors,resource,DATA/'scene.xml')
    return db


def enroll(args):
    import cv2
    from identity import Faces,Speaker,audio_bytes
    name=identity(args.identity)
    image=cv2.imread(str(local_path(args.face)))
    if image is None:raise ValueError('Cannot read enrollment image')
    faces=Faces().detect(image)
    if len(faces)!=1:raise ValueError('Enrollment requires exactly one detected face')
    audio=audio_bytes(local_path(args.voice).read_bytes())
    from audio_ros import DoubleWake
    if '小北小北' not in DoubleWake().detect(audio,16000):raise ValueError('Enrollment voice must clearly contain 小北小北')
    voices=read_json(DATA/'voice-gallery.json',{});faces_gallery=read_json(DATA/'face-gallery.json',{})
    voices[name]=Speaker().profile(audio,voices.get(name));faces_gallery[name]=faces[0]['embedding']
    write_json(DATA/'voice-gallery.json',voices);write_json(DATA/'face-gallery.json',faces_gallery)
    from audio_pipeline import save_clip
    save_clip(DATA,name,audio)
    print(json.dumps({'enrolled':name,'face_only':True,'voice_quality':voices[name]['quality']},ensure_ascii=False))


def play(args):
    from identity import Speaker,match,audio_bytes
    from audio_pipeline import enqueue_audio,clip_path
    actors=read_json(DATA/'actors.json',{}).get('actors',[])
    actor=next((a for a in actors if a['slot']==args.slot),None)
    if actor is None:raise ValueError('No actor in this slot')
    audio=audio_bytes((local_path(args.wav) if args.wav else clip_path(DATA,actor['id'])).read_bytes())
    who,score,scores=match(Speaker().embed(audio),read_json(DATA/'voice-gallery.json',{}))
    if who!=actor['id']:raise ValueError('Speaker mismatch: '+str(who or 'unknown'))
    request=enqueue_audio(DATA,audio,args.slot,monitor=True)
    print(json.dumps({'request':request,'source_identity':who,'entry_score':score,'note':'Robot infers identity again from propagated four-channel audio; receiver channel 0 is saved locally.'}),flush=True)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        meta=read_json(DATA/'audio-monitor'/(request+'.json'),{})
        if meta.get('status')=='failed':raise ValueError(meta.get('message','Acoustic rendering failed'))
        if meta.get('status')=='ready':
            print(str(DATA/'audio-monitor'/(request+'.wav')));return
        time.sleep(.2)
    raise ValueError('No simulated receiver audio within 20 seconds; check simulation state')


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    run=subs.add_parser('run');run.add_argument('--manifest',default='examples/session.json');run.add_argument('--mode',choices=['fresh','saved'],default='fresh');run.add_argument('--database');run.add_argument('--initial-pose',default='')
    e=subs.add_parser('enroll');e.add_argument('--identity',required=True);e.add_argument('--face',required=True);e.add_argument('--voice',required=True)
    p=subs.add_parser('play');p.add_argument('--slot',type=int,required=True);p.add_argument('--wav')
    subs.add_parser('status');subs.add_parser('cancel')
    goal=subs.add_parser('goal');goal.add_argument('x',type=float);goal.add_argument('y',type=float)
    args=parser.parse_args()
    if args.command=='enroll':return enroll(args)
    if args.command=='play':return play(args)
    if args.command=='status':
        print(json.dumps({k:read_json(DATA/(k+'.json'),{}) for k in ['ready','mission-state','localization','audio-perception']},ensure_ascii=False,indent=2));return
    if args.command=='cancel':(DATA/'cancel-request').touch();print('Cancellation requested; poll status for acknowledgement.');return
    if args.command=='goal':
        if not read_json(DATA/'ready.json',{}).get('ready'):raise ValueError('Map registration / navigation not ready')
        state=read_json(DATA/'mission-state.json',{})
        if state.get('busy') or state.get('identity'):raise ValueError('Finish or cancel the current person-search task first')
        return subprocess.call(['python3','/workspace/scripts/goal.py',str(args.x),str(args.y),'--odom-topic','/g1_odometry_publisher/odom','--position-tolerance','.5','--output',str(DATA/'goal.json')])
    with (ROOT/'.runtime/core-run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=prepare(args.manifest,args.mode,args.database)
        command=['ros2','launch','/workspace/core/launch/session.launch.py','mode:='+args.mode,'database:='+str(db)]
        if args.initial_pose:command.append('initial_pose:='+args.initial_pose)
        return subprocess.call(['python3','/workspace/scripts/run_launch.py',*command])


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,BlockingIOError) as exc:raise SystemExit(str(exc))
