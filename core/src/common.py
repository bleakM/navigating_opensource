"""Local workbench storage; recording data never enters Git or external services."""
import json,os,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'.runtime/session'
DATA.mkdir(parents=True,exist_ok=True)


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def write_json(path,data):atomic(path,json.dumps(data,ensure_ascii=False,indent=2).encode())


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
