"""Add the original world model to each verified CS2 editor package."""
import argparse
import ast
import hashlib
import io
import json
import re
import struct
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generator',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--packages',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--index',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    program=ast.parse(args.generator.read_text())
    scope={'struct':struct}
    exec(compile(ast.Module(body=[n for n in program.body if
         isinstance(n,ast.FunctionDef) and n.name=='vpk' or
         isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets)],
         type_ignores=[]),str(args.generator),'exec'),scope)
    names={f'v_{stem}.mdl':'cs2_'+key for key,(_,stem) in scope['NAMES'].items()}
    expected={p['weapon']:p for p in json.loads(args.index.read_text())['packages']}
    catalog=[]
    for group in sorted(args.sources.glob('*.zip')):
        with ZipFile(group) as source:
            for member in source.namelist():
                if not member.lower().endswith('.vpk'): continue
                files=scope['vpk'](source.read(member))
                matches=[names[p.rsplit('/',1)[-1]] for p in files if p.rsplit('/',1)[-1] in names]
                assert len(matches)==1,member
                weapon=matches[0]
                package=args.packages/f'{weapon}-v1.zip'
                raw=package.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==expected[weapon]['sha256'],weapon
                ground={p:b for p,b in files.items() if p.startswith('models/') and p.rsplit('/',1)[-1].startswith('w_')}
                assert ground and any(p.endswith('.mdl') for p in ground),weapon
                for path,model in ground.items():
                    if not path.endswith('.mdl'):continue
                    assert model[:4]==b'IDST'
                    for ext,offset in (('.vvd',8),('.dx90.vtx',16)):
                        companion=ground[path[:-4]+ext]
                        assert companion[offset:offset+4]==model[8:12],(weapon,path,ext)
                # Restore the original materials for the original world UVs.
                ground.update({p:b for p,b in files.items() if p.startswith('materials/')})
                with ZipFile(io.BytesIO(raw)) as old:
                    template=f'assets/{weapon}_view/template.zip'
                    with ZipFile(io.BytesIO(old.read(template))) as source_template:
                        entries={p:source_template.read(p) for p in source_template.namelist()}
                    for path in entries.keys() & ground.keys():
                        if entries[path]!=ground[path]:
                            assert path.endswith('.vmt') and re.search(rb'skinmaker_view',entries[path]),(weapon,path)
                    new_template=io.BytesIO()
                    with ZipFile(new_template,'w',ZIP_DEFLATED,compresslevel=1) as z:
                        for path,data in (entries|ground).items(): z.writestr(path,data)
                    target=args.out/f'{weapon}-v2.zip'
                    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=1) as z:
                        for path in old.namelist():
                            data=new_template.getvalue() if path==template else old.read(path)
                            if path.endswith('/info.json'):
                                info=json.loads(data)
                                info.update(view_only=False, world_skin='original')
                                data=json.dumps(info).encode()
                            z.writestr(path,data)
                with ZipFile(target) as z:
                    assert z.testzip() is None
                    with ZipFile(io.BytesIO(z.read(template))) as t:
                        for path,data in ground.items():assert t.read(path)==data
                sha=hashlib.sha256(target.read_bytes()).hexdigest()
                size=target.stat().st_size
                catalog.append(dict(weapon=weapon,size=size,sha256=sha,
                    parts=[dict(name=target.name,bytes=size,sha256=sha)]))
                print(weapon,len([p for p in ground if p.endswith('.mdl')]),flush=True)
    assert sorted(p['weapon'] for p in catalog)==sorted(expected)
    (args.out/'catalog.json').write_text(json.dumps(dict(version=2,packages=sorted(catalog,key=lambda p:p['weapon'])),indent=2)+'\n')

if __name__=='__main__':main()
