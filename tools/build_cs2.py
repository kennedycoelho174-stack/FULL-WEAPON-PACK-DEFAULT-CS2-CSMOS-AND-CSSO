"""Build independent editor/download packages from the user's Source conversions.

The native view mesh is the export authority. Original OBJ geometry is verified
as the reference, while UV seams are split without moving weighted vertices.
"""
import argparse, collections, gzip, hashlib, io, json, math, re, struct, zipfile
from pathlib import Path
import numpy as np
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from native_helpers import I, H, put, align, models, triangles, native_mesh
from vtf_png import decode

SIZE=2048
NAMES={'ak47':('AK-47','rif_ak47'),'aug':('AUG','rif_aug'),
 'famas':('FAMAS','rif_famas'),'galil':('Galil AR','rif_galilar'),
 'm4a1s':('M4A1-S','rif_m4a1_silencer'),'m4a4':('M4A4','rif_m4a4'),
 'sg553':('SG 553','rif_sg556'),'deagle':('Desert Eagle','pist_deagle'),
 'glock':('Glock-18','pist_glock18'),'p250':('P250','pist_p250'),
 'usp':('USP-S','pist_usp'),'p2000':('P2000','pist_hkp2000'),
 'dual':('Dual Berettas','pist_elite'),'tec9':('Tec-9','pist_tec9'),
 'fiveseven':('Five-SeveN','pist_fiveseven'),'cz75':('CZ75-Auto','pist_cz_75'),
 'r8':('R8 Revolver','pist_revolver'),'zeus':('Zeus x27','eq_taser'),
 'bizon':('PP-Bizon','smg_bizon'),'mac10':('MAC-10','smg_mac10'),
 'mp5':('MP5-SD','smg_mp5sd'),'mp7':('MP7','smg_mp7'),'mp9':('MP9','smg_mp9'),
 'p90':('P90','smg_p90'),'ump45':('UMP-45','smg_ump45'),
 'awp':('AWP','snip_awp'),'g3sg1':('G3SG1','snip_g3sg1'),
 'scar20':('SCAR-20','snip_scar20'),'ssg':('SSG 08','snip_ssg08'),
 'mag7':('MAG-7','shot_mag7'),'nova':('Nova','shot_nova'),
 'sawedoff':('Sawed-Off','shot_sawedoff'),'xm1014':('XM1014','shot_xm1014'),
 'm249':('M249','mach_m249para'),'negev':('Negev','mach_negev')}

def digest(b):return hashlib.sha256(b).hexdigest()
def text(b,p):return b[p:b.index(0,p)].decode()

def vpk(data):
 sig,version,length=struct.unpack_from('<III',data)
 assert sig==0x55aa1234 and version in (1,2)
 header=28 if version==2 else 12
 tree=data[header:header+length];cursor=0;out={}
 def string():
  nonlocal cursor
  end=tree.index(0,cursor);value=tree[cursor:end].decode();cursor=end+1;return value
 while (ext:=string()):
  while (directory:=string()):
   while (name:=string()):
    crc,pre,archive,offset,size,terminator=struct.unpack_from('<IHHIIH',tree,cursor)
    cursor+=18;preload=tree[cursor:cursor+pre];cursor+=pre
    assert archive==32767 and terminator==65535
    path=(directory+'/' if directory!=' ' else '')+name+'.'+ext
    assert not path.startswith('/') and '..' not in path.split('/')
    payload=preload+data[header+length+offset:header+length+offset+size]
    assert len(payload)==pre+size
    import zlib
    assert zlib.crc32(payload)&0xffffffff==crc,path
    out[path]=payload
 return out

def read_geometry(files,prefix):
 m=files[prefix+'.mdl'];v=files[prefix+'.vvd'];x=files[prefix+'.dx90.vtx']
 assert m[:4]==b'IDST' and I(m,4)==48 and I(v,48)==0
 groups=[]
 for bi,mi,at in models(m):
  for mesh_index in range(I(m,at+72)):
   mesh=at+I(m,at+76)+mesh_index*116
   count=I(m,mesh+8);start=I(m,at+84)//48+I(m,mesh+12)
   records=[v[I(v,56)+(start+k)*48:I(v,56)+(start+k+1)*48] for k in range(count)]
   tri=triangles(x,bi,mi,mesh_index)
   if not len(tri):continue
   values=np.array([struct.unpack_from('<8f',r,16) for r in records],np.float32)
   tangents=[v[I(v,60)+(start+k)*16:I(v,60)+(start+k+1)*16] for k in range(count)]
   groups.append(dict(body=bi,model=mi,at=at,mesh=mesh_index,material=I(m,mesh),tangents=tangents,
    records=records,pos=values[:,:3],norm=values[:,3:6],uv=values[:,6:],tri=tri))
 assert groups
 return groups

def source_images(files,mdl):
 lower={p.lower():p for p in files}
 dirs=[text(mdl,I(mdl,I(mdl,216)+k*4)).replace('\\','/').strip('/') for k in range(I(mdl,212))]
 images={};details=[]
 for k in range(I(mdl,204)):
  at=I(mdl,208)+k*64;name=text(mdl,at+I(mdl,at)).replace('\\','/')
  candidates=['materials/'+(d+'/' if d else '')+name+'.vmt' for d in dirs]+['materials/'+name+'.vmt']
  path=next((lower[p.lower()] for p in candidates if p.lower() in lower),None)
  assert path,('Missing VMT',name,candidates)
  match=re.search(r'\$basetexture"?\s+"?([^"\s}]+)',files[path].decode(errors='replace'),re.I)
  assert match,path
  texture='materials/'+match[1].replace('\\','/')+'.vtf'
  assert texture.lower() in lower,texture
  image=np.array(decode(files[lower[texture.lower()]])[0])
  images[k]=image;details.append(dict(material=name,texture=texture))
 return images,details

def raster(groups,images):
 result=np.zeros((SIZE,SIZE,4),np.uint8);result[:]=(255,255,255,255)
 owner=np.zeros((SIZE,SIZE),bool);overlaps=0
 for group in groups:
  if not group['paintable']:continue
  image=images[group['material']];height,width=image.shape[:2]
  for t in group['tri']:
   uv=group['atlas'][t];source=group['source_uv'][t]
   lo=np.floor(uv.min(0)*SIZE).astype(int).clip(0,SIZE-1)
   hi=np.ceil(uv.max(0)*SIZE).astype(int).clip(0,SIZE)
   yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
   d=uv[1:]-uv[0];det=d[0,0]*d[1,1]-d[0,1]*d[1,0]
   if abs(det)<1e-14:continue
   dx=(xx+.5)/SIZE-uv[0,0];dy=(yy+.5)/SIZE-uv[0,1]
   b=(dx*d[1,1]-dy*d[1,0])/det;c=(d[0,0]*dy-d[0,1]*dx)/det
   inside=(np.minimum(np.minimum(1-b-c,b),c)>=-1e-7)
   if not inside.any():continue
   xy=np.column_stack((1-b[inside]-c[inside],b[inside],c[inside]))@source
   sx=np.floor((xy[:,0]%1)*width).astype(int).clip(0,width-1)
   sy=np.floor((xy[:,1]%1)*height).astype(int).clip(0,height-1)
   y=yy[inside];x=xx[inside];overlaps+=int(owner[y,x].sum())
   result[y,x]=image[sy,sx];result[y,x,3]=255;owner[y,x]=True
 assert owner.sum()>SIZE*SIZE*.2
 # Duplicate alternate bodygroup surfaces can have identical triangles, but
 # xatlas packs their charts independently. Interior overlap indicates failure.
 assert overlaps < owner.sum()*.002,('UV overlap',overlaps,int(owner.sum()))
 distance,nearest=distance_transform_edt(~owner,return_indices=True)
 pad=(~owner)&(distance<=4);result[pad]=result[nearest[0][pad],nearest[1][pad]]
 out=io.BytesIO();Image.fromarray(result).save(out,format='PNG',compress_level=1)
 return out.getvalue(),int(owner.sum())

def tangent_records(pos,norm,uv,tri):
 tangent=np.zeros_like(pos);bitangent=np.zeros_like(pos)
 for t in tri:
  edge=pos[t[1:]]-pos[t[0]];d=uv[t[1:]]-uv[t[0]];det=d[0,0]*d[1,1]-d[0,1]*d[1,0]
  if abs(det)<1e-14:continue
  tangent[t]+=(edge[0]*d[1,1]-edge[1]*d[0,1])/det
  bitangent[t]+=(edge[1]*d[0,0]-edge[0]*d[1,0])/det
 tangent-=norm*(tangent*norm).sum(1)[:,None]
 length=np.linalg.norm(tangent,axis=1);bad=length<1e-12
 if bad.any():
  axes=np.zeros_like(tangent[bad]);axes[:,0]=1
  parallel=np.abs(norm[bad,0])>.9;axes[parallel]=(0,1,0)
  tangent[bad]=np.cross(norm[bad],axes);length=np.linalg.norm(tangent,axis=1)
 tangent/=np.maximum(length[:,None],1e-12)
 sign=np.where((np.cross(norm,tangent)*bitangent).sum(1)<0,-1.,1.)
 return [struct.pack('<4f',*tangent[k],sign[k]) for k in range(len(pos))]

def unwrap_groups(groups):
 paintable=[g for g in groups if g['paintable']]
 atlas=xatlas.Atlas()
 for g in paintable:atlas.add_mesh(g['pos'],g['tri'],normals=g['norm'])
 options=xatlas.PackOptions();options.resolution=SIZE;options.padding=8
 atlas.generate(pack_options=options)
 for g in groups:
  if not g['paintable']:g['atlas']=g['uv'];g['source_uv']=g['uv']
 for k,g in enumerate(paintable):
  mapping,tri,uv=atlas[k];mapping=np.asarray(mapping)
  g['source_uv']=g['uv'][mapping];g['atlas']=uv.copy();g['tri']=tri.copy()
  g['pos']=g['pos'][mapping];g['norm']=g['norm'][mapping]
  source=g['records'];g['records']=[]
  for i,j in enumerate(mapping):
   record=bytearray(source[int(j)]);struct.pack_into('<2f',record,40,*uv[i]);g['records'].append(bytes(record))
  g['tangents']=tangent_records(g['pos'],g['norm'],uv,g['tri'])
 return groups

def chunks(g,max_bones):
 """Respect Source's 16-bit mesh indices and per-strip bone palette."""
 used={};bones=set();indices=[]
 def emit():
  order=list(used)
  return [g['records'][k] for k in order],[g['tangents'][k] for k in order],np.array(indices,np.uint32),g['target_material']
 for tri in g['tri']:
  needed=set(int(k) for k in tri)
  palette={b for k in needed for b in g['records'][k][12:12+g['records'][k][15]]}
  assert len(palette)<=max_bones
  if indices and (len(set(used)|needed)>60000 or len(bones|palette)>max_bones):
   yield emit();used={};bones=set();indices=[]
  bones|=palette
  indices.append([used.setdefault(int(k),len(used)) for k in tri])
 if indices:yield emit()

def patch_native(files,prefix,groups,id):
 old=files[prefix+'.mdl'];mdl=bytearray(old);ov=files[prefix+'.vvd']
 geom=collections.defaultdict(list)
 maxbones=min(H(files[prefix+e],8) for e in ('.dx90.vtx','.dx80.vtx','.sw.vtx'))
 for g in groups:geom[g['body'],g['model']].extend(chunks(g,maxbones))
 vertex=bytearray();tangent=bytearray();cursor=0;mesh_id=0
 original_model=list(models(old))
 for bi,mi,at in original_model:
  parts=geom[bi,mi];table=align(mdl);local=0
  for recs,tans,tri,material_id in parts:
   source_mesh=at+I(old,at+76);dest=len(mdl);mdl.extend(old[source_mesh:source_mesh+116])
   put(mdl,dest,material_id,at-dest,len(recs),local);put(mdl,dest+32,mesh_id);mesh_id+=1
   for lod in range(8):put(mdl,dest+52+lod*4,len(recs))
   vertex.extend(b''.join(recs));tangent.extend(b''.join(tans));local+=len(recs)
  put(mdl,at+72,len(parts),table-at,local,cursor*48,cursor*16);cursor+=local
 # Keep original optical materials and append the controlled painted surface.
 material=align(mdl);count=I(old,204)
 for k in range(count):
  at=I(old,208)+k*64;dest=len(mdl);mdl.extend(old[at:at+64]);put(mdl,dest,at+I(old,at)-dest)
 dest=len(mdl);mdl.extend(b'\0'*64);name=len(mdl);mdl.extend(b'skinmaker_view\0');put(mdl,dest,name-dest)
 put(mdl,204,count+1,material);skin=align(mdl);families=max(1,I(old,224));refs=I(old,220)
 assert refs==count
 for family in range(families):
  at=I(old,228)+family*refs*2;mdl.extend(old[at:at+refs*2]);mdl.extend(struct.pack('<H',count))
 put(mdl,220,refs+1,families,skin)
 directory=f'models/weapons/v_models/cs2_{id}/'
 path_table=align(mdl);paths=I(old,212)
 mdl.extend(old[I(old,216):I(old,216)+paths*4]);slot=len(mdl);mdl.extend(b'\0'*4)
 name=len(mdl);mdl.extend(directory.encode()+b'\0')
 put(mdl,slot,name);put(mdl,212,paths+1,path_table)
 put(mdl,76,len(mdl));files[prefix+'.mdl']=bytes(mdl)
 header=bytearray(ov[:64]);put(header,12,1)
 for lod in range(8):put(header,16+lod*4,cursor)
 put(header,48,0,0,64,64+len(vertex));files[prefix+'.vvd']=bytes(header+vertex+tangent)
 for ext in ('.dx80.vtx','.dx90.vtx','.sw.vtx'):
  oldx=files[prefix+ext];buf=bytearray(oldx)
  for bi,mi,at in original_model:
   mesh_offsets=[native_mesh(buf,recs,tri,ext!='.sw.vtx',H(oldx,8)) for recs,tans,tri,mat in geom[bi,mi]]
   table=align(buf)
   for mesh in mesh_offsets:
    dest=len(buf);buf.extend(buf[mesh:mesh+9]);put(buf,dest+4,mesh+I(buf,mesh+4)-dest)
   bp=I(oldx,32)+bi*8;model=bp+I(oldx,bp+4)+mi*8
   for li in range(I(oldx,model)):
    lod=model+I(oldx,model+4)+li*12;put(buf,lod,len(mesh_offsets),table-lod)
  files[prefix+ext]=bytes(buf)
 files['materials/'+directory+'skinmaker_view.vmt']=('VertexLitGeneric\n{\n$model 1\n$basetexture "'+directory+'skinmaker_view"\n$phong 0\n}\n').encode()
 return cursor

def glb(groups,png,id,source_images):
 blob=bytearray();views=[];access=[]
 def view(data):
  align(blob);i=len(views);views.append(dict(buffer=0,byteOffset=len(blob),byteLength=len(data)));blob.extend(data);return i
 def acc(data,n,integer=False):
  a=np.asarray(data,'<u4' if integer else '<f4').reshape(-1,n)
  record=dict(bufferView=view(a.tobytes()),componentType=5125 if integer else 5126,count=len(a),type={1:'SCALAR',2:'VEC2',3:'VEC3'}[n])
  if n==3:record.update(min=a.min(0).tolist(),max=a.max(0).tolist())
  i=len(access);access.append(record);return i
 transparent=io.BytesIO();Image.new('RGBA',(1,1)).save(transparent,format='PNG')
 images=[dict(bufferView=view(png),mimeType='image/png'),dict(bufferView=view(transparent.getvalue()),mimeType='image/png')]
 materials=[dict(name=f'CS2{id}Layer{k}',extras=dict(skinMakerLayer=k),alphaMode='OPAQUE' if k==0 else 'BLEND',
  pbrMetallicRoughness=dict(baseColorTexture=dict(index=0 if k==0 else 1),metallicFactor=0,roughnessFactor=.7)) for k in range(3)]
 nodes=[];meshes=[]
 # Empty alternatives are animation toggles in the game. The editor shows the
 # first nonempty alternative of each bodypart, avoiding duplicate previews.
 alternatives={b:min(g['model'] for g in groups if g['body']==b) for b in {g['body'] for g in groups}}
 for layer,gap in enumerate((0.,.003,.009)):
  for g in groups:
   if g['model']!=alternatives[g['body']]:continue
   if not g['paintable'] and layer!=0:continue
   pos=(g['pos']+g['norm']*gap)[:,[1,2,0]];norm=g['norm'][:,[1,2,0]]
   attr=dict(POSITION=acc(pos,3),NORMAL=acc(norm,3),TEXCOORD_0=acc(g['atlas'],2),TEXCOORD_1=acc(g['source_uv'],2))
   mat=layer
   if not g['paintable']:
    image=io.BytesIO();Image.fromarray(source_images[g['material']]).save(image,format='PNG')
    image_index=len(images);images.append(dict(bufferView=view(image.getvalue()),mimeType='image/png'))
    mat=len(materials);materials.append(dict(name='CS2Protected',extras=dict(skinMakerLayer=0),
     pbrMetallicRoughness=dict(baseColorTexture=dict(index=image_index),metallicFactor=0,roughnessFactor=.2),alphaMode='BLEND'))
   primitive=dict(attributes=attr,indices=acc(g['tri'][:,[0,2,1]],1,True),material=mat)
   meshes.append(dict(primitives=[primitive]));name=f'CS2{id}Layer{layer}' if g['paintable'] else f'CS2{id}Protected{len(meshes)}'
   nodes.append(dict(name=name,mesh=len(meshes)-1))
 doc=dict(asset=dict(version='2.0',generator='SkinMaker CS2 native view atlas'),scene=0,
  scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,materials=materials,
  images=images,textures=[dict(source=k) for k in range(len(images))],accessors=access,bufferViews=views,buffers=[dict(byteLength=len(blob))])
 raw=json.dumps(doc,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4);align(blob)
 return struct.pack('<III',0x46546c67,2,28+len(raw)+len(blob))+struct.pack('<II',len(raw),0x4e4f534a)+raw+struct.pack('<II',len(blob),0x004e4942)+blob

def build(id,raw,out,reference=None):
 files=vpk(raw);prefix='models/weapons/v_'+NAMES[id][1]
 groups=read_geometry(files,prefix);original={name:files[name] for name in files if name.startswith('models/')}
 images,materials=source_images(files,files[prefix+'.mdl'])
 areas=collections.Counter()
 for g in groups:areas[g['material']]+=len(g['tri'])
 primary=areas.most_common(1)[0][0];new_material=I(files[prefix+'.mdl'],204)
 for g in groups:
  g['paintable']=g['material']==primary
  g['target_material']=new_material if g['paintable'] else g['material']
 source_records=collections.Counter(r[:40] for g in groups for r in g['records'])
 faces=sum(len(g['tri']) for g in groups)
 reference_score=None
 if reference is not None:
  uv=np.array([struct.unpack_from('<2f',r,40) for g in groups for r in g['records']])
  obj=reference.copy();obj[:,1]=1-obj[:,1]
  reference_score=float((cKDTree(uv).query(obj)[0]<.00015).mean())
  assert reference_score>.95,(id,'OBJ reference UV mismatch',reference_score)
 groups=unwrap_groups(groups);png,covered=raster(groups,images)
 assert all(r[:40] in source_records for g in groups for r in g['records'])
 count=patch_native(files,prefix,groups,id)
 assert sum(len(g['tri']) for g in groups)==faces
 # Reparse the actual exported topology and compare weighted positions/normals.
 reparsed=read_geometry(files,prefix)
 assert sum(len(g['tri']) for g in reparsed)==faces
 assert all(r[:40] in source_records for g in reparsed for r in g['records'])
 for name,data in original.items():
  if name not in {prefix+ext for ext in ('.mdl','.vvd','.dx90.vtx','.dx80.vtx','.sw.vtx')}:assert files[name]==data
 data=glb(groups,png,id,images);weapon='cs2_'+id;directory=out/weapon;directory.mkdir(parents=True,exist_ok=True)
 template=io.BytesIO()
 with zipfile.ZipFile(template,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,b in files.items():
   if name.startswith('models/') and not name.rsplit('/',1)[-1].startswith('w_'):z.writestr(name,b)
   elif name.startswith('materials/') and name!=f'materials/models/weapons/v_models/cs2_{id}/skinmaker_view.vmt':z.writestr(name,b)
  material=f'materials/models/weapons/v_models/{weapon}/skinmaker_view.vmt';z.writestr(material,files[material])
 report=dict(weapon=weapon,name=NAMES[id][0],source_sha256=digest(raw),vertices=count,triangles=faces,
  covered_pixels=covered,source_materials=materials,rig_and_positions_preserved=True,view_only=True,visual_validation='pending in CSMOS')
 report['obj_uv_correspondence']=reference_score
 report['protected_materials']=[k for k in areas if k!=primary]
 package=out/(weapon+'-v1.zip')
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  z.writestr(f'assets/models/{weapon}_three.glb',data)
  z.writestr(f'assets/{weapon}_view/template.zip',template.getvalue())
  z.writestr(f'assets/{weapon}_view/vanilla.png',png)
  z.writestr(f'assets/{weapon}_view/info.json',json.dumps(report,indent=2))
 b=package.read_bytes();assert zipfile.ZipFile(io.BytesIO(b)).testzip() is None
 record=dict(weapon=weapon,size=len(b),sha256=digest(b),base='https://github.com/kennedycoelho174-stack/FULL-WEAPON-PACK-DEFAULT-CS2-CSMOS-AND-CSSO/releases/download/skinmaker-cs2-v1/',
  parts=[dict(name=package.name,bytes=len(b),sha256=digest(b))])
 (directory/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 return record

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('ids',nargs='*');args=parser.parse_args()
 args.out.mkdir(parents=True,exist_ok=True);wanted=args.ids or list(NAMES);records=[]
 references={}
 geometry=args.sources/'geometry.zip'
 if geometry.exists():
  assert digest(geometry.read_bytes())=='f06ed57135a3875658eaee4175c46b203d4c91f83a7f287ef6663dee11b79bbe'
  with zipfile.ZipFile(geometry) as z:
   for id in wanted:
    name={'cz75':'pist_cz75a','usp':'pist_usp_silencer','m249':'mach_m249','zeus':'pist_taser'}.get(id,NAMES[id][1])
    references[id]=np.array([list(map(float,line.split()[1:3])) for line in z.read('weapon_'+name+'.obj').splitlines() if line.startswith(b'vt ')])
 found=set()
 for path in sorted(args.sources.glob('*.zip')):
  if path.name=='geometry.zip':continue
  with zipfile.ZipFile(path) as z:
   for n in z.namelist():
    if not n.endswith('.vpk'):continue
    raw=z.read(n);files=vpk(raw)
    for id in wanted:
     if 'models/weapons/v_'+NAMES[id][1]+'.mdl' in files:
      assert id not in found,id;found.add(id)
      records.append(build(id,raw,args.out,references.get(id)))
 for id in wanted:
  assert id in found,id
 (args.out/'catalog.json').write_text(json.dumps(dict(version=1,packages=records),indent=2))
if __name__=='__main__':main()
