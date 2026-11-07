import struct
import numpy as np
I=lambda b,o:struct.unpack_from('<i',b,o)[0]
H=lambda b,o:struct.unpack_from('<H',b,o)[0]
def put(b,o,*v):struct.pack_into('<'+'i'*len(v),b,o,*map(int,v))
def align(b):
 while len(b)%4:b.append(0)
 return len(b)
def models(mdl):
 for bi in range(I(mdl,232)):
  bp=I(mdl,236)+bi*16
  for mi in range(I(mdl,bp+4)):
   at=bp+I(mdl,bp+12)+mi*148
   if I(mdl,at+80):yield bi,mi,at

def triangles(vtx,bi,mi,mesh_index=0):
 bp=I(vtx,32)+bi*8;m=bp+I(vtx,bp+4)+mi*8;lod=m+I(vtx,m+4);mesh=lod+I(vtx,lod+4)+mesh_index*9;out=[]
 for gi in range(I(vtx,mesh)):
  g=mesh+I(vtx,mesh+4)+gi*25
  verts=[H(vtx,g+I(vtx,g+4)+k*9+4) for k in range(I(vtx,g))]
  ix=[H(vtx,g+I(vtx,g+12)+k*2) for k in range(I(vtx,g+8))]
  for si in range(I(vtx,g+16)):
   s=g+I(vtx,g+20)+si*27;seq=ix[I(vtx,s+4):I(vtx,s+4)+I(vtx,s)]
   assert vtx[s+18]&1
   for k in range(0,len(seq),3):
    t=[verts[q] for q in seq[k:k+3]]
    if len(set(t))==3:out.append(t)
 return np.asarray(out,np.uint32)

def native_mesh(buf,records,ix,hardware,max_bones):
 mesh=align(buf);buf.extend(b'\0'*9);group=len(buf);buf.extend(b'\0'*25)
 bones=sorted({bone for rec in records for bone in rec[12:12+rec[15]]});assert len(bones)<=max_bones
 palette={b:k for k,b in enumerate(bones)};vo=len(buf)
 assert len(records)<65536
 for k,r in enumerate(records):
  nb=r[15];assert 0<nb<=3
  ids=[palette[b] if hardware else b for b in r[12:12+nb]]+[0]*(3-nb)
  buf.extend(struct.pack('<4BH3B',0,1,2,nb,k,*ids))
 io=len(buf);flat=ix.ravel();buf.extend(struct.pack('<'+'H'*len(flat),*map(int,flat)))
 so=len(buf);buf.extend(b'\0'*27);bo=len(buf)
 if hardware:
  for bone in bones:buf.extend(struct.pack('<ii',palette[bone],bone))
 # numBones is the maximum influences PER VERTEX, not the palette size.
 struct.pack_into('<4ihBii',buf,so,len(flat),0,len(records),0,max(r[15] for r in records),1,len(bones) if hardware else 0,bo-so if hardware else 0)
 struct.pack_into('<6iB',buf,group,len(records),vo-group,len(flat),io-group,1,so-group,2 if hardware else 0)
 put(buf,mesh,1,group-mesh)
 return mesh


