"""Decode the largest first-frame image of Source VTF sticker textures."""
import io, struct
from PIL import Image
import texture2ddecoder as decoder

def u32(b, p): return struct.unpack_from('<I', b, p)[0]
def u16(b, p): return struct.unpack_from('<H', b, p)[0]
def image_size(fmt, w, h):
    if fmt in (13,20): return max(1,(w+3)//4)*max(1,(h+3)//4)*8
    if fmt in (14,15): return max(1,(w+3)//4)*max(1,(h+3)//4)*16
    return w*h*{0:4,1:4,2:3,3:3,4:2,5:1,6:2,8:1,11:4,12:4,16:4}.get(fmt,0)

def decode(b):
    if b[:4]!=b'VTF\0' or len(b)<64: raise ValueError('Invalid VTF')
    major,minor=u32(b,4),u32(b,8)
    if major!=7: raise ValueError('Unsupported VTF version')
    w,h=u16(b,16),u16(b,18);fmt=u32(b,52);mips=b[56]
    if not(0<w<=4096 and 0<h<=4096 and 0<mips<=16): raise ValueError('Invalid dimensions')
    frames=max(1,u16(b,24)); depth=max(1,u16(b,63)) if minor>=2 else 1
    faces=6 if u32(b,20)&0x4000 else 1
    if faces>1 and minor<5 and u16(b,26)!=65535: faces=7
    offset=None
    if minor>=3:
        count=u32(b,68)
        if count>32 or 80+8*count>len(b): raise ValueError('Invalid resource table')
        for i in range(count):
            p=80+8*i
            if b[p:p+3]==b'\x30\0\0': offset=u32(b,p+4)
    if offset is None:
        lowfmt=u32(b,57)
        offset=u32(b,12)+(image_size(lowfmt,b[61],b[62]) if lowfmt!=0xffffffff and b[61] and b[62] else 0)
    for mip in range(mips-1,0,-1):
        n=image_size(fmt,max(1,w>>mip),max(1,h>>mip))
        if not n: raise ValueError(f'Unsupported VTF format {fmt}')
        offset+=n*frames*faces*max(1,depth>>mip)
    size=image_size(fmt,w,h)
    if not size or offset+size>len(b): raise ValueError(f'Unsupported/truncated VTF format {fmt}')
    pixels=b[offset:offset+size]
    if fmt in (13,20,14,15):
        if fmt==14: im=Image.frombytes('RGBA',(w,h),pixels,'bcn',(2,'DXT3'))
        else:
            func=decoder.decode_bc3 if fmt==15 else decoder.decode_bc1
            im=Image.frombytes('RGBA',(w,h),func(pixels,w,h),'raw','BGRA')
    elif fmt in (0,1,2,3,11,12,16):
        mode={0:'RGBA',1:'ABGR',2:'RGB',3:'BGR',11:'ARGB',12:'BGRA',16:'BGRX'}[fmt]
        im=Image.frombytes('RGBA' if fmt not in (2,3) else 'RGB',(w,h),pixels,'raw',mode).convert('RGBA')
    elif fmt==5: im=Image.frombytes('L',(w,h),pixels).convert('RGBA')
    elif fmt==8:
        im=Image.new('RGBA',(w,h),'white');im.putalpha(Image.frombytes('L',(w,h),pixels))
    elif fmt==6: im=Image.frombytes('LA',(w,h),pixels).convert('RGBA')
    elif fmt==4: im=Image.frombytes('RGB',(w,h),pixels,'raw','BGR;16').convert('RGBA')
    else: raise ValueError(f'Unsupported VTF format {fmt}')
    return im, {'width':w,'height':h,'format':fmt,'frames':frames}

def png(b, thumbnail=False, max_edge=None, tint=None):
    im,info=decode(b)
    if thumbnail: im.thumbnail((128,128))
    if max_edge: im.thumbnail((max_edge,max_edge))
    if tint is not None and tuple(tint)!=(255,255,255):
        r,g,blue,alpha=im.split()
        channels=[]
        for channel,strength in zip((r,g,blue),tint):
            channels.append(channel.point([round(v*max(0,min(255,strength))/255) for v in range(256)]))
        im=Image.merge('RGBA',(*channels,alpha))
    out=io.BytesIO();im.save(out,format='PNG',compress_level=1);return out.getvalue(),info


