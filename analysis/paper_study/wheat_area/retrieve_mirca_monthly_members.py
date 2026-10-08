"""Retrieve wheat members of the public non-solid RAR via verified HTTP ranges.

The assembled per-member archives are derived containers. Member bytes and
RAR CRCs are verified; no checksum of the entire remote archive is claimed.
"""
from pathlib import Path
import io, json, hashlib, struct, zlib, subprocess, time
import requests, rarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'
CATALOG=json.loads((DATA/'mirca_os_v2_filelist.json').read_text())
SOURCE=next(x for x in CATALOG['results'] if x['file_name']=='Monthly Growing Area Grids.rar')
URL=SOURCE['url'].replace('http:','https:',1)

class RemoteRangeFile(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.size=SOURCE['size'];self.session=requests.Session();self.buffer=b'';self.start=-1;self.requests=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def fetch(self,start,end):
        r=self.session.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=60)
        if r.status_code in (401,403):raise RuntimeError(f'Public source denied access: {r.status_code}')
        r.raise_for_status()
        expected=f'bytes {start}-{end}/{self.size}'
        if r.status_code!=206 or r.headers.get('Content-Range')!=expected or len(r.content)!=end-start+1:
            raise RuntimeError(f'Incorrect byte range: {r.status_code}, {r.headers.get("Content-Range")} expected {expected}')
        self.requests.append(dict(start=start,end=end,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),etag=r.headers.get('ETag')))
        return r.content
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        size=min(size,self.size-self.pos)
        if size<=0:return b''
        if not (self.start<=self.pos and self.pos+size<=self.start+len(self.buffer)):
            self.start=self.pos;self.buffer=self.fetch(self.pos,min(self.size-1,self.pos+max(size,8192)-1))
        out=self.buffer[self.pos-self.start:self.pos-self.start+size];self.pos+=len(out);return out

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    receipt_path=DATA/'mirca_monthly_wheat_download_manifest.json'
    if receipt_path.exists():
        archived=json.loads(receipt_path.read_text())
        if archived['source']!=SOURCE:raise RuntimeError('Provider metadata differs from the acquired source snapshot')
        for item in archived['members']:
            p=ROOT/item['local_path'];b=p.read_bytes()
            if hashlib.sha256(b).hexdigest()!=item['sha256'] or f'{zlib.crc32(b):08x}'!=item['crc32']:
                raise RuntimeError('Archived member verification failed: '+str(p))
        print(json.dumps(dict(status='existing_archived_members_verified',members=len(archived['members']),network_requests=0)),flush=True)
        return
    f=RemoteRangeFile()
    beginning=f.fetch(0,8191)
    local=rarfile.RarFile(io.BytesIO(beginning),errors='stop')
    parser=local._file_parser
    main=parser._main
    # RAR5 locator points to an uncompressed header cache near the archive end.
    # The provider archive is non-solid, so only the desired member streams
    # need downloading. Each cached member header is compared with its actual
    # original header before extraction.
    main_bytes=beginning[main.header_offset:main.data_offset]
    pos=len(main_bytes)-main.block_extra_size
    recsize,pos=rarfile.load_vint(main_bytes,pos)
    kind,pos=rarfile.load_vint(main_bytes,pos)
    flags,pos=rarfile.load_vint(main_bytes,pos)
    if kind!=1 or not flags&1:raise RuntimeError('Missing RAR5 quick-open locator')
    distance,pos=rarfile.load_vint(main_bytes,pos)
    qo_offset=main.header_offset+distance
    tail=f.fetch(qo_offset,f.size-1)
    (DATA/'mirca_monthly_archive_quick_open.bin').write_bytes(tail)
    qo=parser._parse_block_header(io.BytesIO(tail))
    if qo.filename!='QO' or qo.compress_type!=rarfile.RAR_M0:raise RuntimeError('Unexpected quick-open compression')
    cache=tail[qo.data_offset:qo.data_offset+qo.compress_size]
    if qo.file_flags & 4 and zlib.crc32(cache)!=qo.CRC:raise RuntimeError('Quick-open data CRC mismatch')
    members=[];cached_headers={};pos=0
    while pos<len(cache):
        start=pos;crc,pos=rarfile.load_le32(cache,pos);length,pos=rarfile.load_vint(cache,pos)
        finish=pos+length
        if zlib.crc32(cache[start+4:finish])!=crc:raise RuntimeError('Cached structure CRC mismatch')
        cacheflags,pos=rarfile.load_vint(cache,pos);offset,pos=rarfile.load_vint(cache,pos)
        hsize,pos=rarfile.load_vint(cache,pos);header=cache[pos:pos+hsize]
        if pos+hsize!=finish:raise RuntimeError('Unexpected cached structure length')
        info=parser._parse_block_header(io.BytesIO(header));original_offset=qo_offset-offset
        if info and info.block_type==rarfile.RAR5_BLOCK_FILE:
            info.header_offset=original_offset;info.data_offset=original_offset+info.header_size
            members.append(info);cached_headers[info.filename]=header
        pos=finish
    if local.is_solid():raise RuntimeError('Solid archive cannot be independently extracted')
    index=[dict(filename=x.filename,header_offset=x.header_offset,header_size=x.header_size,
        data_offset=x.data_offset,packed_bytes=x.compress_size,unpacked_bytes=x.file_size,
        crc32=x.CRC,solid=bool(x.flags & rarfile.RAR_FILE_SOLID)) for x in members]
    (HERE/'mirca_monthly_archive_index.json').write_text(json.dumps(dict(source=SOURCE,members=index),indent=2)+'\n')
    targets=[x for x in members if 'Wheat' in x.filename and '_2020_' in x.filename and x.filename.endswith('.nc')]
    if len(targets)!=4 or any(x.flags & rarfile.RAR_FILE_SOLID for x in targets):
        raise RuntimeError(f'Cannot independently extract expected 4 non-solid wheat members: n={len(targets)}')
    # A minimal new main header removes the original quick-open locator.
    main_body=b'\x03\x01\x00\x00'
    prefix=rarfile.RAR5_ID+struct.pack('<I',zlib.crc32(main_body))+main_body
    end_body=b'\x03\x05\x00\x00';end=struct.pack('<I',zlib.crc32(end_body))+end_body
    out=DATA/'mirca_monthly_wheat_2020';out.mkdir(exist_ok=True)
    extracted=[]
    for x in targets:
        block=f.fetch(x.header_offset,x.data_offset+x.compress_size-1)
        if block[:x.header_size]!=cached_headers[x.filename]:raise RuntimeError('Original and cached member headers differ')
        container=out/(Path(x.filename).stem+'.member.rar');container.write_bytes(prefix+block+end)
        subprocess.run(['/usr/bin/tar','-xf',str(container),'-C',str(out)],check=True)
        path=out/x.filename;b=path.read_bytes()
        if len(b)!=x.file_size or zlib.crc32(b)!=x.CRC:raise RuntimeError('Extracted member CRC/size mismatch: '+x.filename)
        extracted.append(dict(filename=x.filename,local_path=str(path.relative_to(ROOT)),bytes=len(b),
            crc32=f'{x.CRC:08x}',sha256=hashlib.sha256(b).hexdigest(),
            header_offset=x.header_offset,data_offset=x.data_offset,packed_bytes=x.compress_size,
            derived_member_container_sha256=hashlib.sha256(container.read_bytes()).hexdigest()))
        print(json.dumps(extracted[-1]),flush=True)
    receipt=dict(source=SOURCE,url=URL,acquisition='HTTP 206 ranges on advertised public non-solid archive',
        entire_archive_downloaded=False,entire_archive_checksum_verified=False,
        range_receipts=f.requests,members=extracted,license='CC BY 4.0',repository='https://www.hydroshare.org/resource/e4582ca0042148338bb5e0148b749ed6/')
    (DATA/'mirca_monthly_wheat_download_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(members=len(extracted),range_requests=len(f.requests),downloaded_bytes=sum(x['bytes'] for x in f.requests))),flush=True)

if __name__=='__main__':main()
