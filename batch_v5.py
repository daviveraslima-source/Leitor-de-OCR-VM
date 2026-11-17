"""Greenish OCR V5: second-pass focused recognition. Preserves V2/V4 predictions.
Uses family/group rules only to normalize the 4-digit subgroup; suffix remains visual.
This is NOT neural network training. Images/labels never leave private Railway volume.
"""
import os, re, io, json, time, traceback, zipfile, threading
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import cv2, numpy as np, pytesseract
from flask import jsonify, Response, request
from app import app, ROOT, TOKEN, load_catalog
from ref_group_rules import build_group_index,read_group

V5RESULT=ROOT/'results_v5.jsonl'
V5STATUS=ROOT/'status_v5.json'
ZIP=ROOT/'dataset.zip'
_ongoing=None
_lock=threading.Lock()
DIGITAL=str.maketrans({'O':'0','Q':'0','D':'0','B':'8','S':'5','I':'1','L':'1','Z':'2','G':'6','@':'0'})
REF_LABEL=re.compile(r'(?:\bR\s*[E3]\s*F\b|\bREF\b)\s*[:=._/-]*\s*([A-Z]{2,12}[A-Z0-9@©$]{4,20})')
COR_LABEL=re.compile(r'\bC\s*[O0Q]\s*R\b\s*[:=._/-]*\s*([0-9OQDBSILZG@]{4})')
REF_TOKEN=re.compile(r'(?<![A-Z0-9])([A-Z]{2,12}[A-Z0-9@©$]{4,20})(?![A-Z0-9])')
COR_ALIGNED=re.compile(r'\bCOR\s*[:=._/-]*\s*(\d{4})(?!\d)')
SPECIAL=re.compile(r'(?<![A-Z0-9])((?:GR\d{2,3}|LINUS(?:INF\d|SHBG\d|\d)?))(?![A-Z0-9])')

def writejson(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)

def load_previous(file):
    old={}
    if file.exists():
        for line in file.read_text('utf-8').splitlines():
            try:
                record=json.loads(line); old[record['image']]=record
            except Exception: pass
    return old

def recognize_roi(blob):
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(blob))).convert('RGB')
    im.thumbnail((2100,2100),Image.Resampling.LANCZOS)
    rgb=np.array(im); h,w=rgb.shape[:2]
    scale=min(1.0,1050/max(h,w))
    sm=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else rgb
    hsv=cv2.cvtColor(sm,cv2.COLOR_RGB2HSV)
    mask=np.uint8((hsv[:,:,1]<125)&(hsv[:,:,2]>110))*255
    k=max(3,int(min(sm.shape[:2])*.012))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)),iterations=1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    scored=[]
    for ct in cnts:
        x,y,bw,bh=cv2.boundingRect(ct)
        area=bw*bh; fraction=area/(sm.shape[0]*sm.shape[1])
        if not (.035<=fraction<=.90):continue
        if bw<100 or bh<85 or not (.26<bw/bh<2.25):continue
        fill=cv2.contourArea(ct)/max(area,1)
        if fill<.46:continue
        score=fraction*fill
        pad=int(.025*max(bw,bh))
        xy=(max(0,round((x-pad)/scale)),max(0,round((y-pad)/scale)),min(w,round((x+bw+pad)/scale)),min(h,round((y+bh+pad)/scale)))
        if xy[2]-xy[0]>100 and xy[3]-xy[1]>95:scored.append((score,xy))
    scored.sort(reverse=True)
    regions=[]
    for index,(_,rect) in enumerate(scored[:2]):
        reg=im.crop(rect);rw,rh=reg.size
        regions.append((f'label{index}_top',reg.crop((0,0,rw,int(rh*.62)))))
        regions.append((f'label{index}_whole',reg))
    # Fallback to center and upper portion, including tags on white apparel.
    regions.append(('center_top',im.crop((round(w*.12),round(h*.07),round(w*.88),round(h*.62)))))
    if not scored:regions.append(('full',im))
    # Avoid running 10+ OCR variants for the same region.
    regions=regions[:5]
    lines=[]
    for idx,(name,reg) in enumerate(regions):
        rot=reg
        gray=ImageOps.autocontrast(ImageOps.grayscale(rot),cutoff=1)
        if max(gray.size)<1500:
            fac=min(3.0,1500/max(gray.size))
            gray=gray.resize((max(1,round(gray.width*fac)),max(1,round(gray.height*fac))),Image.Resampling.LANCZOS)
        variants=[('contrast',ImageEnhance.Contrast(gray).enhance(1.7),'--psm 6')]
        if idx<2:
            variants.append(('threshold',gray.point(lambda pix:255 if pix>160 else 0),'--psm 6'))
        if name=='full':variants=[('sparse',gray,'--psm 11')]
        for typ,image,conf in variants:
            try:
                txt=pytesseract.image_to_string(image,config=conf,timeout=14).upper()
            except (RuntimeError,Exception):continue
            for ln in txt.splitlines():
                ln=' '.join(ln.split()).strip()
                if ln:lines.append((ln,name,typ))
    return lines

def extract_candidates(lines,refs,groups,lengths):
    reference=defaultdict(float);color=defaultdict(float);reasons_r=defaultdict(list);reasons_c=defaultdict(list)
    for ln,region,variant in lines:
        source=f'{region}/{variant}'
        for match in REF_LABEL.finditer(ln):
            raw=match.group(1)
            res=read_group(raw,groups,lengths)
            if res and res.get('REF'):
                v=res['REF'];reference[v]+=3.5;reasons_r[v].append(f'REF:{source}')
            elif raw in refs:
                reference[raw]+=3.5;reasons_r[raw].append(f'REF_ESPECIAL:{source}')
        for raw in REF_TOKEN.findall(ln):
            res=read_group(raw,groups,lengths)
            if res and res.get('REF'):
                v=res['REF'];reference[v]+=1.0;reasons_r[v].append(f'TOKEN:{source}')
        for raw in SPECIAL.findall(ln):
            if raw in refs:reference[raw]+=2.5;reasons_r[raw].append(f'ESPECIAL:{source}')
        for mat in COR_LABEL.finditer(ln):
            raw=mat.group(1).translate(DIGITAL)
            if len(raw)==4 and raw.isdigit():
                color[raw]+=4.0;reasons_c[raw].append(f'COR:{source}')
    ref_sorted=sorted(reference,key=lambda key:reference[key],reverse=True)
    cor_sorted=sorted(color,key=lambda key:color[key],reverse=True)
    return {'refs':ref_sorted[:5],'cors':cor_sorted[:5],
            'ref_scores':dict(reference),'cor_scores':dict(color),
            'ref_sources':dict(reasons_r),'cor_sources':dict(reasons_c)}

def merge_results(v4,v2,cands):
    oldref=v4.get('REF');oldcor=v4.get('COR')
    refs=cands['refs'];cors=cands['cors']
    rr=cands['ref_scores'];cs=cands['cor_scores']
    newref=refs[0] if refs else None; newcor=cors[0] if cors else None
    # High confidence only when text was found by several image treatments.
    nr=rr.get(newref,0);nc=cs.get(newcor,0)
    rref=oldref;rcor=oldcor;refsource='v4';corsource='v4'
    if newref:
        if not oldref:
            rref=newref;refsource='v5_focus_provisional'
        elif oldref==newref:refsource='v4_v5_agree'
        elif nr>=7 and (not len(refs)>1 or nr>=rr[refs[1]]*1.8):
            rref=newref;refsource='v5_focus_strong_override'
        else:refsource='v4_v5_disagreement_review'
    if newcor:
        if not oldcor:
            rcor=newcor;corsource='v5_focus_provisional'
        elif oldcor==newcor:corsource='v4_v5_agree'
        elif nc>=8 and (not len(cors)>1 or nc>=cs[cors[1]]*1.7):
            rcor=newcor;corsource='v5_focus_strong_override'
        else:corsource='v4_v5_disagreement_review'
    if rref and rcor:status='REF_COR_CANDIDATOS_REVISAR'
    elif rref or rcor:status='INCOMPLETO_REVISAR'
    else:status='SEM_LEITURA'
    if 'disagreement' in refsource or 'disagreement' in corsource:status='DIVERGENCIA_REVISAR'
    return {'REF':rref,'COR':rcor,'status':status,
        'prior_v4_REF':oldref,'prior_v4_COR':oldcor,
        'prior_v2_REF':v2.get('REF'),'prior_v2_COR':v2.get('COR'),
        'origem_REF':refsource,'origem_COR':corsource,
        'v5_ref_candidates':refs,'v5_cor_candidates':cors,
        'v5_ref_scores':{k:round(rr[k],1) for k in refs},
        'v5_cor_scores':{k:round(cs[k],1) for k in cors}}

def do_v5():
    state={'stage':'starting','processed':0,'total':0,'version':'v5','started_at':time.time(),'result':'results_v5.jsonl'}
    try:
        if not ZIP.exists():raise FileNotFoundError('dataset.zip ausente')
        refs,colors=load_catalog()
        if not refs:raise ValueError('catalog.json ausente')
        groups,lengths=build_group_index(tuple(sorted(refs)))
        prev4=load_previous(ROOT/'results_v4.jsonl')
        if len(prev4)<965:raise ValueError(f'V4 incompleta: {len(prev4)} registros; nao iniciar V5')
        prev2=load_previous(ROOT/'results.jsonl')
        done=load_previous(V5RESULT)
        with zipfile.ZipFile(ZIP) as z:
            entries=[f for f in z.infolist() if not f.is_dir() and f.filename.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
            state.update(stage='processing',processed=len(done),total=len(entries),catalog_refs=len(refs))
            writejson(V5STATUS,state)
            with V5RESULT.open('a',encoding='utf8') as out:
                for idx,f in enumerate(entries):
                    name=Path(f.filename).name
                    if name in done:continue
                    t=time.monotonic()
                    try:
                        with z.open(f) as file:
                            blob=file.read(30_000_001)
                        if len(blob)>30_000_000:raise ValueError('Imagem maior que 30MB')
                        lines=recognize_roi(blob)
                        cand=extract_candidates(lines,refs,groups,lengths)
                        value=merge_results(prev4.get(name,{}),prev2.get(name,{}),cand)
                        value['evidencias_texto_v5']=[{'linha':line[:110],'regiao':where} for line,where,_ in lines[:15]]
                    except Exception as exc:
                        value={'REF':prev4.get(name,{}).get('REF'),'COR':prev4.get(name,{}).get('COR'),'status':'ERRO_V5_PRESERVA_V4','error':str(exc)[:150]}
                    row={'image':name,'zip_index':idx,**value}
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                    done[name]=row;state.update(processed=len(done),last_image=name,last_seconds=round(time.monotonic()-t,1),last_update=time.time())
                    if len(done)%5==0 or len(done)==len(entries):
                        writejson(V5STATUS,state)
                        print('GREENISH_V5_PROGRESS '+json.dumps({'processed':len(done),'total':len(entries),'image':name}),flush=True)
        state.update(stage='completed',completed_at=time.time());writejson(V5STATUS,state)
        print('GREENISH_V5_COMPLETED '+json.dumps({'processed':state['processed'],'total':state['total']}),flush=True)
    except Exception as exc:
        state.update(stage='failed',error=str(exc),trace=traceback.format_exc()[-900:]);writejson(V5STATUS,state)
        print('GREENISH_V5_FAILED '+str(exc),flush=True)

def allowed():return TOKEN and request.headers.get('Authorization','')=='Bearer '+TOKEN
@app.get('/status-v5')
def statusv5():
    if not allowed():return jsonify({'error':'unauthorized'}),401
    return jsonify(json.loads(V5STATUS.read_text('utf8')) if V5STATUS.exists() else {'stage':'not_started','total':965,'processed':0})
@app.get('/results-v5')
def resultsv5():
    if not allowed():return jsonify({'error':'unauthorized'}),401
    if not V5RESULT.exists():return jsonify({'error':'not_ready'}),404
    return Response(V5RESULT.read_bytes(),mimetype='application/x-ndjson',headers={'Content-Disposition':'attachment; filename="greenish_resultados_v5.jsonl"'})
@app.post('/start-v5')
def startv5():
    global _ongoing
    if not allowed():return jsonify({'error':'unauthorized'}),401
    with _lock:
        if _ongoing and _ongoing.is_alive():return jsonify({'error':'already_processing'}),409
        _ongoing=threading.Thread(target=do_v5,daemon=True);_ongoing.start()
    return jsonify({'stage':'started'}),202
if __name__=='__main__':
    from waitress import serve
    if os.environ.get('AUTO_V5')=='1':
        _ongoing=threading.Thread(target=do_v5,daemon=True);_ongoing.start()
    serve(app,host='0.0.0.0',port=int(os.environ.get('PORT','8080')))