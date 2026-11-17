"""Leitor contextual V2: nova OCR independente; JSONL anterior serve só como pista."""
import re,time
from io import BytesIO
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image,ImageOps
import pytesseract
from rapidfuzz import process,fuzz

EXPLICIT_REF=re.compile(r'\bR[E3]F\s*[:=._-]?\s*([A-Z0-9@©$]{5,24})')
ANY_REF=re.compile(r'(?<![A-Z0-9])([A-Z]{2,9}[A-Z0-9]{4,16})(?![A-Z0-9])')
EXPLICIT_COR=re.compile(r'\bC[O0Q]R\s*[:=._-]?\s*([0-9OQDBSILZG]{4,5})(?![0-9])')
NUM_NAME=re.compile(r'(?<!\d)(\d{4})\s*[-–:]\s*([A-Z]{3,24})')
CODE4=re.compile(r'\b(\d{4})\b')
DIGITS=str.maketrans({'O':'0','Q':'0','D':'0','B':'8','S':'5','I':'1','L':'1','Z':'2','G':'6','@':'0'})

def regions(raw):
    img=ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
    img.thumbnail((1950,1950))
    rgb=np.asarray(img);h,w=rgb.shape[:2]
    sc=min(1,1400/max(h,w))
    sm=cv2.resize(rgb,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc<1 else rgb
    hsv=cv2.cvtColor(sm,cv2.COLOR_RGB2HSV)
    white=np.uint8((hsv[:,:,2]>120)&(hsv[:,:,1]<110))*255
    k=max(3,round(min(sm.shape[:2])*.008))
    white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)),iterations=1)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(white,8)
    rects=[]
    for x,y,ww,hh,area in stats[1:]:
        A=ww*hh
        if A<.018*sm.shape[0]*sm.shape[1] or A>.78*sm.shape[0]*sm.shape[1] or ww<65 or hh<35 or area/A<.44:continue
        rects.append((A,(max(0,int((x-15)/sc)),max(0,int((y-15)/sc)),min(w,int((x+ww+15)/sc)),min(h,int((y+hh+15)/sc)))))
    rects.sort(reverse=True)
    crops=[('full',img)]
    for i,(_,bb) in enumerate(rects[:2]):
        patch=img.crop(bb)
        if min(patch.size)<40:continue
        crops.append((f'panel{i}',patch))
        pw,ph=patch.size
        crops.append((f'paneltop{i}',patch.crop((0,0,pw,int(ph*.52)))))
    if len(crops)==1 and max(w,h)>900:crops.append(('center',img.crop((int(w*.12),int(h*.10),int(w*.88),int(h*.90)))))
    return crops

def ocr_lines(raw):
    lines=[]
    variants=[]
    for name,im in regions(raw):
        for angle in (0,90,270):
            rot=im.rotate(angle,expand=True) if angle else im
            variants.append((name,angle,rot,'--psm 11' if name=='full' else '--psm 6'))
    for name,angle,im,config in variants[:12]:
        g=ImageOps.autocontrast(ImageOps.grayscale(im))
        if max(g.size)<650:
            f=min(3.0,900/max(g.size))
            g=g.resize((round(g.width*f),round(g.height*f)),Image.Resampling.LANCZOS)
        try:txt=pytesseract.image_to_string(g,config=config,timeout=16).upper()
        except Exception:continue
        for l in txt.splitlines():
            l=' '.join(l.split()).strip()
            if l:lines.append((l,name,angle))
    return lines

def read_v2(raw,refs,colors,old=None):
    old=old or {}
    st=time.monotonic()
    try:
        lines=ocr_lines(raw)
        counts=defaultdict(float);ev=defaultdict(list)
        # Restrição apenas dos quatro primeiros dígitos depois das letras.
        # Não procurar "a referência completa mais parecida" no catálogo.
        from ref_group_rules import build_group_index, read_group
        groups, suffix_lengths=build_group_index(tuple(sorted(refs)))
        def put(raw,weight,where):
            raw=re.sub('[^A-Z0-9@©$]','',raw.upper())
            if not raw or not any(c.isdigit() for c in raw):return
            info=read_group(raw,groups,suffix_lengths)
            if info and info["REF"]:
                value=info["REF"]  # Todos os dígitos finais vieram do OCR.
                adjusted=info["status"]=="GRUPO_AJUSTADO_REVISAR"
                score=weight * (.65 if adjusted else 1.0)
                counts[value]+=score
                ev[value].append(("grupo_ajustado_revisar" if adjusted else "grupo_valido",where))
            elif raw in refs: # Produtos especiais fora do padrão de 4 dígitos.
                counts[raw]+=weight
                ev[raw].append(("referencia_excepcional",where))
        for line,name,angle in lines:
            where=f'{name}:{angle}'
            for r in EXPLICIT_REF.findall(line):put(r,6,where)
            for r in ANY_REF.findall(line):put(r,2.5,where)
            for r in EXPLICIT_REF.findall(line.replace(' ','').replace(':','')):put(r,4,where)
        if old.get('REF') in counts:counts[old['REF']]+=.3
        ropts=sorted(counts,key=lambda v:counts[v],reverse=True)
        ref=ropts[0] if ropts else None
        cc=defaultdict(float);cv=defaultdict(list)
        refcolors=colors.get(ref or '',{})
        for line,name,angle in lines:
            where=f'{name}:{angle}'
            for raw_cor in EXPLICIT_COR.findall(line):
                c=raw_cor.translate(DIGITS)
                if len(c)==4 and c.isdigit():
                    cc[c]+=6 if c in refcolors else 4;cv[c].append(('explicit',where))
            for c,colorname in NUM_NAME.findall(line):
                if not refcolors or c in refcolors:
                    cc[c]+=3;cv[c].append(('numeric_named',where))
            if any(w in line for w in ('COR','PRETO','BRANCO','NATURAL','CHUMBO')):
                for c in CODE4.findall(line):
                    if refcolors and c not in refcolors:continue
                    cc[c]+=1.5;cv[c].append(('context',where))
        copts=sorted(cc,key=lambda v:cc[v],reverse=True)
        cor=copts[0] if copts else None
        pairs=[(r,c) for r in ropts[:5] for c in copts[:5] if c in colors.get(r,{})]
        if pairs:
            ref,cor=max(pairs,key=lambda p:counts[p[0]]+cc[p[1]])
        confirmed_catalog=bool(ref and cor and cor in colors.get(ref,{}))
        if not ref and not cor:status='SEM_LEITURA'
        elif not ref or not cor:status='INCOMPLETO_REVISAR'
        elif confirmed_catalog:status='CANDIDATO_CATALOGO_REVISAR'
        else:status='DIVERGENCIA_REVISAR'
        return {'REF':ref,'COR':cor,'status':status,'catalog_pair':confirmed_catalog,
                'prior_REF':old.get('REF'),'prior_COR':old.get('COR'),
                'candidate_REF':ropts[:4],'candidate_COR':copts[:4],
                'ref_evidence':ev.get(ref,[])[:8],'cor_evidence':cv.get(cor,[])[:8],
                'seconds':round(time.monotonic()-st,1)}
    except Exception as exc:
        return {'REF':None,'COR':None,'status':'ERRO','error':str(exc)[:180]}
