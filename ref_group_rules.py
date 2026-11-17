"""Regra Greenish: letras da família + quatro dígitos do subgrupo + final variável.
O catálogo é usado apenas para aprender a lista de SUBGRUPOS por família.
NUNCA completar o sufixo a partir de uma referência parecida.
"""
import re
from collections import defaultdict, Counter
from functools import lru_cache

VALID_CODE = re.compile(r'^([A-Z]{2,12})(\d{4})(\d*)$')
TOKEN = re.compile(r'(?<![A-Z0-9])([A-Z]{2,12}[A-Z0-9@©$]{4,18})(?![A-Z0-9])')
DIGIT_CONFUSIONS = str.maketrans({
    'O':'0','Q':'0','D':'0','@':'0','©':'0',
    'I':'1','L':'1','Z':'2','S':'5','$':'5','B':'8','G':'6'
})

@lru_cache(maxsize=2)
def build_group_index(ref_tuple):
    groups=defaultdict(Counter)
    suffix_lengths=defaultdict(Counter)
    for r in ref_tuple:
        m=VALID_CODE.fullmatch(r)
        if m:
            family,group,suffix=m.groups()
            groups[family][group]+=1
            suffix_lengths[(family,group)][len(suffix)]+=1
    return dict(groups),dict(suffix_lengths)

def read_group(token, groups, suffix_lengths):
    """Só normaliza caracteres ambíguos nos QUATRO dígitos do subgrupo."""
    token=re.sub(r'[\s.:;/_-]+','',token.upper())
    family=next((f for f in sorted(groups,key=len,reverse=True) if token.startswith(f)),None)
    if not family:return None
    tail=token[len(family):]
    possible=sorted(groups[family],key=lambda g:(-groups[family][g],g))
    if len(tail)<4:return dict(token=token,family=family,group=None,REF=None,status='SUBGRUPO_INCOMPLETO',possible_groups=possible)
    rawgroup=tail[:4]
    group=rawgroup.translate(DIGIT_CONFUSIONS)
    suffix=tail[4:]
    if not group.isdigit():
        return dict(token=token,family=family,group=None,REF=None,status='SUBGRUPO_ILEGIVEL',possible_groups=possible)
    if group not in groups[family]:
        return dict(token=token,family=family,group=group,REF=None,status='SUBGRUPO_NOVO_REVISAR',possible_groups=possible)
    if not suffix or not suffix.isdigit() or len(suffix) not in suffix_lengths.get((family,group),{}):
        return dict(token=token,family=family,group=group,REF=None,status='SUFIXO_INCOMPLETO_REVISAR',possible_groups=[group])
    REF=family+group+suffix
    status='GRUPO_AJUSTADO_REVISAR' if rawgroup!=group else 'GRUPO_LIDO'
    return dict(token=token,family=family,group=group,REF=REF,status=status,possible_groups=[group])

def scan_patterns(text,refs):
    groups,lengths=build_group_index(tuple(sorted(refs)))
    outputs=[]
    seen=set()
    for raw in TOKEN.findall(str(text).upper()):
        if raw in seen:continue
        seen.add(raw)
        info=read_group(raw,groups,lengths)
        if info:outputs.append(info)
    return outputs

def family_summary(refs):
    groups,_=build_group_index(tuple(sorted(refs)))
    return {f:dict(c) for f,c in groups.items()}
