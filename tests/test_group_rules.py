"""Testes de regressão da regra prefixo + 4 dígitos + sufixo visual.
Usa dados sintéticos; catálogo e fotos privados não entram no GitHub.
"""
from ref_group_rules import build_group_index,read_group,scan_patterns

sample=("CAM01022060","CAM03102060","CAMV0102508","CAMV2008012",
        "BER10011889","BER70010077","BERV1001120","BERV7001025",
        "SHO10020446","SHOV1002015","LINUS2")
groups,lengths=build_group_index(tuple(sorted(sample)))
def ck(v):
    return read_group(v,groups,lengths)

assert ck("CAM01022060")["REF"]=="CAM01022060"
assert ck("CAM03102060")["group"]=="0310"
assert ck("CAMV0102508")["family"]=="CAMV"
assert ck("BERV7001025")["group"]=="7001"
assert ck("BER1@011889")["REF"]=="BER10011889"
assert ck("BER1@011889")["status"]=="GRUPO_AJUSTADO_REVISAR"
assert ck("CAM0102")["REF"] is None
assert ck("CAM0102")["status"]=="SUFIXO_INCOMPLETO_REVISAR"
assert ck("CAM04012060")["REF"] is None
assert ck("CAM04012060")["status"]=="SUBGRUPO_NOVO_REVISAR"
assert ck("CAM0310ABCD")["REF"] is None
assert ck("LINUS2") is None # Exceções têm tratamento separado no OCR principal.
assert any(x["REF"]=="CAMV2008012" for x in scan_patterns("REF CAMV2008012 COR 2000",sample))
print("Testes de padrões Greenish: OK")
