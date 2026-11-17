"""Greenish REF/COR OCR runner. Public code; private upload and results."""
import os, re, json, time, zipfile, threading, tempfile, shutil, traceback
from collections import Counter
from pathlib import Path
from flask import Flask, request, jsonify, Response
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from rapidfuzz import process, fuzz

app = Flask(__name__)
ROOT = Path(os.getenv("GREENISH_DATA_DIR", "/data/greenish_ocr"))
ROOT.mkdir(parents=True, exist_ok=True)
TOKEN = os.getenv("GREENISH_STATUS_TOKEN", "")
STATE = {"stage":"awaiting_dataset", "processed":0, "total":0, "started_at":None}
LOCK = threading.Lock()
REF_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,9}[A-Z0-9]{1,16})(?![A-Z0-9])")
TAG_REF = re.compile(r"R[E3]?F\s*[:.=/-]?\s*([A-Z0-9]{4,24})")
TAG_COR = re.compile(r"C[O0]R\s*[:.=/-]?\s*([0-9OQDBSILZG]{4})")
CODE4 = re.compile(r"(?<![0-9])([0-9]{4})(?![0-9])")
CONV = str.maketrans({"O":"0","Q":"0","D":"0","B":"8","S":"5","I":"1","L":"1","Z":"2","G":"6"})
MAX_UPLOAD = 400 * 1024 * 1024

def authorized():
    header = request.headers.get("Authorization", "")
    return bool(TOKEN) and header == "Bearer " + TOKEN

def protect():
    if not authorized():
        return jsonify({"error":"unauthorized"}), 401

def load_catalog():
    p = ROOT / "catalog.json"
    if not p.exists():
        return set(), {}
    try:
        j=json.loads(p.read_text("utf-8"))
        return set(j.get("refs",[])), j.get("colors",{})
    except Exception:
        return set(), {}

def read_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((2300,2300))
    out=[]
    for rotation in (0, 90, 270):
        rot=img.rotate(rotation,expand=True) if rotation else img
        gray=ImageOps.autocontrast(ImageOps.grayscale(rot))
        for variant in (gray, ImageEnhance.Contrast(gray).enhance(2)):
            try:
                out.append(pytesseract.image_to_string(variant,config="--psm 11",timeout=17).upper())
            except RuntimeError:
                pass
    return "\n".join(out)

def candidates(text, refs, colors):
    rawrefs=TAG_REF.findall(text)
    rawrefs.extend(t for t in REF_RE.findall(text) if len(t)>5)
    known=[]
    for tok in rawrefs:
        tok=tok.strip().upper()
        if tok in refs: known.append((tok, "exact"))
        elif refs:
            family=tok[:3]
            subset=(r for r in refs if r.startswith(family) and abs(len(r)-len(tok))<=1)
            match=process.extractOne(tok,subset,scorer=fuzz.ratio,score_cutoff=93)
            if match: known.append((match[0],"suggestion"))
        else: known.append((tok, "unverified"))
    rawcolors=[c.translate(CONV) for c in TAG_COR.findall(text)]
    ref=Counter(r for r,k in known if k=="exact").most_common(1)
    if not ref:
        ref=Counter(r for r,k in known).most_common(1)
    ref=ref[0][0] if ref else None
    cor=Counter(c for c in rawcolors if len(c)==4 and c.isdigit()).most_common(1)
    cor=cor[0][0] if cor else None
    if not cor and ref and ref in colors:
        # Without visually read 4 digits, never fill them in from the catalog.
        pass
    status="REVISAR"
    if ref and cor and (not refs or (ref in refs and cor in colors.get(ref,{}))):
        if (ref, "exact") in known and cor in TAG_COR.findall(text):
            status="CANDIDATO_EXATO"
        else: status="VERIFICAR_VISUALMENTE"
    return {"REF":ref,"COR":cor,"status":status,"raw_ref_candidates":rawrefs[:12],"raw_color_candidates":rawcolors[:12]}

def run_job(zip_path):
    try:
        with LOCK: STATE.update(stage="reading_zip",processed=0,total=0,started_at=time.time())
        refs, colors=load_catalog()
        with zipfile.ZipFile(zip_path) as z:
            entries=[x for x in z.infolist() if x.filename.lower().endswith((".jpg",".jpeg",".png",".webp")) and not x.is_dir()]
            if len(entries)>5000 or sum(x.file_size for x in entries)>3_000_000_000: raise ValueError("ZIP too large")
            with LOCK: STATE.update(stage="processing",total=len(entries),catalog_refs=len(refs))
            out=(ROOT/"results.jsonl").open("w",encoding="utf-8")
            try:
                for i,entry in enumerate(entries):
                    if entry.file_size>30_000_000:
                        result={"REF":None,"COR":None,"status":"SKIPPED","error":"file too large"}
                    else:
                        try:
                            from io import BytesIO
                            with z.open(entry) as src:
                                blob=src.read(30_000_001)
                            if len(blob)>30_000_000:raise ValueError("image too large")
                            result=candidates(read_image(BytesIO(blob)),refs,colors)
                        except Exception as e:
                            result={"REF":None,"COR":None,"status":"ERRO","error":str(e)[:180]}
                    out.write(json.dumps({"image":Path(entry.filename).name,**result},ensure_ascii=False)+"\n")
                    out.flush()
                    with LOCK: STATE.update(processed=i+1,last_image=Path(entry.filename).name)
            finally: out.close()
        with LOCK: STATE.update(stage="completed",completed_at=time.time())
    except Exception as e:
        with LOCK: STATE.update(stage="failed",error=str(e),trace=traceback.format_exc()[-900:])

@app.get("/health")
def health(): return jsonify({"ok":True})

@app.get("/")
def home():
    return Response("""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Greenish OCR</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:720px;margin:38px auto;padding:0 20px;background:#fafcf9;color:#153b2c}section{background:white;border:1px solid #dce8df;border-radius:12px;padding:20px;margin:18px 0}input,button{display:block;margin:12px 0;width:100%;box-sizing:border-box;padding:12px;font:inherit}button{cursor:pointer;border:0;border-radius:8px;background:#15573d;color:white}small{color:#53655b}pre{white-space:pre-wrap;word-break:break-word}</style>
<h1>Greenish — OCR REF / COR</h1><p>Envie o ZIP de etiquetas para analisar REF e COR. O código deste serviço é público; <strong>suas fotos e catálogo não são publicados no GitHub.</strong></p>
<section><label>Token de acesso (Railway → Variables → GREENISH_STATUS_TOKEN)<input id="token" type="password" placeholder="Cole o token de acesso"></label>
<label>Catálogo JSON (opcional, melhora validação)<input id="catalog" type="file" accept=".json"></label><button onclick="sendCatalog()">Enviar catálogo</button>
<label>ZIP com as fotos<input id="zip" type="file" accept=".zip"></label><button onclick="sendZip()">Enviar ZIP e iniciar processamento</button>
<small>Os dados serão enviados diretamente ao serviço protegido. Imagens sem leitura segura ficam marcadas para revisão.</small>
<progress id="progress" max="100" value="0" style="width:100%"></progress><div id="message"></div></section>
<section><button onclick="status()">Consultar andamento</button><button onclick="results()">Baixar resultados (JSONL)</button><pre id="state">Aguardando...</pre></section>
<script>
const token=()=>document.getElementById("token").value.trim();
const auth=()=>({"Authorization":"Bearer "+token()});
const msg=s=>document.getElementById("message").textContent=s;
async function sendCatalog(){let f=document.getElementById("catalog").files[0];if(!f)return msg("Escolha o catálogo.");let r=await fetch("/catalog",{method:"POST",headers:{...auth(),"Content-Type":"application/json"},body:f});msg(await r.text());}
function sendZip(){let f=document.getElementById("zip").files[0];if(!f)return msg("Escolha o ZIP.");let xhr=new XMLHttpRequest();xhr.open("POST","/upload");xhr.setRequestHeader("Authorization","Bearer "+token());xhr.setRequestHeader("Content-Type","application/zip");xhr.upload.onprogress=e=>{if(e.lengthComputable)document.getElementById("progress").value=100*e.loaded/e.total};xhr.onload=()=>msg(xhr.responseText);xhr.onerror=()=>msg("Falha de rede no envio");xhr.send(f);msg("Enviando ZIP...");}
async function status(){let r=await fetch("/status",{headers:auth()});document.getElementById("state").textContent=await r.text();}
async function results(){let r=await fetch("/results",{headers:auth()});if(!r.ok)return msg(await r.text());let b=await r.blob(),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="greenish_ref_cor_resultados.jsonl";a.click();URL.revokeObjectURL(a.href);}
</script></html>""",mimetype="text/html")

@app.get("/status")
def status():
    unauthorized=protect()
    if unauthorized:return unauthorized
    with LOCK: return jsonify(dict(STATE))

@app.post("/catalog")
def catalog():
    unauthorized=protect()
    if unauthorized:return unauthorized
    body=request.get_data(cache=False)
    if len(body)>15_000_000:return jsonify({"error":"catalog_too_large"}),413
    j=json.loads(body.decode("utf-8"))
    if not isinstance(j.get("refs"),list) or not isinstance(j.get("colors"),dict):
        return jsonify({"error":"invalid_catalog"}),400
    (ROOT/"catalog.json").write_bytes(body)
    return jsonify({"refs":len(j["refs"])})

@app.post("/upload")
def upload():
    unauthorized=protect()
    if unauthorized:return unauthorized
    if request.content_length is None or request.content_length>MAX_UPLOAD:
        return jsonify({"error":"invalid_upload_size"}),413
    with LOCK:
        if STATE.get("stage") in ("extracting","processing"):return jsonify({"error":"busy"}),409
        STATE.update(stage="uploading",processed=0,total=0)
    zip_path=ROOT/"dataset.zip"
    try:
        with zip_path.open("wb") as out:
            while True:
                chunk=request.stream.read(1024*1024)
                if not chunk:break
                out.write(chunk)
                if out.tell()>MAX_UPLOAD:raise ValueError("upload too large")
        if not zipfile.is_zipfile(zip_path):raise ValueError("invalid zip")
    except Exception as e:
        with LOCK:STATE["stage"]="awaiting_dataset"
        return jsonify({"error":str(e)}),400
    threading.Thread(target=run_job,args=(zip_path,),daemon=True).start()
    return jsonify({"stage":"started"}),202

@app.get("/results")
def results():
    unauthorized=protect()
    if unauthorized:return unauthorized
    p=ROOT/"results.jsonl"
    if not p.exists():return jsonify({"error":"not_ready"}),404
    return Response(p.read_bytes(),mimetype="application/x-ndjson")

if __name__=="__main__":
    from waitress import serve
    serve(app,host="0.0.0.0",port=int(os.getenv("PORT","8080")))
