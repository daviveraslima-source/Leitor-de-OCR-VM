"""Execucao isolada de segunda leitura (V4) das fotos ja armazenadas no Railway.
Nunca altera dataset.zip, catalog.json ou results.jsonl da primeira leitura.
Nomes de arquivos e fotos permanecem em volume privado.
"""
import os,json,time,zipfile,threading,traceback
from pathlib import Path
from io import BytesIO
from flask import jsonify,Response,request
from app import app,ROOT,TOKEN,load_catalog
from reader_v2 import read_v2

RESULTS=ROOT/"results_v4.jsonl"
STATUS=ROOT/"status_v4.json"
INPUT=ROOT/"dataset.zip"
GUARD=threading.Lock()
THREAD=None
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,path)

def get_status():
    if STATUS.exists():
        try:return json.loads(STATUS.read_text("utf-8"))
        except Exception:pass
    return {"stage":"not_started","processed":0,"total":0}

def job():
    state={"stage":"starting","processed":0,"total":0,"started_at":time.time(),"version":"V4","source":"dataset.zip","output":"results_v4.jsonl"}
    try:
        if not INPUT.is_file():raise FileNotFoundError("dataset.zip nao esta no volume de dados")
        refs,colors=load_catalog()
        if not refs:raise ValueError("catalog.json vazio ou ausente")
        old={}
        original=ROOT/"results.jsonl"
        if original.exists():
            for line in original.read_text("utf-8").splitlines():
                try:
                    r=json.loads(line)
                    old[r.get("image")]=r
                except Exception:pass
        completed=set()
        if RESULTS.exists():
            for line in RESULTS.read_text("utf-8").splitlines():
                try:completed.add(json.loads(line).get("image"))
                except Exception:pass
        with zipfile.ZipFile(INPUT) as archive:
            entries=[x for x in archive.infolist() if not x.is_dir() and x.filename.lower().endswith((".jpg",".jpeg",".png",".webp"))]
            if len(entries)>5000:raise ValueError("Muitas imagens no ZIP")
            state.update(stage="processing",processed=len(completed),total=len(entries),catalog_refs=len(refs),already_done=len(completed))
            atomic_json(STATUS,state)
            with RESULTS.open("a",encoding="utf-8") as out:
                for i,entry in enumerate(entries):
                    name=Path(entry.filename).name
                    if name in completed:continue
                    started=time.monotonic()
                    try:
                        with archive.open(entry) as image:
                            raw=image.read(30_000_001)
                        if len(raw)>30_000_000:raise ValueError("Imagem acima de 30 MB")
                        r=read_v2(raw,refs,colors,old=old.get(name))
                    except Exception as exc:
                        r={"REF":None,"COR":None,"status":"ERRO","error":str(exc)[:200]}
                    row={"image":name,"zip_index":i,**r}
                    out.write(json.dumps(row,ensure_ascii=False)+"\n")
                    out.flush()
                    completed.add(name)
                    state.update(processed=len(completed),last_image=name,last_seconds=round(time.monotonic()-started,1),last_update=time.time())
                    if len(completed)%5==0 or len(completed)==len(entries):
                        atomic_json(STATUS,state)
                        print("GREENISH_V4_PROGRESS "+json.dumps({k:state.get(k) for k in ("processed","total","last_seconds","last_image")}),flush=True)
        state.update(stage="completed",completed_at=time.time())
        atomic_json(STATUS,state)
        print("GREENISH_V4_COMPLETED "+json.dumps({"processed":state["processed"],"total":state["total"]}),flush=True)
    except Exception as exc:
        state.update(stage="failed",error=str(exc),trace=traceback.format_exc()[-750:])
        atomic_json(STATUS,state)
        print("GREENISH_V4_FAILED "+str(exc),flush=True)

def auth():
    return TOKEN and request.headers.get("Authorization","")==("Bearer "+TOKEN)

@app.get("/status-v4")
def status_v4():
    if not auth():return jsonify({"error":"unauthorized"}),401
    return jsonify(get_status())

@app.get("/results-v4")
def results_v4():
    if not auth():return jsonify({"error":"unauthorized"}),401
    if not RESULTS.exists():return jsonify({"error":"not_ready"}),404
    return Response(RESULTS.read_bytes(),mimetype="application/x-ndjson",headers={"Content-Disposition":'attachment; filename="greenish_resultados_v4.jsonl"'})

@app.post("/start-v4")
def start_v4():
    global THREAD
    if not auth():return jsonify({"error":"unauthorized"}),401
    with GUARD:
        if THREAD and THREAD.is_alive():return jsonify({"stage":"already_processing"}),409
        THREAD=threading.Thread(target=job,daemon=True)
        THREAD.start()
    return jsonify({"stage":"started"}),202

if __name__=="__main__":
    from waitress import serve
    if os.environ.get("AUTO_V4","")=="1":
        with GUARD:
            THREAD=threading.Thread(target=job,daemon=True)
            THREAD.start()
    serve(app,host="0.0.0.0",port=int(os.environ.get("PORT","8080")))
