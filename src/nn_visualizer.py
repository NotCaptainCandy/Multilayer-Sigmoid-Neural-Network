"""nn_visualizer.py
~~~~~~~~~~~~~~~~~~~

An interactive, 3Blue1Brown-style visualizer for the feedforward MNIST
network defined in ``network2.py``.  It lets you:

  * watch the test image flow through the network,
  * see each neuron's activation as its brightness,
  * see every weight as a blue (negative) -> red (positive) edge,
  * click any neuron to view its incoming weights as an image grid
    (for the first hidden layer that is the classic 28x28 "what this
    neuron looks for" picture),
  * scrub through the MNIST test set or draw your own digit,
  * TRAIN new models from the GUI (architecture / epochs / eta / lambda /
    amount of all-zero-target noise) and switch between them from a
    dropdown of every model trained or cached on disk.

This file does NOT modify any existing file.  It reuses ``network2`` for
training and ``../data/mnist.pkl.gz`` for data.  Trained models are cached
next to this script (``viz_model_<arch>_noise<N>.json``) so they reappear
in the dropdown on later runs.

Run it (from anywhere):

    uv run python src/nn_visualizer.py

Flags only set the *default* model trained when no cache exists yet; once
running, everything is controllable from the browser.
"""

import argparse
import glob
import gzip
import json
import os
import pickle
import queue
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so we can import the sibling network2 module
import network2  # noqa: E402  (reused, never modified)

DATA_PATH = os.path.join(HERE, "..", "data", "mnist.pkl.gz")


def cache_path(arch, n_noise):
    """A config-specific cache file so each model persists across runs."""
    tag = "-".join(str(s) for s in arch)
    return os.path.join(HERE, "viz_model_%s_noise%d.json" % (tag, n_noise))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
RAW_TR = RAW_VA = RAW_TE = None          # raw (images, labels) tuples
VAL_X = VAL_Y = TEST_X = TEST_Y = None   # (784, N) / (N,) for fast accuracy


def load_data():
    global RAW_TR, RAW_VA, RAW_TE, VAL_X, VAL_Y, TEST_X, TEST_Y
    with gzip.open(DATA_PATH, "rb") as f:
        RAW_TR, RAW_VA, RAW_TE = pickle.load(f, encoding="latin1")
    VAL_X = np.asarray(RAW_VA[0]).T
    VAL_Y = np.asarray(RAW_VA[1])
    TEST_X = np.asarray(RAW_TE[0]).T
    TEST_Y = np.asarray(RAW_TE[1])


def _vectorized(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def acc_vec(weights, biases, X, Y):
    """Vectorised accuracy: a fast forward pass over all columns of X."""
    a = X
    for w, b in zip(weights, biases):
        a = 1.0 / (1.0 + np.exp(-(w @ a + b)))
    return float((a.argmax(0) == Y).mean())


def make_noise(n, seed):
    """n random-noise images (uniform [0,1]) with the all-zero target."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    z = np.zeros((10, 1))
    return [(rng.random((784, 1), dtype=np.float32), z) for _ in range(n)]


# --------------------------------------------------------------------------
# Model registry (thread-safe) + training worker
# --------------------------------------------------------------------------
REG = {}            # id -> entry dict (holds metadata + numpy weights)
ORDER = []          # ids in display order
LOCK = threading.Lock()
JOBQ = queue.Queue()
_counter = 0


def _new_id():
    global _counter
    _counter += 1
    return "m%d" % _counter


def _meta(e):
    return {k: e[k] for k in
            ("id", "name", "sizes", "noise", "epochs", "eta", "lmbda",
             "status", "accuracy", "progress")}


def add_ready(name, sizes, noise, weights, biases, path, epochs=None, eta=None,
              lmbda=None, accuracy=None):
    mid = _new_id()
    with LOCK:
        REG[mid] = dict(id=mid, name=name, sizes=list(sizes), noise=noise,
                        epochs=epochs, eta=eta, lmbda=lmbda, status="ready",
                        accuracy=accuracy, progress=None,
                        weights=weights, biases=biases, path=path)
        ORDER.append(mid)
    return mid


def add_pending(sizes, epochs, eta, lmbda, noise):
    mid = _new_id()
    name = "%s · noise%d" % ("-".join(str(s) for s in sizes), noise)
    with LOCK:
        REG[mid] = dict(id=mid, name=name, sizes=list(sizes), noise=noise,
                        epochs=epochs, eta=eta, lmbda=lmbda, status="queued",
                        accuracy=None, progress=None, weights=None, biases=None,
                        path=cache_path(sizes, noise))
        ORDER.append(mid)
    JOBQ.put(mid)
    return mid


def delete_model(mid):
    """Remove a model from the registry and delete its cache file."""
    with LOCK:
        e = REG.get(mid)
        if e is None:
            return 404, "no such model"
        if e["status"] in ("queued", "training"):
            return 409, "cannot delete a model while it is training"
        path = e.get("path")
        del REG[mid]
        if mid in ORDER:
            ORDER.remove(mid)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as ex:
            print("Could not remove %s: %s" % (path, ex))
    print("Deleted model %s" % mid)
    return 200, "deleted"


def do_train(mid):
    with LOCK:
        e = REG[mid]
        sizes, epochs = e["sizes"], e["epochs"]
        eta, lmbda, noise = e["eta"], e["lmbda"], e["noise"]
        save_to = e["path"]
        e["status"] = "training"

    base = list(zip([x.reshape(784, 1) for x in RAW_TR[0]],
                    [_vectorized(y) for y in RAW_TR[1]]))
    data = base + make_noise(noise, seed=1)

    net = network2.Network(sizes, cost=network2.CrossEntropyCost)
    for ep in range(epochs):
        net.SGD(data, 1, 10, eta, lmbda=lmbda)
        acc = acc_vec(net.weights, net.biases, VAL_X, VAL_Y)
        with LOCK:
            e["progress"] = {"epoch": ep + 1, "epochs": epochs, "eval_acc": acc}

    test_acc = acc_vec(net.weights, net.biases, TEST_X, TEST_Y)
    with LOCK:
        e["accuracy"] = test_acc
        e["weights"] = net.weights
        e["biases"] = net.biases
        e["status"] = "ready"
    try:
        net.save(save_to)
    except Exception:
        traceback.print_exc()
    print("Trained %s -> test accuracy %.2f%%" % (e["name"], 100 * test_acc))


def train_worker():
    while True:
        mid = JOBQ.get()
        try:
            do_train(mid)
        except Exception as ex:
            traceback.print_exc()
            with LOCK:
                REG[mid]["status"] = "error"
                REG[mid]["error"] = str(ex)


def load_cached_models():
    """Populate the registry from any viz_model_*.json files on disk."""
    for path in sorted(glob.glob(os.path.join(HERE, "viz_model*.json"))):
        try:
            net = network2.load(path)
        except Exception:
            continue
        m = re.search(r"noise(\d+)", os.path.basename(path))
        noise = int(m.group(1)) if m else 0
        acc = acc_vec(net.weights, net.biases, TEST_X, TEST_Y)
        name = "%s · noise%d (cached)" % ("-".join(map(str, net.sizes)), noise)
        add_ready(name, net.sizes, noise, net.weights, net.biases, path,
                  accuracy=acc)
        print("Loaded cache %s  (%.2f%%)" % (os.path.basename(path), 100 * acc))


def model_json(mid):
    with LOCK:
        e = REG.get(mid)
        if e is None or e["weights"] is None:
            return None
        weights, biases, sizes = e["weights"], e["biases"], e["sizes"]
    return json.dumps({
        "sizes": list(sizes),
        "weights": [w.round(6).tolist() for w in weights],
        "biases": [b.round(6).tolist() for b in biases],
    })


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------
def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/models":
                with LOCK:
                    out = [_meta(REG[i]) for i in ORDER]
                self._send(json.dumps(out))
            elif u.path == "/api/model":
                mid = qs.get("id", [None])[0]
                if mid is None:
                    with LOCK:
                        ready = [i for i in ORDER if REG[i]["status"] == "ready"]
                    mid = ready[0] if ready else None
                body = model_json(mid) if mid else None
                if body is None:
                    self._send("{}", code=404)
                else:
                    self._send(body)
            elif u.path == "/api/meta":
                self._send(json.dumps({"num_test": int(len(TEST_Y))}))
            elif u.path == "/api/image":
                i = int(qs.get("i", ["0"])[0]) % len(TEST_Y)
                self._send(json.dumps({
                    "index": i, "label": int(TEST_Y[i]),
                    "pixels": np.asarray(RAW_TE[0][i]).astype(float).round(5).tolist(),
                }))
            else:
                self._send("not found", code=404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/delete":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    mid = json.loads(self.rfile.read(length) or b"{}").get("id")
                except Exception:
                    self._send(json.dumps({"ok": False, "msg": "bad request"}), code=400)
                    return
                code, msg = delete_model(mid)
                self._send(json.dumps({"ok": code == 200, "msg": msg}), code=code)
                return
            if u.path != "/api/train":
                self._send("not found", code=404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                cfg = json.loads(self.rfile.read(length) or b"{}")
                arch = [int(x) for x in cfg["arch"]]
                epochs = int(cfg["epochs"])
                eta = float(cfg["eta"])
                lmbda = float(cfg["lmbda"])
                noise = int(cfg["noise"])
                assert len(arch) >= 2 and arch[0] == 784 and arch[-1] == 10, \
                    "arch must start with 784 and end with 10"
                assert all(s > 0 for s in arch), "layer sizes must be positive"
                assert 1 <= epochs <= 200, "epochs out of range (1-200)"
                assert eta > 0, "eta must be > 0"
                assert lmbda >= 0, "lambda must be >= 0"
                assert 0 <= noise <= 200000, "noise out of range (0-200000)"
            except Exception as ex:
                self._send("Bad request: %s" % ex, "text/plain", code=400)
                return
            mid = add_pending(arch, epochs, eta, lmbda, noise)
            self._send(json.dumps({"id": mid}))

    return Handler


# --------------------------------------------------------------------------
# Front-end
# --------------------------------------------------------------------------
PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST Network Visualizer</title>
<style>
  :root{--bg:#000;--fg:#e9e9e9;--muted:#8a8a8a;--accent:#3aa6c4;
    --pos:#e25b56;--neg:#4aa3e0;--panel:#0c0c0e;--line:#222;}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font-family:Georgia,"Times New Roman",serif}
  header{padding:12px 20px;border-bottom:1px solid var(--line);
    display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  header h1{font-size:19px;margin:0;font-weight:normal}
  header .sub{color:var(--muted);font-size:13px}
  .wrap{display:flex;gap:16px;padding:14px;align-items:stretch}
  .netbox{flex:1 1 auto;min-width:0;background:#000;border:1px solid var(--line);
    border-radius:8px;padding:6px}
  canvas{display:block}
  .side{display:flex;flex-direction:column;gap:14px;width:330px;flex:0 0 330px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
  .card h2{font-size:14px;margin:0 0 10px;font-weight:normal;color:#cfcfcf}
  .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0}
  button{background:#161619;color:var(--fg);border:1px solid #2c2c30;border-radius:6px;
    padding:6px 10px;font-family:inherit;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.active{border-color:var(--accent);color:#fff}
  select,input[type=number],input[type=text]{background:#161619;color:var(--fg);
    border:1px solid #2c2c30;border-radius:6px;padding:5px 6px;font-family:inherit}
  input[type=number]{width:70px}
  input[type=range]{width:150px}
  label.ctl{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px}
  .guess{font-size:15px}.guess b{font-size:26px}
  .ok{color:#7fd17f}.bad{color:#e58181}
  .imgwrap{display:flex;gap:14px;align-items:center}
  .legend{font-size:11px;color:var(--muted);line-height:1.6}
  .swatch{display:inline-block;width:11px;height:11px;border-radius:2px;
    vertical-align:middle;margin:0 4px}
  .hint{font-size:11px;color:#666;margin-top:6px}
  .toggles{display:flex;flex-wrap:wrap;gap:6px}
  #gridTitle{font-size:12px;color:var(--muted);margin-bottom:8px;min-height:30px}
  details{margin-top:8px}summary{cursor:pointer;color:#bbb;font-size:13px}
  .form label{font-size:12px;color:var(--muted);display:flex;align-items:center;
    justify-content:space-between;gap:8px;margin:6px 0}
  #trainStatus{font-size:12px;margin-top:6px;min-height:16px}
</style>
</head>
<body>
<header>
  <h1>MNIST Neural Network &mdash; live visualizer</h1>
  <span class="sub" id="archLabel"></span>
</header>

<div class="wrap">
  <div class="netbox" id="netbox">
    <canvas id="net"></canvas>
  </div>

  <div class="side">
    <div class="card">
      <h2>Models</h2>
      <div class="row">
        <select id="modelSel" style="flex:1 1 auto;min-width:0"></select>
        <button id="refreshModels" title="refresh">&#x21bb;</button>
        <button id="deleteModel" title="delete selected model">Delete</button>
      </div>
      <div id="modelInfo" class="sub"></div>
      <details>
        <summary>Train a new model</summary>
        <div class="form">
          <label>architecture <input type="text" id="tArch" value="784,16,16,10" style="width:150px"></label>
          <label>epochs <input type="number" id="tEpochs" value="12" min="1" max="200"></label>
          <label>learning rate &eta; <input type="number" id="tEta" value="0.5" step="0.1"></label>
          <label>regularization &lambda; <input type="number" id="tLmbda" value="5.0" step="0.5"></label>
          <label>noise images <input type="number" id="tNoise" value="40000" min="0" step="5000"></label>
        </div>
        <button id="trainBtn" style="margin-top:6px">Train new model</button>
        <div id="trainStatus" class="sub"></div>
        <div class="hint">Noise images use the all-zero target so the net learns to
          keep every output low on structureless input.</div>
      </details>
    </div>

    <div class="card">
      <h2>Input &amp; prediction</h2>
      <div class="imgwrap">
        <canvas id="inImg" width="112" height="112"
                style="border:2px solid var(--accent);border-radius:4px"></canvas>
        <div>
          <div class="guess">Guess&nbsp;&rarr;&nbsp;<b id="guess">?</b></div>
          <div id="conf" class="sub" style="margin-top:6px"></div>
          <div id="truth" class="sub" style="margin-top:4px"></div>
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <button id="prev">&larr; Prev</button>
        <button id="next">Next &rarr;</button>
        <button id="rand">Random</button>
        <label class="ctl">idx <input type="number" id="idx" value="0" min="0"></label>
      </div>
      <div class="toggles" style="margin-top:8px">
        <button id="modeTest" class="active">Test set</button>
        <button id="modeDraw">Draw your own</button>
      </div>
      <div id="drawArea" style="display:none;margin-top:10px">
        <canvas id="pad" width="196" height="196"
                style="background:#000;border:1px solid #333;border-radius:4px;cursor:crosshair"></canvas>
        <div class="row"><button id="clearPad">Clear</button>
          <span class="hint">draw a digit (white on black)</span></div>
      </div>
    </div>

    <div class="card">
      <h2>Output activations</h2>
      <canvas id="bars" width="300" height="200"></canvas>
    </div>

    <div class="card">
      <h2>Incoming weights of selected neuron</h2>
      <div id="gridTitle">Click any neuron in the network.</div>
      <canvas id="grid" width="252" height="252"
              style="background:#000;border:1px solid #222;border-radius:4px"></canvas>
      <div class="legend" style="margin-top:8px">
        <span class="swatch" style="background:var(--neg)"></span>negative &nbsp;
        <span class="swatch" style="background:#000;border:1px solid #333"></span>~0 &nbsp;
        <span class="swatch" style="background:var(--pos)"></span>positive
      </div>
    </div>

    <div class="card">
      <h2>Controls</h2>
      <label class="ctl">Neuron brightness (gamma)
        <input type="range" id="gamma" min="0.2" max="1.6" step="0.05" value="0.65"></label>
      <label class="ctl">Weak-weight cutoff
        <input type="range" id="thresh" min="0" max="0.9" step="0.02" value="0"></label>
      <label class="ctl">Edge opacity
        <input type="range" id="ealpha" min="0.05" max="1" step="0.05" value="0.85"></label>
      <div class="toggles" style="margin-top:6px">
        <button id="tInput" class="active">Input&rarr;hidden edges</button>
        <button id="tContrib" class="active">Color by contribution</button>
      </div>
    </div>
  </div>
</div>

<script>
let MODEL=null, MODEL_ID=null, MODELS=[], NTEST=0;
let curIndex=0, pixels=null, trueLabel=null, acts=null, selected=null, mode="test";
const opt={gamma:0.65, thresh:0, ealpha:0.85, inputEdges:true, contrib:true};
let LW=900, LH=620, layout=null, inputIdx=[], maxAbs=[];
const net=document.getElementById('net'), nctx=net.getContext('2d');

// ---------- math ----------
const sig=z=>1/(1+Math.exp(-z));
function feed(input){
  const a=[input.slice()]; let cur=input;
  for(let l=0;l<MODEL.weights.length;l++){
    const W=MODEL.weights[l], b=MODEL.biases[l], out=new Array(W.length);
    for(let j=0;j<W.length;j++){const wj=W[j];let s=b[j][0];
      for(let k=0;k<wj.length;k++)s+=wj[k]*cur[k]; out[j]=sig(s);}
    a.push(out); cur=out;
  }
  return a;
}
const argmax=a=>{let m=0;for(let i=1;i<a.length;i++)if(a[i]>a[m])m=i;return m;};
function wColor(w,m,alpha){
  let t=Math.max(-1,Math.min(1,w/(m||1))); const a=alpha*Math.min(1,Math.abs(t));
  return t>=0?`rgba(226,91,86,${a})`:`rgba(74,163,224,${a})`;
}
function grayFill(act){const v=Math.pow(Math.max(0,Math.min(1,act)),opt.gamma);
  const c=Math.round(v*255);return `rgb(${c},${c},${c})`;}

// ---------- layout ----------
function ysEven(n,top,bot){const ys=[];for(let j=0;j<n;j++)
  ys.push(n===1?(top+bot)/2:top+(bot-top)*j/(n-1));return ys;}
function ysGap(n,top,bot){ // split into two halves with a gap in the middle
  const half=n/2, mid=(top+bot)/2, gap=Math.max(28,(bot-top)*0.07);
  const ys=[], t2=mid-gap/2, b1=mid+gap/2;
  for(let i=0;i<half;i++)ys.push(top+(t2-top)*(half===1?0:i/(half-1)));
  for(let i=0;i<half;i++)ys.push(b1+(bot-b1)*(half===1?0:i/(half-1)));
  return ys;
}
function buildLayout(){
  const sizes=MODEL.sizes, L=sizes.length, xPad=72, xR=66;
  const xs=[];for(let i=0;i<L;i++)xs.push(xPad+(LW-xPad-xR)*(i/(L-1)));
  let shownIn=Math.min(sizes[0], Math.max(8, Math.floor((LH-70)/20)));
  shownIn-=shownIn%2;                                   // keep even for the gap
  const shown=sizes.map((n,i)=>i===0?shownIn:n);
  inputIdx=[];for(let i=0;i<shownIn;i++)inputIdx.push(Math.round(i*(sizes[0]-1)/(shownIn-1)));
  const top=34, bot=LH-34, pos=[];
  for(let i=0;i<L;i++){
    const n=shown[i], ys=(i===0)?ysGap(n,top,bot):ysEven(n,top,bot), col=[];
    for(let j=0;j<n;j++)col.push({x:xs[i],y:ys[j],layer:i,index:j});
    pos.push(col);
  }
  layout={pos,shown,inMidY:(top+bot)/2,inX:xs[0]};
  maxAbs=MODEL.weights.map(Wm=>{let m=1e-9;for(const r of Wm)for(const v of r)m=Math.max(m,Math.abs(v));return m;});
}
const nodeAct=(l,j)=>l===0?pixels[inputIdx[j]]:acts[l][j];

function sizeCanvas(){
  const box=document.getElementById('netbox');
  LW=Math.max(560, box.clientWidth-14);
  LH=Math.max(560, window.innerHeight-110);
  const dpr=window.devicePixelRatio||1;
  net.width=LW*dpr; net.height=LH*dpr; net.style.width=LW+'px'; net.style.height=LH+'px';
  nctx.setTransform(dpr,0,0,dpr,0,0);
}

// ---------- network drawing ----------
function drawNet(){
  if(!MODEL||!acts)return;
  nctx.clearRect(0,0,LW,LH);
  const pos=layout.pos;
  for(let l=0;l<MODEL.weights.length;l++){
    if(l===0&&!opt.inputEdges)continue;
    const W=MODEL.weights[l], m=maxAbs[l], src=pos[l], dst=pos[l+1];
    for(let dj=0;dj<dst.length;dj++)for(let si=0;si<src.length;si++){
      const realK=(l===0)?inputIdx[si]:si, w=W[dj][realK], t=Math.abs(w/m);
      if(t<opt.thresh)continue;
      let a=opt.ealpha;
      if(opt.contrib){const sa=(l===0)?pixels[realK]:acts[l][si];
        a*=Math.min(1,Math.abs(w)*sa/(m*0.6));}
      nctx.strokeStyle=wColor(w,m,a); nctx.lineWidth=0.6+1.4*t;
      nctx.beginPath();nctx.moveTo(src[si].x,src[si].y);nctx.lineTo(dst[dj].x,dst[dj].y);nctx.stroke();
    }
  }
  for(let l=0;l<pos.length;l++){
    const isOut=(l===pos.length-1), r=isOut?11:9;
    for(let j=0;j<pos[l].length;j++){
      const p=pos[l][j];
      nctx.beginPath();nctx.arc(p.x,p.y,r,0,7);nctx.fillStyle=grayFill(nodeAct(l,j));nctx.fill();
      const sel=selected&&selected.layer===l&&selected.index===j;
      nctx.lineWidth=sel?2.5:1.2; nctx.strokeStyle=sel?'#ffd23f':'#cfcfcf'; nctx.stroke();
      if(isOut){nctx.fillStyle='#cfcfcf';nctx.font='15px Georgia';nctx.textBaseline='middle';
        nctx.fillText(String(j),p.x+16,p.y);}
    }
  }
  // input abbreviation: middle ellipsis + 784 label
  nctx.fillStyle='#cfcfcf';nctx.font='20px Georgia';nctx.textAlign='center';nctx.textBaseline='middle';
  nctx.fillText('⋮',layout.inX,layout.inMidY);
  nctx.font='13px Georgia';nctx.fillStyle='#9a9a9a';
  nctx.fillText(String(MODEL.sizes[0]),layout.inX-46,layout.inMidY);
  nctx.textAlign='left';
  // box around the predicted output
  const g=argmax(acts[acts.length-1]), gp=pos[pos.length-1][g];
  nctx.strokeStyle='#ffd23f';nctx.lineWidth=2;nctx.strokeRect(gp.x+12,gp.y-11,22,22);
}

// ---------- side panels ----------
function drawInputImage(){
  const c=document.getElementById('inImg'), x=c.getContext('2d'), S=c.width/28;
  for(let r=0;r<28;r++)for(let q=0;q<28;q++){const v=Math.round(pixels[r*28+q]*255);
    x.fillStyle=`rgb(${v},${v},${v})`;x.fillRect(q*S,r*S,S+1,S+1);}
}
function drawBars(){
  const c=document.getElementById('bars'), x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const out=acts[acts.length-1], g=argmax(out), n=out.length, h=c.height/n, bx=24, bw=c.width-60;
  x.font='12px Georgia';x.textBaseline='middle';
  for(let i=0;i<n;i++){const y=i*h+h/2;
    x.fillStyle='#aaa';x.fillText(String(i),4,y);
    x.fillStyle='#1c1c20';x.fillRect(bx,y-6,bw,12);
    x.fillStyle=i===g?'#ffd23f':'#5a5a64';x.fillRect(bx,y-6,bw*out[i],12);
    x.fillStyle='#bbb';x.fillText(out[i].toFixed(2),bx+bw+6,y);}
}
function drawGrid(){
  const c=document.getElementById('grid'), x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height);
  const title=document.getElementById('gridTitle');
  if(!selected){title.textContent='Click any neuron in the network.';return;}
  const {layer,index}=selected;
  if(layer===0){title.textContent='Input pixel #'+inputIdx[index]+' — no incoming weights.';return;}
  const w=MODEL.weights[layer-1][index], inLen=w.length;
  const cols=inLen===784?28:Math.ceil(Math.sqrt(inLen)), rows=Math.ceil(inLen/cols);
  let m=1e-9;for(const v of w)m=Math.max(m,Math.abs(v));
  const cell=Math.floor(Math.min(c.width/cols,c.height/rows));
  const ox=(c.width-cell*cols)/2, oy=(c.height-cell*rows)/2;
  for(let i=0;i<inLen;i++){const r=Math.floor(i/cols),q=i%cols,t=w[i]/m;let col;
    if(t>=0){const a=Math.min(1,t);col=`rgb(${Math.round(226*a)},${Math.round(70*a)},${Math.round(66*a)})`;}
    else{const a=Math.min(1,-t);col=`rgb(${Math.round(60*a)},${Math.round(140*a)},${Math.round(224*a)})`;}
    x.fillStyle=col;x.fillRect(ox+q*cell,oy+r*cell,cell,cell);}
  const lname=layer===MODEL.sizes.length-1?'output '+index:'hidden L'+layer+' #'+index;
  title.innerHTML='Neuron <b>'+lname+'</b>: its '+inLen+' incoming weights'+
    (inLen===784?' as a 28&times;28 image (what it looks for).':' ('+cols+'&times;'+rows+' grid).');
}

// ---------- orchestration ----------
function recompute(){
  if(!MODEL)return;
  acts=feed(pixels); drawNet(); drawInputImage(); drawBars(); drawGrid();
  const out=acts[acts.length-1], g=argmax(out), mx=out[g], low=mx<0.5;
  const gEl=document.getElementById('guess');
  gEl.textContent=g; gEl.style.opacity=low?0.4:1;
  document.getElementById('conf').innerHTML='max output: <b>'+mx.toFixed(2)+'</b>'+
    (low?' &middot; <span class="bad">no clear digit</span>':'');
  const tEl=document.getElementById('truth');
  if(mode==='test'){const ok=g===trueLabel;
    tEl.innerHTML='actual: <b>'+trueLabel+'</b> &middot; '+(ok?'<span class="ok">correct</span>':'<span class="bad">wrong</span>');}
  else tEl.textContent='(your drawing)';
}
async function loadTest(i){
  curIndex=((i%NTEST)+NTEST)%NTEST;
  const d=await (await fetch('/api/image?i='+curIndex)).json();
  pixels=d.pixels; trueLabel=d.label; document.getElementById('idx').value=curIndex; recompute();
}

// ---------- models ----------
async function loadModelData(id){
  const m=await (await fetch('/api/model?id='+id)).json();
  if(!m.sizes)return;
  MODEL=m; MODEL_ID=id; selected=null; buildLayout();
  document.getElementById('archLabel').textContent=m.sizes.join(' → ');
  if(!pixels) pixels=new Array(m.sizes[0]).fill(0);
  recompute();
}
function modelLabel(m){
  let s=m.sizes.join('-')+' · noise'+m.noise;
  if(m.status==='ready') s+=' · '+(100*m.accuracy).toFixed(1)+'%';
  else if(m.status==='training'){const p=m.progress;
    s+=' · training '+(p?p.epoch+'/'+p.epochs+' ('+(100*p.eval_acc).toFixed(0)+'%)':'…');}
  else if(m.status==='queued') s+=' · queued';
  else if(m.status==='error') s+=' · error';
  return s;
}
async function refreshModels(){
  MODELS=await (await fetch('/api/models')).json();
  const sel=document.getElementById('modelSel'), prev=sel.value;
  sel.innerHTML='';
  if(MODELS.length===0){
    MODEL=null;MODEL_ID=null;nctx.clearRect(0,0,LW,LH);
    document.getElementById('archLabel').textContent='(no models — train one below)';
    document.getElementById('modelInfo').textContent='';return;
  }
  for(const m of MODELS){const o=document.createElement('option');o.value=m.id;o.textContent=modelLabel(m);sel.appendChild(o);}
  let pick=prev;
  if(!MODELS.some(m=>m.id===pick)) pick=(MODELS.find(m=>m.status==='ready')||MODELS[0]||{}).id;
  if(pick)sel.value=pick;
  const cur=MODELS.find(m=>m.id===sel.value);
  document.getElementById('modelInfo').innerHTML = cur ?
    ('selected: '+cur.sizes.join('-')+' · noise '+cur.noise+
     (cur.accuracy!=null?' · test acc <b>'+(100*cur.accuracy).toFixed(2)+'%</b>':'')) : '';
  if(cur&&cur.status==='ready'&&MODEL_ID!==cur.id) await loadModelData(cur.id);
  if(MODELS.some(m=>m.status==='training'||m.status==='queued')) setTimeout(refreshModels,1500);
}

// ---------- interaction ----------
net.addEventListener('click',e=>{
  const rect=net.getBoundingClientRect(), mx=e.clientX-rect.left, my=e.clientY-rect.top;
  let best=null,bd=15*15;
  for(const col of layout.pos)for(const p of col){const d=(p.x-mx)**2+(p.y-my)**2;if(d<bd){bd=d;best=p;}}
  if(best){selected={layer:best.layer,index:best.index};drawNet();drawGrid();}
});
document.getElementById('prev').onclick=()=>loadTest(curIndex-1);
document.getElementById('next').onclick=()=>loadTest(curIndex+1);
document.getElementById('rand').onclick=()=>loadTest(Math.floor(Math.random()*NTEST));
document.getElementById('idx').onchange=e=>loadTest(parseInt(e.target.value||'0'));
window.addEventListener('keydown',e=>{if(mode!=='test')return;
  if(e.key==='ArrowLeft')loadTest(curIndex-1);
  if(e.key==='ArrowRight')loadTest(curIndex+1);
  if(e.key==='r')loadTest(Math.floor(Math.random()*NTEST));});
function bindSlider(id,k){document.getElementById(id).addEventListener('input',e=>{opt[k]=parseFloat(e.target.value);if(acts)drawNet();});}
bindSlider('gamma','gamma');bindSlider('thresh','thresh');bindSlider('ealpha','ealpha');
function bindToggle(id,k){const b=document.getElementById(id);
  b.onclick=()=>{opt[k]=!opt[k];b.classList.toggle('active',opt[k]);drawNet();};}
bindToggle('tInput','inputEdges');bindToggle('tContrib','contrib');

document.getElementById('modelSel').onchange=async e=>{
  const m=MODELS.find(x=>x.id===e.target.value);
  if(m&&m.status==='ready')await loadModelData(m.id);
  if(m)document.getElementById('modelInfo').innerHTML='selected: '+m.sizes.join('-')+' · noise '+m.noise+
    (m.accuracy!=null?' · test acc <b>'+(100*m.accuracy).toFixed(2)+'%</b>':' · '+m.status);
};
document.getElementById('refreshModels').onclick=refreshModels;
document.getElementById('deleteModel').onclick=async()=>{
  const id=document.getElementById('modelSel').value, m=MODELS.find(x=>x.id===id), info=document.getElementById('modelInfo');
  if(!m)return;
  if(m.status==='training'||m.status==='queued'){info.innerHTML='<span class="bad">can\'t delete while training</span>';return;}
  if(!confirm('Delete model '+m.sizes.join('-')+' · noise'+m.noise+'?\nThis also removes its cache file on disk.'))return;
  const r=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  const res=await r.json().catch(()=>({}));
  if(!r.ok||!res.ok){info.innerHTML='<span class="bad">delete failed: '+(res.msg||r.status)+'</span>';return;}
  if(MODEL_ID===id)MODEL_ID=null;     // force refreshModels to load another
  await refreshModels();
};
document.getElementById('trainBtn').onclick=async()=>{
  const st=document.getElementById('trainStatus');
  const arch=document.getElementById('tArch').value.split(',').map(s=>parseInt(s.trim()));
  const body={arch, epochs:+document.getElementById('tEpochs').value,
    eta:+document.getElementById('tEta').value, lmbda:+document.getElementById('tLmbda').value,
    noise:+document.getElementById('tNoise').value};
  const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){st.innerHTML='<span class="bad">'+(await r.text())+'</span>';return;}
  const {id}=await r.json(); st.textContent='Training '+arch.join('-')+' … watch the dropdown.';
  await refreshModels(); document.getElementById('modelSel').value=id; refreshModels();
};

// ---------- draw pad ----------
const pad=document.getElementById('pad'), pctx=pad.getContext('2d'); let drawing=false;
function clearPad(){pctx.fillStyle='#000';pctx.fillRect(0,0,pad.width,pad.height);}
clearPad();
function padDraw(e){if(!drawing)return;const r=pad.getBoundingClientRect();
  const x=(e.clientX-r.left)*pad.width/r.width, y=(e.clientY-r.top)*pad.height/r.height;
  pctx.fillStyle='#fff';pctx.beginPath();pctx.arc(x,y,11,0,7);pctx.fill();samplePad();}
pad.addEventListener('mousedown',e=>{drawing=true;padDraw(e);});
window.addEventListener('mouseup',()=>{drawing=false;});
pad.addEventListener('mousemove',padDraw);
document.getElementById('clearPad').onclick=()=>{clearPad();pixels=new Array(784).fill(0);recompute();};
function samplePad(){const img=pctx.getImageData(0,0,pad.width,pad.height).data, cell=pad.width/28, px=new Array(784).fill(0);
  for(let r=0;r<28;r++)for(let q=0;q<28;q++){let s=0,c=0;
    for(let yy=0;yy<cell;yy++)for(let xx=0;xx<cell;xx++){const sx=Math.floor(q*cell+xx),sy=Math.floor(r*cell+yy);s+=img[(sy*pad.width+sx)*4];c++;}
    px[r*28+q]=s/c/255;}
  pixels=px;trueLabel=null;recompute();}
function setMode(m){mode=m;
  document.getElementById('modeTest').classList.toggle('active',m==='test');
  document.getElementById('modeDraw').classList.toggle('active',m==='draw');
  document.getElementById('drawArea').style.display=m==='draw'?'block':'none';
  if(m==='test')loadTest(curIndex);else{clearPad();pixels=new Array(784).fill(0);recompute();}}
document.getElementById('modeTest').onclick=()=>setMode('test');
document.getElementById('modeDraw').onclick=()=>setMode('draw');

// ---------- boot ----------
let _rt=null;
window.addEventListener('resize',()=>{clearTimeout(_rt);_rt=setTimeout(()=>{
  sizeCanvas();if(MODEL){buildLayout();drawNet();}},120);});
(async function(){
  sizeCanvas();
  NTEST=(await (await fetch('/api/meta')).json()).num_test;
  document.getElementById('idx').max=NTEST-1;
  await refreshModels();          // populates dropdown + loads a ready model
  await loadTest(0);              // sets the first test image and renders
})();
</script>
</body>
</html>
'''


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Interactive MNIST network visualizer")
    p.add_argument("--arch", default="784,16,16,10",
                   help="default model shape used only if no cache exists yet")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--lmbda", type=float, default=5.0)
    p.add_argument("--noise", type=int, default=40000,
                   help="default number of all-zero-target noise images")
    p.add_argument("--retrain", action="store_true",
                   help="queue a fresh default model at startup even if caches exist")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    args.arch = [int(s) for s in args.arch.split(",")]
    if args.arch[0] != 784 or args.arch[-1] != 10:
        p.error("--arch must start with 784 and end with 10")
    return args


def main():
    args = parse_args()
    print("Loading MNIST...")
    load_data()

    threading.Thread(target=train_worker, daemon=True).start()
    load_cached_models()
    if args.retrain or not ORDER:
        print("Queueing a default model: %s noise%d" % (args.arch, args.noise))
        add_pending(args.arch, args.epochs, args.eta, args.lmbda, args.noise)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler())
    url = "http://127.0.0.1:%d/" % args.port
    print("\nServing visualizer at %s   (Ctrl+C to stop)" % url)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
