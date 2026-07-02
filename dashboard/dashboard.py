# app.py — Alert2Source · Source Attribution Console (Streamlit, live file-reading)

import os, json, pathlib
import pandas as pd, numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Alert2Source Console", layout="wide")

# ============================ CONFIG  ============================
BASE = os.getenv("ALERT2SOURCE_OUTPUT_DIR", "outputs")
DEFAULT_REPORTS = {
    "SHAP":           f"{BASE}/reports/full_shap_only.json",
    "SHAP+RAG":       f"{BASE}/reports/full_shap_rag.json",
    "SHAP+RAG+Image": f"{BASE}/reports/full_shap_rag_image.json",
}
DEFAULT_GOLD   = os.getenv("ALERT2SOURCE_GOLD_CSV", "data/cams_reg_source_gold.csv")
DEFAULT_MASTER = os.getenv("ALERT2SOURCE_MASTER_CSV", "data/processed/final_master_data_FINAL.csv")
# ====================================================================================

EMISSION = ["traffic", "urban_anthropogenic", "industrial", "port_shipping"]
CAT2GOLD = {"traffic": "traffic", "industrial": "industrial",
            "urban_anthropogenic": "urban", "port_shipping": "port_shipping"}
GOLD2CAT = {v: k for k, v in CAT2GOLD.items()}
POL2PFX  = {"no2": "nox", "pm10": "pm10"}


@st.cache_data(show_spinner=False)
def _load_json(p): return json.load(open(p))
@st.cache_data(show_spinner=False)
def _load_csv(p):  return pd.read_csv(p)


def assemble(report_files: dict, gold_csv: str, master_csv: str) -> dict:
    reps = {c: _load_json(p) for c, p in report_files.items() if pathlib.Path(p).exists()}
    if not reps:
        return {"cases": [], "metrics": {}}
    base_key = "SHAP+RAG+Image" if "SHAP+RAG+Image" in reps else list(reps)[0]
    base = reps[base_key]
    gold = _load_csv(gold_csv).set_index("AirQualityStation")
    mst  = _load_csv(master_csv).set_index("AirQualityStation")

    def station_meta(st_id):
        if st_id in mst.index:
            r = mst.loc[st_id]
            if isinstance(r, pd.DataFrame): r = r.iloc[0]
            return float(r["Latitude"]), float(r["Longitude"]), {p: float(r[p]) for p in ["no2", "o3", "pm10"]}
        return None, None, {}

    def cams_block(st_id, pol):
        if pol not in POL2PFX or st_id not in gold.index: return None
        r = gold.loc[st_id]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        pfx = POL2PFX[pol]; items = []
        for cat in EMISSION:
            suf = CAT2GOLD[cat]
            items.append({"cat": cat, "share": float(r[f"{pfx}_share_{suf}"]),
                          "present": bool(r[f"{pfx}_present_{suf}"])})
        items.sort(key=lambda x: -x["share"])
        return {"items": items, "dom": GOLD2CAT.get(r[f"{pfx}_dom_modeled"], r[f"{pfx}_dom_modeled"]),
                "present": [it["cat"] for it in items if it["present"]]}

    def ranked_of(entry):
        """LLM ranked_sources for an entry, cleaned to a 4-permutation."""
        sr = entry.get("structured_reports")
        sr0 = (sr[0] if isinstance(sr, list) and sr else sr) or {}
        raw = sr0.get("ranked_sources") or []
        out = []
        for it in raw:
            if isinstance(it, str):
                c = it.strip().lower().replace(" ", "_")
                if c in EMISSION and c not in out:
                    out.append(c)
        for c in EMISSION:
            if c not in out:
                out.append(c)
        return out

    cases = []
    for ck, e in base.items():
        pol = e.get("pollutant"); st_id = (e.get("eligibility") or {}).get("station_id")
        lat, lon, obs = station_meta(st_id)
        if lat is None:
            el = e.get("eligibility") or {}; lat, lon = el.get("Latitude"), el.get("Longitude")
        tg = e.get("top_groups") or []
        shap = sorted([{"cat": g["source_category"], "val": float(g.get("positive_shap_sum", 0))} for g in tg],
                      key=lambda x: -x["val"])
        _ssum = sum(s["val"] for s in shap) or 1.0
        for s in shap:
            s["share"] = s["val"] / _ssum
        # LLM ranked_sources per condition (the real output now; differs across conditions)
        ranked = {c: ranked_of(reps[c].get(ck, {}) or {}) for c in reps}
        cases.append({"case_key": ck, "station": st_id, "pollutant": pol,
                      "lat": lat, "lon": lon, "observed": obs.get(pol),
                      "shap": shap, "cams": cams_block(st_id, pol),
                      "ranked": ranked,
                      "relerr": (e.get("eligibility") or {}).get("relerr_fusion"),
                      "reports": {c: " ".join((reps[c].get(ck, {}) or {}).get("reports") or []) for c in reps}})

    # Dashboard dedup: one marker per (station_id, pollutant), keep most accurate (min relerr).
    _best = {}
    for _c in cases:
        _key = (_c["station"], _c["pollutant"])
        _r = _c.get("relerr"); _r = _r if isinstance(_r, (int, float)) else float("inf")
        if _key not in _best or _r < _best[_key][0]:
            _best[_key] = (_r, _c)
    cases = [_c for _, _c in _best.values()]

    conds = list(reps.keys())

    # Per-case, per-condition: root (LLM top-1) and divergence vs CAMS dominant.
    for _c in cases:
        _c["root_by_cond"] = {}
        _c["div_by_cond"] = {}
        dom = _c["cams"]["dom"] if _c["cams"] else None
        for cond in conds:
            r = _c["ranked"].get(cond) or []
            top1 = r[0] if r else None
            # O3 has no emission gold; mark root as secondary for display
            _c["root_by_cond"][cond] = ("secondary" if _c["pollutant"] == "o3" else top1)
            _c["div_by_cond"][cond] = (bool(top1 and dom and top1 != dom)
                                       if (_c["pollutant"] in POL2PFX and dom) else None)

    # Per-condition metrics from LLM ranked_sources (NO2/PM10 with gold only).
    def cond_metrics(cond):
        ac1 = ac3 = mrr = rec3 = prec = 0.0; n = 0
        for c in cases:
            if c["pollutant"] not in POL2PFX or not c["cams"]:
                continue
            rank = c["ranked"].get(cond) or []
            if not rank:
                continue
            dom = c["cams"]["dom"]; pres = set(c["cams"]["present"]); n += 1
            if dom and rank[0] == dom: ac1 += 1
            if dom in rank[:3]: ac3 += 1
            rp = rank.index(dom) + 1 if dom in rank else None
            mrr += (1 / rp if rp else 0)
            if rank[0] in pres: prec += 1
            if pres: rec3 += len(set(rank[:3]) & pres) / len(pres)
        if n == 0:
            return {"n": 0}
        return {"n": n, "ac1": round(ac1/n, 3), "ac3": round(ac3/n, 3),
                "mrr": round(mrr/n, 3), "rec3": round(rec3/n, 3),
                "prec": round(prec/n, 3)}

    metrics_by_cond = {cond: cond_metrics(cond) for cond in conds}
    return {"cases": cases, "conds": conds, "metrics_by_cond": metrics_by_cond}


# --------------------------- the interactive dashboard (HTML/JS) ---------------------------
TPL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#0E1117;--panel:#171C25;--panel2:#1F2632;--line:#2A323F;--ink:#E7ECF3;--mut:#B4BECE;--dim:#8A95A8;--sig:#3DD6C4;
--traffic:#4F9DE8;--urban_anthropogenic:#E0A33E;--industrial:#E8765A;--port_shipping:#3DD6C4;--geographic_context:#7E8AA0;--meteorological_condition:#A98BD6;--no2:#E0A33E;--o3:#5BA8E0;--pm10:#B57BD6;
--mono:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;--sans:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.45}
.grid{display:grid;grid-template-columns:1.15fr 1fr;gap:14px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;overflow:hidden}
.card .hd{display:flex;align-items:center;justify-content:space-between;padding:10px 13px;border-bottom:1px solid var(--line)}
.card .hd .t{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut)}
.filterbar{display:flex;gap:6px;flex-wrap:wrap;padding:9px 13px;border-bottom:1px solid var(--line)}
.fbtn{font-family:var(--mono);font-size:10.5px;padding:5px 10px;border-radius:7px;border:1px solid var(--line);background:transparent;color:var(--mut);cursor:pointer;display:flex;align-items:center;gap:6px}
.fbtn i{width:8px;height:8px;border-radius:50%;display:inline-block}
.fbtn .ct{color:var(--dim)}.fbtn.on{background:var(--panel2);color:var(--ink);border-color:var(--sig)}
#map{height:430px;background:#0f141c}#sat{height:170px;background:#0f141c;border-bottom:1px solid var(--line)}
.leaflet-container{background:#0f141c;font-family:var(--sans)}
.leaflet-control-layers,.leaflet-bar{border:1px solid var(--line)!important;background:var(--panel)!important;color:var(--ink)!important}
.legend{display:flex;flex-wrap:wrap;gap:12px;padding:9px 13px;border-top:1px solid var(--line);font-size:11.5px;color:var(--mut)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.formulas{padding:11px 14px;border-top:1px solid var(--line);background:var(--panel2);font-size:11px}
.formulas .fttl{color:var(--ink);font-weight:600;margin-bottom:8px;font-size:11.5px}
.formulas .fsub{display:block;color:var(--dim);font-weight:400;font-size:9.5px;margin-top:3px}
.formulas .frow{display:flex;align-items:baseline;gap:10px;padding:4px 0;border-top:1px solid rgba(255,255,255,0.04)}
.formulas .fk{font-family:var(--mono);color:var(--sig);min-width:78px;font-size:11px}
.formulas .fx{font-family:var(--mono);color:var(--mut);min-width:270px;font-size:10.5px}
.formulas .fx sub{font-size:8px}
.formulas .fd{color:var(--dim);font-size:10px}
.empty{padding:42px 20px;text-align:center;color:var(--dim);font-size:13px}
.sh{padding:12px 14px;border-bottom:1px solid var(--line)}.sh .row1{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.badge{font-family:var(--mono);font-size:10.5px;font-weight:600;padding:3px 8px;border-radius:6px;text-transform:uppercase;color:#0E1117}
.sh h2{margin:7px 0 0;font-size:13px;font-family:var(--mono);font-weight:600;word-break:break-all}
.sh .meta{margin-top:4px;font-size:12px;color:var(--mut);font-family:var(--mono)}.sh .obs b{color:var(--ink)}
.dual{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}@media(max-width:520px){.dual{grid-template-columns:1fr}}
.col{background:var(--panel);padding:11px 13px}.col h3{margin:0 0 9px;font-size:11px;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase}
.col h3 .sub{display:block;color:var(--dim);font-size:9.5px;margin-top:2px;text-transform:none}
.bar{display:flex;align-items:center;gap:7px;margin:5px 0;font-size:11.5px}
.bar .rk{font-family:var(--mono);color:var(--dim);width:13px;text-align:right;font-size:10px}
.bar .nm{width:62px;font-size:11px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar .track{flex:1;height:9px;border-radius:5px;background:#10151d;overflow:hidden}.bar .fill{height:100%;border-radius:5px}
.bar .vv{font-family:var(--mono);font-size:10.5px;color:var(--mut);width:42px;text-align:right}.bar.dom .nm{color:var(--ink);font-weight:700}
.nocams{color:var(--dim);font-size:11.5px;padding:8px 0;line-height:1.5}
.verdict{padding:9px 14px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11.5px;display:flex;align-items:center;gap:8px;background:var(--panel2)}
.dot{width:8px;height:8px;border-radius:50%;flex:none}.rep{padding:12px 14px}
.tabs{display:flex;gap:6px;margin-bottom:9px;flex-wrap:wrap}
.condsel{display:flex;gap:6px;padding:9px 14px 0;flex-wrap:wrap}
.ctab{font-family:var(--mono);font-size:10.5px;padding:5px 11px;border-radius:7px;border:1px solid var(--line);background:transparent;color:var(--mut);cursor:pointer;font-weight:600}
.ctab.on{background:var(--sig);color:#0B0E14;border-color:var(--sig)}
.shaphint{padding:6px 14px 10px;font-size:10px;color:var(--dim);border-bottom:1px solid var(--line)}
.tab{font-family:var(--mono);font-size:10.5px;padding:5px 9px;border-radius:7px;border:1px solid var(--line);background:transparent;color:var(--mut);cursor:pointer}
.tab.on{background:var(--sig);color:#0E1117;border-color:var(--sig);font-weight:600}
.reptext{font-size:12px;line-height:1.6;color:#C7CFDB;max-height:210px;overflow:auto;white-space:pre-wrap;background:#10151d;border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.reptext strong{color:var(--ink)}::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:#2A323F;border-radius:5px}
</style></head><body>
<div class="grid">
 <div class="card"><div class="hd"><span class="t">Europe · root-cause map</span><span class="t" id="cnt"></span></div>
  <div class="filterbar" id="filterbar"></div>
  <div id="map"></div><div class="legend" id="legend"></div>
  <div class="formulas">
   <div class="fttl">Evaluation metric definitions <span class="fsub">N = NO2·PM10 cases, dominant = CAMS top emitter, present = set of CAMS present emitters, rank = LLM ranked_sources</span></div>
   <div class="frow"><span class="fk">AC@k</span><span class="fx">= (1/N) &middot; &Sigma;<sub>i</sub> &#120128;[ dominant<sub>i</sub> &isin; rank<sub>i</sub>[:k] ]</span><span class="fd">Fraction where the top emitter is within the top k (k=1,3)</span></div>
   <div class="frow"><span class="fk">MRR</span><span class="fx">= (1/N) &middot; &Sigma;<sub>i</sub> 1 / rank<sub>i</sub>(dominant)</span><span class="fd">Mean reciprocal rank of the top emitter (0 if absent)</span></div>
   <div class="frow"><span class="fk">recall@k</span><span class="fx">= (1/N) &middot; &Sigma;<sub>i</sub> |rank<sub>i</sub>[:k] &cap; present<sub>i</sub>| / |present<sub>i</sub>|</span><span class="fd">Fraction of present emitters captured within the top k</span></div>
   <div class="frow"><span class="fk">SHAP&ne;CAMS</span><span class="fx">= (1/N) &middot; &Sigma;<sub>i</sub> &#120128;[ rank<sub>i</sub>[0] &ne; dominant<sub>i</sub> ] = 1 &minus; AC@1</span><span class="fd">Predicted #1 &ne; top emitter (red outline on map) — structural divergence between concentration driver and emission source</span></div>
  </div></div>
 <div class="card"><div class="hd"><span class="t">station readout</span><span class="t">satellite + rankings</span></div>
  <div id="sat"></div><div id="detailbody"><div class="empty">Select a station on the map →</div></div></div>
</div>
<script>
const DATA=__DATA__;
const CATLBL={traffic:"Traffic",urban_anthropogenic:"Urban",industrial:"Industrial",port_shipping:"Port/Ship",geographic_context:"Geographic",meteorological_condition:"Meteorology"};
const POLLBL={no2:"NO\u2082",o3:"O\u2083",pm10:"PM\u2081\u2080"};
const ROOTS=["traffic","urban_anthropogenic","industrial","port_shipping","secondary"];
let curCond=(DATA.conds||[]).slice(-1)[0]||'SHAP';  // default to richest condition; defined early (used by markers)
const ROOTLBL={traffic:"Traffic",urban_anthropogenic:"Urban",industrial:"Industrial",port_shipping:"Port",secondary:"Secondary (O\u2083)"};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const catC=c=>css('--'+c)||'#888';const polC=p=>css('--'+p)||'#888';
const rootC=r=>!r?'#5C6677':(r==='secondary'?css('--o3'):catC(r));
document.getElementById('legend').innerHTML=ROOTS.map(r=>`<span><i style="background:${rootC(r)}"></i>${ROOTLBL[r]}</span>`).join('')+`<span style="color:var(--dim)">· color = identified root · circle size ∝ concentration</span>`;
document.getElementById('cnt').textContent=DATA.cases.length+" stations";

const carto=L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{subdomains:'abcd',maxZoom:19,attribution:'© OSM, © CARTO'});
const esri=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Esri'});
const main=L.map('map',{layers:[carto]});L.control.layers({'Dark map':carto,'Satellite':esri},null,{position:'topright'}).addTo(main);
const markers=[];const valid=DATA.cases.map((c,i)=>({c,i})).filter(o=>o.c.lat!=null&&o.c.lon!=null);
const omax=Math.max(...valid.map(o=>o.c.observed||0),1);
valid.forEach(({c,i})=>{const dv=c.div_by_cond&&c.div_by_cond[curCond]===true;const rc=c.root_by_cond?c.root_by_cond[curCond]:null;const m=L.circleMarker([c.lat,c.lon],{radius:4+Math.sqrt((c.observed||0)/omax)*6,color:dv?'#E8765A':'#0E1117',weight:dv?2.5:1,fillColor:rootC(rc),fillOpacity:0.85});
 m.addTo(main);m.on('click',()=>selectCase(i));m.bindTooltip(`${POLLBL[c.pollutant]} · ${curCond} root: ${ROOTLBL[rc]||'-'}${dv?' · ⚠ SHAP≠CAMS':''} · ${c.station||''}`,{direction:'top'});markers[i]=m;});
const sat=L.map('sat',{zoomControl:false,attributionControl:false,layers:[L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19})]}).setView([50,10],4);
let satMarker=null;setTimeout(()=>{main.invalidateSize();sat.invalidateSize();},150);

// ----- root filter (+ divergence) -----
let curFilter='all';
function _rc(c){return c.root_by_cond?c.root_by_cond[curCond]:null;}
function _dv(c){return c.div_by_cond&&c.div_by_cond[curCond]===true;}
function _show(c){return curFilter==='all'||(curFilter==='__div__'?_dv(c):_rc(c)===curFilter);}
function fitShown(){const pts=valid.filter(o=>_show(o.c)).map(o=>[o.c.lat,o.c.lon]);
 if(pts.length)main.fitBounds(L.latLngBounds(pts).pad(0.15));}
function applyFilter(){valid.forEach(({c,i})=>{const show=_show(c);
 if(show){if(!main.hasLayer(markers[i]))markers[i].addTo(main);}else if(main.hasLayer(markers[i]))main.removeLayer(markers[i]);});fitShown();}
function renderFilterBar(){const counts={};valid.forEach(o=>{const rc=_rc(o.c);counts[rc]=(counts[rc]||0)+1;});
 const ndiv=valid.filter(o=>_dv(o.c)).length;
 const btns=[`<button class="fbtn ${curFilter==='all'?'on':''}" data-f="all">All <span class="ct">${valid.length}</span></button>`]
  .concat(ROOTS.filter(r=>counts[r]).map(r=>`<button class="fbtn ${curFilter===r?'on':''}" data-f="${r}"><i style="background:${rootC(r)}"></i>${ROOTLBL[r]} <span class="ct">${counts[r]}</span></button>`))
  .concat(ndiv?[`<button class="fbtn ${curFilter==='__div__'?'on':''}" data-f="__div__" style="border-color:#E8765A;color:#E8765A"><i style="background:#E8765A"></i>SHAP≠CAMS <span class="ct">${ndiv}</span></button>`]:[]);
 const fb=document.getElementById('filterbar');fb.innerHTML=btns.join('');
 fb.querySelectorAll('.fbtn').forEach(b=>b.onclick=()=>{curFilter=b.dataset.f;applyFilter();renderFilterBar();});}
renderFilterBar();fitShown();

function bars(items,key,maxV,dom,showShare){return items.map((it,idx)=>{const v=it[key],w=Math.max(2,(v/(maxV||1))*100);
 let vt;
 if(key==='share')vt=(v*100).toFixed(0)+'%';
 else if(showShare)vt=((it.share||0)*100).toFixed(0)+'%';
 else vt=v.toFixed(3);
 return `<div class="bar ${dom&&it.cat===dom?'dom':''}"><span class="rk">${idx+1}</span><span class="nm" title="${CATLBL[it.cat]}">${CATLBL[it.cat]}</span><span class="track"><span class="fill" style="width:${w.toFixed(0)}%;background:${catC(it.cat)}"></span></span><span class="vv">${vt}</span></div>`;}).join('');}
let cur={};
function rankBars(ranked,dom){
 // render the LLM ranked_sources (ordered list) for the selected condition
 return (ranked||[]).map((cat,idx)=>{const w=Math.max(8,100-idx*22);
  return `<div class="bar ${dom&&cat===dom?'dom':''}"><span class="rk">${idx+1}</span><span class="nm" title="${CATLBL[cat]}">${CATLBL[cat]}</span><span class="track"><span class="fill" style="width:${w}%;background:${catC(cat)}"></span></span><span class="vv">#${idx+1}</span></div>`;}).join('');
}
function bars(items,key,maxV,dom,showShare){return items.map((it,idx)=>{const v=it[key],w=Math.max(2,(v/(maxV||1))*100);
 let vt;
 if(key==='share')vt=(v*100).toFixed(0)+'%';
 else if(showShare)vt=((it.share||0)*100).toFixed(0)+'%';
 else vt=v.toFixed(3);
 return `<div class="bar ${dom&&it.cat===dom?'dom':''}"><span class="rk">${idx+1}</span><span class="nm" title="${CATLBL[it.cat]}">${CATLBL[it.cat]}</span><span class="track"><span class="fill" style="width:${w.toFixed(0)}%;background:${catC(it.cat)}"></span></span><span class="vv">${vt}</span></div>`;}).join('');}
function selectCase(i){const c=DATA.cases[i];cur={c,i,sel:curCond};
 markers.forEach((m,j)=>{if(m&&main.hasLayer(m)){const dj=DATA.cases[j].div_by_cond&&DATA.cases[j].div_by_cond[curCond]===true;m.setStyle({color:dj?'#E8765A':'#0E1117',weight:dj?2.5:1});}});if(markers[i])markers[i].setStyle({color:'#fff',weight:2.5}).bringToFront();
 if(c.lat!=null){sat.setView([c.lat,c.lon],15);if(satMarker)sat.removeLayer(satMarker);satMarker=L.circleMarker([c.lat,c.lon],{radius:8,color:'#3DD6C4',weight:2.5,fillColor:'#3DD6C4',fillOpacity:0.18}).addTo(sat);}
 renderDetail();}
function renderDetail(){const c=cur.c;
 const ranked=(c.ranked&&c.ranked[curCond])||[];
 const top1=c.root_by_cond?c.root_by_cond[curCond]:(ranked[0]||null);
 let camsHtml,verdict;
 if(c.cams){const cmax=Math.max(...c.cams.items.map(x=>x.share),1e-9);camsHtml=bars(c.cams.items,'share',cmax,c.cams.dom);
  const match=top1&&top1===c.cams.dom;
  verdict=`<div class="verdict"><span class="dot" style="background:${match?css('--sig'):'#E8A24A'}"></span><span><b style="color:var(--ink)">${match?'Match':'divergence'}</b> — <b style="color:var(--sig)">${curCond}</b> predicted #1 <b style="color:${catC(top1||'')}">${top1?CATLBL[top1]:'-'}</b>, CAMS top <b style="color:${catC(c.cams.dom)}">${CATLBL[c.cams.dom]}</b></span></div>`;
 }else{camsHtml=`<div class="nocams">No CAMS emission label. O3 is a secondary photochemical pollutant, so its <b style="color:var(--mut)">local emission source is undefined</b>.</div>`;
  verdict=`<div class="verdict"><span class="dot" style="background:var(--o3)"></span><span>Negative control (${curCond} predicted #1 <b style="color:${catC(ranked[0]||'')}">${ranked[0]?CATLBL[ranked[0]]:'-'}</b>)</span></div>`;}
 const condTabs=(DATA.conds||[]).map(k=>`<button class="ctab ${k===curCond?'on':''}" data-c="${k}">${k}</button>`).join('');
 document.getElementById('detailbody').innerHTML=`<div class="sh"><div class="row1"><span class="badge" style="background:${polC(c.pollutant)}">${POLLBL[c.pollutant]}</span><span class="meta">${curCond} predicted root: ${CATLBL[top1]||'-'} · ${c.station||'—'}</span></div><h2>${c.case_key}</h2><div class="meta obs">${c.lat!=null?c.lat.toFixed(3)+'°N, '+c.lon.toFixed(3)+'°E':'no coordinates'}${c.observed!=null?' · observed <b>'+c.observed.toFixed(1)+' µg/m³</b>':''}</div></div>
  <div class="condsel">${condTabs}</div>
  <div class="dual"><div class="col"><h3 style="color:var(--sig)">Our Prediction<span class="sub">${curCond} · LLM ranking</span></h3>${rankBars(ranked,c.cams?c.cams.dom:null)}</div><div class="col"><h3 style="color:#E8A24A">Label root (CAMS Data)<span class="sub">emission share</span></h3>${camsHtml}</div></div>${verdict}
  <div class="shaphint">SHAP concentration contribution (reference): ${c.shap.filter(s=>['traffic','urban_anthropogenic','industrial','port_shipping'].includes(s.cat)).slice(0,3).map(s=>`${CATLBL[s.cat]} ${(s.share*100).toFixed(0)}%`).join(' · ')}</div>
  <div class="rep"><div class="tabs">${(DATA.conds||[]).map(k=>`<button class="tab ${k===curCond?'on':''}" data-k="${k}">${k}</button>`).join('')}</div><div class="reptext" id="reptext"></div></div>`;
 renderReport();
 // condition selector (both the pill row and the report tabs switch condition)
 document.querySelectorAll('.ctab,.tab').forEach(t=>t.onclick=()=>{curCond=t.dataset.c||t.dataset.k;selectCase(cur.i);refreshMarkersForCond();});
}
function refreshMarkersForCond(){valid.forEach(({c,i})=>{if(markers[i]&&main.hasLayer(markers[i])){const dj=c.div_by_cond&&c.div_by_cond[curCond]===true;markers[i].setStyle({color:dj?'#E8765A':'#0E1117',weight:dj?2.5:1});}});
 if(cur.i!=null&&markers[cur.i])markers[cur.i].setStyle({color:'#fff',weight:2.5}).bringToFront();}
function renderReport(){const c=cur.c;let t=(c.reports[curCond]||'(no report)');t=t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');document.getElementById('reptext').innerHTML=t;}
const f=DATA.cases.findIndex(c=>c.pollutant==='no2'&&c.cams);if(f>=0)selectCase(f);else if(DATA.cases.length)selectCase(0);
</script></body></html>"""


# ----------------------------------- UI -----------------------------------
st.markdown("#### Alert2Source · Source-Attribution Console")

with st.sidebar:
    st.markdown("**File paths (editable)**")
    rfiles = {}
    for cond, dp in DEFAULT_REPORTS.items():
        rfiles[cond] = st.text_input(cond, value=dp)
    gold_csv   = st.text_input("CAMS gold", value=DEFAULT_GOLD)
    master_csv = st.text_input("master",   value=DEFAULT_MASTER)
    if st.button("🔄 Reload data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("After regenerating reports, change the paths or press Reload to apply. "
               "Buttons on the map = root filter (shows only stations identified with that root).")

try:
    DATA = assemble(rfiles, gold_csv, master_csv)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

if not DATA["cases"]:
    st.warning("No cases found. Check the file paths in the sidebar.")
    st.stop()

mbc = DATA["metrics_by_cond"]
conds = DATA["conds"]
_any = next((mbc[c] for c in conds if mbc.get(c, {}).get("n")), None)
if _any:
    st.markdown("**Performance comparison by condition** — LLM ranked_sources vs CAMS-REG (NO2·PM10, O3 excluded)")
    tbl = []
    for c in conds:
        mm = mbc.get(c, {})
        if not mm.get("n"):
            continue
        tbl.append({"Condition": c, "n": mm["n"], "AC@1": mm["ac1"], "AC@3": mm["ac3"],
                    "MRR": mm["mrr"], "recall@3": mm["rec3"]})
    tdf = pd.DataFrame(tbl).set_index("Condition")
    st.dataframe(tdf, use_container_width=True)
    # headline metrics for the richest condition
    rich = conds[-1] if conds else None
    if rich and mbc.get(rich, {}).get("n"):
        st.caption(f"Table above: compares how metrics change as evidence is added (SHAP→+RAG→+Image). "
                   f"AC@1·AC@3·MRR are based on the CAMS dominant emitter, recall@3 on the present-set (NO2·PM10, O3 excluded). "
                   f"You can switch conditions for the map and detail views below (default {rich}). "
                   f"See the formulas at the bottom of the map.")

html = TPL.replace("__DATA__", json.dumps(DATA, ensure_ascii=False))
components.html(html, height=720, scrolling=False)
