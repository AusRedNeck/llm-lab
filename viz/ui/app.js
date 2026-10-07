const $ = id => document.getElementById(id);
let snapshot = null, runDetail = null, selectedRun = "", viewName = "training", thinkReady = false, compareReady = false;
const palette = {lime:"#c5ef70", orange:"#ffae68", cyan:"#78d9d1", purple:"#a98be8", red:"#f47b83", grid:"#302d3c", muted:"#a29eb0"};

document.querySelectorAll(".nav").forEach(button => button.addEventListener("click", () => {
  viewName = button.dataset.view;
  document.querySelectorAll(".nav").forEach(item => item.classList.toggle("active", item === button));
  $("training-view").classList.toggle("hidden", viewName !== "training");
  $("think-view").classList.toggle("hidden", viewName !== "think");
  $("activity-view").classList.toggle("hidden", viewName !== "activity");
  $("crumb-view").textContent = viewName.toUpperCase();
  if(viewName === "think") {
    if(!thinkReady) loadThink();
    else refreshThinkStatus();
  }
  if(viewName === "activity") loadActivity();
}));

function addStat(label, value, note) {
  const card = document.createElement("div"); card.className = "stat";
  const name = document.createElement("div"); name.className = "stat-label"; name.textContent = label;
  const number = document.createElement("div"); number.className = "stat-value"; number.textContent = value;
  const detail = document.createElement("div"); detail.className = "stat-note"; detail.textContent = note;
  card.append(name, number, detail); return card;
}
function selected() { return runDetail?.name === selectedRun ? runDetail : null; }
function line(run, key, color, factor=1) {
  return {color, points:(run?.rows || []).map(row=>[Number(row.tokens ?? row.step ?? 0), Number.isFinite(row[key]) ? row[key]*factor : null])};
}

function draw(id, series) {
  const canvas=$(id), box=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(box.width*dpr)); canvas.height=Math.round(box.height*dpr);
  const c=canvas.getContext("2d"); c.scale(dpr,dpr);
  const W=box.width,H=box.height,L=38,R=10,T=13,B=25;
  c.clearRect(0,0,W,H);
  const pts=series.flatMap(s=>s.points).filter(p=>p[1]!==null&&Number.isFinite(p[1]));
  c.strokeStyle=palette.grid;c.lineWidth=1;
  for(let i=0;i<4;i++){let y=T+(H-T-B)*i/3;c.beginPath();c.moveTo(L,y);c.lineTo(W-R,y);c.stroke();}
  if(!pts.length){c.fillStyle=palette.muted;c.font="12px system-ui";c.fillText("No data recorded for this run",L+8,H/2);return;}
  let xmin=Math.min(...pts.map(p=>p[0])),xmax=Math.max(...pts.map(p=>p[0]));
  let ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1]));
  if(xmin===xmax)xmax=xmin+1;if(ymin===ymax){ymin-=.5;ymax+=.5;}
  const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R), y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
  series.forEach(s=>{c.strokeStyle=s.color;c.lineWidth=1.8;c.beginPath();let pen=false;
    s.points.forEach(p=>{if(p[1]===null||!Number.isFinite(p[1])){pen=false;return;}pen?c.lineTo(x(p[0]),y(p[1])):c.moveTo(x(p[0]),y(p[1]));pen=true;});c.stroke();});
  c.fillStyle=palette.muted;c.font="10px system-ui";c.fillText(ymax.toFixed(2),3,T+4);c.fillText(ymin.toFixed(2),3,H-B);
  c.fillText(`${(xmax/1e6).toFixed(1)}M tokens`,Math.max(L,W-100),H-5);
}

function renderRun() {
  const meta=(snapshot?.runs||[]).find(item=>item.name===selectedRun), run=selected();
  const select=$("run-select"), runs=snapshot?.runs||[];
  if(select.options.length!==runs.length){select.replaceChildren();runs.forEach(item=>{let o=document.createElement("option");o.value=item.name;o.textContent=item.name;select.append(o);});}
  if(meta){select.value=meta.name;$("run-title").textContent=meta.name;$("run-badge").textContent=meta.live?"LIVE":"RECENT";$("run-badge").classList.toggle("good",!!meta.live);}
  else{$("run-title").textContent="No runs found";$("run-badge").textContent="NO RUN";}
  const selectedRow=(run?.rows||[]).at(-1), seen=meta?.tokens_seen;
  const coverage=seen!=null&&meta?.train_tokens?`${(seen/meta.train_tokens).toFixed(2)}× corpus`:"not recorded";
  const gap=selectedRow?.avg50!=null&&selectedRow?.val_bpb!=null&&run?.bpb_factor!=null
    ?`${(selectedRow.avg50*run.bpb_factor-selectedRow.val_bpb).toFixed(3)} bpb`:"not recorded";
  const age=meta?.last_update_age_s;
  const freshness=age==null?"unknown":age<60?`${age}s ago`:age<3600?`${Math.floor(age/60)}m ago`:`${Math.floor(age/3600)}h ago`;
  $("run-health").replaceChildren(addStat("TOKENS SEEN",seen==null?"not recorded":`${(seen/1e6).toFixed(1)}M`,"step × effective batch × context"),
    addStat("CORPUS PASSES",coverage,"estimated from run tokens / corpus tokens"),
    addStat("TRAIN–VAL GAP",gap,"latest available readings; not a diagnosis"),
    addStat("LAST LOG UPDATE",freshness,meta?.live?"writer active":"stale / stopped"));
  const loss=[];if(run?.bpb_factor)loss.push(line(run,"avg50",palette.lime,run.bpb_factor));loss.push(line(run,"val_bpb",palette.orange));
  draw("loss-chart",loss);draw("stability-chart",[line(run,"gnorm",palette.cyan),line(run,"lmax",palette.orange),line(run,"ent",palette.purple)]);
  draw("loader-chart",[line(run,"served",palette.cyan),line(run,"random_train",palette.red)]);
}

function renderArms() {
  const host=$("arms-table"), arms=snapshot?.arms||[], problems=snapshot?.problems||[];
  $("problem-count").textContent=`${problems.length} CHECKS`;
  host.replaceChildren();if(!arms.length){host.textContent="No experiment arms in registry.";return;}
  const table=document.createElement("table"), head=document.createElement("tr");
  ["ARM","VERDICT","BEST BPB","STATUS","GOAL"].forEach(label=>{let th=document.createElement("th");th.textContent=label;head.append(th);});
  const thead=document.createElement("thead");thead.append(head);table.append(thead);
  const body=document.createElement("tbody");
  arms.forEach(arm=>{const tr=document.createElement("tr");
    const values=[arm.id,arm.verdict||"—",arm.best_bpb==null?"—":Number(arm.best_bpb).toFixed(4),arm.live?"LIVE":"RECORDED",arm.goal||""];
    values.forEach((value,i)=>{let td=document.createElement("td");td.textContent=String(value);if(i===1)td.className="verdict "+String(arm.verdict||"");tr.append(td);});body.append(tr);});
  table.append(body);host.append(table);
}

function renderCompareOptions() {
  const runs=snapshot?.runs||[];
  [$("compare-a"),$("compare-b")].forEach(select=>{
    const prior=select.value;
    if(select.options.length===runs.length&&Array.from(select.options).every((o,i)=>o.value===runs[i].name))return;
    select.replaceChildren();runs.forEach(run=>{const option=document.createElement("option");option.value=run.name;option.textContent=run.name;select.append(option);});
    if(runs.some(run=>run.name===prior))select.value=prior;
  });
  if(!compareReady){
    if(runs.some(run=>run.name===selectedRun))$("compare-a").value=selectedRun;
    const a=runs.find(run=>run.name===$("compare-a").value);
    const other=runs.find(run=>run.name!==a?.name&&a?.corpus&&run.corpus===a.corpus)||runs.find(run=>run.name!==a?.name);
    if(other)$("compare-b").value=other.name;
    compareReady=true;
  }else if($("compare-b").value===$("compare-a").value){
    const a=runs.find(run=>run.name===$("compare-a").value);
    const other=runs.find(run=>run.name!==a?.name&&a?.corpus&&run.corpus===a.corpus)||runs.find(run=>run.name!==a?.name);
    if(other)$("compare-b").value=other.name;
  }
}
function compareRuns() {
  const runs=snapshot?.runs||[], a=runs.find(run=>run.name===$("compare-a").value), b=runs.find(run=>run.name===$("compare-b").value);
  const result=$("compare-results");result.replaceChildren();
  if(!a||!b){$("compare-note").textContent="Select two available runs to compare.";return;}
  const keys=["corpus","tokenizer","val_frac"], matches=keys.every(key=>a[key]!=null&&b[key]!=null&&String(a[key])===String(b[key]));
  $("compare-note").textContent=matches
    ?"Logged corpus, tokenizer, and validation fraction match. Confirm both runs used the same held-out slice before treating BPB deltas as a fair ranking."
    :"Not safely comparable on BPB: corpus, tokenizer, or validation fraction differs or is missing. Compare trends, not a winner.";
  const table=document.createElement("table"), head=document.createElement("tr");
  ["METRIC",a.name,b.name].forEach(text=>{const th=document.createElement("th");th.textContent=text;head.append(th);});
  const thead=document.createElement("thead");thead.append(head);table.append(thead);
  const body=document.createElement("tbody");
  const rows=[["Best validation (bpb)",a.best_bpb==null?"not recorded":Number(a.best_bpb).toFixed(4),b.best_bpb==null?"not recorded":Number(b.best_bpb).toFixed(4)],
    ["Best step",a.best_step??"not recorded",b.best_step??"not recorded"],
    ["Tokens seen",a.tokens_seen==null?"not recorded":`${(a.tokens_seen/1e6).toFixed(1)}M`,b.tokens_seen==null?"not recorded":`${(b.tokens_seen/1e6).toFixed(1)}M`],
    ["Parameters (M)",a.params_m??"not recorded",b.params_m??"not recorded"],
    ["Last update",a.last_update_age_s==null?"unknown":`${Math.floor(a.last_update_age_s/60)}m ago`,b.last_update_age_s==null?"unknown":`${Math.floor(b.last_update_age_s/60)}m ago`]];
  rows.forEach(values=>{const tr=document.createElement("tr");values.forEach(value=>{const td=document.createElement("td");td.textContent=String(value);tr.append(td);});body.append(tr);});
  table.append(body);result.append(table);
}
$("compare-runs").addEventListener("click",compareRuns);

function render() {
  const runs=snapshot?.runs||[], live=runs.filter(run=>run.live).length, problems=snapshot?.problems||[];
  $("stats").replaceChildren(addStat("RUNS",runs.length,"indexed from local logs"),addStat("ACTIVE",live,live?"updating now":"no live writers"),addStat("EXPERIMENT ARMS",(snapshot?.arms||[]).length,"from registry"),addStat("REGISTRY FLAGS",problems.length,problems.length?"inspect below":"no detected issues"));
  $("updated-at").textContent=snapshot?.updated_at?new Date(snapshot.updated_at).toLocaleTimeString():"—";
  renderCompareOptions();renderRun();renderArms();
  $("notice").textContent=problems.length?problems.slice(0,3).join(" · "):`Watching ${runs.length} local run${runs.length===1?"":"s"}; refreshed automatically.`;
  $("notice").classList.toggle("error",problems.length>0);
}

async function loadRunDetail(name) {
  if(!name){runDetail=null;return;}
  const response=await fetch(`/api/runs/${encodeURIComponent(name)}`,{cache:"no-store"});
  if(!response.ok)throw new Error(`run detail returned ${response.status}`);
  runDetail=await response.json();
}

$("run-select").addEventListener("change",async event=>{
  selectedRun=event.target.value;runDetail=null;renderRun();
  try{await loadRunDetail(selectedRun);renderRun();}
  catch(error){$("notice").textContent=`Could not read selected run: ${error.message}`;$("notice").classList.add("error");}
});
window.addEventListener("resize",()=>snapshot&&renderRun());

async function refresh() {
  if(refresh.busy)return;refresh.busy=true;
  try{
    const response=await fetch("/api/dashboard",{cache:"no-store"});
    if(!response.ok)throw new Error(`dashboard returned ${response.status}`);
    snapshot=await response.json();
    if(!snapshot.runs.some(run=>run.name===selectedRun))selectedRun=(snapshot.runs.find(run=>run.live)||snapshot.runs[snapshot.runs.length-1])?.name||"";
    await loadRunDetail(selectedRun);
    $("connection").textContent="CONNECTED";render();
  }catch(error){
    $("connection").textContent="RETRYING";$("notice").textContent=`Could not read run data: ${error.message}`;$("notice").classList.add("error");
  }finally{refresh.busy=false;}
}
refresh.busy=false;refresh();window.setInterval(refresh,5000);

let generationBusy=false, latestTrace=null;
async function api(path, options={}) {
  const response=await fetch(path,{cache:"no-store",...options});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||`request failed (${response.status})`);
  return data;
}
function showModelStatus(state) {
  const loaded=!!state.loaded;
  $("model-badge").textContent=loaded?`MODEL LOADED · ${state.kind==="huggingface"?"OPEN WEIGHTS":"LAB CHECKPOINT"} · ${String(state.device).toUpperCase()}`:"NO MODEL LOADED";
  $("load-model").disabled=!!state.generating;
  $("unload-model").disabled=!loaded||!!state.generating;
  $("generate-button").disabled=!loaded||generationBusy||!!state.generating;
  $("inspect-trace").disabled=!loaded||!!state.generating;
  $("compare-checkpoints").disabled=!thinkReady||!!state.generating||state.device==="cuda";
  $("model-status").textContent=loaded?`${state.checkpoint_id} · ${state.device} · ${state.config?.num_layers||"?"} layers · ctx ${state.config?.context_length||"?"}`:"Choose a trusted local checkpoint. CUDA holds the training lock until unload.";
}
async function loadThink() {
  $("think-notice").textContent="Scanning local checkpoints and device availability…";
  try{
    const [items,devices,state]=await Promise.all([api("/api/checkpoints"),api("/api/devices"),api("/api/model/status")]);
    const select=$("checkpoint-select");select.replaceChildren();
    items.forEach(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=`${item.label||item.id} · ${(item.size_bytes/1048576).toFixed(0)} MB`;select.append(option);});
    const compareA=$("compare-model-a"),compareB=$("compare-model-b");
    [compareA,compareB].forEach(list=>{list.replaceChildren();items.forEach(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=item.label||item.id;list.append(option);});});
    const own=items.find(item=>item.id.includes("pythia_tokenizer_202609302055_step15136"));
    const upstream=items.find(item=>item.id==="hf:pythia70m-step1000");
    if(own){compareA.value=own.id;select.value=own.id;}
    if(upstream)compareB.value=upstream.id;
    const preferred=own;
    if(preferred)select.value=preferred.id;
    const cuda=Array.from($("device-select").options).find(o=>o.value==="cuda");
    cuda.disabled=!devices.cuda_available||devices.training_busy;
    cuda.textContent=!devices.cuda_available?"CUDA · unavailable":devices.training_busy?"CUDA · training busy":"CUDA · available";
    thinkReady=true;showModelStatus(state);$("think-notice").classList.remove("error");
    $("think-notice").textContent=devices.training_busy?"A training process owns the GPU. CPU inference remains available.":"Fast generation is ready when a model is loaded.";
  }catch(error){$("think-notice").textContent=`Think workbench unavailable: ${error.message}`;$("think-notice").classList.add("error");}
}
async function compareCheckpoints() {
  const button=$("compare-checkpoints"),output=$("checkpoint-compare-result");
  button.disabled=true;button.textContent="COMPARING…";output.replaceChildren();
  $("checkpoint-compare-note").classList.remove("error");
  try{
    const result=await postJSON("/api/compare",{checkpoint_ids:[$("compare-model-a").value,$("compare-model-b").value],prompt:$("prompt-input").value});
    $("checkpoint-compare-note").textContent=result.token_ids_match
      ?`Tokenization matches (${result.models[0].token_ids.length} prompt tokens). Top-token match: ${result.same_top_prediction?"yes":"no"}.`
      :"Tokenization differs between checkpoints; inspect each distribution separately, not token-for-token.";
    result.models.forEach(model=>{
      const card=document.createElement("article");card.className="model-compare-card";
      const title=document.createElement("h3");title.textContent=model.checkpoint_id;card.append(title);
      const entropy=document.createElement("p");entropy.className="subhead small";entropy.textContent=`Entropy ${model.entropy.toFixed(2)} nats · prompt IDs ${model.token_ids.slice(0,12).join(", ")}${model.token_ids.length>12?" …":""}`;card.append(entropy);
      model.top.forEach(item=>{const row=document.createElement("div");row.className="model-top-token";row.textContent=`${item.text||"∅"} · id ${item.id} · ${(item.probability*100).toFixed(2)}%`;card.append(row);});
      output.append(card);
    });
  }catch(error){$("checkpoint-compare-note").textContent=error.message;$("checkpoint-compare-note").classList.add("error");}
  finally{button.textContent="COMPARE MODELS";await refreshThinkStatus();}
}
$("compare-checkpoints").addEventListener("click",compareCheckpoints);

function postJSON(path, body) {
  return api(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
}
function traceMetric(value) { return Array.isArray(value)?value[Number($("trace-token").value)]?.toFixed(2)??"—":"not exposed"; }
function renderTraceSelection() {
  if(!latestTrace)return;
  const tokenIndex=Number($("trace-token").value), layerIndex=Number($("trace-layer").value), headIndex=Number($("trace-head").value);
  const layer=latestTrace.layers[layerIndex];
  const weights=layer.attention_by_head[headIndex]||[];
  const bars=$("attention-bars");bars.replaceChildren();
  weights.forEach((weight,index)=>{
    const row=document.createElement("div");row.className="attention-row";
    const label=document.createElement("span");label.className="attention-label";label.textContent=`${index===tokenIndex?"▶ ":""}${index}: ${latestTrace.tokens[index]} [${latestTrace.token_ids[index]}]`;
    const track=document.createElement("div");track.className="attention-track";
    const fill=document.createElement("div");fill.className="attention-fill";fill.style.width=`${Math.max(0,Math.min(100,weight*100))}%`;track.append(fill);
    const val=document.createElement("span");val.className="attention-value";val.textContent=`${(weight*100).toFixed(1)}%`;
    row.append(label,track,val);bars.append(row);
  });
  const grid=$("pathway-layers");grid.replaceChildren();
  latestTrace.layers.forEach((item,index)=>{
    const cell=document.createElement("div");cell.className="pathway-cell";
    const title=document.createElement("b");title.textContent=`LAYER ${index+1}${index===layerIndex?" · SELECTED":""}`;cell.append(title);
    [["hidden ‖h‖",item.hidden_norms], ["attention Δ",item.attention_delta_norms], ["FFN Δ",item.ffn_delta_norms], ["GELU RMS",item.ffn_activation_rms]].forEach(([label,values])=>{
      const line=document.createElement("span");const value=Array.isArray(values)?values[tokenIndex]?.toFixed(2):"not exposed";line.textContent=`${label}: ${value??"—"}`;cell.append(line);
    });
    grid.append(cell);
  });
  $("trace-note").textContent=`${latestTrace.checkpoint_id} · ${latestTrace.capture_kind}. Attention is a routing measurement, not causal proof.${latestTrace.capture_kind.includes("open-weights")?" Feed-forward branch internals are not exposed by this backend.":""}`;
}
$("inspect-trace").addEventListener("click",async()=>{
  const button=$("inspect-trace");button.disabled=true;button.textContent="CAPTURING…";
  $("trace-note").textContent="Running one bounded, opt-in forward pass…";
  try{
    latestTrace=await postJSON("/api/trace",{prompt:$("prompt-input").value,max_context:128});
    const tokens=$("trace-token");tokens.replaceChildren();
    latestTrace.tokens.forEach((token,index)=>{const option=document.createElement("option");option.value=index;option.textContent=`${index}: ${token||"∅"} [${latestTrace.token_ids[index]}]`;tokens.append(option);});
    tokens.value=String(latestTrace.selected_index);
    const layers=$("trace-layer");layers.replaceChildren();latestTrace.layers.forEach((_,index)=>{const option=document.createElement("option");option.value=index;option.textContent=`Layer ${index+1}`;layers.append(option);});
    const heads=$("trace-head");heads.replaceChildren();(latestTrace.layers[0]?.attention_by_head||[]).forEach((_,index)=>{const option=document.createElement("option");option.value=index;option.textContent=`Head ${index+1}`;heads.append(option);});
    tokens.disabled=layers.disabled=heads.disabled=false;renderTraceSelection();
  }catch(error){$("trace-note").textContent=error.message;$("trace-note").classList.add("error");}
  finally{button.textContent="INSPECT PROMPT";button.disabled=false;await refreshThinkStatus();}
});
$("trace-token").addEventListener("change",renderTraceSelection);
$("trace-layer").addEventListener("change",renderTraceSelection);
$("trace-head").addEventListener("change",renderTraceSelection);
$("load-model").addEventListener("click",async()=>{
  const button=$("load-model");button.disabled=true;$("model-status").textContent="Loading local checkpoint…";
  try{showModelStatus(await postJSON("/api/model/load",{checkpoint_id:$("checkpoint-select").value,device:$("device-select").value}));$("think-notice").textContent="Checkpoint loaded. Ready to generate.";}
  catch(error){$("model-status").textContent=error.message;$("think-notice").classList.add("error");}
  finally{button.disabled=false;}
});
$("unload-model").addEventListener("click",async()=>{
  try{showModelStatus(await postJSON("/api/model/unload",{}));$("think-notice").textContent="Model unloaded; GPU lock released.";}
  catch(error){$("think-notice").textContent=error.message;$("think-notice").classList.add("error");}
});

function appendTokenEvent(event) {
  const card=document.createElement("article");card.className="token-event";
  const head=document.createElement("div");head.className="token-event-head";
  const tok=document.createElement("strong");tok.textContent=`${event.step}. ${event.token||`[${event.token_id}]`}`;
  const certainty=document.createElement("span");certainty.textContent=`p ${(event.probability*100).toFixed(1)}%`;
  head.append(tok,certainty);card.append(head);
  const ent=document.createElement("small");ent.textContent=`token ${event.token_id} · entropy ${event.entropy.toFixed(2)} nats`;card.append(ent);
  const alternatives=document.createElement("div");alternatives.className="alternatives";
  (event.top||[]).forEach(item=>{const pill=document.createElement("span");pill.className="alternative";pill.textContent=item.text||`[${item.id}]`;const p=document.createElement("b");p.textContent=`${(item.probability*100).toFixed(0)}%`;pill.append(p);alternatives.append(pill);});
  card.append(alternatives);$("token-events").append(card);
}

async function generateSample() {
  if(generationBusy)return;
  generationBusy=true;$("generate-button").disabled=true;$("token-events").replaceChildren();
  $("token-count").textContent="0 TOKENS";$("generation-state").textContent="Starting";$("generated-text").textContent="";
  try{
    const task=await postJSON("/api/generations",{prompt:$("prompt-input").value,
      max_new_tokens:Number($("max-tokens").value),temperature:Number($("temperature").value),
      top_k:Number($("top-k").value),seed:Number($("seed").value)});
    let cursor=0,result;
    do{
      await new Promise(resolve=>setTimeout(resolve,250));
      result=await api(`/api/generations/${encodeURIComponent(task.id)}?after=${cursor}`);
      result.events.forEach(appendTokenEvent);cursor=result.cursor;
      $("generated-text").textContent=result.text||"";$("token-count").textContent=`${cursor} TOKENS`;
      $("generation-state").textContent=result.status.toUpperCase();
      if(result.prompt_truncated)$("think-notice").textContent=`Prompt shortened to the last ${result.prompt_tokens} context tokens.`;
      if(result.status==="error")throw new Error(result.error||"generation failed");
    }while(result.status==="running");
    $("think-notice").classList.remove("error");
  }catch(error){$("generation-state").textContent="ERROR";$("think-notice").textContent=error.message;$("think-notice").classList.add("error");}
    finally{generationBusy=false;await refreshThinkStatus();}
}
$("generate-button").addEventListener("click",generateSample);

async function refreshThinkStatus() {
  try {
    const [state, devices] = await Promise.all([api("/api/model/status"), api("/api/devices")]);
    showModelStatus(state);
    const cuda=Array.from($("device-select").options).find(o=>o.value==="cuda");
    if(cuda) {
      cuda.disabled=!devices.cuda_available||devices.training_busy;
      cuda.textContent=!devices.cuda_available?"CUDA · unavailable":devices.training_busy?"CUDA · training busy":"CUDA · available";
    }
    $("think-notice").textContent=devices.training_busy?"A training process owns the GPU. CPU inference remains available.":"Fast generation is ready when a model is loaded.";
    $("think-notice").classList.remove("error");
  } catch(error) {
    $("think-notice").textContent=`Think workbench unavailable: ${error.message}`;
    $("think-notice").classList.add("error");
  }
}
window.setInterval(()=>{if(viewName==="think"&&thinkReady&&!generationBusy)refreshThinkStatus();},5000);

/* ---------------- Activity workspace ---------------- */
const streamColor = {inference:"#c5ef70", tools:"#ffae68", memory:"#78d9d1"};
let activityPayload = null, activityBusy = false;

function hhmm(ts) {
  if(ts === null || ts === undefined) return "—";
  const s = String(ts), i = s.indexOf("T");
  return i >= 0 ? s.slice(i + 1, i + 9) : s.slice(0, 8);
}
function statCards(target, cards) {
  const box = $(target); box.replaceChildren();
  cards.forEach(([label, value, note]) => box.append(addStat(label, value, note)));
}
function tr(cells, key) {
  const row = document.createElement("tr");
  if(key) row.dataset.key = key;
  cells.forEach(c => { const td = document.createElement("td"); td.textContent = c; row.append(td); });
  return row;
}
function streamDown(prefix, error) {
  $(prefix + "-badge").textContent = "OFFLINE";
  $(prefix + "-badge").classList.remove("good");
  $(prefix + "-feed").replaceChildren();
  $(prefix + "-stats").replaceChildren();
  const n = $(prefix + "-notice");
  n.textContent = `Source unavailable: ${error || "unknown"}`;
  n.classList.add("error");
}

function renderInference(s) {
  if(!s.ok) return streamDown("inference", s.error);
  const live = s.freshness === "LIVE";
  $("inference-badge").textContent = live ? "LIVE" : "FILE";
  $("inference-badge").classList.toggle("good", live);
  statCards("inference-stats", [
    ["LAST HOUR", String(s.stats.last_hour_count ?? 0), "requests"],
    ["TOKENS", String(s.stats.tokens_total ?? 0), "prompt + completion"],
    ["SPEND", ((s.stats.cost_cents ?? 0) / 100).toFixed(4) + " $", "routed spend"],
    ["ERRORS", String(s.stats.errors ?? 0), "last hour"],
  ]);
  const feed = $("inference-feed"); feed.replaceChildren();
  s.events.forEach(e => feed.append(tr(
    [hhmm(e.ts), e.provider, e.model,
     String((e.tokens_prompt || 0) + (e.tokens_completion || 0)),
     String(e.latency_ms), (e.cost_cents || 0).toFixed(1) + " ¢"],
    "inference:" + e.ts_epoch_ms)));
  const n = $("inference-notice"); n.classList.remove("error");
  n.textContent = live ? "Covers HFM-routed calls only."
    : "HFM proxy down — showing last file state. Covers HFM-routed calls only.";
}

function renderTools(s) {
  if(!s.ok) return streamDown("tools", s.error);
  $("tools-badge").textContent = "LIVE";
  $("tools-badge").classList.add("good");
  const top = Object.entries(s.stats.by_tool || {}).sort((a, b) => b[1] - a[1])[0];
  statCards("tools-stats", [
    ["EVENTS", String(s.stats.total ?? 0), "in window"],
    ["INVOCATIONS", String(s.stats.invocations ?? 0), "calls made"],
    ["RESULTS", String(s.stats.results ?? 0), "returns"],
    ["TOP TOOL", top ? top[0] : "—", top ? top[1] + " events" : ""],
  ]);
  const feed = $("tools-feed"); feed.replaceChildren();
  s.events.forEach(e => feed.append(tr(
    [hhmm(e.iso || e.timestamp), e.tool_name || "—", e.kind,
     (e.session_title || e.session_id || "—")],
    "tools:" + e.id)));
  const n = $("tools-notice"); n.classList.remove("error");
  n.textContent = "Invocation/result pairs from session transcripts.";
}

function renderMemory(s) {
  if(!s.ok) return streamDown("memory", s.error);
  $("memory-badge").textContent = "LIVE";
  $("memory-badge").classList.add("good");
  statCards("memory-stats", [
    ["RECALLED", String(s.stats.recalled_total ?? 0), "across window"],
    ["WORKING", String(s.stats.working ?? 0), "events shown"],
    ["EPISODIC", String(s.stats.episodic ?? 0), "events shown"],
    ["GRAPH", String((s.stats.graph_edges ?? 0) + (s.stats.triples ?? 0)), "edges + triples"],
  ]);
  const feed = $("memory-feed"); feed.replaceChildren();
  s.events.forEach(e => feed.append(tr(
    [hhmm(e.last_recalled || e.timestamp), e.tier, e.preview,
     String(e.recall_count || 0)],
    "memory:" + e.tier + ":" + e.id)));
  const n = $("memory-notice"); n.classList.remove("error");
  n.textContent = "Recent memories, most-recalled first in window.";
}

function highlightRow(key) {
  const row = document.querySelector(`#activity-view tr[data-key="${CSS.escape(key)}"]`);
  if(!row) return;
  row.scrollIntoView({block: "center", behavior: "smooth"});
  row.classList.add("row-flash");
  window.setTimeout(() => row.classList.remove("row-flash"), 1600);
}

function renderActivity(payload) {
  activityPayload = payload;
  $("activity-updated-at").textContent = String(payload.updated_at || "").replace("T", " ").slice(0, 19);
  renderInference(payload.inference || {ok: false, error: "missing"});
  renderTools(payload.tools || {ok: false, error: "missing"});
  renderMemory(payload.memory || {ok: false, error: "missing"});
  if(typeof drawMap === "function") drawMap(payload.map || {nodes: [], links: []});
  const n = $("activity-notice"); n.classList.remove("error");
  n.textContent = "Three streams · click a node in the map to inspect the event behind it.";
}

async function loadActivity() {
  if(activityBusy) return;
  activityBusy = true;
  try {
    renderActivity(await api("/api/activity"));
  } catch(error) {
    const n = $("activity-notice");
    n.textContent = `Activity unavailable: ${error.message}`;
    n.classList.add("error");
  } finally { activityBusy = false; }
}
window.setInterval(() => { if(viewName === "activity") loadActivity(); }, 5000);

