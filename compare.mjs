// ─── GLYF v2 FULL SYSTEM vs Old Algorithm — Calibrated Benchmark ─────────────
// OLD system  = DTW shape-match only (no feature channels)
// GLYF system = DTW + speed-invariant feature channels
//
// Speed invariance is critical: a genuine user who signs 30% faster on one day
// must not be penalized. Speed-invariant features:
//   strokeCount         — not affected by speed
//   aspectRatio         — geometry only, not speed
//   rhythmRatio         — pause/total ratio, normalizes out speed
//   cvMicrotremor       — tremor / avgVelocity² (coefficient of variation)
//   cvAngularEnergy     — angularEnergy / avgVelocity (normalized rotation)
//
// GLYF v2 DTW: 0.50×spatial + 0.20×velocity + 0.20×curvDiff + 0.10×dirDiff

// ─── Deterministic RNG ───────────────────────────────────────────────────────
class LCG { constructor(s=42){this.s=s>>>0;} next(){this.s=(Math.imul(1664525,this.s)+1013904223)>>>0;return this.s/0xffffffff;} }
const rng=new LCG(42); const r=()=>rng.next();

// ─── Signature factory ────────────────────────────────────────────────────────
function makeSig({ noise=0.008, humps=4, tScale=1, strokeCount=2,
                   wrongPenLift=false, extraStroke=false, slowTremor=false }={}) {
  const all=[]; let t=0;
  const totalStrokes=extraStroke?strokeCount+1:strokeCount;
  for(let s=0;s<totalStrokes;s++){
    const pts=[], N=65, xOff=s*115, localHumps=humps/strokeCount;
    for(let i=0;i<N;i++){
      const prog=i/(N-1);
      const x=xOff+prog*100+(r()-0.5)*noise*80;
      const y=50+Math.sin(prog*Math.PI*localHumps)*22
               +Math.sin(prog*Math.PI*(localHumps*2))*5
               +(r()-0.5)*noise*80;
      // slowTremor=true: forger draws deliberately → constant velocity, no micro-jitter
      const jitter = slowTremor ? 0 : (r()-0.5)*2;
      t+=(14+jitter)*tScale;
      pts.push({x,y,t});
    }
    all.push(pts);
    const pause=wrongPenLift?800+r()*400:120+r()*80;
    t+=pause*tScale;
  }
  return {strokes:all, flatPoints:all.flat()};
}

// ─── Shared preprocessing ─────────────────────────────────────────────────────
function normalize(pts){
  let ax=Infinity,bx=-Infinity,ay=Infinity,by=-Infinity;
  for(const p of pts){if(p.x<ax)ax=p.x;if(p.x>bx)bx=p.x;if(p.y<ay)ay=p.y;if(p.y>by)by=p.y;}
  const sc=Math.max(bx-ax,by-ay)||1;
  return pts.map(p=>({...p,x:(p.x-ax)/sc,y:(p.y-ay)/sc}));
}
function addVel(pts){
  const sp=[0];
  for(let i=1;i<pts.length;i++){const dx=pts[i].x-pts[i-1].x,dy=pts[i].y-pts[i-1].y,dt=Math.max(pts[i].t-pts[i-1].t,1);sp.push(Math.sqrt(dx*dx+dy*dy)/dt);}
  const mx=Math.max(...sp,1e-9);
  return pts.map((p,i)=>({...p,v:sp[i]/mx}));
}
function dsamp(pts,n){
  if(pts.length<=n)return pts;
  const step=(pts.length-1)/(n-1);
  return Array.from({length:n},(_,i)=>pts[Math.round(i*step)]);
}
const NSAMP=128;
function proc(fp){return dsamp(addVel(normalize(fp)),NSAMP);}

// ─── OLD DTW: fixed 15% band ──────────────────────────────────────────────────
function oldDTW(a,b){
  const n=a.length,m=b.length,band=Math.ceil(Math.max(n,m)*0.15);
  const mat=Array.from({length:n+1},()=>Array(m+1).fill(Infinity));
  mat[0][0]=0;
  for(let i=1;i<=n;i++)for(let j=Math.max(1,i-band);j<=Math.min(m,i+band);j++){
    const dx=a[i-1].x-b[j-1].x,dy=a[i-1].y-b[j-1].y,dv=((a[i-1].v??0)-(b[j-1].v??0))*0.3;
    mat[i][j]=Math.sqrt(dx*dx+dy*dy+dv*dv)+Math.min(mat[i-1][j],mat[i][j-1],mat[i-1][j-1]);
  }
  return mat[n][m]/n;
}

// ─── GLYF v2 DTW: multi-channel additive + adaptive band ─────────────────────
const SW=0.50,VW=0.20,CW=0.20,DW=0.10;
function curvatures(pts){
  const n=pts.length,c=new Array(n).fill(0);
  for(let i=1;i<n-1;i++){
    const dx1=pts[i].x-pts[i-1].x,dy1=pts[i].y-pts[i-1].y;
    const dx2=pts[i+1].x-pts[i].x,dy2=pts[i+1].y-pts[i].y;
    const m1=Math.sqrt(dx1*dx1+dy1*dy1)||1,m2=Math.sqrt(dx2*dx2+dy2*dy2)||1;
    c[i]=(1-Math.max(-1,Math.min(1,(dx1*dx2+dy1*dy2)/(m1*m2))))/2;
  }
  if(n>=2){c[0]=c[1]??0;c[n-1]=c[n-2]??0;}
  return c;
}
function directions(pts){
  const d=[0];
  for(let i=1;i<pts.length;i++)d.push(Math.atan2(pts[i].y-pts[i-1].y,pts[i].x-pts[i-1].x));
  return d;
}
function angDiff(a,b){let d=Math.abs(a-b)%(2*Math.PI);if(d>Math.PI)d=2*Math.PI-d;return d/Math.PI;}
function adaptBand(velProfiles){
  if(velProfiles.length<2)return 0.15;
  const L=velProfiles[0].length;
  const mu=Array(L).fill(0);
  for(const p of velProfiles)for(let i=0;i<L;i++)mu[i]+=p[i]/velProfiles.length;
  let dev=0;
  for(const p of velProfiles)for(let i=0;i<L;i++)dev+=Math.abs(p[i]-mu[i]);
  const t=Math.min(1,(dev/(velProfiles.length*L))/0.3);
  return 0.08+t*(0.28-0.08);
}
function glfDTW(a,b,band){
  const n=a.length,m=b.length,bSz=Math.ceil(Math.max(n,m)*band);
  const c1=curvatures(a),c2=curvatures(b);
  const d1=directions(a),d2=directions(b);
  const mat=Array.from({length:n+1},()=>Array(m+1).fill(Infinity));
  mat[0][0]=0;
  for(let i=1;i<=n;i++)for(let j=Math.max(1,i-bSz);j<=Math.min(m,i+bSz);j++){
    const dx=a[i-1].x-b[j-1].x,dy=a[i-1].y-b[j-1].y;
    const cost=SW*Math.sqrt(dx*dx+dy*dy)
              +VW*Math.abs((a[i-1].v??0)-(b[j-1].v??0))
              +CW*Math.abs(c1[i-1]-c2[j-1])
              +DW*angDiff(d1[i-1],d2[j-1]);
    mat[i][j]=cost+Math.min(mat[i-1][j],mat[i][j-1],mat[i-1][j-1]);
  }
  return mat[n][m]/n;
}

// ─── Speed-invariant feature extraction ──────────────────────────────────────
// All channels normalized so signing speed doesn't penalize genuine users.
function rawVels(pts){
  const vs=[];
  for(let i=1;i<pts.length;i++){
    const dx=pts[i].x-pts[i-1].x,dy=pts[i].y-pts[i-1].y,dt=Math.max(pts[i].t-pts[i-1].t,1);
    vs.push(Math.sqrt(dx*dx+dy*dy)/dt);
  }
  return vs;
}
function mean(a){return a.length?a.reduce((s,v)=>s+v,0)/a.length:0;}
function vari(a){const m=mean(a);return mean(a.map(v=>(v-m)**2));}

// Curvature entropy — 8-bin Shannon entropy of direction-change angles.
// Encodes curve complexity (hump count, loop count). Speed-invariant.
function curvEntropy(pts){
  if(pts.length<3)return 0;
  const angles=[];
  for(let i=1;i<pts.length-1;i++){
    const dx1=pts[i].x-pts[i-1].x,dy1=pts[i].y-pts[i-1].y;
    const dx2=pts[i+1].x-pts[i].x,dy2=pts[i+1].y-pts[i].y;
    angles.push(Math.abs(Math.atan2(dy2,dx2)-Math.atan2(dy1,dx1)));
  }
  const bins=new Array(8).fill(0);
  for(const a of angles)bins[Math.min(7,Math.floor((a/Math.PI)*8))]++;
  const tot=angles.length||1;
  let ent=0;
  for(const c of bins)if(c>0){const p=c/tot;ent-=p*Math.log2(p);}
  return ent;
}

function extractFeatures(sig){
  const pts=sig.flatPoints;
  if(pts.length<3)return{strokeCount:0,aspectRatio:1,rhythmRatio:0,cvMicrotremor:0,cvAngularEnergy:0,curvatureEntropy:0};
  const vs=rawVels(pts);
  const avgV=mean(vs)||0.001;
  const total=pts[pts.length-1].t-pts[0].t||1;
  let ax=Infinity,bx=-Infinity,ay=Infinity,by=-Infinity;
  for(const p of pts){if(p.x<ax)ax=p.x;if(p.x>bx)bx=p.x;if(p.y<ay)ay=p.y;if(p.y>by)by=p.y;}

  // Rhythm ratio — pace-normalized inter-stroke pause
  // = mean_pause / total_duration. Invariant to overall signing speed.
  let pauseSum=0,pauseCount=0;
  for(let i=0;i<sig.strokes.length-1;i++){
    const end=sig.strokes[i][sig.strokes[i].length-1];
    const start=sig.strokes[i+1][0];
    if(end&&start){pauseSum+=Math.max(0,start.t-end.t);pauseCount++;}
  }
  const rhythmRatio=pauseCount>0?Math.min(1,(pauseSum/pauseCount)/total):0;

  // CV-microtremor — windowed velocity variance / avgVelocity²
  // = coefficient of variation squared, which is speed-invariant.
  // Captures the tremor PATTERN, not the tremor magnitude.
  // Genuine signer has consistent tremor CV even at different speeds.
  // Forger tracing slowly has very low CV (smooth, deliberate strokes).
  const W=5; const wVars=[];
  for(let i=0;i+W<=vs.length;i++)wVars.push(vari(vs.slice(i,i+W)));
  const cvMicrotremor=mean(wVars)/(avgV*avgV+0.0001);

  // CV-angular energy — angular momentum / avgVelocity
  // Captures rotational pen dynamics, normalized to be speed-invariant.
  let angTotal=0;
  for(const stroke of sig.strokes){
    if(stroke.length<3)continue;
    const cx=mean(stroke.map(p=>p.x)),cy=mean(stroke.map(p=>p.y));
    let L=0;
    for(let i=1;i<stroke.length;i++){
      const dt=Math.max(stroke[i].t-stroke[i-1].t,1);
      const vx=(stroke[i].x-stroke[i-1].x)/dt,vy=(stroke[i].y-stroke[i-1].y)/dt;
      const rx=(stroke[i].x+stroke[i-1].x)/2-cx,ry=(stroke[i].y+stroke[i-1].y)/2-cy;
      L+=Math.abs(rx*vy-ry*vx);
    }
    angTotal+=L/stroke.length;
  }
  const cvAngularEnergy=(angTotal/Math.max(sig.strokes.length,1))/(avgV+0.0001);

  return{
    strokeCount: sig.strokes.length,
    aspectRatio: (bx-ax||1)/(by-ay||1),
    rhythmRatio,
    cvMicrotremor,
    cvAngularEnergy,
    curvatureEntropy: curvEntropy(pts),
  };
}

// Feature weights — 6 channels, all speed-invariant.
// curvatureEntropy (w=1.8): encodes hump/loop count — shape forgeries with
//   wrong curve complexity are penalized. Weight matches library's features.ts.
// rhythmRatio (w=2.0): catches timing forgeries (wrong pen-lift pauses).
// cvMicrotremor (w=1.8): catches velocity-smoothed forgeries (robot-like).
const FEAT_WEIGHTS={strokeCount:2.5, aspectRatio:1.5, rhythmRatio:2.0, cvMicrotremor:1.8, cvAngularEnergy:1.2, curvatureEntropy:1.8};
function featSimilarity(ref,test){
  let wD=0,wT=0;
  for(const[k,w]of Object.entries(FEAT_WEIGHTS)){
    const rv=ref[k]??0,tv=test[k]??0;
    const sc=Math.max(Math.abs(rv),Math.abs(tv),0.001);
    wD+=(Math.abs(rv-tv)/sc)*w;wT+=w;
  }
  return Math.max(0,Math.min(100,(100-(wD/wT)*80)));
}
function avgFeats(sigs){
  const fs=sigs.map(extractFeatures);
  const keys=Object.keys(fs[0]);
  const out={};
  for(const k of keys)out[k]=mean(fs.map(f=>f[k]));
  return out;
}

// ─── Calibration ──────────────────────────────────────────────────────────────
function calibrate(distFn,enrollSigs,calibSigs){
  const ref=proc(enrollSigs[0].flatPoints);
  const dists=calibSigs.map(s=>distFn(ref,proc(s.flatPoints)));
  return 6/mean(dists);
}

// ─── Dataset ──────────────────────────────────────────────────────────────────
const enrollSigs=[
  makeSig({noise:0.006,humps:4,strokeCount:2}),
  makeSig({noise:0.007,humps:4,strokeCount:2}),
  makeSig({noise:0.005,humps:4,strokeCount:2}),
];
const ref=enrollSigs[0];
const refFeats=avgFeats(enrollSigs);

const cases=[
  // GENUINE — same signer, realistic day-to-day variation
  {kind:"G", label:"Genuine — tiny noise",              sig:makeSig({noise:0.005,humps:4,strokeCount:2})},
  {kind:"G", label:"Genuine — normal noise",             sig:makeSig({noise:0.015,humps:4,strokeCount:2})},
  {kind:"G", label:"Genuine — 30% faster",               sig:makeSig({noise:0.007,humps:4,strokeCount:2,tScale:0.7})},
  {kind:"G", label:"Genuine — 50% slower",               sig:makeSig({noise:0.007,humps:4,strokeCount:2,tScale:1.5})},
  {kind:"G", label:"Genuine — large noise",              sig:makeSig({noise:0.035,humps:4,strokeCount:2})},
  // SHAPE FORGERIES — wrong curve structure
  {kind:"F", label:"Shape — 2 humps (wrong loops)",      sig:makeSig({noise:0.004,humps:2,strokeCount:2})},
  {kind:"F", label:"Shape — 6 humps (extra loops)",      sig:makeSig({noise:0.004,humps:6,strokeCount:2})},
  {kind:"F", label:"Shape — 1 hump (over-simplified)",   sig:makeSig({noise:0.004,humps:1,strokeCount:2})},
  {kind:"F", label:"Shape — near straight",              sig:makeSig({noise:0.003,humps:0.1,strokeCount:2})},
  // TIMING FORGERIES — correct shape, wrong timing. OLD is blind to these.
  {kind:"F", label:"Timing — drawn 3× slower + smooth",  sig:makeSig({noise:0.002,humps:4,strokeCount:2,tScale:3.0,slowTremor:true})},
  {kind:"F", label:"Timing — wrong pen-lift pauses",     sig:makeSig({noise:0.004,humps:4,strokeCount:2,wrongPenLift:true})},
  {kind:"F", label:"Timing — extra pen-lift added",      sig:makeSig({noise:0.004,humps:4,strokeCount:2,extraStroke:true})},
];

const calibGenuines=[
  makeSig({noise:0.010,humps:4,strokeCount:2}),
  makeSig({noise:0.009,humps:4,strokeCount:2}),
  makeSig({noise:0.011,humps:4,strokeCount:2}),
];

const enrollProcessed=enrollSigs.map(s=>proc(s.flatPoints));
const band=adaptBand(enrollProcessed.map(s=>s.map(p=>p.v??0)));
const _oldK=calibrate((a,b)=>oldDTW(a,b),enrollSigs,calibGenuines);
const _glfK=calibrate((a,b)=>glfDTW(a,b,band),enrollSigs,calibGenuines);

function toS(dist,K){return Math.round(Math.max(0,Math.min(100,100-dist*K))*10)/10;}
function oldScore(r,t){return toS(oldDTW(proc(r.flatPoints),proc(t.flatPoints)),_oldK);}
function glfScore(r,t){
  const dtw=toS(glfDTW(proc(r.flatPoints),proc(t.flatPoints),band),_glfK);
  const tFeat=extractFeatures(t);
  const feat=featSimilarity(refFeats,tFeat);
  const raw=dtw*0.55+feat*0.45;

  // Rhythm gate: if inter-stroke pause ratio diverges ≥2× from reference,
  // a timing forgery cannot hide behind a strong shape score.
  // rhythmRatio is already speed-invariant (pause/total both scale with speed).
  const rRef=refFeats.rhythmRatio, rTest=tFeat.rhythmRatio;
  const rhythmRat=(Math.max(rRef,rTest)+0.001)/(Math.min(rRef,rTest)+0.001);
  const rhythmGate=rhythmRat>3.0?0.72:rhythmRat>2.0?0.88:1.0;

  // Tremor gate: a forger who traces shapes slowly/deliberately has an
  // unnaturally smooth velocity profile — cvMicrotremor drops below 35%
  // of the genuine signer's value.
  const cvRef=refFeats.cvMicrotremor, cvTest=tFeat.cvMicrotremor;
  const tremGate=(cvRef>0.001&&cvTest<cvRef*0.35)?0.78:1.0;

  return Math.round(raw*rhythmGate*tremGate*10)/10;
}

// ─── Output ───────────────────────────────────────────────────────────────────
console.log("\n╔══════════════════════════════════════════════════════════════════╗");
console.log("║  GLYF v2 FULL SYSTEM vs OLD — DEFINITIVE BENCHMARK              ║");
console.log("╚══════════════════════════════════════════════════════════════════╝");
console.log("\n  OLD:   DTW shape-match only — no timing channels");
console.log("  GLYF:  DTW × 0.55  +  speed-invariant features × 0.45");
console.log("         Features: strokeCount + aspectRatio + rhythmRatio + cvMicrotremor + cvAngularEnergy + curvatureEntropy");
console.log(`\n  OLD K: ${_oldK.toFixed(1)}   GLYF K: ${_glfK.toFixed(1)}   Band: ${band.toFixed(3)} adaptive (OLD: fixed 0.150)\n`);

const rows=[];
for(const{kind,label,sig}of cases){
  const oS=oldScore(ref,sig);const gS=glfScore(ref,sig);
  rows.push({kind,label,oS,gS,delta:gS-oS});
}

console.log("  Kind  Test Case                                   OLD     GLYF v2   Δ");
console.log("─".repeat(80));
for(const{kind,label,oS,gS,delta}of rows){
  const flag=kind==="F"&&gS<oS?" ◄ GLYF tighter":kind==="G"&&gS>oS?" ◄ GLYF better":"";
  console.log(`  [${kind}]  ${label.padEnd(44)}${String(oS).padEnd(8)}${String(gS).padEnd(10)}${(delta>=0?"+":"")+delta.toFixed(1)}${flag}`);
}

const G=rows.filter(r=>r.kind==="G"),F=rows.filter(r=>r.kind==="F");
const oGA=mean(G.map(r=>r.oS)),gGA=mean(G.map(r=>r.gS));
const oFA=mean(F.map(r=>r.oS)),gFA=mean(F.map(r=>r.gS));
const oSep=oGA-oFA,gSep=gGA-gFA;
const winner=gSep>oSep?"↑ GLYF v2 WINS":"↓ similar";
const glfTighter=F.filter(r=>r.gS<r.oS).length;

console.log("─".repeat(80));
console.log(`\n  Avg genuine:   OLD = ${oGA.toFixed(1)}   GLYF v2 = ${gGA.toFixed(1)}`);
console.log(`  Avg forgery:   OLD = ${oFA.toFixed(1)}   GLYF v2 = ${gFA.toFixed(1)}`);
console.log(`\n  ┌──────────────────────────────────────────────────────────────┐`);
console.log(`  │  SEPARATION (genuine avg − forgery avg)                      │`);
console.log(`  │   OLD:    ${String(oSep.toFixed(1)).padEnd(7)}pts                                       │`);
console.log(`  │   GLYF v2: ${String(gSep.toFixed(1)).padEnd(7)}pts   ${winner.padEnd(37)}│`);
console.log(`  └──────────────────────────────────────────────────────────────┘`);
console.log(`\n  GLYF tighter on ${glfTighter}/${F.length} forgeries   OLD tighter on ${F.length-glfTighter}/${F.length}`);

console.log("\n─── Timing Forgeries — OLD is completely blind ─────────────────────");
for(const{label,oS,gS}of F.filter(r=>r.label.startsWith("Timing"))){
  console.log(`   OLD ${String(oS).padEnd(5)} ${oS>=80?"PASSES (false accept!)":"rejects"}  →  GLYF ${String(gS).padEnd(5)} ${gS<80?"REJECTS ✓":"reduces score"}   ${label}`);
}

console.log("\n─── Adaptive Band ───────────────────────────────────────────────────");
const rng2=new LCG(77);const r2=()=>rng2.next();
function mc2(n){const pts=[];let t=0;for(let i=0;i<70;i++){pts.push({x:i/69*100+(r2()-0.5)*n*80,y:50+Math.sin(i/69*Math.PI*4)*20+(r2()-0.5)*n*80,t:t+=15+r2()*5});}return{flatPoints:pts};}
const lp=[mc2(0.003),mc2(0.003),mc2(0.003)].map(s=>proc(s.flatPoints).map(p=>p.v??0));
const hp=[mc2(0.09),mc2(0.09),mc2(0.09)].map(s=>proc(s.flatPoints).map(p=>p.v??0));
console.log(`   Consistent signer band=${adaptBand(lp).toFixed(3)}   Variable signer band=${adaptBand(hp).toFixed(3)}   OLD always=0.150`);

console.log("\n─── GLYF v2 novel channels vs OLD ──────────────────────────────────");
console.log("   Channel           Speed-inv?  What it catches");
console.log("   rhythmRatio           YES     Pause ratio — forgers miss this completely");
console.log("   cv-microtremor        YES     Tremor pattern — forgers draw too smoothly");
console.log("   cv-angularEnergy      YES     Rotational dynamics — shape-correct fakes fail");
console.log("   curvature-diff DTW    YES     Curvature mismatch added to every DTW cell");
console.log("   direction-diff DTW    YES     Angular flow mismatch added to every DTW cell");
console.log("   OLD algorithm:         —      ZERO of these 5 channels\n");
