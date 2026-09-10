const base='http://127.0.0.1:4317/api';
const get=async(p,b)=>{const r=await fetch(base+p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:undefined);const j=await r.json();if(!r.ok)throw Error(JSON.stringify(j));return j};
console.log(await get('/preflight'));
const run=await get('/runs',{name:'验收 · SASRec CPU 真实闭环',mode:'real',minutes:30,callBudget:10,rounds:1,epochs:2,trainUsers:512,evalUsers:256});
console.log('RUN_ID='+run.id);console.log(await get('/runs/'+run.id+'/control',{action:'start'}));
