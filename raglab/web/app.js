const $=id=>document.getElementById(id);let generation=0;let runPoll;
const show=(id,value)=>$(id).textContent=typeof value==='string'?value:JSON.stringify(value,null,2);
async function api(path,body,method=body?'POST':'GET',headers={}){const token=$('bearer-token').value;const res=await fetch('/api/v1/'+path,{method,headers:{'Content-Type':'application/json',...(token?{Authorization:'Bearer '+token}:{}),...headers},body:body?JSON.stringify(body):undefined});const value=await res.json();if(!res.ok)throw new Error(`${res.status} · ${value.error||'request_failed'}: ${value.message||JSON.stringify(value.details)}`);return value;}
function bind(id,fn){$(id).addEventListener('click',async()=>{$('error').hidden=true;$(id).disabled=true;try{await fn();}catch(e){show('error',e.message);$('error').hidden=false;}finally{$(id).disabled=false;}});}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(s=>s.hidden=s.id!==b.dataset.tab);document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('selected',x===b));});
function params(){return{top_k:Number($('top-k').value),min_relevance:Number($('min-score').value)};}
function renderAnswer(result){$('answer-panel').hidden=false;show('answer-status',`${result.status} · ${result.reason||'本次证据通过检查'} · ${result.validation}`);show('answer-text',result.answer);$('evidence').replaceChildren();for(const cite of result.citations){const card=document.createElement('div');card.className='evidence-card';const title=document.createElement('small');title.textContent=`${cite.source} · ${cite.chunk_id.slice(0,12)} · index ${cite.index_version.slice(0,8)}`;const quote=document.createElement('p');quote.textContent=cite.quote;card.append(title,quote);$('evidence').append(card);}show('answer-detail',{resolved_question:result.resolved_question,trace_id:result.trace_id,stages_ms:result.stages_ms,degradation:result.degradation});}
bind('build',async()=>show('index-result',await api('index/rebuild',{chunk_size:Number($('chunk-size').value),overlap:Number($('overlap').value)})));
bind('ask',async()=>{const own=++generation;const result=await api('answer',{question:$('question').value,conversation_id:$('conversation').value,params:params()});if(own===generation)renderAnswer(result);});
bind('compare',async()=>{const result=await api('compare',{question:$('question').value,left:params(),right:{...params(),top_k:6,mmr_lambda:0.7}});$('compare-panel').hidden=false;show('compare-result',Object.fromEntries(Object.entries(result).map(([k,v])=>[k,{status:v.status,evidence:v.evidence.map(h=>({source:h.chunk.source,relevance:h.relevance,fusion:h.fusion_score,channels:h.channels})),stages_ms:v.stages_ms}])));});
function editMemory(card,m){
  const editor=document.createElement('div'),input=document.createElement('textarea');
  input.value=m.content;input.maxLength=2000;input.setAttribute('aria-label','纠正记忆内容');
  const save=document.createElement('button'),cancel=document.createElement('button');
  save.textContent='保存纠正';cancel.textContent='取消编辑';cancel.className='secondary';
  save.onclick=async()=>{try{await api('memories/'+m.memory_id,{content:input.value,expected_version:m.version},'PATCH');renderMemories((await api('memories')).memories);}catch(e){show('error',e.message);$('error').hidden=false;}};
  cancel.onclick=()=>editor.remove();editor.append(input,save,cancel);card.append(editor);input.focus();return false;
}
function renderMemories(memories){
  $('memory-list').replaceChildren();if(!memories.length)show('memory-list','暂无匹配的有效记忆。');
  for(const m of memories){
    const card=document.createElement('div');card.className='memory-card';
    const p=document.createElement('p');p.textContent=m.content;
    const note=document.createElement('small');note.textContent=`${m.kind} · v${m.version} · ${m.enabled?'启用':'禁用'} · ${m.source}`;card.append(p,note);
    for(const [label,action] of [['纠正',()=>editMemory(card,m)],[m.enabled?'禁用':'启用',()=>api('memories/'+m.memory_id,{enabled:!m.enabled,expected_version:m.version},'PATCH')],['删除',()=>api('memories/'+m.memory_id,null,'DELETE')]]){
      const button=document.createElement('button');button.className='secondary';button.textContent=label;
      button.onclick=async()=>{try{if(await action()!==false)renderMemories((await api('memories')).memories);}catch(e){show('error',e.message);$('error').hidden=false;}};card.append(button);
    }$('memory-list').append(card);
  }
}
bind('save-memory',async()=>{await api('memories',{content:$('memory-content').value,kind:$('memory-kind').value});renderMemories((await api('memories')).memories);});
bind('refresh-memory',async()=>renderMemories((await api('memories')).memories));
bind('search-memory',async()=>renderMemories((await api('memories/search',{question:$('memory-query').value})).memories));
function renderRun(run){$('run-id').value=run.run_id;show('run-summary',`${run.status} · epoch ${run.topic_epoch} · ${run.recoverable?'可恢复':run.resume_reason||'无待恢复步骤'}`);show('run-result',run);if(['succeeded','partial','failed','cancelled','timed_out','needs_reconciliation','awaiting_review','interrupted'].includes(run.status))clearInterval(runPoll);}
function poll(){clearInterval(runPoll);runPoll=setInterval(async()=>{try{renderRun(await api('agent/runs/'+$('run-id').value));}catch(e){clearInterval(runPoll);show('error',e.message);$('error').hidden=false;}},600);}
bind('submit-run',async()=>{const body={message:$('run-message').value,conversation_id:$('run-conversation').value,switch_topic:$('switch-topic').checked,pause_for_review:$('pause').checked,timeout_seconds:120};if($('run-plan').value.trim())body.tasks=JSON.parse($('run-plan').value);renderRun(await api('agent/runs',body,'POST',{'Idempotency-Key':crypto.randomUUID().replaceAll('-','')}));poll();});
bind('get-run',async()=>renderRun(await api('agent/runs/'+$('run-id').value)));
bind('cancel-run',async()=>renderRun(await api('agent/runs/'+$('run-id').value+'/cancel',{})));
bind('resume-run',async()=>{renderRun(await api('agent/runs/'+$('run-id').value+'/resume',{approve:true}));poll();});
bind('refresh-diagnostics',async()=>{show('diagnostic-result',await api('diagnostics'));show('trace-result',await api('traces'));});
function refreshIdentity(){api('config').then(c=>{show('mode',`${c.mode} · ${c.embedding} · ${c.generator} · ${c.principal}`);$('build').disabled=c.auth_mode==='bearer';}).catch(e=>show('mode',e.message));api('index').then(x=>show('index-result',x)).catch(()=>{});}
$('bearer-token').addEventListener('change',refreshIdentity);refreshIdentity();
