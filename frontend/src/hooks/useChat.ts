import {useState} from 'react';
export type Message={role:'assistant'|'user';content:string;kind?:'summary'};
export type Prediction={probability:number;risk_label:string;feature_importances:Record<string,number>;agreement_score:number;model_probabilities:Record<string,number>};
type PendingRequest=
 |{kind:'chat';content:string;messageId:string;restart:boolean}
 |{kind:'research';content:string;screeningId?:number};
const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8000';
const WELCOME='Hi, I’m here to help with PCOS questions and a brief risk screening. You can ask me a question first; if you want to begin, just say “start”. I will ask about waist and hip measurements too, but you can skip them if you don’t know.';

function isRestartCommand(value:string){
 const raw=value.toLowerCase();
 if(['फिर से शुरू','नया टेस्ट','नई चैट','नया स्क्रीनिंग'].some(phrase=>raw.includes(phrase)))return true;
 const text=value.toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
 return /\b(restart|start over|start again|new chat|new conversation|new test|new screening|test again|take test again|start new test|start new screening)\b/.test(text)
  || /\b(phir se shuru|dobara shuru|naya test|nayi chat|फिर से शुरू|नया टेस्ट|नई चैट)\b/.test(text);
}
function normalizeAssistantText(value:string){return value.replace(/\bI[’']ll\b/gi,'I will')}
export function isScreeningStateMessage(value:string){
 const text=value.toLowerCase();
 return /\b(summary|summari[sz]e|recorded|what did i answer|what (?:was|is) my|did i report|have i (?:reported|answered)|which answers|not provided|update my|correct my|change my|my correct|my (?:age|weight|height|period|cycle|acne|hip|waist|hair|skin|exercise)|current.{0,20}(?:estimate|score|risk))\b/.test(text)
  || /(meri summary|mere jawab|maine kya|kya record kiya|sahi wazan|sahi value|galat|badal do)/.test(text);
}

async function responseBody(response:Response):Promise<any>{
 try{return await response.json()}catch{return {}}
}

async function fetchWithRetry(url:string,init:RequestInit):Promise<Response>{
 let lastError:unknown;
 for(let attempt=0;attempt<3;attempt++){
  const controller=new AbortController();
  const timeout=window.setTimeout(()=>controller.abort(),70000);
  try{
   const response=await fetch(url,{...init,signal:controller.signal});
   if(response.status===401)return response;
   const transient=response.status===408||response.status===425||response.status===429||response.status>=500;
   if(!transient||attempt===2)return response;
   lastError=new Error('Temporary service error');
  }catch(error){
   lastError=error;
   if(attempt===2)throw error;
  }finally{window.clearTimeout(timeout)}
  await new Promise(resolve=>window.setTimeout(resolve,500*(attempt+1)));
 }
 throw lastError instanceof Error?lastError:new Error('Temporary service error');
}

function userFacingRequestError(){
 return 'We could not complete that request. Your message is still visible in this chat. Retry it here without refreshing the page.';
}

export function useChat(token:string,onUnauthorized:()=>void){
 const [messages,setMessages]=useState<Message[]>([{role:'assistant',content:WELCOME}]);
 const [sessionId,setSessionId]=useState<string>();const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [complete,setComplete]=useState(false);const [prediction,setPrediction]=useState<Prediction|null>(null);const [awaitingConfirmation,setAwaitingConfirmation]=useState(false);
 const [pendingRequest,setPendingRequest]=useState<PendingRequest|null>(null);
 const send=async(content:string)=>{
  if(!content.trim()||busy)return;const restart=isRestartCommand(content);setError('');setPendingRequest(null);
  if(restart){setMessages([{role:'assistant',content:WELCOME}]);setSessionId(undefined);setComplete(false);setPrediction(null);setAwaitingConfirmation(false)}
  setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  const request:PendingRequest={kind:'chat',content,messageId:typeof crypto.randomUUID==='function'?crypto.randomUUID():`${Date.now()}-${Math.random().toString(36).slice(2)}`,restart};
  const run=async()=>{
   try{
    const response=await fetchWithRetry(`${API}/chat`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({session_id:request.restart?undefined:sessionId,message_id:request.messageId,message:request.content})});
    if(response.status===401){onUnauthorized();return undefined}
    const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`HTTP ${response.status}`);
    const reply=normalizeAssistantText(data.reply||'');setSessionId(data.session_id);setComplete(Boolean(data.complete));if(data.prediction)setPrediction(data.prediction);else if(request.restart)setPrediction(null);setAwaitingConfirmation(Boolean(data.awaiting_confirmation));setMessages(m=>[...m,{role:'assistant',content:reply,kind:data.message_type==='summary'?'summary':undefined}]);setPendingRequest(null);setError('');return reply;
   }catch{setPendingRequest(request);setError(userFacingRequestError());return undefined}finally{setBusy(false)}
  };
  return run();
 };
 const askResearch=async(content:string,screeningId?:number)=>{
  if(!content.trim()||busy)return;setError('');setPendingRequest(null);setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  const request:PendingRequest={kind:'research',content,screeningId};
  const run=async()=>{
   try{
    const response=await fetchWithRetry(`${API}/chat/research`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({question:request.content,conversation:messages.slice(-10),screening_id:request.screeningId})});
    if(response.status===401){onUnauthorized();return undefined}
    const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`HTTP ${response.status}`);
    const answer=normalizeAssistantText(data.answer||'');setMessages(m=>[...m,{role:'assistant',content:answer,kind:request.screeningId!==undefined||/summary/i.test(request.content)?'summary':undefined}]);setPendingRequest(null);setError('');return answer;
   }catch{setPendingRequest(request);setError(userFacingRequestError());return undefined}finally{setBusy(false)}
  };
  return run();
 };
 const retry=async()=>{
  if(!pendingRequest||busy)return undefined;
  const current=pendingRequest;setError('');setBusy(true);
  try{
   const url=current.kind==='chat'?`${API}/chat`:`${API}/chat/research`;
   const body=current.kind==='chat'
    ?{session_id:current.restart?undefined:sessionId,message_id:current.messageId,message:current.content}
    :{question:current.content,conversation:messages.slice(0,-1).slice(-10),screening_id:current.screeningId};
   const response=await fetchWithRetry(url,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify(body)});
   if(response.status===401){onUnauthorized();return undefined}
   const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`HTTP ${response.status}`);
   const reply=normalizeAssistantText(current.kind==='chat'?data.reply||'':data.answer||'');
   if(current.kind==='chat'){setSessionId(data.session_id);setComplete(Boolean(data.complete));if(data.prediction)setPrediction(data.prediction);setAwaitingConfirmation(Boolean(data.awaiting_confirmation));setMessages(m=>[...m,{role:'assistant',content:reply,kind:data.message_type==='summary'?'summary':undefined}])}
   else setMessages(m=>[...m,{role:'assistant',content:reply,kind:current.screeningId!==undefined||/summary/i.test(current.content)?'summary':undefined}]);
   setPendingRequest(null);setError('');return reply;
  }catch{setError(userFacingRequestError());return undefined}finally{setBusy(false)}
 };
 const restart=()=>send('start a new test');
 return {messages,busy,error,send,askResearch,restart,retry,hasPendingRequest:Boolean(pendingRequest),complete,prediction,awaitingConfirmation};
}
