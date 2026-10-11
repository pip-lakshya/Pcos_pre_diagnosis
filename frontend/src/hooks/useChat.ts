import {useState} from 'react';
export type Message={role:'assistant'|'user';content:string;kind?:'summary'};
export type Prediction={probability:number;risk_label:string;feature_importances:Record<string,number>;agreement_score:number;model_probabilities:Record<string,number>};
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

function readableError(error:unknown):string{
 if(error instanceof TypeError)return `Can't connect to the PCOS backend at ${API}. Check that Uvicorn is running and VITE_API_URL points to it.`;
 return error instanceof Error?error.message:'Something went wrong.';
}
async function responseBody(response:Response):Promise<any>{
 try{return await response.json()}catch{return {}}
}

export function useChat(token:string,onUnauthorized:()=>void){
 const [messages,setMessages]=useState<Message[]>([{role:'assistant',content:WELCOME}]);
 const [sessionId,setSessionId]=useState<string>();const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [complete,setComplete]=useState(false);const [prediction,setPrediction]=useState<Prediction|null>(null);const [awaitingConfirmation,setAwaitingConfirmation]=useState(false);
 const send=async(content:string)=>{
  if(!content.trim()||busy)return;const restart=isRestartCommand(content);setError('');
  if(restart){setMessages([{role:'assistant',content:WELCOME}]);setSessionId(undefined);setComplete(false);setPrediction(null);setAwaitingConfirmation(false)}
  setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  try{
   const response=await fetch(`${API}/chat`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({session_id:restart?undefined:sessionId,message:content})});
   if(response.status===401){onUnauthorized();throw new Error('Your session expired, please log in again.')}
   const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`The screening service returned HTTP ${response.status}.`);
   const reply=normalizeAssistantText(data.reply||'');setSessionId(data.session_id);setComplete(Boolean(data.complete));if(data.prediction)setPrediction(data.prediction);else if(restart)setPrediction(null);setAwaitingConfirmation(Boolean(data.awaiting_confirmation));setMessages(m=>[...m,{role:'assistant',content:reply,kind:data.message_type==='summary'?'summary':undefined}]);return reply;
  }catch(err){setError(readableError(err));return undefined}finally{setBusy(false)}
 };
 const askResearch=async(content:string,screeningId?:number)=>{
  if(!content.trim()||busy)return;setError('');setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  try{
   const response=await fetch(`${API}/chat/research`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({question:content,conversation:messages.slice(-10),screening_id:screeningId})});
   if(response.status===401){onUnauthorized();throw new Error('Your session expired, please log in again.')}
   const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`The research service returned HTTP ${response.status}.`);
   const answer=normalizeAssistantText(data.answer||'');setMessages(m=>[...m,{role:'assistant',content:answer,kind:screeningId!==undefined||/summary/i.test(content)?'summary':undefined}]);return answer;
  }catch(err){setError(readableError(err));return undefined}finally{setBusy(false)}
 };
 const restart=()=>send('start a new test');
 return {messages,busy,error,send,askResearch,restart,complete,prediction,awaitingConfirmation};
}
