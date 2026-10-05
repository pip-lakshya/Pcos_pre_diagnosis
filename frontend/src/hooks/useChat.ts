import {useState} from 'react';
export type Message={role:'assistant'|'user';content:string};
export type Prediction={probability:number;risk_label:string;feature_importances:Record<string,number>;agreement_score:number;model_probabilities:Record<string,number>};
const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8000';

function readableError(error:unknown):string{
 if(error instanceof TypeError)return `Can't connect to the PCOS backend at ${API}. Check that Uvicorn is running and VITE_API_URL points to it.`;
 return error instanceof Error?error.message:'Something went wrong.';
}
async function responseBody(response:Response):Promise<any>{
 try{return await response.json()}catch{return {}}
}

export function useChat(token:string,onUnauthorized:()=>void){
 const [messages,setMessages]=useState<Message[]>([{role:'assistant',content:'Hi, I’m here to help with PCOS questions and a brief risk screening. You can ask me a question first; if you want to begin, just say “start”. I’ll ask about waist and hip measurements too, but you can skip them if you don’t know.'}]);
 const [sessionId,setSessionId]=useState<string>();const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [complete,setComplete]=useState(false);const [prediction,setPrediction]=useState<Prediction|null>(null);const [awaitingConfirmation,setAwaitingConfirmation]=useState(false);
 const send=async(content:string)=>{
  if(!content.trim()||busy)return;setError('');setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  try{
   const response=await fetch(`${API}/chat`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({session_id:sessionId,message:content})});
   if(response.status===401){onUnauthorized();throw new Error('Your session expired, please log in again.')}
   const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`The screening service returned HTTP ${response.status}.`);
   setSessionId(data.session_id);setComplete(Boolean(data.complete));setPrediction(data.prediction||null);setAwaitingConfirmation(Boolean(data.awaiting_confirmation));setMessages(m=>[...m,{role:'assistant',content:data.reply}]);return data.reply as string;
  }catch(err){setError(readableError(err));return undefined}finally{setBusy(false)}
 };
 const askResearch=async(content:string)=>{
  if(!content.trim()||busy)return;setError('');setMessages(m=>[...m,{role:'user',content}]);setBusy(true);
  try{
   const response=await fetch(`${API}/chat/research`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify({question:content,conversation:messages.slice(-10)})});
   if(response.status===401){onUnauthorized();throw new Error('Your session expired, please log in again.')}
   const data=await responseBody(response);if(!response.ok)throw new Error(data.detail||`The research service returned HTTP ${response.status}.`);
   setMessages(m=>[...m,{role:'assistant',content:data.answer}]);return data.answer as string;
  }catch(err){setError(readableError(err));return undefined}finally{setBusy(false)}
 };
 return {messages,busy,error,send,askResearch,complete,prediction,awaitingConfirmation};
}
