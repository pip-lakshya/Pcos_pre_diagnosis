import {useCallback,useEffect,useRef,useState} from 'react';

export interface VoiceProvider {
 start(onText:(text:string)=>void,onError:(message:string)=>void):void;
 stop():void;
 speak(text:string,voiceName?:string):void;
 cancelSpeech():void;
 prepareSpeech?():void;
}
const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8000';

function sentenceChunks(text:string):string[]{
 const sentences=text.match(/[^.!?]+[.!?]+|[^.!?]+$/g)||[text];const chunks:string[]=[];
 for(const sentence of sentences){let rest=sentence.trim();while(rest.length>220){let split=rest.lastIndexOf(' ',220);if(split<80)split=220;chunks.push(rest.slice(0,split).trim());rest=rest.slice(split).trim()}if(rest)chunks.push(rest)}return chunks;
}
function preferredVoice(voices:SpeechSynthesisVoice[]):SpeechSynthesisVoice|undefined{
 const english=voices.filter(v=>/^en([_-]|$)/i.test(v.lang));const preferredNames=['google uk english female','samantha','zira','aria','jenny','neerja','google us english'];
 for(const name of preferredNames){const found=english.find(v=>v.name.toLowerCase().includes(name));if(found)return found}
 return english.find(v=>v.name.toLowerCase().includes('female'))||english[0]||voices.find(v=>v.default)||voices[0];
}
class BrowserVoiceProvider implements VoiceProvider {
 private recognition:any=null;private activeUtterance:SpeechSynthesisUtterance|null=null;private speechRun=0;
 start(onText:(text:string)=>void,onError:(message:string)=>void){this.cancelSpeech();const C=(window as any).SpeechRecognition||(window as any).webkitSpeechRecognition;if(!C){onError('Speech recognition is not available in this browser.');return;}this.recognition=new C();this.recognition.lang='en-US';this.recognition.interimResults=false;this.recognition.onresult=(e:any)=>onText(e.results[0][0].transcript);this.recognition.onerror=()=>onError('I could not hear that. Please try again.');this.recognition.onend=()=>{this.recognition=null};this.recognition.start();}
 stop(){try{this.recognition?.stop()}catch{}this.recognition=null;this.cancelSpeech()}
 cancelSpeech(){this.speechRun++;if('speechSynthesis'in window){window.speechSynthesis.cancel();this.activeUtterance=null}}
 speak(text:string,voiceName?:string){if(!('speechSynthesis'in window))return;this.cancelSpeech();const run=this.speechRun;const synthesis=window.speechSynthesis;const readyVoices=()=>new Promise<SpeechSynthesisVoice[]>(resolve=>{const available=synthesis.getVoices();if(available.length){resolve(available);return}let settled=false;const finish=()=>{if(settled)return;settled=true;synthesis.removeEventListener('voiceschanged',finish);resolve(synthesis.getVoices())};synthesis.addEventListener('voiceschanged',finish);window.setTimeout(finish,1000)});void readyVoices().then(available=>{if(run!==this.speechRun)return;const english=available.filter(v=>/^en([_-]|$)/i.test(v.lang));const voice=english.find(v=>v.name===voiceName)||preferredVoice(english);if(!voice){this.activeUtterance=null;return}const chunks=sentenceChunks(text);let index=0;const playNext=()=>{if(run!==this.speechRun||index>=chunks.length)return;const utterance=new SpeechSynthesisUtterance(chunks[index++]);utterance.voice=voice;utterance.lang=voice.lang||'en-US';utterance.rate=1;utterance.pitch=1;utterance.onend=playNext;this.activeUtterance=utterance;synthesis.speak(utterance)};playNext()})}
}
class ServerTtsProvider implements VoiceProvider {
 private browser=new BrowserVoiceProvider();private audio:HTMLAudioElement|null=null;private audioUrl:string|null=null;private controller:AbortController|null=null;private token:string;private unauthorized:()=>void;private onError:(message:string)=>void;
 constructor(token:string,unauthorized:()=>void,onError:(message:string)=>void){this.token=token;this.unauthorized=unauthorized;this.onError=onError}
 start(a:(text:string)=>void,b:(message:string)=>void){this.browser.start(a,b)}
 stop(){this.browser.stop();this.cancelSpeech()}
 cancelSpeech(){this.browser.cancelSpeech();this.controller?.abort();this.controller=null;if(this.audio){this.audio.pause();this.audio.src='';this.audio=null}if(this.audioUrl){URL.revokeObjectURL(this.audioUrl);this.audioUrl=null}}
 speak(text:string){this.cancelSpeech();const controller=new AbortController();this.controller=controller;fetch(`${API}/tts`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${this.token}`},body:JSON.stringify({text}),signal:controller.signal}).then(async response=>{if(response.status===401){this.unauthorized();return}if(!response.ok)throw new Error('Speech playback is unavailable.');const blob=await response.blob();if(controller.signal.aborted)return;this.audioUrl=URL.createObjectURL(blob);this.audio=new Audio(this.audioUrl);this.audio.onended=()=>this.cancelSpeech();await this.audio.play()}).catch(error=>{if(error?.name!=='AbortError')this.onError('Spoken playback is unavailable right now.')})}
}

class MagpieTtsProvider implements VoiceProvider {
 private browser=new BrowserVoiceProvider();private controller:AbortController|null=null;private context:AudioContext|null=null;private sources=new Set<AudioBufferSourceNode>();private run=0;private nextStart=0;
 private token:string;private unauthorized:()=>void;private onError:(message:string)=>void;
 constructor(token:string,unauthorized:()=>void,onError:(message:string)=>void){this.token=token;this.unauthorized=unauthorized;this.onError=onError}
 prepareSpeech(){try{this.context??=new AudioContext({sampleRate:22050});void this.context.resume()}catch{}}
 start(onText:(text:string)=>void,onError:(message:string)=>void){this.browser.start(onText,onError)}
 stop(){this.browser.stop();this.cancelSpeech()}
 cancelSpeech(){this.run++;this.controller?.abort();this.controller=null;for(const source of this.sources){try{source.stop()}catch{}}this.sources.clear();this.nextStart=0;this.browser.cancelSpeech()}
 speak(text:string){this.cancelSpeech();this.prepareSpeech();const run=this.run;const controller=new AbortController();this.controller=controller;void this.play(text,run,controller)}
 private async play(text:string,run:number,controller:AbortController){
  try{
   const response=await fetch(`${API}/tts`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${this.token}`},body:JSON.stringify({text}),signal:controller.signal});
   if(response.status===401){this.unauthorized();return}
   if(!response.ok||!response.body)throw new Error('Streaming speech is unavailable.');
   const context=this.context;if(!context)throw new Error('Audio playback is not available in this browser.');
   await context.resume();const reader=response.body.getReader();let carry=new Uint8Array(0);let scheduled=false;
   while(true){
    const {done,value}=await reader.read();if(run!==this.run)return;if(done)break;
    const joined=new Uint8Array(carry.length+value.length);joined.set(carry);joined.set(value,carry.length);const usable=joined.length-(joined.length%2);carry=joined.slice(usable);
    if(!usable)continue;
    const view=new DataView(joined.buffer,joined.byteOffset,usable);const samples=new Float32Array(usable/2);for(let i=0;i<samples.length;i++)samples[i]=view.getInt16(i*2,true)/32768;
    const buffer=context.createBuffer(1,samples.length,22050);buffer.getChannelData(0).set(samples);const source=context.createBufferSource();source.buffer=buffer;source.connect(context.destination);source.onended=()=>this.sources.delete(source);this.sources.add(source);
    const startsAt=Math.max(context.currentTime+0.08,this.nextStart);source.start(startsAt);this.nextStart=startsAt+buffer.duration;scheduled=true;
   }
   if(!scheduled&&run===this.run)throw new Error('NVIDIA returned no speech audio.');
  }catch(error){if((error as {name?:string})?.name!=='AbortError'&&run===this.run)this.onError('Magpie voice playback failed. You can try again or select a browser voice in settings.')}
 }
}

export function useVoice(token:string,onUnauthorized:()=>void){
 const unauthorizedRef=useRef(onUnauthorized);unauthorizedRef.current=onUnauthorized;
 const provider=useRef<VoiceProvider>(new BrowserVoiceProvider());
 const [listening,setListening]=useState(false);const [spoken,setSpokenState]=useState(true);const [voiceError,setVoiceError]=useState('');const [voices,setVoices]=useState<SpeechSynthesisVoice[]>([]);const [selectedVoice,setSelectedVoiceState]=useState(()=>localStorage.getItem('pcos_voice_name')||'');const [ttsProvider,setTtsProvider]=useState<'browser'|'edge'|'magpie'>('browser');
 useEffect(()=>{if(!('speechSynthesis'in window))return;const synthesis=window.speechSynthesis;const refresh=()=>{const values=synthesis.getVoices().filter(v=>/^en([_-]|$)/i.test(v.lang));if(values.length){setVoices(values);setSelectedVoiceState(current=>{const chosen=current&&values.some(v=>v.name===current)?current:preferredVoice(values)?.name||'';if(chosen)localStorage.setItem('pcos_voice_name',chosen);return chosen})}};refresh();synthesis.addEventListener('voiceschanged',refresh);const timer=window.setTimeout(refresh,1200);return()=>{window.clearTimeout(timer);synthesis.removeEventListener('voiceschanged',refresh)}},[]);
 useEffect(()=>{let cancelled=false;fetch(`${API}/tts/config`,{headers:{Authorization:`Bearer ${token}`}}).then(async r=>{if(r.status===401){unauthorizedRef.current();return}if(r.ok){const data=await r.json();if(cancelled)return;const mode=data.provider==='magpie'?'magpie':data.provider==='edge'?'edge':'browser';setTtsProvider(mode);if(mode==='magpie')provider.current=new MagpieTtsProvider(token,()=>unauthorizedRef.current(),setVoiceError);else if(mode==='edge')provider.current=new ServerTtsProvider(token,()=>unauthorizedRef.current(),setVoiceError)}}).catch(()=>{});return()=>{cancelled=true}},[token]);
 const stop=useCallback(()=>{provider.current.stop();setListening(false)},[]);
 const start=useCallback((onText:(text:string)=>void)=>{setVoiceError('');provider.current.cancelSpeech();setListening(true);provider.current.start(text=>{setListening(false);onText(text)},err=>{setListening(false);setVoiceError(err)})},[]);
 const setSelectedVoice=useCallback((name:string)=>{setSelectedVoiceState(name);localStorage.setItem('pcos_voice_name',name)},[]);
 const setSpoken=useCallback((value:boolean)=>{setSpokenState(value);if(!value)provider.current.cancelSpeech()},[]);
 useEffect(()=>()=>provider.current.stop(),[]);
 return {listening,spoken,setSpoken,voiceError,start,stop,voices,selectedVoice,setSelectedVoice,ttsProvider,prepareSpeech:()=>provider.current.prepareSpeech?.(),speak:(s:string)=>{if(spoken)provider.current.speak(s,selectedVoice);else provider.current.cancelSpeech()},cancelSpeech:()=>provider.current.cancelSpeech()};
}
