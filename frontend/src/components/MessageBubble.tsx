import type {ReactNode} from 'react';
import type {Message} from '../hooks/useChat';

function inline(text:string):ReactNode[]{
 return text.split(/(\*\*[^*]+\*\*|__[^_]+__)/g).filter(Boolean).map((part,index)=>{
  const bold=(part.startsWith('**')&&part.endsWith('**'))||(part.startsWith('__')&&part.endsWith('__'));
  return bold?<strong key={index}>{part.slice(2,-2)}</strong>:<span key={index}>{part}</span>;
 });
}

function tableCells(line:string){return line.trim().replace(/^\||\|$/g,'').split('|').map(cell=>cell.trim())}
function isSeparatorRow(line:string){return tableCells(line).every(cell=>/^:?-{3,}:?$/.test(cell))}

export function MessageBubble({message}: {message:Message}){
 const user=message.role==='user';
 const lines=message.content.split('\n');
 const body:ReactNode[]=[];
 for(let index=0;index<lines.length;){
  const line=lines[index].trim();
  if(!line){index++;continue}
  if(line.startsWith('|')){
   const rows:string[][]=[];
   while(index<lines.length&&lines[index].trim().startsWith('|')){
    const current=lines[index].trim();
    if(!isSeparatorRow(current))rows.push(tableCells(current));
    index++;
   }
   const headers=rows.shift()||[];
   body.push(<div key={`table-${index}`} className="my-2 space-y-2 rounded-xl bg-bg/70 p-3">{rows.map((row,rowIndex)=><div key={rowIndex} className="grid gap-2 border-b border-accent-soft pb-2 last:border-0 last:pb-0">{row.map((cell,cellIndex)=><p key={cellIndex} className="text-sm leading-5">{headers[cellIndex]&&<strong className="mr-1">{headers[cellIndex]}:</strong>}{inline(cell)}</p>)}</div>)}</div>);
   continue;
  }
  const heading=line.match(/^#{1,3}\s+(.+)/);
  const bullet=line.match(/^(?:[-*]|\d+\.)\s+(.+)/);
  if(heading)body.push(<h3 key={index} className="mt-2 font-semibold">{inline(heading[1])}</h3>);
  else if(bullet)body.push(<p key={index} className="pl-3 before:mr-2 before:content-['•']">{inline(bullet[1])}</p>);
  else body.push(<p key={index}>{inline(line)}</p>);
  index++;
 }
 return <div className={`flex ${user?'justify-end':'justify-start'}`}><div className={`max-w-[90%] space-y-2 rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${user?'rounded-br-md bg-accent text-ink':'rounded-bl-md border border-accent-soft bg-white text-ink'}`}>{body}</div></div>;
}
