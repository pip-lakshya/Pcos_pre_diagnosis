import {Component,type ErrorInfo,type ReactNode} from 'react';

type Props={children:ReactNode};
type State={failed:boolean};

export class ChatErrorBoundary extends Component<Props,State>{
 state:State={failed:false};
 static getDerivedStateFromError(){return {failed:true}}
 componentDidCatch(error:Error,info:ErrorInfo){console.error('Chat interface rendering failed',error,info.componentStack)}
 render(){
  if(this.state.failed)return <main className="grid min-h-screen place-items-center bg-bg px-5 text-ink"><section role="alert" className="max-w-md rounded-2xl border border-accent-soft bg-white p-6 text-center"><h1 className="text-lg font-semibold">The chat could not be displayed</h1><p className="mt-2 text-sm">Refresh the page to try again. Your saved screening history is not deleted.</p><button onClick={()=>window.location.reload()} className="mt-4 rounded-xl bg-accent px-4 py-2 text-sm font-medium">Refresh page</button></section></main>;
  return this.props.children;
 }
}
