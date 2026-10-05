import {LandingPage,PolicyPage} from './LandingPage';
import PortalApp from './PortalApp';

export default function App(){
 const path=window.location.pathname.replace(/\/+$/,'')||'/';
 if(path==='/app'||path==='/login'||path==='/register')return <PortalApp/>;
 if(path==='/privacy')return <PolicyPage kind="privacy"/>;
 if(path==='/legal')return <PolicyPage kind="legal"/>;
 return <LandingPage/>;
}
