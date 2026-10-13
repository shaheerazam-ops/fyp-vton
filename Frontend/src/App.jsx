import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Camera, Check, CheckCircle2, ChevronDown, Download, ImagePlus, LoaderCircle, LockKeyhole, RotateCcw, Shirt, Sparkles, UploadCloud, X, AlertCircle, RefreshCw, MoveRight, SlidersHorizontal } from 'lucide-react';
import './style.css';

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPT = ['image/jpeg', 'image/png', 'image/webp'];
const DEMO_GARMENT = '/demo-garment.jpg';
const DEMO_PERSON = '/demo-person.jpg';

async function urlToFile(url, name) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Could not load the sample image.');
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || 'image/jpeg' });
}
function Brand({small=false}) {return <div className={'brand '+(small?'brand-small':'')}><span className="brand-mark" aria-hidden="true"><span className="brand-mark-rust"/><span className="brand-mark-ink"/></span><span className="brand-word">Fit<span>Layer</span></span></div>}
function PreviewUpload({ label, hint, file, image, onChange, disabled=false, compact=false }) {
  const input = useRef(null), [drag, setDrag] = useState(false);
  const pick = f => { if (f && !disabled) onChange(f); };
  return <div className={'input-photo '+(compact?'compact':'')}>
    <div className="input-photo-heading"><div><strong>{label}</strong><span>{hint}</span></div>{file && <span className="selected-label"><Check size={12}/> Added</span>}</div>
    <div className={'photo-zone '+(image?'with-image ':'')+(drag?'drag ':'')+(disabled?'disabled':'')} role="button" tabIndex={disabled?-1:0} aria-label={`Upload ${label}`} onClick={()=>!disabled&&input.current?.click()} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();input.current?.click();}}} onDragOver={e=>{e.preventDefault();if(!disabled)setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);pick(e.dataTransfer.files?.[0]);}}>
      <input hidden type="file" accept="image/png,image/jpeg,image/webp" ref={input} onChange={e=>{pick(e.target.files?.[0]);e.target.value='';}} />
      {image ? <><img src={image} alt={`${label} selected`} /><span className="replace-chip"><RefreshCw size={13}/> Replace photo</span></> : <div className="upload-placeholder"><span className="upload-symbol"><UploadCloud size={22}/></span><strong>Choose a photo</strong><span>or drag and drop here</span><small>JPG, PNG or WEBP · Max 10MB</small></div>}
    </div>
    {file && <div className="file-attachment"><CheckCircle2 size={15}/><span title={file.name}>{file.name}</span><button type="button" disabled={disabled} aria-label={`Remove ${label}`} onClick={()=>onChange(null)}><X size={15}/></button></div>}
  </div>
}
export default function App() {
  const [step,setStep]=useState('setup');
  const [person,setPerson]=useState(null), [garment,setGarment]=useState(null);
  const [personPreview,setPersonPreview]=useState(''), [garmentPreview,setGarmentPreview]=useState(DEMO_GARMENT);
  const [result,setResult]=useState('');
  const [error,setError]=useState(''),[status,setStatus]=useState('idle'),[online,setOnline]=useState(null),[latency,setLatency]=useState('');
  const [showAdvanced,setShowAdvanced]=useState(false),[category,setCategory]=useState('top'),[description,setDescription]=useState('Blue denim button-down shirt'),[seed,setSeed]=useState('42');
  const controller=useRef(null);
  const busy=status==='loading';
  useEffect(()=>{if(!person){setPersonPreview('');return;}const u=URL.createObjectURL(person);setPersonPreview(u);return()=>URL.revokeObjectURL(u)},[person]);
  useEffect(()=>{if(!garment){setGarmentPreview(DEMO_GARMENT);return;}const u=URL.createObjectURL(garment);setGarmentPreview(u);return()=>URL.revokeObjectURL(u)},[garment]);
  useEffect(()=>()=>{controller.current?.abort()},[]);
  useEffect(()=>()=>{if(result)URL.revokeObjectURL(result)},[result]);
  useEffect(()=>{let mounted=true;fetch('/api/health').then(r=>{if(!r.ok)throw Error('Not ready');return r.json()}).then(()=>mounted&&setOnline(true)).catch(()=>mounted&&setOnline(false));return()=>{mounted=false}},[]);
  function changePerson(file){if(file&&(!ACCEPT.includes(file.type)||file.size>MAX_BYTES)){setError('Choose a JPG, PNG, or WEBP image under 10 MB.');return;}setPerson(file);clearResult();}
  function changeGarment(file){if(file&&(!ACCEPT.includes(file.type)||file.size>MAX_BYTES)){setError('Choose a JPG, PNG, or WEBP image under 10 MB.');return;}setGarment(file);clearResult();}
  function clearResult(){setError('');setStatus('idle');setResult('');setStep('setup');setLatency('');}
  async function useSample(){try{changePerson(await urlToFile(DEMO_PERSON,'sample-person.jpg'));}catch(e){setError(e.message)}}
  async function generate(){if(!person||busy)return;setError('');setStatus('loading');setStep('result');setResult('');const ctrl=new AbortController();controller.current=ctrl;
    try {const selectedGarment=garment||await urlToFile(DEMO_GARMENT,'demo-denim-shirt.jpg');const form=new FormData();form.append('person_photo',person);form.append('garment_photo',selectedGarment);form.append('category',category);form.append('description',description.trim()||'clothing');form.append('seed',seed);
      const response=await fetch('/api/v1/try-on',{method:'POST',body:form,signal:ctrl.signal});if(!response.ok){let msg=await response.text();try{let j=JSON.parse(msg);msg=typeof j.detail==='string'?j.detail:msg}catch{}throw Error(`Generation failed (${response.status}): ${msg.slice(0,180)}`)}const blob=await response.blob();if(!blob.type.startsWith('image/'))throw Error('The backend returned an unexpected response.');setResult(URL.createObjectURL(blob));setLatency(response.headers.get('X-VTON-Latency-Ms')||'');setStatus('done');setOnline(true);
    }catch(e){if(e.name!=='AbortError'){setError(e.message);setStatus('error');}}finally{controller.current=null;}}
  return <main className="widget-page">
    <section className="vton-modal widget-only" aria-label="FitLayer virtual try-on widget">
      <header className="modal-header"><Brand small/><span className="modal-header-caption">VIRTUAL FITTING ROOM</span><span className={'health '+(online===true?'is-on':online===false?'is-off':'')}><i/>{online===true?'API connected':online===false?'API offline':'Checking API'}</span></header>
      <div className="modal-body">
        <div className="modal-heading"><span className="eyebrow"><span className="eyebrow-dot"/> YOUR VIRTUAL FITTING ROOM</span><h2>{step==='result'?(busy?'Creating your look':status==='done'?'This is your look':'We couldn’t finish'):'See it. Wear it.'}</h2><p>{step==='result'?(busy?'Hold tight while we prepare your virtual preview.':status==='done'?'Take a closer look, or try another photo.':'Check the issue below and try again.'):'Add your photo and get a preview wearing this piece.'}</p></div>
        {step==='setup'?<div className="modal-workspace">
          <div className="modal-inputs"><div className="mini-step">01 <span>ADD YOUR PHOTO</span></div><PreviewUpload label="Your photo" hint="A clear, front-facing image works best" file={person} image={personPreview} onChange={changePerson}/><button type="button" className="sample-link" onClick={useSample}><ImagePlus size={16}/> Use sample photo instead <ArrowRight size={15}/></button>
            <div className="fine-divider"/><div className="mini-step">02 <span>YOUR SELECTED ITEM</span></div>
            <div className="selected-product"><div className="selected-product-thumb"><img src={garmentPreview} alt="Selected denim shirt"/></div><div className="selected-product-details"><span>SELECTED GARMENT</span><strong>Everyday denim shirt</strong><small>Washed indigo · Top</small></div><span className="product-check"><Check size={15}/></span></div>
            <button type="button" className="customize-toggle" aria-expanded={showAdvanced} onClick={()=>setShowAdvanced(!showAdvanced)}><SlidersHorizontal size={15}/> Testing options <ChevronDown size={15} className={showAdvanced?'rotated':''}/></button>
            {showAdvanced&&<div className="advanced"><PreviewUpload compact label="Replace garment" hint="For two-image API testing" file={garment} image={garment?garmentPreview:''} onChange={changeGarment}/><div className="adv-row"><label>Category<select value={category} onChange={e=>setCategory(e.target.value)}><option value="top">Top</option><option value="bottom">Bottom</option><option value="dress">Dress</option></select></label><label>Seed<input type="number" value={seed} onChange={e=>setSeed(e.target.value)}/></label></div><label>Description<input value={description} onChange={e=>setDescription(e.target.value)}/></label></div>}
          </div>
          <aside className="modal-visual"><div className="visual-topline"><span>YOUR PREVIEW</span><Sparkles size={17}/></div><div className="visual-inner">{personPreview?<img src={personPreview} alt="Your uploaded photo"/>:<div className="empty-visual"><div className="avatar-graphic"><Camera size={33}/></div><strong>Your fitting room awaits</strong><span>Your photo will appear here</span></div>}</div><div className="visual-bottom"><span>BEFORE</span><span>Ready for the transformation <MoveRight size={15}/></span></div></aside>
        </div>:<div className="result-view"><div className="result-preview">{busy?<div className="result-wait"><div className="processing-orb"><LoaderCircle size={31} className="spin"/></div><strong>Working on your preview</strong><p>Your photos are being processed.</p></div>:result?<img src={result} alt="Generated virtual try-on result"/>:<div className="result-wait"><AlertCircle size={32}/><strong>Unable to generate preview</strong><p>Please try again.</p></div>}</div><div className="result-side"><div className="result-side-label">YOUR TRY-ON</div><h3>{status==='done'?'Meet your new look.':busy?'A little style magic.':'Let’s try that again.'}</h3><p>{status==='done'?'Here’s your virtual preview of the selected garment.':busy?'We’re sending your selected photos to the FitLayer backend.':'Your uploaded images are still here, so you can retry.'}</p><div className="mini-product"><img src={garmentPreview} alt="Selected shirt"/><div><span>TRYING ON</span><strong>Everyday denim shirt</strong></div></div>{status==='done'&&<a className="download-result" href={result} download="fitlayer-tryon.png"><Download size={17}/> Download image</a>}{!busy&&<button className="revisit-button" type="button" onClick={clearResult}><RotateCcw size={16}/> {status==='done'?'Try another photo':'Back to photos'}</button>}{latency&&status==='done'&&<small className="latency">Generated in {latency} ms</small>}</div></div>}
        {error&&<div className="inline-error" role="alert"><AlertCircle size={17}/>{error}</div>}
      </div>
      <footer className="modal-footer"><div className="privacy-notice"><LockKeyhole size={15}/><span>Demo images are sent to your local API</span></div>{step==='setup'?<button type="button" className="generate-cta" disabled={!person||!Number.isSafeInteger(Number(seed))||seed.trim()===''} onClick={generate}>Generate my try-on <Sparkles size={17}/></button>:<span className="footer-stage">{busy?<><LoaderCircle className="spin" size={16}/> Processing</>:status==='done'?<><CheckCircle2 size={16}/> Preview ready</>:null}</span>}</footer>
      <div className="modal-powered">POWERED BY <strong>FITLAYER</strong> <span>·</span> LOCAL CPU DEMO</div>
    </section>
  </main>
}
