"""Serial actual-file paired timing, no scoring or in-loop report writes."""
import argparse,hashlib,json,time
import torch
from probe_v319_value import ROOT,load_solution
from build_v324_candidate import candidate_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',choices=('bank','full','wide'),default='bank');a=p.parse_args()
    torch.set_num_threads(4);src=candidate_source();assert src==(ROOT/'solution/solution_v324.py').read_text()==(ROOT/'solution/solution.py').read_text()
    mods={n:load_solution(ROOT/'solution'/f'solution_v{v}.py') for n,v in (('parent',322),('candidate',324))}
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/linear.pt',weights_only=False);ids=list(range(len(bank)))
    if a.shape=='full':
        from build_model_proxy_dataset import MODELS,build_linear_group
        layers=[(m,l) for m in MODELS for l in m.linear_layers];ids=[0,9]
        for gi in ids:
            model,layer=layers[gi];bank[gi]=build_linear_group(model,layer,2026090317+gi,'online-shape',(1,10,128,512,1024),'checkpoint',ROOT/'local_model_tensor_cache')
    if a.shape=='wide':
        from build_model_proxy_dataset import DEEPSEEK_V4_FLASH,build_linear_group
        from build_fitted_dataset import quantize_nvfp4
        model=DEEPSEEK_V4_FLASH;cache=ROOT/'local_model_tensor_cache';ids=[0]
        g=build_linear_group(model,model.linear_layers[0],2026090317,'online-shape',(1,10,128,512,1024),'checkpoint',cache)
        matrix=torch.load(cache/'deepseek-v4-flash-0731--attn_wo_b_tp8.pt',weights_only=True);g['weight']=quantize_nvfp4(matrix.T.contiguous().float());bank[0]=g
    totals={n:dict(calibration=0.,dynamic=0.) for n in mods};rows=[]
    for gi in ids:
        g=bank[gi];row=dict(group=gi,shape=list(g['weight'][0].shape))
        for name in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
            m=mods[name];t=time.perf_counter();state=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);cal=time.perf_counter()-t
            t=time.perf_counter()
            for packed in g['test_activation_list']:out=m.hif4_dynamic_quantize_activation(*packed,state['activation_state'])
            dyn=time.perf_counter()-t;del out;row[name]=dict(calibration=cal,dynamic=dyn);totals[name]['calibration']+=cal;totals[name]['dynamic']+=dyn
        rows.append(row);print(gi,totals,flush=True)
    ratio=sum(totals['candidate'].values())/sum(totals['parent'].values())
    report=dict(complete=True,sha256=hashlib.sha256(src.encode()).hexdigest(),shape=a.shape,rows=rows,totals=totals,ratio=ratio,score_prediction_allowed=False,warning='Linear-only Mac 4-thread timing; do not multiply whole online runtime by this ratio. wide=transposed checkpoint runtime stress, not actual layer quality.')
    path=ROOT/f'docs/v324-{a.shape}-interface-timing.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('sha256')!=report['sha256']:
            archive=path.with_name(path.stem+'-'+previous['sha256'][:8]+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    path.write_text(json.dumps(report,indent=2)+'\n');print('ratio',ratio)
if __name__=='__main__':main()
