"""Reproducible reference-generated A–F validation plus warm/naive AD trials."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from ptycho_jax.DM.tests.synthetic import make_case
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.dm import run_dm
from ptycho_jax.DM.init import initialize_state
from ptycho_jax.DM.stages import exit_waves
from ptycho_jax.DM.metrics import amplitude_metrics,aligned_object_error,coherence_overlap
from ptycho_jax.DM.operators import coverage_map,object_diagonal
from ptycho_jax.DM.handoff import to_ad_parameters,evaluate_ad,refine_ad


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def run_ladder(output,n_iter=200,ad_steps=50,frame=64,side=10):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for label in ('A','B','C','D','E','F_fixed','F_per_position'):
        case=label[0]
        c=make_case(case,frame=frame,side=side,seed=10)
        g,d=c['geom'],c['data'];k=len(c['probe'])
        state=initialize_state(c['probe_guess'],k,d,g,seed=12)
        cfg=DMConfig(n_inner=3,object_cg_iterations=8)
        if label in ('A','B','F_fixed'):cfg=replace(cfg,weight_strategy='fixed')
        if label=='C':cfg=replace(cfg,per_position_weights_start=10**9)
        if label in ('A','B'):
            # A/B validate the known fixed-weight inverse problem. Freezing
            # measured transmitted counts instead would prescribe wrong flux.
            amplitudes=jnp.array(c['a'],copy=True)
            state=state._replace(a=amplitudes,psi=exit_waves(state.probe,state.obj,amplitudes,g))
        if label=='F_fixed':
            amplitudes=jnp.broadcast_to(jnp.sqrt(jnp.mean(c['a']**2,axis=0)),state.a.shape)
            state=state._replace(a=amplitudes,psi=exit_waves(state.probe,state.obj,amplitudes,g))
        initial_metrics=amplitude_metrics(state.psi,d)
        naive,fixed=to_ad_parameters(state,g)
        start=time.perf_counter()
        state,history=run_dm(state,d,g,cfg,n_iter,log_every=20)
        elapsed=time.perf_counter()-start
        recovered=amplitude_metrics(exit_waves(state.probe,state.obj,state.a,g),d)
        truth=amplitude_metrics(exit_waves(c['probe'],c['obj'],c['a'],g),d)
        params,fx=to_ad_parameters(state,g)
        handoff=evaluate_ad(params,fx,g,d,batch_size=10)
        _,warm_history,warm=refine_ad(params,fx,g,d,n_steps=ad_steps,batch_size=10)
        _,cold_history,cold=refine_ad(naive,fixed,g,d,n_steps=ad_steps,batch_size=10)
        p=np.asarray(state.probe);norms=np.linalg.norm(p.reshape(k,-1),axis=1)
        punit=p/np.maximum(norms[:,None,None],1e-30)
        rec_w=np.asarray(state.a)**2*norms[None]**2;true_w=np.asarray(c['a'])**2
        overlap=abs(punit.reshape(k,-1).conj()@np.asarray(c['probe']).reshape(k,-1).T)
        row,col=linear_sum_assignment(-overlap)
        correlations=[]
        for i,j in zip(row,col):
            r=(rec_w[:,i]/rec_w.sum(axis=1)).astype(float)
            t=(true_w[:,j]/true_w.sum(axis=1)).astype(float)
            correlation=None
            if overlap[i,j]>=.8 and np.ptp(r)>1e-8 and np.ptp(t)>1e-8:
                correlation=float(np.corrcoef(r,t)[0,1])
            correlations.append({'recovered_mode':int(i),'true_mode':int(j),'overlap':float(overlap[i,j]),'relative_weight_correlation':correlation})
        illumination=np.asarray(object_diagonal(c['probe'],c['a'],g))
        evaluation_mask=illumination>.05*illumination.max()
        result={'case':label,'seed':10,'frame':frame,'positions':len(g.n),'iterations':n_iter,
            'initial_mse':float(initial_metrics['model_mse']),'dm_mse':float(recovered['model_mse']),
            'model_r':float(recovered['model_r']),'noise_floor_r':float(truth['model_r']),
            'object_error':aligned_object_error(state.obj,c['obj'],evaluation_mask),
            'object_error_full_geometric_coverage':aligned_object_error(state.obj,c['obj'],coverage_map(g)>0),
            'evaluation_support':'truth normal diagonal above 5% of maximum, fixed before recovery',
            'probe_coherence_overlap':coherence_overlap(p,rec_w.mean(axis=0)/(norms**2),c['probe'],true_w.mean(axis=0)),
            'weight_matches':correlations,'handoff_mse':handoff['loss'],
            'handoff_relative_difference':float(abs(handoff['loss']-float(recovered['model_mse']))/max(float(recovered['model_mse']),1e-30)),
            'warm_ad_final_mse':warm['final_loss'],'naive_ad_final_mse':cold['final_loss'],
            'ad_steps':ad_steps,'dm_seconds_including_compile':elapsed}
        threshold={'A':1e-3,'B':3e-3,'C':3e-3,'D':5e-3,'E':1.1*float(truth['model_r']),'F':5e-3}[case]
        result['indicative_R_pass']=result['model_r']<threshold
        summaries.append(result)
        folder=output/label;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'arrays.npz',obj=np.asarray(state.obj),probe=p,a=np.asarray(state.a),
            truth_obj=np.asarray(c['obj']),truth_probe=np.asarray(c['probe']),truth_a=np.asarray(c['a']),
            **{f'dm_{key}':value for key,value in history.items()},warm_ad_loss=warm_history,naive_ad_loss=cold_history)
        (folder/'metrics.json').write_text(json.dumps(result,default=serial,indent=2,allow_nan=False)+'\n')
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        axes[0].semilogy(history['model_r']);axes[0].set(title=label+' DM model R',xlabel='Iteration')
        axes[1].imshow(np.angle(c['obj']),cmap='twilight');axes[1].set_title('True phase')
        axes[2].imshow(np.angle(state.obj),cmap='twilight');axes[2].set_title('Recovered phase')
        fig.savefig(folder/'reconstruction.png',dpi=140);plt.close(fig)
        print('LADDER',json.dumps(result,default=serial),flush=True)
        (output/'summary.json').write_text(json.dumps(summaries,default=serial,indent=2,allow_nan=False)+'\n')
    lines=['# Synthetic validation ladder','','All intensities were generated by the untouched AD reference. Positions stay fixed.','',
           '| Case | R | Truth noise R | Object error | Coherence overlap | DM MSE | AD after DM | AD naive | R threshold |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in summaries:
        lines.append(f"| {s['case']} | {s['model_r']:.4g} | {s['noise_floor_r']:.4g} | {s['object_error']:.4g} | "
                     f"{s['probe_coherence_overlap']:.5f} | {s['dm_mse']:.5g} | {s['warm_ad_final_mse']:.5g} | "
                     f"{s['naive_ad_final_mse']:.5g} | {s['indicative_R_pass']} |")
    lines+=['','Object error removes scalar and affine phase on a fixed illuminated support (normal diagonal >5% maximum); '
            'the full-geometric-coverage error is also retained in JSON. Unaligned translation is not removed. '
            'Mode-specific correlations are reported only for matched modes (overlap >=0.8); '
            'unmatched/constant cases are null. F_fixed is also the negative control for the same varying-weight truth used in D. '
            'Thresholds are indicative, and a failed entry remains visible rather than changing a tolerance. '
            'The scalar/mode gauges can limit parameter recovery even when the measured-data fit is accurate.','']
    (output/'ladder_report.md').write_text('\n'.join(lines))
    return summaries

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/dm_validation_ladder'))
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--ad-steps',type=int,default=50)
    parser.add_argument('--frame',type=int,default=64)
    parser.add_argument('--side',type=int,default=10)
    args=parser.parse_args()
    run_ladder(args.output,args.iterations,args.ad_steps,args.frame,args.side)
