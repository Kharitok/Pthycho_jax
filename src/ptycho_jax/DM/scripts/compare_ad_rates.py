"""Continue saved DM/naive starts at the original NanoMAX AD learning rates."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pipeline_nanomax_upd as base
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.init import make_data
from ptycho_jax.DM.handoff import refine_ad
from ptycho_jax.DM.plotting import save_snapshot
from ptycho_jax.DM.nanomax import save_json


def compare(root,epochs=400):
    root=Path(root)
    config=json.loads((root/'configuration.json').read_text())['reconstruction']
    config['datapath']=Path(config['datapath']);config['output_dir']=root
    config['crop_shape']=tuple(config['crop_shape'])
    if config['crop_center'] is not None:config['crop_center']=tuple(config['crop_center'])
    cfg=base.ReconstructionConfig(**config)
    loaded=base.load_reconstruction_data(cfg)
    data=make_data(loaded['measured_intensities'],loaded['detector_mask'],dtype=jnp.float32)
    summaries=[]
    for case in json.loads((root/'summary.json').read_text()):
        folder=root/case['case']
        with np.load(folder/'dm_factors.npz') as f:
            geom=make_geometry(f['n'],f['delta'],f['sample'].shape,loaded['mean_probe'].shape,cfg.shift_margin)
        result={'case':case['case'],'initialization_defocus_m':case['initialization_defocus_m'],
                'epochs':epochs,'learning_rates':{'sample':.03,'probe_modes':.3,'modal_weights':.003}}
        for stage,label in (('dm','warm'),('initial','naive')):
            with np.load(folder/stage/'parameters.npz') as f:
                params={key:jnp.asarray(f[key]) for key in ('sample','probe_modes','modal_weights')}
                fixed={'sample_positions':jnp.asarray(f['fixed_sample_positions']),
                       'scan_mistakes':jnp.asarray(f['fixed_scan_mistakes'])}
                origin=f['object_pixel_origin']
            final,history,metrics=refine_ad(params,fixed,geom,data,n_steps=epochs,batch_size=cfg.batch_size,
                                           seed=cfg.seed,learning_rates=result['learning_rates'])
            save_snapshot(folder/f'ad_original_rates_{label}',final,fixed,loaded['reconstruction_pix_size'],origin,
                          title=f'Original AD rates, {label} start',frame_indices=loaded['frame_indices'],margin=geom.margin)
            np.savez_compressed(folder/f'ad_original_rates_{label}'/'history.npz',model_mse=history)
            result[label]=metrics
        summaries.append(result)
        print('ORIGINAL_RATES',result,flush=True)
        save_json(root/'original_ad_rates_summary.json',summaries)
    lines=['# Original AD learning-rate control','','Sample/probe/weight rates: 0.03 / 0.3 / 0.003. '
           'Same fixed positions, two modes, full shot matrices and equal AD epochs for both starts. '
           'These are additional continuations from the saved DM/initial parameters; DM is not rerun.','',
           '| Initial defocus | Initial AD from DM | Final AD from DM | Initial naive AD | Final naive AD |',
           '| --- | --- | --- | --- | --- |']
    for r in summaries:lines.append(f"| {r['initialization_defocus_m']*1e3:.3f} mm | {r['warm']['initial_loss']:.6g} | "
        f"{r['warm']['final_loss']:.6g} | {r['naive']['initial_loss']:.6g} | {r['naive']['final_loss']:.6g} |")
    (root/'original_ad_rates_comparison.md').write_text('\n'.join(lines)+'\n')
    return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);p.add_argument('--epochs',type=int,default=400)
    a=p.parse_args();compare(a.run_dir,a.epochs)
