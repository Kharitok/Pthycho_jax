"""Standalone NanoMAX DM reconstruction; no AD optimization is performed."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
from pathlib import Path
import jax
import numpy as np
import pipeline_nanomax_upd as base
from ptycho_jax.DM.nanomax import make_start,save_json
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.dm import run_dm
from ptycho_jax.DM.handoff import to_ad_parameters,evaluate_ad
from ptycho_jax.DM.plotting import save_snapshot

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datapath',type=Path,default=base.DEFAULT_DATAPATH)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--probe-defocus-m',type=float,default=.0015)
    parser.add_argument('--crop',type=int,default=512)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Output already exists; choose a fresh directory.')
    args.output_dir.mkdir(parents=True)
    cfg=base.ReconstructionConfig(datapath=args.datapath,output_dir=args.output_dir,crop_shape=(args.crop,args.crop))
    loaded=base.load_reconstruction_data(cfg)
    state,data,geom,origin=make_start(loaded,cfg,args.probe_defocus_m)
    state,history=run_dm(state,data,geom,DMConfig(),args.iterations,debug_dir=args.output_dir/'debug')
    params,fixed=to_ad_parameters(state,geom)
    save_snapshot(args.output_dir,params,fixed,loaded['reconstruction_pix_size'],origin,
                   title=f'DM only, {args.iterations} iterations',frame_indices=loaded['frame_indices'],margin=geom.margin)
    np.savez_compressed(args.output_dir/'history.npz',**history)
    metrics=evaluate_ad(params,fixed,geom,data)
    save_json(args.output_dir/'metrics.json',{'iterations':args.iterations,'reference_mse':metrics['loss'],
                                            'probe_defocus_m':args.probe_defocus_m,'positions_frozen':True})
