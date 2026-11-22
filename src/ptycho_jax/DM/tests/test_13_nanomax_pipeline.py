import json
import h5py
import numpy as np
from ptycho_jax.DM import nanomax


def test_nanomax_DM_AD_two_defocus_guesses_and_fixed_positions(tmp_path):
    path=tmp_path/'scan.h5'
    y,x=np.indices((24,24))
    frame=(500*np.exp(-((x-12)**2+(y-12)**2)/15)+1).astype(np.uint32)
    frames=np.stack([frame,frame*2,frame*3]);frames[:,0,0]=np.iinfo(np.uint32).max
    with h5py.File(path,'w') as f:
        f['/entry/measurement/eiger4m/frames']=frames
        f['/entry/measurement/sx']=[-.1,0,.1]
        f['/entry/measurement/sy']=[-.1,.1,0]
        f['/entry/snapshots/pre_scan/energy_raw']=[10000.]
    out=tmp_path/'run'
    assert nanomax.main(['--datapath',str(path),'--output-dir',str(out),'--crop-shape','16','16',
                         '--crop-center','12','12','--dm-iterations','2','--ad-steps','2',
                         '--batch-size','2','--dm-inner','1','--cg-iterations','3','--device','cpu'])==0
    results=json.loads((out/'summary.json').read_text())
    assert len(results)==2
    for result,factor in zip(results,(1.,1.3)):
        assert result['nominal_defocus_m']==.0015
        assert result['initialization_defocus_m']==.0015*factor
        assert result['positions_frozen']
        assert result['handoff_relative_difference']<1e-4
        folder=out/result['case']
        with np.load(folder/'initial/parameters.npz') as initial,np.load(folder/'ad/parameters.npz') as final:
            np.testing.assert_array_equal(initial['fixed_scan_mistakes'],final['fixed_scan_mistakes'])
            assert final['probe_modes'].shape==(2,16,16)
            assert final['modal_weights'].shape==(3,2,2)
        assert (folder/'convergence.png').stat().st_size>1000
    assert (out/'results.md').is_file()
