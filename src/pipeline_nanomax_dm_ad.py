"""NanoMAX DM -> AD driver. Solver and experiment implementation live in ptycho_jax/DM."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from ptycho_jax.DM.nanomax import main

if __name__=='__main__':
    raise SystemExit(main())
