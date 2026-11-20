"""
Implementation of a data loader for NanoMax HDF5 datasets using h5py and hdf5plugin.
Provides convenient access to commonly used datasets such as detector frames, scan positions, detector distance, energy, and defocus.
Supports context-managed file access to ensure proper resource handling.
This module defines two main classes:
- `H5DataLoader`: A context-managed HDF5 file loader that ensures proper opening and closing of files.
- `NanoMaxDataset`: A high-level interface for accessing NanoMax-specific datasets, providing properties for detector frames, scan positions, detector distance, energy, and defocus.
This module is intended to simplify the process of loading and accessing NanoMax experimental data stored in HDF5 format, abstracting away the low-level file handling and providing a user-friendly interface for common datasets.

Example usage:

```python
from data_loading.nanomax_dataloader import NanoMaxDataset

datapath = "path/to/nanomax/data.h5"
dataset = NanoMaxDataset(datapath)
detector_data = dataset.detector_data
scan_x_data = dataset.scan_x_data
scan_y_data = dataset.scan_y_data
```

# if anything is changed in the hdf5 structure, inherit from `NanoMaxDataset` and override the relevant dataset paths accordingly
as in

```python
class CustomNanoMaxDataset(NanoMaxDataset):
    detector_path: str = "/entry/measurement/custom_eiger4m/frames"
    scan_x_path: str = "/entry/measurement/custom_sx"
    scan_y_path: str = "/entry/measurement/custom_sy"
```

# The `CustomNanoMaxDataset` class can now be used in the same way as `NanoMaxDataset`, but it will read from the custom dataset paths specified.


"""



from pathlib import Path

import h5py
import hdf5plugin


class H5DataLoader:
    def __init__(self, filepath: str | Path, mode: str = "r") -> None:
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: h5py.File | None = None

    def __enter__(self) -> "H5DataLoader":
        if self._file is not None:
            raise RuntimeError("H5DataLoader context is already open.")
        self._file = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError(
                "Cannot access 'self.file' outside of a 'with' block."
            )
        return self._file

    def read_dataset(self, key: str):
        return self.file[key][:]


class NanoMaxDataset:

    detector_path: str = "/entry/measurement/eiger4m/frames"
    scan_x_path: str = "/entry/measurement/sx"
    scan_y_path: str = "/entry/measurement/sy"
    detector_distance_path: str = "/entry/measurement/detector_distance"
    energy_path: str = "/entry/measurement/energy"
    defocus_path: str = "/entry/measurement/defocus"




    def __init__(self, datapath: str | Path) -> None:
        self.datapath = Path(datapath)

    def __getitem__(self, key: str):
        # Context manager lives inside the class; file closes automatically on exit
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(key)

    @property
    def detector_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.detector_path)
    
    @property
    def scan_x_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.scan_x_path)

    @property
    def scan_y_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.scan_y_path)

    @property
    def detector_distance_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.detector_distance_path)

    @property
    def energy_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.energy_path)

    @property
    def defocus_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.defocus_path)



