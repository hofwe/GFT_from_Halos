import h5py
import numpy as np

def myRead(f_base):
    """Summarizes VELOCIRaptor hdf5 halo data and returns the positions and the masses.

    Args:
        f_base (str): A prefix of the name of the halo data.

    Returns:
        np.ndarray(dtype='float64') of size (N, 3): Positions
        np.ndarray(dtype='float64') of size (N): Masses
    """
    pos = []
    mass = []
    
    for i in range(2):
        filename = f_base + '.' + str(i)
        with h5py.File(filename) as f:
            x = f['/Xc'][:]
            y = f['/Yc'][:]
            z = f['/Zc'][:]
            m = f['/Mass_FOF'][:]
            pos.append(np.stack([x, y, z]))
            mass.append(m)

    pos = np.concatenate(pos, axis=1).T
    mass = np.concatenate(mass)
    return pos, mass

