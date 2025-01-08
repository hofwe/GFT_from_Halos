from myRead import myRead
import torch
from nbodykit.lab import *
import numpy as np

def edge_func(a):
    return 1./a
def gft(pos, mass, func=edge_func):
    """Returns the Graph Fourier transform of given halo data.

    Args:
        pos (torch.Tensor(dtype='float64') of size (N, 3)): Positions of the halo data.
        mass (torch.Tensor(dtype='float64') of size (N)): Masses of the halo data.
        func (function or lambda): (Optional) The function which calculates edge weights from distances. Defaults to edge_func (inverse).

    Returns:
        np.ndarray(dtype='float64') of size (N): Eigenvalues of the complete graph.
        np.ndarray(dtype='float64') of size (N): Graph Fourier transform.
    """
    distance = torch.cdist(pos, pos)

    diag_mask = torch.eye(distance.size(0), distance.size(1), dtype=torch.bool)
    laplacian = torch.zeros_like(distance)
    laplacian[~diag_mask] = -func(distance[~diag_mask])
    degree = torch.matmul(-laplacian, torch.ones_like(distance[0]))
    laplacian[diag_mask] = degree

    e, v = torch.linalg.eigh(laplacian)
    
    gft = torch.matmul(v, mass)
    return e, gft

def power_spectrum(pos, mass, boxsize=0, Nmesh=256, resampler="cic", dk=0.05, kmin=0.01):
    """Returns the power spectrum of given halo data.

    Args:
        pos (torch.Tensor(dtype='float64') of size (N, 3)): Positions of the halo data.
        mass (torch.Tensor(dtype='float64') of size (N)): Masses of the halo data.
        boxsize (scalar or 3-vector): Size of the box for simulation in Mpc/h
        Nmesh (int): The number of cells per side on the mesh
        resampler (str): The string specifying which resampler interpolation scheme to use
        dk (float): The linear spacing of k bins to use
        kmin (float): The lower edge of the first k bin to use

    Returns:
        np.ndarray(dtype='float64') of size (N): k values.
        np.ndarray(dtype='float64') of size (N): Power spectrum.
    """
    # convert NumPy array to ArrayCatalog
    nhalo=len(mass)
    halocat = np.empty(nhalo, dtype=[("Position", ("f8", 3)),("Mass", "f8")])
    halocat["Position"] = pos.cpu().numpy().copy()
    halocat["Mass"] = mass.cpu().numpy().copy()



    ps_dict = {
        "Nhalo": nhalo,
        "halocat": halocat
    }

    arraycat = ArrayCatalog(ps_dict["halocat"])

    #convert ArrayCatalog to mesh
    if boxsize==0:
        boxsize=Boxsize(pos)
    mesh=arraycat.to_mesh(resampler=resampler, BoxSize=boxsize, Nmesh=Nmesh, weight="Mass", interlaced=True)

    #get power spectrum
    r = FFTPower(mesh, mode='1d', dk=dk, kmin=kmin)
    Pk = r.power
    return Pk["k"], Pk["power"].real

def Boxsize(pos):
    """Returns the boxsize for given position.

    Args:
        pos (torch.Tensor(dtype='float64') of size (N, 3)): Positions of the halo data.

    Returns:
        boxsize (float): Scalar boxsize
    """
    min_pos = torch.min(pos, dim=0)[0]  
    max_pos = torch.max(pos, dim=0)[0]  
    BoxSize = (max_pos - min_pos).max().cpu().numpy()
    return BoxSize

if __name__ == '__main__':
    f_base = '../samples/cola/f5000.properties'
    pos, mass = myRead(f_base)
    
    pos = torch.tensor(pos).to('cuda')
    mass = torch.tensor(mass).to('cuda')

    e, gft = gft(pos, mass, edge_func)
    #print(e)
    #print(gft)

    k,pk=power_spectrum(pos,mass)
    #np.savetxt("k.txt",k)
    #np.savetxt("pk.txt",pk)

