from myRead import myRead
import torch


def edge_func(a):
    return 1./a
def gft(pos, mass, func=edge_func):
    """Returns the Graph Fourier transform of given halo data.

    Args:
        pos (np.ndarray(dtype='float64') of size (N, 3)): Positions of the halo data.
        mass (np.ndarray(dtype='float64') of size (N)): Masses of the halo data.
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

def power_spectrum(pos, mass):
    """Returns the power spectrum of given halo data.

    Args:
        pos (np.ndarray(dtype='float64') of size (N, 3)): Positions of the halo data.
        mass (np.ndarray(dtype='float64') of size (N)): Masses of the halo data.

    Returns:
        np.ndarray(dtype='float64') of size (N): Power spectrum.
ph.
    """
    pass



if __name__ == '__main__':
    f_base = '../samples/cola/f5000.properties'
    pos, mass = myRead(f_base)
    
    pos = torch.tensor(pos).to('cuda')
    mass = torch.tensor(mass).to('cuda')

    e, gft = gft(pos, mass, edge_func)
    print(e)
    print(gft)
