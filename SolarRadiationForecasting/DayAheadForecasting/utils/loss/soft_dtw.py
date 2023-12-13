import numpy as np
import torch
# from numba import cuda
from torch.autograd import Function

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))

# @cuda.jit
def compute_softdtw(D, gamma):
  device = D.device
  N = D.shape[0]
  M = D.shape[1]
  R = torch.zeros((N + 2, M + 2), device=device) + 1e8 ####
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      # rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
      # softmin = - gamma * (np.log(rsum) + rmax)
      softmin = - gamma * (torch.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R


# @cuda.jit
def compute_softdtw_backward(D_, R, gamma):
  device = D_.device
  N = D_.shape[0]
  M = D_.shape[1]
  D = torch.zeros((N + 2, M + 2), device=device) ####
  E = torch.zeros((N + 2, M + 2), device=device) ####
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      # a = np.exp(a0)
      # b = np.exp(b0)
      # c = np.exp(c0)
      a = torch.exp(a0)
      b = torch.exp(b0)
      c = torch.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
        dev = D.device
        print("Device:", dev)
        batch_size,N,N = D.shape
        gamma = torch.tensor([gamma], device=dev)
        # D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N+2 ,N+2), device=dev) 
        for k in range(0, batch_size): # loop over all D in the batch    
            # Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(dev)
            print("Computing softdtw")
            Rk = compute_softdtw(D[k,:,:], g_)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        print("Backward device:", dev)
        D, R, gamma = ctx.saved_tensors
        batch_size,N,N = D.shape
        # D_ = D.detach().cpu().numpy()
        # R_ = R.detach().cpu().numpy()

        g_ = gamma.item()

        E = torch.zeros((batch_size, N ,N), device=dev)
        for k in range(batch_size):         
            # Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(dev)
            Ek = compute_softdtw_backward(D[k,:,:], R[k,:,:], g_)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None

