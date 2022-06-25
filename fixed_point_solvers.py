import torch

def fixed_point_iteration(f, x0, max_iter=50, tol=1e-4):
    x_prev, x = x0, f(x0)
    for _ in range(max_iter):
        x_prev, x = x, f(x)

        res = (x_prev - x).norm().item()
        if (res < tol):
            break
    return x

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    # http://implicit-layers-tutorial.org/deep_equilibrium_models/
    bsz, d,  = x0.shape
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res = (F[:,k%m] - X[:,k%m]).norm().item()
        if (res < tol):
            break
    return X[:,k%m].view_as(x0)