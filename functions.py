import numpy as np

def ip(p,q):
    """
    p: B batches of N' points (B,1 or N',n+1,1)
    q: B batches of N' points (B,1 or N',n+1,1)
    out: B batches of Minkowski inner products of p and q (B,1 or N',1,1)
    """
    newq=q.copy()
    newq[:,:,0,:]*=-1
    out=np.sum(p*newq,axis=2,keepdims=True)
    return out

def norm(v):
    """
    v: B batches of N' vectors (B,N',n+1,1)
    out: B batches of Minkowski norms of v (B,N',1,1)
    """
    out=np.sqrt(np.absolute(ip(v,v)))
    return out

def fmean(Y,w,tol=1e-6):
    """
    Y: B batches of N data points in hyperboloid (B,N,n+1,1)
    w: B batches of N weights (B,N,1,1)
    out: B batches of weighted Fr'echet means  (B,1,n+1,1)
    """
    old=Y[:,0,:,:]
    old=np.expand_dims(old,1) #(B,1,n+1,1)
    current=np.copy(old) #(B,1,n+1,1)
    count=1
    while count==1 or np.sum(np.arccosh(-ip(old,current))>tol)>0:
        prod=-ip(Y,current) #(B,N,1,1)
        paran=2*np.arccosh(prod)/np.sqrt(prod**2-1) #(B,N,1,1)
        indices=np.where(np.abs(prod-1)<tol)[0:2]
        paran[indices]=2
        u=w*paran*Y #(B,N,n+1,1)
        usum=np.sum(u,axis=1,keepdims=True) #(B,1,n+1,1)
        denom=np.sqrt(np.absolute(-ip(usum,usum)))
        old=np.copy(current)
        current=usum/denom #(B,1,n+1,1)
        count+=1
    out=current
    return out

def exp(p,v,tol=1e-6):
    """
    p: B batches of points in the hyperboloid (B,N',n+1,1)
    v: B batches of tangent vectors at p (B,N',n+1,1)
    out: B batches of exp(p,v) (B,N',n+1,1)
    """
    vnorm=norm(v) #(B,N',1,1)
    out=np.cosh(vnorm)*p+np.sinh(vnorm)*v/vnorm #(B,N',n+1,1)
    indices=np.where(vnorm<tol)[0:2]
    out[indices]=p[indices]
    return out

def fp(Y,p,betaxip):
    """
    Y: B batches of N data points in the hyperboloid (B,N,n+1,1)
    p: B batches of p (B,1,n+1,1)
    betaxip: B batches of beta*xi_p at some p (B,1,n+1,1)
    out: B batches of beta*xi_Y (B,N,n+1,1)
    """
    beta=norm(betaxip) #(B,1,1,1)
    xip=betaxip/beta #(B,1,n+1,1)
    plus=p+xip #(B,1,n+1,1)
    vect=plus+ip(Y,plus)*Y #(B,N,n+1,1)
    out=beta*vect/norm(vect) #(B,N,n+1,1)
    return out

def shift(Y,p,betaxip,tol=1e-6):
    """
    Y: B batches of N data points in hyperboloid (B,N,n+1,1)
    p: B batches of p (B,1,n+1,1)
    betaxip: B batches of beta*xi_p at some p (B,1,n+1,1)
    out: B batches of exp(Y,beta*xi_Y) (B,N,n+1,1)
    """
    out=exp(Y,fp(Y,p,betaxip),tol) #(B,N,n+1,1)
    return out

def meanshift(Y,w,p,betaxip,tol=1e-6):
    """
    Y: B batches of N data points in hyperboloid (B,N,n+1,1)
    w: B batches of N weights (B,N,1,1)
    p: B batches of p (B,1,n+1,1)
    betaxip: B batches of beta*xi_p at some p (B,1,n+1,1)
    out: B batches of weighted sample mean shifts of Y by betaxi (B,1,n+1,1)
    """
    out=fmean(shift(Y,p,betaxip,tol),w,tol) #(B,1,n+1,1)
    return out

def log(p1,p2,tol=1e-6):
    """
    p1 and p2: B batches of points in hyperboloid (B,1,n+1,1)
    """
    pq=ip(p1,p2) #(B,1,1.1)
    vect=p2+pq*p1 #(B,1,n+1,1)
    dist=np.arccosh(-pq) #(B,1,1,1)
    out=dist*vect/norm(vect) #(B,1,n+1,1)
    indices=np.where(dist<tol)[0:2]
    out[indices]=0
    return out

def invbroyden(Y,w,target,tol=1e-6):
    """
    Y: B batches of N data points in hyperboloid (B,N,n+1,1)
    w: B batches of N weights (B,N,1,1)
    target: B batches of points in hyperboloid (B,1,n+1,1)
    out: B batches of betaxis for which meanshift(Y,w,betaxi)=target (B,1,n+1,1)
    """
    B=Y.shape[0]
    N=Y.shape[1]
    n=Y.shape[2]-1
    p=fmean(Y,w,tol) #(B,1,n+1,1)
    xold=log(p,target,tol) #(B,1,n+1,1)
    target=target[:,:,1:,:] #(B,1,n,1)
    origin=np.zeros([B,1,n+1,1]) #(B,1,n+1,1)
    origin[:,:,0,:]=1
    xold=fp(origin,p,xold) #(B,1,n+1,1)
    xold[:,:,0,:]=0 #ensure it is in tangent space at origin
    xcurrent=1.1*xold
    delx=(xcurrent-xold)[:,:,1:,:] #(B,1,n,1)
    fold=meanshift(Y,w,origin,xold)[:,:,1:,:]-target #(B,1,n,1)
    Jinvold=np.tile(np.expand_dims(np.identity(n),0),(B,1,1,1)) #(B,1,n,n)
    count=1
    while np.sum(np.sum(delx*delx,2)>tol)>0:
        fcurrent=meanshift(Y,w,origin,xcurrent,tol)[:,:,1:,:]-target #(B,1,n,1)
        delf=fcurrent-fold #(B,1,n,1)
        delx=(xcurrent-xold)[:,:,1:,:] #(B,1,n,1)
        Jinvcurrent=Jinvold+np.matmul(((delx-np.matmul(Jinvold,delf))/(np.sum(delf*delf,2,keepdims=True))),np.transpose(delf,(0,1,3,2))) #(B,1,n,n)
        zs=np.zeros((B,1,1,1))
        xnew=xcurrent-np.concatenate((zs,np.matmul(Jinvcurrent,fcurrent)),axis=2) #(B,1,n,1)
        xnew[:,:,0,:]=0 #ensure it is in tangent space at origin
        xold=xcurrent.copy()
        xcurrent=xnew.copy()
        fold=fcurrent.copy()
        Jinvold=Jinvcurrent.copy()
        count+=1
    out=xcurrent
    return out
