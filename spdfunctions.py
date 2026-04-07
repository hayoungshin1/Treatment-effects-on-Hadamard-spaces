import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
import dipy.reconst.dti as dti
from dipy.data import get_fnames
import os
import nibabel as nib
import pandas as pd
from dipy.reconst.dti import color_fa, fractional_anisotropy
from dipy.data import get_sphere
from dipy.viz import actor, window
from scipy import stats

def spdip(pinv, v1, v2):
    """
    pinv: B batches of inverses p^{-1} of N' points, directly inputed to save time (B,1 or N',m,m)
    v1, v2: B batches of 1 or N' tangent vectors in T_pM (B,1 or N',m,m)
    out: inner products (B,1 or N',1,1)
    """
    t1=np.matmul(pinv,v1)
    t2=np.matmul(pinv,v2)
    out=np.trace(np.matmul(t1,t2),axis1=-2,axis2=-1)
    out=np.expand_dims(np.expand_dims(out, axis=-1),axis=-1)
    return out

def spdnorm(pinv, v):
    """
    pinv: B batches of inverses p^{-1} of N' points, directly inputed to save time (B,1 or N',m,m)
    v: B batches of 1 or N' tangent vectors in T_pM (B,1 or N',m,m)
    out: B batches of norms of v (B,1 or N',1,1)
    """
    out=np.sqrt(np.absolute(spdip(pinv,v,v)))
    return out

def matrix_exp(A):
    """
    A: batch of SPD matrices (B,N',m,m)
    out: matrix exponentials of those matrices (B,N',m,m)
    """
    m=A.shape[-1]
    A=(A+np.transpose(A,(0,1,3,2)))/2
    L, V=np.linalg.eig(A)
    L=np.real(L) # ensure realness
    V=np.real(V) # ensure realness
    Lbig=np.expand_dims(np.exp(L),-1)
    Leye=np.expand_dims(np.expand_dims(np.eye(m),0),0)
    Lmat=Lbig*Leye
    out=np.matmul(np.matmul(V,Lmat),np.transpose(V,(0,1,3,2)))
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def matrix_log(A):
    """
    A: batch of SPD matrices (B,N',m,m)
    out: matrix logs of those matrices (B,N',m,m)
    """
    m=A.shape[-1]
    A=(A+np.transpose(A,(0,1,3,2)))/2
    L, V=np.linalg.eig(A)
    L=np.real(L) # ensure realness
    L=np.clip(L,a_min=0,a_max=None) # ensure non-negativeness
    V=np.real(V) # ensure realness
    Lbig=np.expand_dims(np.log(L),-1)
    Leye=np.expand_dims(np.expand_dims(np.eye(m),0),0)
    Lmat=Lbig*Leye
    out=np.matmul(np.matmul(V,Lmat),np.transpose(V,(0,1,3,2)))
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def matrix_sqrt(A):
    """
    A: batch of SPD matrices (B,N',m,m)
    out: matrix square roots of those matrices (B,N',m,m)
    """
    m=A.shape[-1]
    A=(A+np.transpose(A,(0,1,3,2)))/2
    L, V=np.linalg.eig(A)
    L=np.real(L) # ensure realness
    L=np.clip(L,a_min=0,a_max=None) # ensure non-negativeness
    V=np.real(V) # ensure realness
    Lbig=np.expand_dims(np.sqrt(L),-1)
    Leye=np.expand_dims(np.expand_dims(np.eye(m),0),0)
    Lmat=Lbig*Leye
    out=np.matmul(np.matmul(V,Lmat),np.transpose(V,(0,1,3,2)))
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def spdexp(phalf, phalfinv, v):
    """
    phalf: B batches of square roots p^{1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    phalfinv: B batches of square roots p^{-1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    v: B batches of 1 or N' tangent vectors in T_pM (B,1 or N',m,m)
    out: each exp_p(v) (B,1 or N',m,m)
    """
    v=(v+np.transpose(v,(0,1,3,2)))/2
    out=matrix_exp(np.matmul(np.matmul(phalfinv,v),phalfinv))
    out=np.matmul(np.matmul(phalf,out),phalf)
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def spdlog(phalf, phalfinv, x):
    """
    phalf: B batches of square roots p^{1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    phalfinv: B batches of square roots p^{-1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    x: B batches of points in P_m (B,1 or N',m,m)
    out: each log_p(x) (B,1 or N',m,m)
    """
    x=(x+np.transpose(x,(0,1,3,2)))/2
    out=matrix_log(np.matmul(np.matmul(phalfinv,x),phalfinv))
    out=np.matmul(np.matmul(phalf,out),phalf)
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def spddistance(phalf,pinv,phalfinv,x):
    """
    phalf: B batches of square roots p^{1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    pinv: B batches of inverses p^{-1} of N' points, directly inputed to save time (B,1 or N',m,m)
    phalfinv: B batches of square roots p^{-1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    x: B batches of points in P_m (B,1 or N',m,m)
    out: each d(p,x) (B,1 or N',1,1)
    """
    v=spdlog(phalf, phalfinv, x)
    out=np.sqrt(spdip(pinv, v, v))
    return out

def spdloss(phalf,pinv,phalfinv,Y,w):
    """
    phalf: B batches of square roots p^{1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    pinv: B batches of inverses p^{-1} of N' points, directly inputed to save time (B,1 or N',m,m)
    phalfinv: B batches of square roots p^{-1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    Y: B batches of N SPD matrices (B,N,m,m)
    w: B batches of N weights (B,N,1,1)
    out: each d(p,x) (B)
    """
    out=np.sum(w*spddistance(phalf,pinv,phalfinv,Y)**2,axis=(1,2,3),keepdims=False)
    return out

def spdgrad(phalf,pinv,phalfinv,Y,w):
    """
    phalf: B batches of square roots p^{1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    pinv: B batches of inverses p^{-1} of N' points, directly inputed to save time (B,1 or N',m,m)
    phalfinv: B batches of square roots p^{-1/2} of N' points, directly inputed to save time (B,1 or N',m,m)
    Y: B batches of N SPD matrices (B,N,m,m)
    w: B batches of N weights (B,N,1,1)
    out: gradient in each T_pM (B,1,m,m)
    """
    out=-np.sum(w*spdlog(phalf,phalfinv,Y),axis=1,keepdims=True)
    return out

def spdfmean(Y, w, tol=1e-10):
    """
    Y: B batches of N SPD matrices (B,N,m,m)
    w: B batches of N weights (B,N,1,1)
    out: B batches of weighted Fr'echet means  (B,1,m,m)
    """
    current_p=np.mean(Y,axis=1,keepdims=True)
    B=Y.shape[0]
    current_phalf=matrix_sqrt(current_p)
    current_phalf=(current_phalf+np.transpose(current_phalf,(0,1,3,2)))/2
    current_pinv=np.linalg.inv(current_p)
    current_pinv=(current_pinv+np.transpose(current_pinv,(0,1,3,2)))/2
    current_phalfinv=np.linalg.inv(current_phalf)
    current_phalfinv=(current_phalfinv+np.transpose(current_phalfinv,(0,1,3,2)))/2
    current_loss=spdloss(current_phalf,current_pinv,current_phalfinv,Y,w)
    step=spdgrad(current_phalf,current_pinv,current_phalfinv,Y,w)
    lr=np.ones((B))
    count=0
    while np.any(spdip(current_pinv,step,step)>tol) and count<100:
        new_p=spdexp(current_phalf,current_phalfinv,-np.expand_dims(np.expand_dims(np.expand_dims(lr,-1),-1),-1)*step)
        new_p=(new_p+np.transpose(new_p,(0,1,3,2)))/2
        new_phalf=matrix_sqrt(new_p)
        new_phalf=(new_phalf+np.transpose(new_phalf,(0,1,3,2)))/2
        new_pinv=np.linalg.inv(new_p)
        new_pinv=(new_pinv+np.transpose(new_pinv,(0,1,3,2)))/2
        new_phalfinv=np.linalg.inv(new_phalf)
        new_phalfinv=(new_phalfinv+np.transpose(new_phalfinv,(0,1,3,2)))/2
        new_loss=spdloss(new_phalf,new_pinv,new_phalfinv,Y,w)
        lr*=1.1*(new_loss<current_loss)+0.5*(new_loss>=current_loss) # adjust learning rate based on whether or not loss was improved
        current_p[new_loss<current_loss,:,:,:]=new_p[new_loss<current_loss,:,:,:]
        current_phalf[new_loss<current_loss,:,:,:]=new_phalf[new_loss<current_loss,:,:,:]
        current_pinv[new_loss<current_loss,:,:,:]=new_pinv[new_loss<current_loss,:,:,:]
        current_phalfinv[new_loss<current_loss,:,:,:]=new_phalfinv[new_loss<current_loss,:,:,:]
        step[new_loss<current_loss,:,:,:]=spdgrad(current_phalf[new_loss<current_loss,:,:,:],current_pinv[new_loss<current_loss,:,:,:],current_phalfinv[new_loss<current_loss,:,:,:],Y[new_loss<current_loss,:,:,:],w[new_loss<current_loss,:,:,:])
        current_loss[new_loss<current_loss]=new_loss[new_loss<current_loss]
        count+=1
    out=current_p
    return out

def spdfp(Yhalf,Yhalfinv,phalf,phalfinv,betaxip):
    """
    Yhalf,Yhalfinv: B batches of various transformations of N SPD matrices (B,N,m,m)
    phalf,phalfinv: B batches of various transformations of SPD matrix p (B,1,m,m)
    betaxip: B batches of beta*xi_p at some p (B,1,m,m)
    out: B batches of beta*xi_Y (B,N,m,m)
    """
    m=Yhalf.shape[-1]
    prod=np.matmul(np.matmul(phalfinv,betaxip),phalfinv)
    D, V=np.linalg.eig(prod)
    D=np.real(D)
    V=np.real(V)
    evals=np.zeros_like(V)
    evecs=np.zeros_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            order=np.argsort(D[i,j,:])[::-1]
            evecs[i,j,:,:]=np.transpose(V[i,j,:,order])
            evals[i,j,:,:]=np.expand_dims(D[i,j,order],-1)*np.eye(m)
    W=np.matmul(np.matmul(Yhalfinv,phalf),evecs)
    W[:,:,:,0]/=np.sqrt(np.sum(W[:,:,:,0]*W[:,:,:,0],axis=-1,keepdims=True))
    for k in range(1,m):
        W[:,:,:,k]-=np.sum(np.sum(W[:,:,:,0:k]*np.expand_dims(W[:,:,:,k],axis=-1),axis=-2,keepdims=True)*W[:,:,:,0:k],axis=-1)
        W[:,:,:,k]/=np.sqrt(np.sum(W[:,:,:,k]*W[:,:,:,k],axis=-1,keepdims=True))
    out=np.matmul(np.matmul(np.matmul(np.matmul(Yhalf,W),evals),np.transpose(W,(0,1,3,2))),Yhalf)
    out=(out+np.transpose(out,(0,1,3,2)))/2
    return out

def spdshift(Yhalf,Yhalfinv,phalf,phalfinv,betaxip):
    """
    Yhalf,Yhalfinv: B batches of various transformations of N SPD matrices (B,N,m,m)
    phalf,phalfinv: B batches of various transformations of SPD matrix p (B,1,m,m)
    betaxip: B batches of beta*xi_p at some p (B,1,m,m)
    out: B batches of exp(Y,beta*xi_Y) (B,N,m,m)
    """
    out=spdexp(Yhalf,Yhalfinv,spdfp(Yhalf,Yhalfinv,phalf,phalfinv,betaxip)) #(B,N,m,m)
    return out

def spdmeanshift(Yhalf,Yhalfinv,w,phalf,phalfinv,betaxip,tol=1e-10):
    """
    Yhalf,Yhalfinv: B batches of various transformations of N SPD matrices (B,N,m,m)
    phalf,phalfinv: B batches of various transformations of SPD matrix p (B,1,m,m)
    betaxip: B batches of beta*xi_p at some p (B,1,m,m)
    w: B batches of N weights (B,N,1,1)
    out: B batches of weighted sample mean shifts of Y by betaxi (B,1,m,m)
    """
    out=spdfmean(spdshift(Yhalf,Yhalfinv,phalf,phalfinv,betaxip),w,tol) #(B,1,m,m)
    return out

# Note that the following is an orthonormal basis in the tangent space at the identity
# [1,0,0]  (1/sqrt(2)) * [0,1,0]  (1/sqrt(2)) * [0,0,1]  [0,0,0]  (1/sqrt(2)) * [0,0,0]  [0,0,0]
# [0,0,0]                [1,0,0]                [0,0,0]  [0,1,0]                [0,0,1]  [0,0,0]
# [0,0,0],               [0,0,0],               [1,0,0], [0,0,0],               [0,1,0], [0,0,1]

def vectorizer(A):
    """
    A: B batches of N 3x3 symmetric matrices (B,N,3,3)
    out: matrices written as vectors in terms of the above orthonormal basis (B,N,6,1)
    """
    out=np.expand_dims(np.concatenate((np.expand_dims(A[:,:,0,0],-1), np.expand_dims(A[:,:,0,1]*np.sqrt(2),-1), np.expand_dims(A[:,:,0,2]*np.sqrt(2),-1), np.expand_dims(A[:,:,1,1],-1), np.expand_dims(A[:,:,1,2]*np.sqrt(2),-1), np.expand_dims(A[:,:,2,2],-1)),2),-1)
    return out

def matricizer(v):
    """
    Inverse of the vectorizer function
    v: (B,N,6,1)
    out: (B,N,3,3)
    """
    v1=np.expand_dims(v[:,:,0,:],-2)*np.array([[[[1,0,0],[0,0,0],[0,0,0]]]])
    v2=np.expand_dims(v[:,:,1,:],-2)*np.array([[[[0,1,0],[1,0,0],[0,0,0]]]])/np.sqrt(2)
    v3=np.expand_dims(v[:,:,2,:],-2)*np.array([[[[0,0,1],[0,0,0],[1,0,0]]]])/np.sqrt(2)
    v4=np.expand_dims(v[:,:,3,:],-2)*np.array([[[[0,0,0],[0,1,0],[0,0,0]]]])
    v5=np.expand_dims(v[:,:,4,:],-2)*np.array([[[[0,0,0],[0,0,1],[0,1,0]]]])/np.sqrt(2)
    v6=np.expand_dims(v[:,:,5,:],-2)*np.array([[[[0,0,0],[0,0,0],[0,0,1]]]])
    out=v1+v2+v3+v4+v5+v6
    return out

def spdinvbroyden(Y,w,target,tol=1e-10):
    """
    Y: B batches of N 3x3 SPD matrices (B,N,3,3)
    w: B batches of N weights (B,N,1,1)
    target: B batches of points in hyperboloid (B,1,3,3)
    out: B batches of betaxis (at identity) for which meanshift maps to target (B,1,3,3)
    """
    B=Y.shape[0]
    m=Y.shape[-1]
    Yhalf=matrix_sqrt(Y)
    Yhalf=(Yhalf+np.transpose(Yhalf,(0,1,3,2)))/2
    Yinv=np.linalg.inv(Y)
    Yinv=(Yinv+np.transpose(Yinv,(0,1,3,2)))/2
    Yhalfinv=np.linalg.inv(Yhalf)
    Yhalfinv=(Yhalfinv+np.transpose(Yhalfinv,(0,1,3,2)))/2
    p=spdfmean(Y,w) # (B,1,3,3)
    phalf=matrix_sqrt(p) # (B,1,3,3)
    phalfinv=np.linalg.inv(phalf) # (B,1,3,3)
    xinitial=spdlog(phalf,phalfinv,target) # (B,1,3,3)
    identity=np.tile(np.expand_dims(np.expand_dims(np.eye(m),0),0),(B,1,1,1)) # (B,1,3,3)
    xinitial=spdfp(identity,identity,phalf,phalfinv,xinitial) # (B,1,3,3)
    xold=xinitial.copy()
    xcurrent=1.1*xold # (B,1,3,3)
    delx=vectorizer(xcurrent-xold) # (B,1,6,1)
    vtarget=vectorizer(target) # (B,1,6,1)
    fold=vectorizer(spdmeanshift(Yhalf,Yhalfinv,w,identity,identity,xold))-vtarget # (B,1,6,1)
    Jinvold=np.tile(np.expand_dims(np.eye(6),0),(B,1,1,1)) #(B,1,6,6)
    count=0
    othercount=0
    while np.any(np.sum(delx*delx,2)>tol) and othercount<100:
        try:
            fcurrent=vectorizer(spdmeanshift(Yhalf,Yhalfinv,w,identity,identity,xcurrent))-vtarget # (B,1,6,1)
        except np.linalg.LinAlgError:
            xinitial*=1.01
            xold=xinitial.copy()
            xcurrent=1.1*xold # (B,1,3,3)
            delx=vectorizer(xcurrent-xold) # (B,1,6,1)
            fold=vectorizer(spdmeanshift(Yhalf,Yhalfinv,w,identity,identity,xold))-vtarget # (B,1,6,1)
            Jinvold=np.tile(np.expand_dims(np.eye(6),0),(B,1,1,1)) #(B,1,6,6)
            count=0
            othercount+=1
            fcurrent=vectorizer(spdmeanshift(Yhalf,Yhalfinv,w,identity,identity,xcurrent))-vtarget # (B,1,6,1)
        delf=fcurrent-fold # (B,1,6,1)
        delx=vectorizer(xcurrent-xold) # (B,1,6,1)
        Jinvcurrent=Jinvold+np.matmul(((delx-np.matmul(Jinvold,delf))/(np.sum(delf*delf,2,keepdims=True))),np.transpose(delf,(0,1,3,2))) #(B,1,6,6)
        xnew=xcurrent-matricizer(np.matmul(Jinvcurrent,fcurrent)) #(B,1,3,3)
        xnew=(xnew+np.transpose(xnew,(0,1,3,2)))/2 # ensure symmetry
        xold=xcurrent.copy()
        xcurrent=xnew.copy()
        fold=fcurrent.copy()
        Jinvold=Jinvcurrent.copy()
        count+=1
    out=xcurrent
    #print(count, othercount)
    return out


def decompose(A):
    """
    A: (N,L,W,H,3,3) np array to be reshaped
    out: reshaped (L*W*H,N,3,3) np array
    """
    N,L,W,H=A.shape[0:4]
    out=np.transpose(A.reshape(N,L*W*H,3,3),(1,0,2,3))
    return out

def recompose(A, L, W, H):
    """
    L, W, H: positive integers
    A: reshaped (L*W*H,N,3,3) np array
    out: original (N,L,W,H,3,3) np array
    """
    N=A.shape[1]
    A=np.transpose(A,(1,0,2,3))
    out=A.reshape(N, L, W, H, 3, 3)
    return out
