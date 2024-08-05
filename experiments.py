np.random.seed(1)

n=2
tol=1e-6
origin=np.zeros([1,1,3,1])
origin[:,:,0,:]=1 #(1,1,n+1,1)

vs=np.identity(2)
vs=np.expand_dims(vs,(0,1)) #(1,1,n,2)

N=2000000
covariates=np.random.uniform(size=(1,N,2,1))-0.5 #(1,N,2,1)
S=2-(np.expand_dims(covariates[:,:,0,:],2)>=0) #(1,N,1,1)
Z=np.zeros((1,N,1,1))
for s in [1,2]:
    mN=np.sum(S==s)
    indices=np.where(S==s)[1]
    Z[:,np.random.choice(indices,int((mN+1)/2),replace=False),:,:]=1

temp=np.zeros([1,1,n,1])
temp[:,:,0,:]=1 #(1,1,n,1)
tans=Z*temp-(1-Z)*temp+np.random.normal(size=(1,N,2,1))+np.matmul(vs,covariates) #(1,N,n,1)
zs=np.zeros((1,N,1,1))
tans=np.concatenate((zs,tans),axis=2)
origins=np.tile(origin,(1,N,1,1))
R=exp(origins,tans,tol) #(1,N,n+1,1)
wC=0.5*(1-Z)*(S==1)/np.sum((1-Z)*(S==1),1,keepdims=True)+0.5*(1-Z)*(S==2)/np.sum((1-Z)*(S==2),1,keepdims=True) #(1,N,1,1)
wT=0.5*Z*(S==1)/np.sum(Z*(S==1),1,keepdims=True)+0.5*Z*(S==2)/np.sum(Z*(S==2),1,keepdims=True) #(1,N,1,1)
trueans=invbroyden(R,wC,fmean(R,wT,tol),tol)
print(trueans)

# classical randomized experiment-consistency

np.random.seed(1)

reps=1000
Ns=[125,250,500,1000]
expconsistency=[0,0,0,0]
for j in range(4):
    N=Ns[j]
    covariates=np.random.uniform(size=(reps,N,2,1))-0.5 #(reps,N,2,1)
    S=2-(np.expand_dims(covariates[:,:,0,:],2)>=0) #(reps,N,1,1)
    Z=np.zeros((reps,N,1,1))
    for s in [1,2]:
        for rep in range(reps):
            mN=np.sum(S[rep,:,:,:]==s)
            indices=np.where(S[rep,:,:,:]==s)[0]
            Z[rep,np.random.choice(indices,int((mN+1)/2),replace=False),:,:]=1
    temp=np.zeros([1,1,n,1])
    temp[:,:,0,:]=1 #(1,1,n,1)
    tans=Z*temp-(1-Z)*temp+np.random.normal(size=(reps,N,2,1))+np.matmul(vs,covariates) #(reps,N,n,1)
    zs=np.zeros((reps,N,1,1))
    tans=np.concatenate((zs,tans),axis=2)
    origins=np.tile(origin,(reps,N,1,1))
    R=exp(origins,tans,tol) #(reps,N,n+1,1)
    wC=0.5*(1-Z)*(S==1)/np.sum((1-Z)*(S==1),1,keepdims=True)+0.5*(1-Z)*(S==2)/np.sum((1-Z)*(S==2),1,keepdims=True) #(1,N,1,1)
    wT=0.5*Z*(S==1)/np.sum(Z*(S==1),1,keepdims=True)+0.5*Z*(S==2)/np.sum(Z*(S==2),1,keepdims=True) #(1,N,1,1)
    ans=invbroyden(R,wC,fmean(R,wT,tol),tol)
    expconsistency[j]=np.mean(np.sqrt(np.sum((trueans-ans)**2,2)),0).item()
    print(expconsistency[j])

# classical randomized experiment-confidence region

np.random.seed(1)

Ns=[125,250,500,1000]
expcounter=[0,0,0,0]
for j in range(4):
    N=Ns[j]
    for k in range(1000):
        covariates=np.random.uniform(size=(1,N,2,1))-0.5 #(1,N,2,1)
        S=2-(np.expand_dims(covariates[:,:,0,:],2)>=0) #(1,N,1,1)
        Z=np.zeros((1,N,1,1))
        for s in [1,2]:
            mN=np.sum(S==s)
            indices=np.where(S==s)[1]
            Z[:,np.random.choice(indices,int((mN+1)/2),replace=False),:,:]=1
        temp=np.zeros([1,1,n,1])
        temp[:,:,0,:]=1 #(1,1,n,1)
        tans=Z*temp-(1-Z)*temp+np.random.normal(size=(1,N,2,1))+np.matmul(vs,covariates) #(1,N,n,1)
        zs=np.zeros((1,N,1,1))
        tans=np.concatenate((zs,tans),axis=2)
        origins=np.tile(origin,(1,N,1,1))
        R=exp(origins,tans,tol) #(1,N,n+1,1)
        wC=0.5*(1-Z)*(S==1)/np.sum((1-Z)*(S==1),1,keepdims=True)+0.5*(1-Z)*(S==2)/np.sum((1-Z)*(S==2),1,keepdims=True) #(1,N,1,1)
        wT=0.5*Z*(S==1)/np.sum(Z*(S==1),1,keepdims=True)+0.5*Z*(S==2)/np.sum(Z*(S==2),1,keepdims=True) #(1,N,1,1)
        ans=invbroyden(R,wC,fmean(R,wT,tol),tol)
        B=1000
        bootR=np.empty((B,N,n+1,1))
        bootS=np.empty((B,N,1,1))
        bootZ=np.empty((B,N,1,1))
        for b in range(B):
            bootindices=np.random.choice(N,N,replace=True)
            bootR[b,:,:,:]=R[0,bootindices,:,:]
            bootS[b,:,:,:]=S[0,bootindices,:,:]
            bootZ[b,:,:,:]=Z[0,bootindices,:,:]
        bootwC=0.5*(1-bootZ)*(bootS==1)/np.sum((1-bootZ)*(bootS==1),1,keepdims=True)+0.5*(1-bootZ)*(bootS==2)/np.sum((1-bootZ)*(bootS==2),1,keepdims=True) #(B,N,1,1)
        bootwT=0.5*bootZ*(bootS==1)/np.sum(bootZ*(bootS==1),1,keepdims=True)+0.5*bootZ*(bootS==2)/np.sum(bootZ*(bootS==2),1,keepdims=True) #(B,N,1,1)
        bootans=invbroyden(bootR,bootwC,fmean(bootR,bootwT,tol),tol)
        if np.squeeze(np.sqrt(np.sum((trueans-ans)**2)))<=np.sort(np.squeeze(np.sqrt(np.sum((bootans-ans)**2,2))))[int(0.95*B)]:
            expcounter[j]+=1
        if k % 10==0:
            print(N, k, expcounter[j])
    print(expcounter[j])
