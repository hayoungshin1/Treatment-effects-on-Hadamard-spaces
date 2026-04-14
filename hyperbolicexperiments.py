# true answer, based on very large sample and strong consistency theorem

np.random.seed(1)

n=2
tol=1e-6
origin=np.zeros([1,1,3,1])
origin[:,:,0,:]=1 #(1,1,n+1,1)

vs=np.identity(2)
vs=np.expand_dims(vs,(0,1)) #(1,1,n,2)

N=20000000
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

# classical randomized experiment-confidence region

np.random.seed(10)

B=1000
Ns=[125,250,500,1000]
expcounter1=[0,0,0,0]
expcounter2=[0,0,0,0]
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
        ans=np.squeeze(invbroyden(R,wC,fmean(R,wT,tol),tol),axis=1)[:,1:3,:]
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
        bootans=np.squeeze(invbroyden(bootR,bootwC,fmean(bootR,bootwT,tol),tol),axis=1)[:,1:3,:]
        if np.squeeze(np.sqrt(np.sum((np.squeeze(trueans,axis=1)[:,1:3,:]-ans)**2)))<=np.sort(np.squeeze(np.sqrt(np.sum((bootans-ans)**2,1))))[int(0.95*B)]:
            expcounter1[j]+=1
        covinv=np.linalg.inv(np.expand_dims(np.cov(np.squeeze(bootans), rowvar=False), axis=0))
        bootstats=np.zeros(B)
        for b in range(B):
            bootstats[b]=np.squeeze(np.matmul(np.matmul(np.transpose(bootans[b:(b+1),:,:]-ans,(0,2,1)),covinv),bootans[b:(b+1),:,:]-ans))
        teststatistic=np.squeeze(np.matmul(np.matmul(np.transpose(ans-np.squeeze(trueans,axis=1)[:,1:3,:],(0,2,1)),covinv),ans-np.squeeze(trueans,axis=1)[:,1:3,:]))
        if teststatistic<=np.sort(bootstats)[int(0.95*B)]:
            expcounter2[j]+=1
        if k % 10==0:
            print(N, k, expcounter1[j], expcounter2[j])
    print(expcounter1[j], expcounter2[j])

# matched observational study-consistency

np.random.seed(1)

reps=1000
Ns=[125,250,500,1000]
OSexpconsistency=[0,0,0,0]
for j in range(4):
    N=Ns[j]
    covariates=np.random.uniform(size=(reps,N,2,1))-0.5 #(reps,N,2,1)
    Z=np.zeros((reps,N,1,1))
    S=np.zeros((reps,N,1,1))
    wC=np.zeros((reps,N,1,1))
    wT=np.zeros((reps,N,1,1))
    for rep in range(reps):
        for i in range(N):
            prob=1/(1+np.exp(-np.sum(covariates[rep,i,:,0])))
            Z[rep,i,0,0]=np.random.binomial(n=1,p=prob)
        df = pd.DataFrame(covariates[rep,:,:,0], columns=['x1', 'x2'])
        df['treat'] = Z[rep,:,0,0]
        names = ['x1', 'x2']
        df_ranked = df.copy()
        df_ranked[names] = df[names].rank()
        matched_results = run_rank_mahalanobis_full_match(df_ranked, 'treat', names)
        totalS=len(matched_results['subclass'].cat.categories)
        S[rep,:,0,0]=matched_results["subclass"].astype(int).to_numpy()
        lambdas=np.zeros(N)
        wCrep=np.zeros((totalS,N,1,1))
        wTrep=np.zeros((totalS,N,1,1))
        for s in range(1,totalS+1):
            lambdas[S[rep,:,0,0]==s]=np.sum(S[rep,:,0,0]==s)/N
            wCrep[s-1,:,0,0]=lambdas*(1-Z[rep,:,0,0])*(S[rep,:,0,0]==s)/np.sum((1-Z[rep,:,0,0])*(S[rep,:,0,0]==s))
            wTrep[s-1,:,0,0]=lambdas*(Z[rep,:,0,0])*(S[rep,:,0,0]==s)/np.sum((Z[rep,:,0,0])*(S[rep,:,0,0]==s))
        wC[rep,:,0,0]=np.sum(wCrep,axis=0)[:,0,0]
        wT[rep,:,0,0]=np.sum(wTrep,axis=0)[:,0,0]
    temp=np.zeros([1,1,n,1])
    temp[:,:,0,:]=1 #(1,1,n,1)
    tans=Z*temp-(1-Z)*temp+np.random.normal(size=(reps,N,2,1))+np.matmul(vs,covariates) #(reps,N,n,1)
    zs=np.zeros((reps,N,1,1))
    tans=np.concatenate((zs,tans),axis=2)
    origins=np.tile(origin,(reps,N,1,1))
    R=exp(origins,tans,tol) #(reps,N,n+1,1)
    ans=invbroyden(R,wC,fmean(R,wT,tol),tol)
    OSexpconsistency[j]=np.mean(np.sqrt(np.sum((trueans-ans)**2,2)),0).item()
    print(OSexpconsistency[j])

# matched observational study-confidence region

B=1000
Ns=[125,250,500,1000]
OSexpcounter1=[0,0,0,0]
OSexpcounter2=[0,0,0,0]
for j in range(4):
    N=Ns[j]
    for k in range(1000):
        np.random.seed(10000*j+k)
        covariates=np.random.uniform(size=(1,N,2,1))-0.5 #(1,N,2,1)
        Z=np.zeros((1,N,1,1))
        S=np.zeros((1,N,1,1))
        wC=np.zeros((1,N,1,1))
        wT=np.zeros((1,N,1,1))
        for i in range(N):
            prob=1/(1+np.exp(-np.sum(covariates[0,i,:,0])))
            Z[0,i,0,0]=np.random.binomial(n=1,p=prob)
        df = pd.DataFrame(covariates[0,:,:,0], columns=['x1', 'x2'])
        df['treat'] = Z[0,:,0,0]
        names = ['x1', 'x2']
        df_ranked = df.copy()
        df_ranked[names] = df[names].rank()
        matched_results = run_rank_mahalanobis_full_match(df_ranked, 'treat', names)
        totalS=len(matched_results['subclass'].cat.categories)
        S[0,:,0,0]=matched_results["subclass"].astype(int).to_numpy()
        lambdas=np.zeros(N)
        wCrep=np.zeros((totalS,N,1,1))
        wTrep=np.zeros((totalS,N,1,1))
        for s in range(1,totalS+1):
            lambdas[S[0,:,0,0]==s]=np.sum(S[0,:,0,0]==s)/N
            wCrep[s-1,:,0,0]=lambdas*(1-Z[0,:,0,0])*(S[0,:,0,0]==s)/np.sum((1-Z[0,:,0,0])*(S[0,:,0,0]==s))
            wTrep[s-1,:,0,0]=lambdas*(Z[0,:,0,0])*(S[0,:,0,0]==s)/np.sum((Z[0,:,0,0])*(S[0,:,0,0]==s))
        wC[0,:,0,0]=np.sum(wCrep,axis=0)[:,0,0]
        wT[0,:,0,0]=np.sum(wTrep,axis=0)[:,0,0]
        temp=np.zeros([1,1,n,1])
        temp[:,:,0,:]=1 #(1,1,n,1)
        tans=Z*temp-(1-Z)*temp+np.random.normal(size=(1,N,2,1))+np.matmul(vs,covariates) #(1,N,n,1)
        zs=np.zeros((1,N,1,1))
        tans=np.concatenate((zs,tans),axis=2)
        origins=np.tile(origin,(1,N,1,1))
        R=exp(origins,tans,tol) #(1,N,n+1,1)
        ans=np.squeeze(invbroyden(R,wC,fmean(R,wT,tol),tol),axis=1)[:,1:3,:]
        bootR=np.empty((B,N,n+1,1))
        bootS=np.empty((B,N,1,1))
        bootZ=np.empty((B,N,1,1))
        bootwC=np.zeros((B,N,1,1))
        bootwT=np.zeros((B,N,1,1))
        for b in range(B):
            bootindices=np.random.choice(N,N,replace=True)
            bootR[b,:,:,:]=R[0,bootindices,:,:]
            bootZ[b,:,:,:]=Z[0,bootindices,:,:]
            df = pd.DataFrame(covariates[0,bootindices,:,0], columns=['x1', 'x2'])
            df['treat'] = bootZ[b,:,0,0]
            names = ['x1', 'x2']
            df_ranked = df.copy()
            df_ranked[names] = df[names].rank()
            matched_results = run_rank_mahalanobis_full_match(df_ranked, 'treat', names)
            totalS=len(matched_results['subclass'].cat.categories)
            bootS[b,:,0,0]=matched_results["subclass"].astype(int).to_numpy()
            lambdas=np.zeros(N)
            wCrep=np.zeros((totalS,N,1,1))
            wTrep=np.zeros((totalS,N,1,1))
            for s in range(1,totalS+1):
                lambdas[bootS[b,:,0,0]==s]=np.sum(bootS[b,:,0,0]==s)/N
                wCrep[s-1,:,0,0]=lambdas*(1-bootZ[b,:,0,0])*(bootS[b,:,0,0]==s)/np.sum((1-bootZ[b,:,0,0])*(bootS[b,:,0,0]==s))
                wTrep[s-1,:,0,0]=lambdas*(bootZ[b,:,0,0])*(bootS[b,:,0,0]==s)/np.sum((bootZ[b,:,0,0])*(bootS[b,:,0,0]==s))
            bootwC[b,:,0,0]=np.sum(wCrep,axis=0)[:,0,0]
            bootwT[b,:,0,0]=np.sum(wTrep,axis=0)[:,0,0]
        bootans=np.squeeze(invbroyden(bootR,bootwC,fmean(bootR,bootwT,tol),tol),axis=1)[:,1:3,:]
        if np.squeeze(np.sqrt(np.sum((np.squeeze(trueans,axis=1)[:,1:3,:]-ans)**2)))<=np.sort(np.squeeze(np.sqrt(np.sum((bootans-ans)**2,1))))[int(0.95*B)]:
            OSexpcounter1[j]+=1
        bootstats=np.zeros(B)
        covinv=np.linalg.inv(np.expand_dims(np.cov(np.squeeze(bootans), rowvar=False), axis=0))
        for b in range(B):
            bootstats[b]=np.squeeze(np.matmul(np.matmul(np.transpose(bootans[b:(b+1),:,:]-ans,(0,2,1)),covinv),bootans[b:(b+1),:,:]-ans))
        teststatistic=np.squeeze(np.matmul(np.matmul(np.transpose(ans-np.squeeze(trueans,axis=1)[:,1:3,:],(0,2,1)),covinv),ans-np.squeeze(trueans,axis=1)[:,1:3,:]))
        if teststatistic<=np.sort(bootstats)[int(0.95*B)]:
            OSexpcounter2[j]+=1
        if k % 10==0:
            print(N, k, OSexpcounter1[j], OSexpcounter2[j])
    print(OSexpcounter1[j], OSexpcounter2[j])

print(OSexpcounter1, OSexpcounter2)
