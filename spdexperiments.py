df=pd.read_csv("Downloads/PDClinicalData.csv")
df["Sex"]=df["Sex"].replace("M",0)
df["Sex"]=df["Sex"].replace("F",1)

treatmentDTs=np.zeros([27,88,88,54,3,3])
controlDTs=np.zeros([26,88,88,54,3,3])

treatmentdf=df.iloc[np.where(df["Group"]=="PD")]
controldf=df.iloc[np.where(df["Group"]=="Control")]

treatmentsubjects=list(treatmentdf["Subject"])
controlsubjects=list(controldf["Subject"])

for i in range(len(treatmentsubjects)):
    patient=treatmentsubjects[i]
    img=nib.load("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_bmatrix_1000.nii.gz")
    data=img.get_fdata()
    bvals=np.loadtxt("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_bval_1000")
    bvecs=np.loadtxt("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_grad_1000").T
    gtab=gradient_table(bvals,bvecs)
    tenmodel=dti.TensorModel(gtab)
    tenfit=tenmodel.fit(data)
    diffusion_tensors=tenfit.quadratic_form
    treatmentDTs[i]=diffusion_tensors
    #print(i)

np.save('Documents/treatmentDTs.npy',treatmentDTs)

for i in range(len(controlsubjects)):
    patient=controlsubjects[i]
    img=nib.load("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_bmatrix_1000.nii.gz")
    data=img.get_fdata()
    bvals=np.loadtxt("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_bval_1000")
    bvecs=np.loadtxt("Downloads/NITRC_PD_DATA/"+patient+"/"+patient+"_grad_1000").T
    gtab=gradient_table(bvals,bvecs)
    tenmodel=dti.TensorModel(gtab)
    tenfit=tenmodel.fit(data)
    diffusion_tensors=tenfit.quadratic_form
    controlDTs[i]=diffusion_tensors
    #print(i)

np.save('Documents/controlDTs.npy',controlDTs)



treatmentDTs=np.load('Documents/treatmentDTs.npy')
controlDTs=np.load('Documents/controlDTs.npy')
treatmentDTs=np.transpose(treatmentDTs,(0,2,3,1,4,5))
controlDTs=np.transpose(controlDTs,(0,2,3,1,4,5))

# slices from near the plane connecting the hemispheres
treatmentDTs=treatmentDTs[:, 40:48, 23:31, 43:44, :, :]
controlDTs=controlDTs[:, 40:48, 23:31, 43:44, :, :]

####### finding ATE

slicetreatmentDTs=decompose(treatmentDTs)
slicecontrolDTs=decompose(controlDTs)
B=slicetreatmentDTs.shape[0]
Nc=slicecontrolDTs.shape[1]
Nt=slicetreatmentDTs.shape[1]
treatmentws=np.ones((1,Nt,1,1))/Nt
controlws=np.ones((1,Nc,1,1))/Nc

ans=np.zeros((B,1,3,3))
for i in range(B):
    ans[i,:,:,:]=spdinvbroyden(slicecontrolDTs[i:(i+1),:,:,:],controlws,spdfmean(slicetreatmentDTs[i:(i+1),:,:,:], treatmentws))[0,:,:,:]

# control DTIs after applying ATE
slicecontrolDTshalf=matrix_sqrt(slicecontrolDTs)
slicecontrolDTshalfinv=np.linalg.inv(slicecontrolDTshalf)
eyes=np.repeat(np.expand_dims(np.eye(3),(0,1)),B,0)
postcontrolDTs=spdexp(slicecontrolDTshalf,slicecontrolDTshalfinv,spdfp(slicecontrolDTshalf, slicecontrolDTshalfinv, eyes, eyes, ans))
postcontrolDTs=recompose(postcontrolDTs,controlDTs.shape[1],controlDTs.shape[2],controlDTs.shape[3])

# draw

j=1

evals, evecs=np.linalg.eig(np.concatenate((controlDTs[j:(j+1),:,:,:,:,:],postcontrolDTs[j:(j+1),:,:,:,:,:]),axis=0))
evals=evals.real
evals/=np.max(evals)
evecs=evecs.real

FA = fractional_anisotropy(evals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, evecs)

sphere = get_sphere(name="repulsion724")

interactive = False

scene = window.Scene()

cfa = RGB/RGB.max()

theseevals = evals[0, :, :, :, :]
theseevecs = evecs[0, :, :, :, :, :]
cfa = cfa[0, :, :, :, :]

scene.add(
    actor.tensor_slicer(theseevals, theseevecs, scalar_colors=cfa, sphere=sphere, scale=1, norm=False)
)
scene.background((255,255,255))

window.show(scene)

window.record(scene=scene, n_frames=1, out_path='Downloads/wholecontrol.png', size=(2000, 2000))

scene.clear()

# draw postcontrolDTs

sphere = get_sphere(name="repulsion724")

interactive = False

scene = window.Scene()

pevals = evals[1, :, :, :, :]
#theseevals=np.transpose(theseevals,(1,2,0,3))
pevecs = evecs[1, :, :, :, :, :]
#theseevecs=np.transpose(theseevecs,(1,2,0,3,4))

scene.add(
    actor.tensor_slicer(pevals, pevecs, scalar_colors=cfa, sphere=sphere, scale=1, norm=False)
)
scene.background((255,255,255))

window.show(scene)

window.record(scene=scene, n_frames=1, out_path='Downloads/wholepostcontrol.png', size=(2000, 2000))

scene.clear()


# inference with just the means of the DTs for each subject

controlmeans=np.transpose(spdfmean(np.transpose(slicecontrolDTs,(1,0,2,3)),np.ones((slicecontrolDTs.shape[1],B,1,1))/B),(1,0,2,3)) #
treatmentmeans=np.transpose(spdfmean(np.transpose(slicetreatmentDTs,(1,0,2,3)),np.ones((slicetreatmentDTs.shape[1],B,1,1))/B),(1,0,2,3))

ate=spdinvbroyden(controlmeans,controlws,spdfmean(treatmentmeans,treatmentws))

controlmeanshalf=matrix_sqrt(controlmeans)
controlmeanshalfinv=np.linalg.inv(controlmeanshalf)
eyes=np.expand_dims(np.eye(3),(0,1))
postcontrolmeans=spdexp(controlmeanshalf,controlmeanshalfinv,spdfp(controlmeanshalf, controlmeanshalfinv, eyes, eyes, ate))

# illustration of this ate

meanevals, meanevecs=np.linalg.eig(np.expand_dims(np.transpose(np.concatenate((controlmeans,postcontrolmeans),axis=0),(1,0,2,3)),2))
meanevals=meanevals.real
meanevals/=np.max(meanevals)
meanevecs=meanevecs.real
#meanevals=meanevals[0:1,:,:,:]
#meanevecs=meanevecs[0:1,:,:,:,:]

FA = fractional_anisotropy(meanevals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, meanevecs)

sphere = get_sphere(name="repulsion724")

interactive = False

scene = window.Scene()

cfa = RGB/RGB.max()

scene.add(
    actor.tensor_slicer(meanevals, meanevecs, scalar_colors=cfa, sphere=sphere, scale=0.5, norm=False)
)
scene.background((255,255,255))

window.show(scene)

window.record(scene=scene, n_frames=1, out_path='Downloads/ateillustration.png', size=(2000, 2000))

scene.clear()

# test based on studentized bootstrap

ate=np.squeeze(vectorizer(spdinvbroyden(controlmeans,controlws,spdfmean(treatmentmeans,treatmentws))),axis=1)
M1=1000
M2=1000
bootstats=np.zeros(M1)
bootates=np.zeros((M1,6,1))
nestbootates=np.zeros((M2,6,1))
for b in range(M1):
    np.random.seed(b)
    boottreatmentindices=np.random.choice(Nt,Nt,replace=True)
    bootcontrolindices=np.random.choice(Nc,Nc,replace=True)
    boottreatmentmeans=treatmentmeans[:,boottreatmentindices,:,:]
    bootcontrolmeans=controlmeans[:,bootcontrolindices,:,:]
    bootates[b,:,:]=vectorizer(spdinvbroyden(bootcontrolmeans,controlws,spdfmean(boottreatmentmeans,treatmentws)))[0,0,:,:]
    for f in range(M2):
        nestboottreatmentindices=np.random.choice(Nt,Nt,replace=True)
        nestbootcontrolindices=np.random.choice(Nc,Nc,replace=True)
        nestboottreatmentmeans=boottreatmentmeans[:,nestboottreatmentindices,:,:]
        nestbootcontrolmeans=bootcontrolmeans[:,nestbootcontrolindices,:,:]
        nestbootates[f,:,:]=vectorizer(spdinvbroyden(nestbootcontrolmeans,controlws,spdfmean(nestboottreatmentmeans,treatmentws)))[0,0,:,:]
    bootcovinv=np.linalg.inv(np.expand_dims(np.cov(np.squeeze(nestbootates), rowvar=False),axis=0)) # (1, 6, 6)
    bootstats[b]=np.squeeze(np.matmul(np.matmul(np.transpose(bootates[b:(b+1),:,:]-ate,(0,2,1)),bootcovinv),bootates[b:(b+1),:,:]-ate))

criticalvalue=np.sort(bootstats)[int(0.95*M1)]
covinv=np.linalg.inv(np.expand_dims(np.cov(np.squeeze(bootates), rowvar=False),axis=0)) # (1, 6, 6)
teststatistic=np.squeeze(np.matmul(np.matmul(np.transpose(ate,(0,2,1)),covinv),ate))
print(teststatistic>criticalvalue)
np.mean(teststatistic<bootstats)

# euclidean comparisons

controlevals=np.linalg.eig(controlmeans)[0]
controlfas=np.squeeze(fractional_anisotropy(controlevals))
controlmds=np.mean(controlevals,(0,2))

treatmentevals=np.linalg.eig(treatmentmeans)[0]
treatmentfas=np.squeeze(fractional_anisotropy(treatmentevals))
treatmentmds=np.mean(treatmentevals,(0,2))

fateststat, fapval=stats.ttest_ind(controlfas, treatmentfas, equal_var=False)
mdteststat, mdpval=stats.ttest_ind(controlmds, treatmentmds, equal_var=False)

print(fapval)
print(mdpval)

treatmentfas=treatmentfas-np.mean(treatmentfas)+np.mean(controlfas) # needed to correctly calculate resample statistics
treatmentmds=treatmentmds-np.mean(treatmentmds)+np.mean(controlmds)

#bootversion
#M=10000
bootfatstats=np.zeros(M)
bootmdtstats=np.zeros(M)
for b in range(M):
    boottreatmentindices=np.random.choice(Nt,Nt,replace=True)
    bootcontrolindices=np.random.choice(Nc,Nc,replace=True)
    boottreatmentfas=treatmentfas[boottreatmentindices]
    bootcontrolfas=controlfas[bootcontrolindices]
    boottreatmentmds=treatmentmds[boottreatmentindices]
    bootcontrolmds=controlmds[bootcontrolindices]
    bootfatstats[b]=stats.ttest_ind(bootcontrolfas,boottreatmentfas, equal_var=False)[0]
    bootmdtstats[b]=stats.ttest_ind(bootcontrolmds,boottreatmentmds, equal_var=False)[0]

print(np.mean(abs(fateststat)<abs(bootfatstats)))
print(np.mean(abs(mdteststat)<abs(bootmdtstats)))
