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

# inference with the means of the DTs for each subject

controlmeans=np.transpose(spdfmean(np.transpose(slicecontrolDTs,(1,0,2,3)),np.ones((slicecontrolDTs.shape[1],B,1,1))/B),(1,0,2,3)) #
treatmentmeans=np.transpose(spdfmean(np.transpose(slicetreatmentDTs,(1,0,2,3)),np.ones((slicetreatmentDTs.shape[1],B,1,1))/B),(1,0,2,3))

ate=spdinvbroyden(controlmeans,controlws,spdfmean(treatmentmeans,treatmentws))

controlmeanshalf=matrix_sqrt(controlmeans)
controlmeanshalfinv=np.linalg.inv(controlmeanshalf)
eyes=np.expand_dims(np.eye(3),(0,1))
postcontrolmeans=spdexp(controlmeanshalf,controlmeanshalfinv,spdfp(controlmeanshalf, controlmeanshalfinv, eyes, eyes, ate))

## illustration of this ate
 
controlevals, controlevecs=np.linalg.eig(controlmeans)
controlevals=controlevals.real
controlevecs=controlevecs.real
postcontrolevals, postcontrolevecs=np.linalg.eig(postcontrolmeans)
postcontrolevals=postcontrolevals.real
postcontrolevecs=postcontrolevecs.real
 
globalmax=max(np.max(controlevals),np.max(postcontrolevals))
controlevals/=globalmax
postcontrolevals/=globalmax
 
# pack the 26 ellipsoids into a 4-row by 7-col grid filled row-major from the top-left, leaving the bottom-left and bottom-right cells empty
controlgridevals=np.zeros((28,3))
controlgridevals[:21]=controlevals[0,:21]
controlgridevals[22:27]=controlevals[0,21:]
controlgridevals=np.expand_dims(controlgridevals.reshape(4,7,3)[::-1].transpose(1,0,2),2) # (7,4,1,3)
controlgridevecs=np.zeros((28,3,3))
controlgridevecs[:21]=controlevecs[0,:21]
controlgridevecs[22:27]=controlevecs[0,21:]
controlgridevecs=np.expand_dims(controlgridevecs.reshape(4,7,3,3)[::-1].transpose(1,0,2,3),2) # (7,4,1,3,3)
 
postcontrolgridevals=np.zeros((28,3))
postcontrolgridevals[:21]=postcontrolevals[0,:21]
postcontrolgridevals[22:27]=postcontrolevals[0,21:]
postcontrolgridevals=np.expand_dims(postcontrolgridevals.reshape(4,7,3)[::-1].transpose(1,0,2),2)
postcontrolgridevecs=np.zeros((28,3,3))
postcontrolgridevecs[:21]=postcontrolevecs[0,:21]
postcontrolgridevecs[22:27]=postcontrolevecs[0,21:]
postcontrolgridevecs=np.expand_dims(postcontrolgridevecs.reshape(4,7,3,3)[::-1].transpose(1,0,2,3),2)
 
sphere = get_sphere(name="repulsion724")
 
interactive = False
 
FA = fractional_anisotropy(controlgridevals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, controlgridevecs)
cfa = RGB/RGB.max()
scene = window.Scene()
scene.add(
    actor.tensor_slicer(controlgridevals, controlgridevecs, scalar_colors=cfa, sphere=sphere, scale=0.5, norm=False)
)
scene.background((255,255,255))
window.show(scene)
window.record(scene=scene, n_frames=1, out_path='Downloads/controlmeans.png', size=(2000, 2000))
scene.clear()
 
FA = fractional_anisotropy(postcontrolgridevals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, postcontrolgridevecs)
cfa = RGB/RGB.max()
scene = window.Scene()
scene.add(
    actor.tensor_slicer(postcontrolgridevals, postcontrolgridevecs, scalar_colors=cfa, sphere=sphere, scale=0.5, norm=False)
)
scene.background((255,255,255))
window.show(scene)
window.record(scene=scene, n_frames=1, out_path='Downloads/postcontrolmeans.png', size=(2000, 2000))
scene.clear()

# bootstrap test

np.random.seed(1)

ate=np.squeeze(vectorizer(spdinvbroyden(controlmeans,controlws,spdfmean(treatmentmeans,treatmentws))),axis=1)
M=1000
bootates=np.zeros((M,6,1))
for b in range(M):
    boottreatmentindices=np.random.choice(Nt,Nt,replace=True)
    bootcontrolindices=np.random.choice(Nc,Nc,replace=True)
    boottreatmentmeans=treatmentmeans[:,boottreatmentindices,:,:]
    bootcontrolmeans=controlmeans[:,bootcontrolindices,:,:]
    bootates[b,:,:]=vectorizer(spdinvbroyden(bootcontrolmeans,controlws,spdfmean(boottreatmentmeans,treatmentws)))[0,0,:,:]

covinv=np.linalg.inv(np.expand_dims(np.cov(np.squeeze(bootates), rowvar=False),axis=0)) # (1, 6, 6)
bootstats=np.zeros(M)
for b in range(M):
    bootstats[b]=np.squeeze(np.matmul(np.matmul(np.transpose(bootates[b:(b+1),:,:]-ate,(0,2,1)),covinv),bootates[b:(b+1),:,:]-ate))

criticalvalue=np.sort(bootstats)[int(0.95*M)]
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
