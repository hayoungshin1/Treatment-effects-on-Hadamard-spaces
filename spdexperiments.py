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
 
# Eigendecompositions for each set of 26 means (shape (1,26,3,3) -> flat (26,3) and (26,3,3))
control_evals_flat, control_evecs_flat = np.linalg.eig(controlmeans[0])
post_evals_flat, post_evecs_flat = np.linalg.eig(postcontrolmeans[0])
control_evals_flat = control_evals_flat.real
control_evecs_flat = control_evecs_flat.real
post_evals_flat = post_evals_flat.real
post_evecs_flat = post_evecs_flat.real
 
# Shared scale: divide every eigenvalue (in both images) by the global max
global_max = max(np.max(control_evals_flat), np.max(post_evals_flat))
control_evals_flat = control_evals_flat / global_max
post_evals_flat = post_evals_flat / global_max
 
def build_grid(evals_flat, evecs_flat):
    """
    Lay 26 ellipsoids into a 4-row x 7-column grid, filled row-major from the
    top-left. Bottom-left and bottom-right cells are left empty (zero eigenvalues
    -> tensor_slicer renders nothing there). Returns evals of shape (7,4,1,3)
    and evecs of shape (7,4,1,3,3) in tensor_slicer's (X,Y,Z,...) convention,
    with Y=3 at the top and Y=0 at the bottom.
    """
    evals_grid = np.zeros((7, 4, 1, 3))
    evecs_grid = np.zeros((7, 4, 1, 3, 3))
    evecs_grid[..., :, :] = np.eye(3)
    for i in range(26):
        if i < 21:
            row = i // 7        
            col = i % 7         
        else:
            row = 3             
            col = (i - 21) + 1  
        x = col
        y = 3 - row             
        evals_grid[x, y, 0, :] = evals_flat[i]
        evecs_grid[x, y, 0, :, :] = evecs_flat[i]
    return evals_grid, evecs_grid
 
control_evals_grid, control_evecs_grid = build_grid(control_evals_flat, control_evecs_flat)
post_evals_grid, post_evecs_grid = build_grid(post_evals_flat, post_evecs_flat)
 
sphere = get_sphere(name="repulsion724")
 
interactive = False
 
def render_grid(evals_grid, evecs_grid, out_path):
    FA = fractional_anisotropy(evals_grid)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs_grid)
    cfa = RGB / RGB.max()
    scene = window.Scene()
    scene.add(
        actor.tensor_slicer(evals_grid, evecs_grid, scalar_colors=cfa, sphere=sphere, scale=0.5, norm=False)
    )
    scene.background((255, 255, 255))
    window.show(scene)
    window.record(scene=scene, n_frames=1, out_path=out_path, size=(2000, 2000))
    scene.clear()
 
render_grid(control_evals_grid, control_evecs_grid, 'Downloads/controlmeans.png')
render_grid(post_evals_grid, post_evecs_grid, 'Downloads/postcontrolmeans.png')

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
