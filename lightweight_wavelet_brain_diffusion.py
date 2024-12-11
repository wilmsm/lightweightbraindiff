import numpy as np
import torch
import torch.nn.functional as F

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc

import monai
from monai.utils import set_determinism
from monai.engines import SupervisedTrainer
from monai.handlers import MeanSquaredError, from_engine

import generative
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import ignite
from ignite.contrib.handlers import ProgressBar

import utils
import sklearn.model_selection
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pywt
from scipy import ndimage

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(8)
set_determinism(8)
random.seed(8)
np.random.seed(8)
torch.set_float32_matmul_precision('high')


#params
lr=1e-5 #fixed lr
channels=64 #number of channels for the Unet
levels=2 #levels of the wavelet decomposition
num_train_timesteps=1000
num_workers=os.cpu_count()//2
pin_memory=torch.cuda.is_available() if num_workers > 0 else False
batch_size_train=5
batch_size_test=batch_size_train
train_epochs=200
gradient_accumulation_steps=2
pix_spacing=1.
amp_mode="amp"
grad_scaler=True if amp_mode=="amp" else False
num_of_files=-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path=None
#load_path='./ukbb_all_cond_model_64ch_cossattention_250epochs.tar'


train_model=False #if false, a model needs to specified in load_path

#get the data
#inhouse data loader; this needs to be adapted when used on own data; image list is the list of filenames and age list has the corresponding chronological age values
image_list, age_list, unique_subs=utils.get_image_label_data('../../pediatric/data/UKBB', "register.nii.gz", '../../pediatric/data/UKBB/ukbb_participants_healthy.tsv', 'AgeAtScan',num_of_files)
age_list=np.float32(age_list)

image_train, image_test, age_train, age_test=sklearn.model_selection.train_test_split(image_list,age_list,test_size=.15,random_state=8)
image_train, image_val, age_train, age_val=sklearn.model_selection.train_test_split(image_train,age_train,test_size=.1,random_state=8)

age_train=np.int32((age_train-46)//1.0)
age_test=np.int32((age_test-46)//1.0)
age_val=np.int32((age_val-46)//1.0)

#convert data into dictionaries to allow for additional flexibility in MONAI
train_files = [{"image": img, "age": label} for img, label in zip(image_train, age_train)]
test_files = [{"image": img, "age": label} for img, label in zip(image_test, age_test)]
val_files = [{"image": img, "age": label} for img, label in zip(image_val, age_val)]

print("train:",len(train_files),"("+str(np.mean(age_train))+"/"+str(np.std(age_train))+")","test:",len(test_files),"("+str(np.mean(age_test))+"/"+str(np.std(age_test))+")","val:",len(val_files),"("+str(np.mean(age_val))+"/"+str(np.std(age_val))+")")

#wavelet decomposition stuff
def pywt_decompose(x):
    packet= pywt.WaveletPacketND(x.get_array(),'haar',axes=(-3,-2,-1),maxlevel=levels)
    x.set_array(torch.tensor(np.concatenate([y.data.view() for y in packet.get_level(levels)],axis=0)/8.))
    gc.collect()
    return x

lambd_pywt = monai.transforms.Lambdad(keys=['image'], func=pywt_decompose)

#transforms for training and testing
train_transforms = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing),mode='bilinear',lazy=True),monai.transforms.DivisiblePadd(keys=["image"],k=64),monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,256,192]),monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),lambd_pywt,monai.transforms.ToTensord(keys=["image"],track_meta=False),monai.transforms.Lambdad(keys=["age"], func=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1))],lazy=False)
test_transforms = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing)),monai.transforms.DivisiblePadd(keys=["image"],k=64),monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,256,192]),monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),lambd_pywt,monai.transforms.ToTensord(keys=["image"],track_meta=False), monai.transforms.Lambdad(keys=["age"], func=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1))])

#Check loader and test DWT process for debugging purposes
#check_ds = monai.data.Dataset(data=train_files[0:2],transform=train_transforms)
#check_loader = monai.data.DataLoader(check_ds, shuffle=False,batch_size=2, num_workers=0, pin_memory=False)
#im_dict = monai.utils.misc.first(check_loader)
#print(im_dict['image'].shape, im_dict['age'])
#writer = monai.data.ITKWriter()
#writer.set_data_array(torch.squeeze(im_dict['image'][0]),channel_dim=0)
#writer.write("./test_dwt.nii.gz")
# dwt_packet=pywt.WaveletPacketND(im_dict['image'],'haar',axes=(2,3,4),maxlevel=2)
# dwt_list=dwt_packet.get_level(2)
#i=0
#for node in dwt_list:
#    i=i+1
#    writer.set_data_array(np.squeeze(node.data),channel_dim=None)
#    writer.write("./test_"+str(node.parent.node_name)+"_"+str(node.node_name)+".nii.gz")

train_ds = monai.data.Dataset(data=train_files,transform=train_transforms)
train_loader = monai.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size_train, num_workers=num_workers, pin_memory=True)

val_ds = monai.data.Dataset(data=test_files,transform=test_transforms)
val_loader = monai.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)

#HERE WE ONLY SELECT ON IMAGE FROM OUR DATA FOR THE TEST - it is subject 2991038 from UK Biobank in our setup
#This has to be adapted...
test_ds = monai.data.Dataset(data=test_files[13:14],transform=test_transforms)
test_loader = monai.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)


#prepare UNet
model = generative.networks.nets.DiffusionModelUNet(
    spatial_dims=3,
    in_channels=channels,
    out_channels=channels,
    num_channels=[128, 128, 256, 256, 512],
    attention_levels=[False, False, False, True, True],
    num_head_channels=[0,0,0,32,32],
    num_res_blocks=2,
    use_flash_attention=False,
    with_conditioning=True,
    cross_attention_dim=1
)

class UnetWrapper(torch.nn.Module):
    def __init__(self, unet) -> None:
        super().__init__()
        self.unet=unet

    def forward(self, x, timesteps,context):
        return self.unet(x,timesteps,context=context)
    
model=UnetWrapper(model)
#load model if available
if load_path is not None:
    model.load_state_dict(torch.load(load_path)['model_state_dict'])
model.to(device)

#set up the training process
scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, schedule="linear_beta",prediction_type="sample")
inferer=utils.FlexibleConditionalDiffusionInfererCross(scheduler)
opt = torch.optim.AdamW(params=model.parameters(), lr=lr)
trainer = SupervisedTrainer(
    device=device,
    max_epochs=train_epochs,
    train_data_loader=train_loader,
    network=model,
    optimizer=opt,
    loss_function=torch.nn.MSELoss(),
    inferer=inferer,
    prepare_batch=utils.SamplePredictionPrepareBatch(num_train_timesteps=num_train_timesteps,condition_name='age'),
    key_train_metric={"train_acc": MeanSquaredError(reduction='mean',output_transform=from_engine(["pred", "label"]))},
    amp=True
)
ignite.metrics.RunningAverage(output_transform=from_engine(["loss"],first=True)).attach(trainer, 'avg. loss')
ignite.contrib.handlers.ProgressBar().attach(trainer,['avg. loss'])

#train the model
if train_model:
    trainer.run()

#trained model is available now, let's do some stuff
model.eval()

#samples a new subject from the learned distribution at 7 different age values (55-80) using DDIM w/ 100 steps
with torch.inference_mode():
    noise = torch.randn((7, channels, 48, 64, 48))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    ddim_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, schedule="linear_beta",prediction_type="sample")
    ddim_scheduler.set_timesteps(num_inference_steps=100)
    #age values predefined
    conditioning=(torch.tensor([50.,55.,60.,65.,70.,75.,80.])-46)
    conditioning=conditioning.unsqueeze(-1).unsqueeze(-1).to(device)
    with torch.cuda.amp.autocast():
        image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=ddim_scheduler, save_intermediates=False,conditioning=conditioning,mode='crossattn')
    image = image.cpu().numpy()

#turn the samples images into "images" - transform from wavelet space into image space
dwt_packet=pywt.WaveletPacketND(np.zeros((1,1,192,256,192)),'haar',axes=(2,3,4),maxlevel=levels)
dwt_list=dwt_packet.get_level(levels)

def recon_image(image, dwt_packet, dwt_list,idx=0):
    i=0
    for node in dwt_list:
        dwt_packet[node.path].data=image[idx,i,:,:,:][np.newaxis,np.newaxis,:,:,:]*8.
        i=i+1

writer = monai.data.ITKWriter() 
for idx in range(noise.shape[0]):
    recon_image(image, dwt_packet, dwt_list,idx)

    dwt_recon=dwt_packet.reconstruct()
    writer.set_data_array(np.squeeze(dwt_recon),channel_dim=None)
    writer.write("./ukbb_test_cross_gen_"+str(idx)+".nii.gz")


model.eval()

#classification of test images
num_classes_overall=np.max(age_test)+1
num_classes=num_classes_overall
model.eval()

runs=50 #do 50 MC runs
reduce_every=5 #drop the least performing half of classes after every 5 runs
reduce_comps=True #False means that we don't reduce the classes
limited_classes=False
scores=[]
test_ages=[]
timesteps_pre=[100,600,200,700,300,800,400,900,500,50]#use fixed timesteps for evaluation - for random choices see below
with torch.inference_mode():
    for test_batch in test_loader:
        for test_idx in range(test_batch['image'].shape[0]):
            num_classes=num_classes_overall
            curr_scores=np.arange(num_classes)*0.
            test_image=test_batch['image'][test_idx,:][None,:]
            test_image_allclass=torch.repeat_interleave(test_image,num_classes,dim=0)
            test_image_allclass= test_image_allclass.to(device)
            test_age=test_batch['age'][test_idx]

            classes=torch.arange(num_classes).to(device)
            classes=torch.arange(num_classes,dtype=torch.float32).to(device)
            classes_to_keep=np.arange(num_classes)

            for r in tqdm(range(runs)):
                if reduce_comps and (r>0) and (num_classes > 3) and ((r % reduce_every) == 0):
                    limited_classes=True
                    scores_idx_sorted=np.argsort(curr_scores[classes_to_keep])

                    classes_to_drop=classes_to_keep[scores_idx_sorted[scores_idx_sorted.shape[0]//2:]]
                    classes_to_keep=classes_to_keep[scores_idx_sorted[0:scores_idx_sorted.shape[0]//2]]

                    curr_scores[classes_to_drop]=np.finfo(np.float64).max
                    num_classes=len(classes_to_keep)
                    test_image_allclass=torch.repeat_interleave(test_image,num_classes,dim=0)
                    test_image_allclass= test_image_allclass.to(device)

                noise = torch.randn((1, channels, 48, 64, 48))
                noise_allclass=torch.repeat_interleave(noise,num_classes,dim=0)
                noise_allclass = noise_allclass.to(device)

                timesteps = torch.tensor([timesteps_pre[r%10]],device=device).long()
                #uncomment if you want to use random timesteps as in the paper instead of the pre-defined ones
                #timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,(1,),device=device).long().expand((num_classes))
                timesteps=torch.repeat_interleave(timesteps,num_classes,dim=0)

                weight_factor=np.exp(-7*(timesteps[0].cpu().numpy()/inferer.scheduler.num_train_timesteps))
                with torch.cuda.amp.autocast():
                    if limited_classes:
                        curr_scores[classes_to_keep]=curr_scores[classes_to_keep]+weight_factor*F.mse_loss(test_image_allclass,inferer(test_image_allclass,model,noise_allclass,timesteps,classes[classes_to_keep].unsqueeze(-1).unsqueeze(-1),mode='crossattn'),reduction='none').sum((1,2,3,4)).cpu().numpy()
                    else:
                        curr_scores[0:num_classes//2]=curr_scores[0:num_classes//2]+weight_factor*F.mse_loss(test_image_allclass[0:num_classes//2,:],inferer(test_image_allclass[0:num_classes//2,:],model,noise_allclass[0:num_classes//2,:],timesteps[0:num_classes//2],classes[0:num_classes//2].unsqueeze(-1).unsqueeze(-1),mode='crossattn'),reduction='none').sum((1,2,3,4)).cpu().numpy()
                        curr_scores[num_classes//2:]=curr_scores[num_classes//2:]+weight_factor*F.mse_loss(test_image_allclass[num_classes//2:,:],inferer(test_image_allclass[num_classes//2:,:],model,noise_allclass[num_classes//2:,:],timesteps[num_classes//2:],classes[num_classes//2:].unsqueeze(-1).unsqueeze(-1),mode='crossattn'),reduction='none').sum((1,2,3,4)).cpu().numpy()
            scores.append(curr_scores)
            test_ages.append(test_age.squeeze())

#compute the MAE
overall_mae=[]
overall_mae_naive=[]
for i in range(len(scores)):
    mae=np.abs(np.argmin(scores[i])-test_ages[i].cpu().numpy())
    overall_mae_naive.append(np.abs(test_ages[i].cpu().numpy()-age_train.mean()))
    print(mae)
    overall_mae.append(mae)
print('MAE',np.mean(overall_mae),np.std(overall_mae),'naive',np.mean(overall_mae_naive),np.std(overall_mae_naive))



#--------------------------------
#code snippets for counterfactual generation
#counterfactual image generation
idx=0
recon_image(test_image, dwt_packet, dwt_list,idx)

#reconstruct the test image first without doing anything 
dwt_recon=dwt_packet.reconstruct()
writer.set_data_array(np.squeeze(dwt_recon),channel_dim=None)
writer.write("./test_image_recon.nii.gz")

#define real age of the test subject
real_age=53
#provide counterfactual ages
ages=[53,46,60,70,80]
#parameters of the DDIM inferer used for counterfactuals
ddim_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, schedule="linear_beta",prediction_type="sample")
ddim_scheduler.set_timesteps(num_inference_steps=200)
for age in ages:
    print(age)
    ddim_scheduler.clip_sample=False
    start_noise=990
    #use reverse DDIM to encode test image
    step_size=5
    noisy_image=test_image.to(device)
    real_conditioning=(torch.tensor([np.float32(real_age)])-46)
    real_conditioning=real_conditioning.unsqueeze(-1).unsqueeze(-1).to(noisy_image.device)

    #actual reverse DDIM
    with torch.inference_mode():
        for t in np.arange(0,start_noise+step_size,step_size):
            model_output = model(noisy_image, torch.Tensor((t,)).to(noisy_image.device), real_conditioning)
            noisy_image, _ = ddim_scheduler.reversed_step(model_output, t, noisy_image)

    #remove artifacts surrounding the encoded brain in the latent space by using a binary mask
    struct1 = ndimage.generate_binary_structure(3, 49)
    noisy_image=noisy_image*torch.tensor(np.float32(ndimage.binary_dilation(np.logical_not(np.isclose(np.abs(test_image[0,0,:,:,:]),1)),struct1))).unsqueeze(0).unsqueeze(0).to(device)
    recon_image(noisy_image.cpu().numpy(), dwt_packet, dwt_list,idx)
    dwt_recon=dwt_packet.reconstruct()
    writer.set_data_array(np.squeeze(dwt_recon),channel_dim=None)
    writer.write("./test_image_noisy_ddim.nii.gz")

    #generate the actual counterfactual
    conditioning=(torch.tensor([np.float32(age)])-46)
    conditioning=conditioning.unsqueeze(-1).unsqueeze(-1).to(noisy_image.device)
    ddim_scheduler.clip_sample=True
    with torch.inference_mode():
        for t in np.arange(start_noise,-step_size,-step_size):
            model_output = model(noisy_image, torch.Tensor((t,)).to(noisy_image.device), conditioning)
            noisy_image, _ = ddim_scheduler.step(model_output, t, noisy_image)

    recon_image(noisy_image.cpu().numpy(), dwt_packet, dwt_list,idx)
    dwt_recon=dwt_packet.reconstruct()
    writer.set_data_array(np.squeeze(dwt_recon),channel_dim=None)
    writer.write("./test_image_noisy_"+str(age)+".nii.gz")




