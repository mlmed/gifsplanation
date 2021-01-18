import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import glob
import numpy as np
import skimage, skimage.filters
import sklearn, sklearn.metrics
import captum, captum.attr
import torch, torch.nn
import pickle
import pandas as pd
import shutil
import os,sys


def compute_attribution(image, method, clf, target, plot=False, ret_params=False, fixrange=None, p=0.0, ae=None, sigma=0, threshold=False):
    
    image = image.clone().detach()
    
    def clean(saliency):
        saliency = np.abs(saliency)
        if sigma > 0:
            saliency = skimage.filters.gaussian(saliency, 
                        mode='constant', 
                        sigma=(sigma, sigma), 
                        truncate=3.5)
        if threshold:
            saliency = thresholdf(saliency, 95)
        return saliency
    
    if "latentshift" in method:
        z = ae.encode(image).detach()
        z.requires_grad = True
        xp = ae.decode(z)
        pred = clf((image*p + xp*(1-p)))[:,clf.pathologies.index(target)]
        dzdxp = torch.autograd.grad((pred), z)[0]
        
        cache = {}
        def compute_shift(lam):
            #print(lam)
            if lam not in cache:
                xpp = ae.decode(z+dzdxp*lam).detach()
                pred1 = clf((image*p + xpp*(1-p)))[:,clf.pathologies.index(target)].detach().cpu().numpy()
                cache[lam] = xpp, pred1
            return cache[lam]
        
        #determine range
        #initial_pred = pred.detach().cpu().numpy()
        _, initial_pred = compute_shift(0)
        
        
        if fixrange:
            lbound,rbound = fixrange
        else:
            #search params
            step = 10

            #left range
            lbound = 0
            last_pred = initial_pred
            while True:
                xpp, cur_pred = compute_shift(lbound)
                #print("lbound",lbound, "last_pred",last_pred, "cur_pred",cur_pred)
                if last_pred < cur_pred:
                    break
                if initial_pred-0.5 > cur_pred:
                    break
                if lbound <= -1000:
                    break
                last_pred = cur_pred
                if np.abs(lbound) < step:
                    lbound = lbound - 1
                else:
                    lbound = lbound - step

            #right range
            rbound = 0
            last_pred = initial_pred
            while True:
                xpp, cur_pred = compute_shift(rbound)
                #print("rbound",rbound, "last_pred",last_pred, "cur_pred",cur_pred)
                if last_pred > cur_pred:
                    break
                if initial_pred+0.05 < cur_pred:
                    break
                if rbound >= 1000:
                    break
                last_pred = cur_pred
                if np.abs(rbound) < step:
                    rbound = rbound + 1
                else:
                    rbound = rbound + step
        
        print(initial_pred, lbound,rbound)
        #lambdas = np.arange(lbound,rbound,(rbound+np.abs(lbound))//10)
        lambdas = np.arange(lbound,rbound+1,np.abs((lbound-rbound)/10))
        ###########################
        
        y = []
        dimgs = []
        xp = ae.decode(z)[0][0].unsqueeze(0).unsqueeze(0).detach()
        for lam in lambdas:
            
            xpp, pred = compute_shift(lam)
            dimgs.append(xpp.cpu().numpy())
            y.append(pred)
            
        if ret_params:
            params = {}
            params["dimgs"] = dimgs
            params["lambdas"] = lambdas
            params["y"] = y
            params["initial_pred"] = initial_pred
            return params
        
        if plot:
            
            plt.imshow(image.detach().cpu()[0][0], interpolation='none', cmap="gray")
            plt.title("image")
            plt.show()
            plt.imshow(xp.detach().cpu()[0][0], interpolation='none', cmap="gray")
            plt.title("image_recon")
            plt.show()
            plt.imshow(((image + xp)/2).detach().cpu()[0][0], interpolation='none', cmap="gray")
            plt.title("image_recon_mix")
            plt.show()
            
            plt.plot(lambdas,y)
            plt.xlabel("lambda shift");
            plt.ylabel("Prediction of " + target);
            plt.show()
        
        if "-max" in method:
            dimage = np.max(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]),0)
        elif "-mean" in method:
            dimage = np.mean(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]),0)
        elif "-mm" in method:
            dimage = np.abs(dimgs[0][0][0] - dimgs[0][0][-1])
        elif "-int" in method:
            dimages = []
            for i in range(len(dimgs)):
                dimages.append(np.abs(dimgs[0][0][0] - dimgs[0][0][1]))
            dimage = np.mean(dimages,0)
        else:
            raise Exception("Unknown mode")
        
        dimage = clean(dimage)
        return dimage
    
    
    if method == "grad":
        image.requires_grad = True
        pred = clf(image)[:,clf.pathologies.index(target)]
        dimage = torch.autograd.grad(torch.abs(pred), image)[0]
        dimage = dimage.detach().cpu().numpy()[0][0]
        dimage = clean(dimage)
        return dimage
    
    if method == "integrated":
        attr = captum.attr.IntegratedGradients(clf)
        dimage = attr.attribute(image, 
                                target=clf.pathologies.index(target),
                                n_steps=100, 
                                return_convergence_delta=False, 
                                internal_batch_size=1)
        dimage = dimage.detach().cpu().numpy()[0][0]
        dimage = clean(dimage)
        return dimage
    
    if method == "guided":
        
        attr = captum.attr.GuidedBackprop(clf)
        dimage = attr.attribute(image, target=clf.pathologies.index(target))
        dimage = dimage.detach().cpu().numpy()[0][0]
        dimage = clean(dimage)
        return dimage
    
def thresholdf(x, percentile):
    return x * (x > np.percentile(x, percentile))

def calc_iou(preds, gt_seg):
    gt_seg = gt_seg.astype(np.bool)
    seg_area_percent = (gt_seg > 0).sum()/(gt_seg != -1).sum() # percent of area
    preds = (thresholdf(preds, (1-seg_area_percent)*100) > 0).astype(np.bool)
    #EPS = 10e-16
    ret = {}
    ret["iou"] = (gt_seg & preds).sum() / ((gt_seg | preds).sum())
    ret["precision"] = sklearn.metrics.precision_score(gt_seg.flatten(),preds.flatten())
    ret["recall"] = sklearn.metrics.recall_score(gt_seg.flatten(),preds.flatten())  
    return ret


def run_eval(target, data, model, ae, to_eval=None, compute_recon=False, pthresh = 0, limit = 40):
    dwhere = np.where(data.csv.has_masks & (data.labels[:,data.pathologies.index(target)]  == 1))[0]
    results = []
    
    if to_eval == None:
        to_eval = [
                   "latentshift-max", 
#                    "latentshift-mean", 
#                    "latentshift-mm", 
#                    "latentshift-int",
                   "grad", "integrated", "guided"
                    ]
    
    for method in to_eval:
        count = 0
        for idx in dwhere:
            #print(method, idx)
            imgresults = []
            sample = data[idx]

            if data.pathologies.index(target) not in sample["pathology_masks"]:
                #print("no mask found")
                continue
            image = torch.from_numpy(sample["img"]).unsqueeze(0).cuda()
            p = model(image)[:,model.pathologies.index(target)].detach().cpu()
            #print(p)
            if p > pthresh:
                dimage = compute_attribution(image, method, model, target, ae=ae)
                #print(method, dimage.shape)
                metrics = calc_iou(dimage, sample["pathology_masks"][data.pathologies.index(target)][0])
                if compute_recon:
                    recon = ae(image)["out"]
                    metrics["mse"] = float(((image-recon)**2).mean().detach().cpu().numpy())
                    metrics["mae"] = float(torch.abs(image-recon).mean().detach().cpu().numpy())
                metrics["idx"] = idx
                metrics["target"] = target
                metrics["method"] = method
                results.append(metrics)
                count += 1
                if count > limit:
                    break
    return pd.DataFrame(results)


def generate_video(image, model, target, ae, temp_path, target_filename=None, border=True, note="", show=False):
    
    params = compute_attribution(image.cuda(), "latentshift", model, target, ret_params=True, ae=ae)
    dimgs = params["dimgs"]
    
    #ffmpeg -i gif-tmp/image-%d-a.png -vcodec libx264 aout.mp4
    temp_path = "/lscratch/joecohen/SDS-2342-ASDAA"
    shutil.rmtree(temp_path, ignore_errors=True) 
    towrite = list(dimgs) + list(reversed(dimgs))
    img = image[0][0].cpu().numpy()

    for idx, dimg in enumerate(towrite):
        if idx % 10 == 0:
            print(idx)
            
        p = model(torch.from_numpy(dimg).cuda())[0,model.pathologies.index(target)].detach().cpu().numpy()
        fig = plt.Figure(figsize=(10, 6), dpi=50)
        gcf = plt.gcf()
        gcf.set_size_inches(10, 6)
        fig.set_canvas(gcf.canvas)
        if border:
            plt.imshow(np.concatenate([img,dimg[0][0]], 1), interpolation='none', cmap='Greys_r')
            plt.title("Pred of {}: {:.2f}".format(target, p),fontsize=25)
        else:
            plt.imshow(dimg[0][0], interpolation='none', cmap='Greys_r')
        plt.axis('off')
        
        if not os.path.exists(temp_path): 
            os.mkdir(temp_path)
        for k in range(3):
            i = idx + len(towrite)*k
            fig.savefig(temp_path +'/image-' + str(i) + '-a.png', bbox_inches='tight', pad_inches=0, transparent=False)
      
    if not target_filename:
        target_filename = "videos/single-{}_{}_{}_{}".format(
            target,
            str(ae),
            str(model),
            note)

    cmd = "module load ffmpeg;ffmpeg -loglevel quiet -stats -y -i {}/image-%d-a.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p '{}.mp4'".format(temp_path,target_filename)
    
    print(cmd)
    os.system(cmd)
    
    if show:
        from IPython.display import Video
        return Video(target_filename + ".mp4", embed=True)
    else:
        return target_filename + ".mp4"

    
    
    
def generate_attributions(sample, model, target, ae, temp_path, dmerge):

    image = torch.from_numpy(sample["img"]).unsqueeze(0).cuda()
    
    p = model(image)[:,model.pathologies.index(target)].detach().cpu()
    print(p)
    methods = ["image", "grad", "guided", "integrated", "latentshift-max"]
    fig, ax = plt.subplots(1,len(methods), figsize=(8,3), dpi=350)
    for i, method in enumerate(methods):

        if method == "image":
            ax[i].imshow(image.detach().cpu()[0][0], interpolation='none', cmap="gray")
            ax[i].set_ylabel(target + "\n" + str(model), fontsize=7)
        else:
            dimage = compute_attribution(image, method, model, target, ae=ae, threshold=True)
            ax[i].imshow(image.detach().cpu()[0][0], interpolation='none', cmap="gray")
            dimage[dimage==0] = np.nan
            ax[i].imshow(dimage, interpolation='none', alpha=0.8, cmap="Reds");
        try:
            ax[i].imshow(sample["pathology_masks"][dmerge.pathologies.index(target)][0], interpolation='none', alpha=0.1);
        except:
            pass
        ax[i].get_xaxis().set_visible(False)
        ax[i].set_yticks([])
        #ax[i].get_yaxis().set_visible(False)
        ax[i].set_title(method, fontsize=8)
    fig.subplots_adjust(wspace=0, hspace=0);
    plt.show()
    
    
    




