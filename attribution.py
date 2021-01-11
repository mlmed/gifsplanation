import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import glob
import numpy as np
import skimage, skimage.filters
import captum, captum.attr
import torch, torch.nn
import pickle
import pandas as pd


def compute_attribution(image, method, clf, target, plot=False, ret_dimgs=False, p=0.0, ae=None, sigma=0):
    
    image = image.clone().detach()
    
    def clean(saliency):
        saliency = np.abs(saliency)
        #saliency = threshold(saliency, 95)
        if sigma > 0:
            saliency = skimage.filters.gaussian(saliency, 
                        mode='constant', 
                        sigma=(sigma, sigma), 
                        truncate=3.5)
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
        initial_pred = pred.detach().cpu().numpy()
        
        #left range
        lbound = 0
        last_pred = pred.detach().cpu().numpy()
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
            lbound = lbound - 10
            
        #right range
        rbound = 0
        last_pred = pred.detach().cpu().numpy()
        while True:
            xpp, cur_pred = compute_shift(rbound)
            #print(rbound, last_pred, cur_pred)
            if last_pred > cur_pred:
                break
            if initial_pred+0.1 < cur_pred:
                break
            if rbound >= 1000:
                break
            last_pred = cur_pred
            rbound = rbound + 10
        
#         rang = 5
#         lambdas = np.arange(-rang,rang+1,rang//5)
        print(initial_pred, lbound,rbound)
        #lambdas = np.arange(lbound,rbound,(rbound+np.abs(lbound))//10)
        lambdas = np.arange(lbound,rbound,np.abs((lbound-rbound)/10))
        ###########################
        
        y = []
        diffs = []
        dimgs = []
        xp = ae.decode(z)[0][0].unsqueeze(0).unsqueeze(0).detach()
        for lam in lambdas:
            
            xpp, pred = compute_shift(lam)
            
            #xpp = ae.decode(z+dzdxp*lam).detach()
            dimgs.append(xpp.cpu().numpy())
            
            #pred = clf((image*p + xpp*(1-p)))[:,clf.pathologies.index(target)]
            y.append(pred)
            
            diff = xpp.cpu()[0][0]
            #diff = np.abs((xp-xpp).cpu()[0][0])
            diffs.append(diff.numpy())  
        #return diffs
        if ret_dimgs:
            return dimgs
        
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
            dimage = np.max(np.abs(xp.cpu().numpy()[0][0] - diffs),0)
        if "-mean" in method:
            dimage = np.mean(np.abs(xp.cpu().numpy()[0][0] - diffs),0)
        if "-mm" in method:
            dimage = np.abs(diffs[0] - diffs[-1])
        if "-int" in method:
            dimages = []
            for i in range(len(diffs)):
                dimages.append(np.abs(diffs[0] - diffs[1]))
            dimage = np.mean(dimages,0)
            
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
                                n_steps=50, 
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
    
def threshold(x, percentile):
    return x * (x > np.percentile(x, percentile))

def calc_iou(locs, segs):
    segs = segs.astype(np.bool)
    seg_area_percent = (segs > 0).sum()/(segs != -1).sum() # percent of area
    locs = (threshold(locs, (1-seg_area_percent)*100) > 0).astype(np.bool) #
    EPS = 10e-16
    iou = (segs & locs).sum() / ((segs | locs).sum() + EPS)                          
    iop = (segs & locs).sum() / (locs.sum() + EPS)                               
    iot = (segs & locs).sum() / (segs.sum() + EPS)       
    return {"iou":iou, "iop":iop,"iot":iot}


def run_eval(target, data, model, ae, pthresh = 0, limit = 20):
    dwhere = np.where(data.csv.has_masks & (data.labels[:,data.pathologies.index(target)]  == 1))[0]
    results = []
    for method in [
                   "latentshift-max", 
#                    "latentshift-mean", 
#                    "latentshift-mm", 
#                    "latentshift-int",
                   "grad", "integrated", "guided"
                    ]:
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
                metrics["idx"] = idx
                metrics["target"] = target
                metrics["method"] = method
                results.append(metrics)
                count += 1
                if count > limit:
                    break
    return pd.DataFrame(results)







