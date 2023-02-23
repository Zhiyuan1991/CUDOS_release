#Adapt code from: https://github.com/rtqichen/beta-tcvae
#Made modification for computing MIG-sup and for 3dshapes
import math
import torch
from torch.autograd import Variable
import lib.utils as utils
import numpy as np
metric_name = 'MIG'

def MIG(mi_normed,remove_large=True):
    for row in mi_normed:
        if row[0]>1:
            print("MI>1 found")
            row[0]=row[1]
            row[1]=row[2]
    gap=mi_normed[:, 0] - mi_normed[:, 1]
    print(gap)
    if remove_large:
        ind=np.where(gap<=1)
        return gap[ind]
    else:
        return gap

def compute_metric_3dshapes(marginal_entropies,cond_entropies,active_units,fac_nums,flags):
    factor_entropies=fac_nums
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / (torch.Tensor(factor_entropies).log()+1e-10)[:, None]
    if flags.task==1:
        mi_normed[:2,:]=0
    elif flags.task==2:
        mi_normed[1,:]=0
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    if flags.task==1:
        metric=metric[2:]
    elif flags.task==2:
        metric=metric[[0,2,3,4,5]] #pick up the new factors only
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2,remove_large=False)
    return metric, metric_sup, mi_normed

def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None,flags=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    #pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(50, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size]
            ,flags)
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        #pbar.update(batch_size)
    #pbar.close()

    entropies /= S

    return entropies

def mutual_info_metric_3dshapes(sess, vae, manager, zlayer=1,task=0,flags=None):
    n_samples=manager.sample_size
    N = manager.sample_size
    K = vae.z_dim
    nparams = vae.q_dist.nparams

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)
    qz_samples = torch.Tensor(N, K)
    n = 0

    indices = list(range(n_samples))
    batch_size=100
    total_batch = n_samples // batch_size

    # Loop over all batches
    for i in range(total_batch):
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]
        xs = manager.get_images(batch_indices)

        z_mean, z_logvar= vae.transform(sess, xs)
        z_sample=vae.inference_z(sess, xs)
        z_logsigma=z_logvar*0.5
        qz_samples[n:n + batch_size,:]=torch.from_numpy(z_sample)
        qz_params[n:n + batch_size,:,0]=torch.from_numpy(z_mean)
        qz_params[n:n + batch_size,:,1]=torch.from_numpy(z_logsigma)
        if flags.sparse_coding_MIG:
            z_spike=vae.inference_spike(sess,xs)
            qz_params[n:n + batch_size,:,2]=torch.from_numpy(z_spike)
        n += batch_size

    if task==0:
        fac_nums=[10,10,10,8,4,15]
    elif task==1:
        fac_nums=[1,1,10,8,4,15]
    elif task==2:
        fac_nums=[9,1,10,8,4,15]
    else:
        print("unknow task for computing MIG")
        quit()

    qz_params = Variable(qz_params.view(fac_nums[0],fac_nums[1],fac_nums[2],fac_nums[3],fac_nums[4],fac_nums[5], K, nparams).cuda())
    qz_samples = Variable(qz_samples.view(fac_nums[0],fac_nums[1],fac_nums[2],fac_nums[3],fac_nums[4],fac_nums[5], K).cuda())

    if flags.sparse_coding_MIG:
        qz_spike=qz_params[:, :, :, :, :, :, :, 2]
        mean_spike=torch.mean(qz_spike.contiguous().view(N, K), dim=0)
        active_units=torch.arange(0, K)[mean_spike > 0.7].long()
    else:
        qz_means = qz_params[:, :, :, :, :, :, :, 0]
        var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
        active_units = torch.arange(0, K)[var > 1e-2].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist,flags=flags)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(6, K)

    print('Estimating conditional entropies for floor_hue.')
    for i in range(fac_nums[0]):
        qz_samples_scale = qz_samples[i, :, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[0], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[0], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[0] += cond_entropies_i.cpu() / fac_nums[0]

    print('Estimating conditional entropies for wall_hue.')
    for i in range(fac_nums[1]):
        qz_samples_scale = qz_samples[:, i, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[1], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[1], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[1] += cond_entropies_i.cpu() / fac_nums[1]

    print('Estimating conditional entropies for object_hue.')
    for i in range(fac_nums[2]):
        qz_samples_scale = qz_samples[:, :, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[2], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[2], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[2] += cond_entropies_i.cpu() / fac_nums[2]

    print('Estimating conditional entropies for scale')
    for i in range(fac_nums[3]):
        qz_samples_scale = qz_samples[:, :, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[3], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[3], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[3] += cond_entropies_i.cpu() / fac_nums[3]

    print('Estimating conditional entropies for shape')
    for i in range(fac_nums[4]):
        qz_samples_scale = qz_samples[:, :, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[4], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[4], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[4] += cond_entropies_i.cpu() / fac_nums[4]

    print('Estimating conditional entropies for orientation')
    for i in range(fac_nums[5]):
        qz_samples_scale = qz_samples[:, :, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // fac_nums[5], K).transpose(0, 1),
            qz_params_scale.view(N // fac_nums[5], K, nparams),
            vae.q_dist,flags=flags)

        cond_entropies[5] += cond_entropies_i.cpu() / fac_nums[5]

    metric = compute_metric_3dshapes(marginal_entropies, cond_entropies,active_units,fac_nums,flags)
    print("metric:",metric)
    return metric, marginal_entropies, cond_entropies, active_units