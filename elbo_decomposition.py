import os
import math
from numbers import Number
from tqdm import tqdm
import torch
from torch.autograd import Variable
import numpy as np

import lib.dist as dist
import lib.flows as flows

def estimate_entropies(qz_samples, qz_params, q_dist):
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
        qz_samples (K, S) Variable
        qz_params  (N, K, nparams) Variable
    """

    # Only take a sample subset of the samples
    # take one sample at a time, do it for S times, that's why M=1 for out loop and
    qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:5000].cuda()))

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    marginal_entropies = torch.zeros(K).cuda()
    joint_entropy = torch.zeros(1).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(50, S - k)
        aa=qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size] #[N,z_dim,S] -> [N,2,batch_size]
        bb=qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size] #[N,z_dim,S,2] -> [N,z_dim,batch_size,2]
        logqz_i = q_dist.log_density(aa,bb) #{N,z_dim,batch_size]
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)
        # computes - log q(z) summed over minibatch
        logqz = logqz_i.sum(1)  # (N, S)
        cc=logsumexp(logqz, dim=0, keepdim=False).data #batch_size
        joint_entropy += (math.log(N) - cc).sum(0)
        pbar.update(batch_size)
    pbar.close()

    marginal_entropies /= S
    joint_entropy /= S

    return marginal_entropies, joint_entropy


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def analytical_NLL(qz_params, q_dist, prior_dist, qz_samples=None):
    """Computes the quantities
        1/N sum_n=1^N E_{q(z|x)} [ - log q(z|x) ]
    and
        1/N sum_n=1^N E_{q(z_j|x)} [ - log p(z_j) ]

    Inputs:
    -------
        qz_params  (N, K, nparams) Variable

    Returns:
    --------
        nlogqz_condx (K,) Variable
        nlogpz (K,) Variable
    """
    pz_params = Variable(torch.zeros(1).type_as(qz_params.data).expand(qz_params.size()), volatile=True)

    nlogqz_condx = q_dist.NLL(qz_params).mean(0)
    nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
    return nlogqz_condx, nlogpz

def elbo_decomposition(sess,vae, manager,check_z_dim=[0],img_size=64):
    S = 1                            # number of latent variable samples
    n_samples = manager.sample_size
    N = manager.sample_size
    nparams = vae.q_dist.nparams
    K = len(check_z_dim)
    #K=3
    n = 0
    indices = list(range(n_samples))
    batch_size = 100
    total_batch = n_samples // batch_size
    logpx = 0
    #N = 48000
    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    for i in range(total_batch):
        #xs = Variable(xs.view(batch_size, -1, img_size, img_size).cuda(), volatile=True)
        #z_params = vae.encoder.forward(xs).view(batch_size, K, nparams)
        #qz_params[n:n + batch_size] = z_params.data

        batch_indices = indices[batch_size * i: batch_size * (i + 1)]
        xs = manager.get_images(batch_indices)
        z_mean, z_logvar= vae.transform(sess, xs)
        z_mean=z_mean[:,check_z_dim]
        z_logvar=z_logvar[:,check_z_dim]

        z_logsigma = z_logvar * 0.5

        qz_params[n:n + batch_size, :, 0] = torch.from_numpy(z_mean)
        qz_params[n:n + batch_size, :, 1] = torch.from_numpy(z_logsigma)
        #zsample_params[n:n + batch_size, :, 0] = torch.from_numpy(z_mean)
        #zsample_params[n:n + batch_size, :, 1] = torch.from_numpy(z_logsigma)
        n += batch_size

        # estimate reconstruction term
        z_params = torch.Tensor(batch_size, K, nparams)
        z_params[:,:, 0] = torch.from_numpy(z_mean)
        z_params[:,:, 1] = torch.from_numpy(z_logsigma)
        for _ in range(S):
            z = vae.q_dist.sample(params=z_params) # z_params should be [batch,10,2] float32 torch tensor
            #x_out=vae.generate_from_singlez(sess, z)
            #x_params=torch.from_numpy(np.reshape(x_out,[batch_size,1,64,64])) # x params should be [batch,1,64,64] float32 torch tensor
            #xs=torch.from_numpy(np.reshape(xs,[batch_size,1,64,64]))
            #logpx += vae.x_dist.log_density(xs, params=x_params).view(batch_size, -1).data.sum()
    # Reconstruction term
    #logpx = logpx / (N * S)

    qz_params = Variable(qz_params.cuda(), volatile=True)
    #zsample_params = Variable(zsample_params.cuda(), volatile=True)

    print('Sampling from q(z).')
    # sample S times from each marginal q(z_j|x_n)
    qz_params_expanded = qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)
    #qz_params_expanded = zsample_params.view(N, K, 1, nparams).expand(N, K, S, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params_expanded) #[48w,2,1]
    qz_samples = qz_samples.transpose(0, 1).contiguous().view(K, N * S)

    print('Estimating entropies.')
    marginal_entropies, joint_entropy = estimate_entropies(qz_samples, qz_params, vae.q_dist)

    if hasattr(vae.q_dist, 'NLL'):
        print("1")
        nlogqz_condx = vae.q_dist.NLL(qz_params).mean(0)
    else:
        print("2")
        nlogqz_condx = - vae.q_dist.log_density(qz_samples,
            qz_params_expanded.transpose(0, 1).contiguous().view(K, N * S)).mean(1)

    if hasattr(vae.prior_dist, 'NLL'):
        print("3")
        pz_params = vae._get_prior_params(N * K).contiguous().view(N, K, -1)
        nlogpz = vae.prior_dist.NLL(pz_params, qz_params).mean(0)
    else:
        print("4")
        nlogpz = - vae.prior_dist.log_density(qz_samples.transpose(0, 1)).mean(0)

    # nlogqz_condx, nlogpz = analytical_NLL(qz_params, vae.q_dist, vae.prior_dist)
    nlogqz_condx = nlogqz_condx.data
    nlogpz = nlogpz.data

    # Independence term
    # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
    dependence = (- joint_entropy + marginal_entropies.sum())[0]

    # Information term
    # KL(q(z|x)||q(z)) = log q(z|x) - log q(z) # I(x,z)=E_q(x)[KL(q(z|x)||q(z))]=E_q(x,z)[log q(z|x) - log q(z)]
    information = (- nlogqz_condx.sum() + joint_entropy)[0]

    # Dimension-wise KL term
    # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
    print(nlogpz,marginal_entropies)
    dimwise_kl = (- marginal_entropies + nlogpz).sum()

    # Compute sum of terms analytically
    # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
    analytical_cond_kl = (- nlogqz_condx + nlogpz).sum()

    print('Dependence: {}'.format(dependence))
    print('Information: {}'.format(information))
    print('Dimension-wise KL: {}'.format(dimwise_kl))
    print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
    print('Estimated  ELBO: {}'.format(logpx - analytical_cond_kl.cpu()))

    return logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy