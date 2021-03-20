import torch
from numpy import arange
import numpy


def nss(prediction, ground_truth):
    """
    Compute NSS score.
        :param ground_truth : ground-truth fixation points(binary map)  4D tensor: (batch, 1, h, w)
        :param prediction : predicted saliency map, 4D tensor: (batch, 1, h, w)
        :return score: NSS score: tensor of (batch)
    """
    sal_map = prediction - prediction.mean(dim=(2, 3)).view(prediction.shape[0], 1, 1)
    sal_map = sal_map / prediction.std(dim=(2, 3), unbiased=True).view(prediction.shape[0], 1, 1)
    sal_map = sal_map * (ground_truth > 0)
    loss = sal_map.sum(dim=(2, 3)) / ground_truth.count_nonzero(dim=(2, 3))
    return loss.sum()


def vcorrcoef(x, y):
    """
    Compute correlation coefficients.
        :param x,y : 2D tensor: (batch, d)
        :return score: tensor: (batch)
    """
    x_sub_mean = x - x.mean(dim=1).view(x.shape[0], 1)
    y_sub_mean = y - y.mean(dim=1).view(y.shape[0], 1)
    r_num = (x_sub_mean * y_sub_mean).sum(dim=1)
    r_den = ((x_sub_mean ** 2).sum(dim=1) * (y_sub_mean ** 2).sum(dim=1)).sqrt()
    return r_num / r_den


def cc(prediction, ground_truth):
    """
    Compute CC score.
        :param prediction : predicted saliency map, 4D tensor: (batch, 1, h, w)
        :param ground_truth : ground-truth saliency map, 4D tensor: (batch, 1, h, w)
        :return score: CC score, tensor: (batch)
    """
    sal_map = prediction - prediction.mean(dim=(2, 3)).view(prediction.shape[0], 1, 1)
    sal_map = sal_map / prediction.std(dim=(2, 3), unbiased=True).view(prediction.shape[0], 1, 1)

    fix_map = ground_truth - ground_truth.mean(dim=(2, 3)).view(ground_truth.shape[0], 1, 1)
    fix_map = fix_map / ground_truth.std(dim=(2, 3), unbiased=True).view(ground_truth.shape[0], 1, 1)

    loss = vcorrcoef(sal_map.view(sal_map.shape[0], -1), fix_map.view(fix_map.shape[0], -1))
    return loss.sum()


def borji(saliency_map, g_truth, num_split=100, step_size=0.1):
    """
    Measures how well the saliencyMap of an image predicts the ground truth human fixations on the image.
        :param saliency_map: the saliency map, 4D tensor: (batch, 1, h, w)
        :param g_truth: the human fixation map (binary matrix), 4D tensor: (batch, 1, h, w)
        :param num_split: number of random splits
        :param step_size: sweeping through saliency map
        :return: score
    """

    s_map = saliency_map.clone().detach()

    # make the saliencyMap the size of the image of fixationMap
    if s_map.shape[2:] != g_truth.shape[2:]:
        import torch.nn.functional as nnf
        s_map[2:] = nnf.interpolate(s_map[2:], size=g_truth.shape[2:], mode='bicubic', align_corners=False)

    # normalize saliency map
    s_map -= s_map.min(dim=(2, 3)).view(s_map.shape[0], 1, 1)
    s_map /= s_map.max(dim=(2, 3)).view(s_map.shape[0], 1, 1)

    # vector of s_map
    s_vec = torch.flatten(s_map)
    g_vec = torch.flatten(g_truth)

    # s_map values at fixation locations
    s_th = s_vec[g_vec > 0]
    num_fixations = s_th.shape[0]
    num_pixels = s_vec.shape[0]

    # for each fixation, sample num_split values from anywhere on the sal map
    random = torch.randint(low=1, high=num_pixels, size=(num_fixations, num_split))
    rand_fix = s_vec[random]

    # calculate AUC per random split (set of random locations)
    auc = []
    for i in range(0, num_split):
        cur_fix = rand_fix[:, i]
        h_bound = max(torch.max(s_th), torch.max(cur_fix))
        all_threshes = torch.flip(torch.FloatTensor([i for i in arange(0, h_bound.item(), step_size)]), dims=[0])
        tp = torch.zeros(all_threshes.shape[0] + 2, 1)
        fp = torch.zeros(all_threshes.shape[0] + 2, 1)
        tp[0] = fp[0] = 0
        tp[-1] = fp[-1] = 1

        for j in range(0, all_threshes.shape[0]):
            thresh = all_threshes[j]
            tp[j + 1] = (s_th >= thresh).sum() / num_fixations
            fp[j + 1] = (cur_fix >= thresh).sum() / num_fixations
        # im not sure about this line:
        auc.append(torch.trapz(fp, tp))

    result = torch.stack(auc, dim=0)
    score = torch.mean(result)

    return score


def judd(saliency_map, g_truth, jitter=False):
    """
    This function measures how well the saliencyMap of an image predicts the ground
    truth human fixations on the image.
        :param saliency_map: the saliency map, 4D tensor: (batch, 1, h, w)
        :param g_truth: the human fixation map (binary matrix), 4D tensor: (batch, 1, h, w)
        :param jitter = True will add tiny non-zero random constant to all map locations
        :return: score
    """

    s_map = saliency_map.clone().detach()

    # make the saliencyMap the size of the image of fixationMap
    if s_map.shape[2:] != g_truth.shape[2:]:
        import torch.nn.functional as nnf
        s_map[2:] = nnf.interpolate(s_map[2:], size=g_truth.shape[2:], mode='bicubic', align_corners=False)

    if jitter:
        s_map = s_map + torch.rand(s_map.shape) / 10000000

    # vector of s_map
    s_vec = torch.flatten(s_map)
    g_vec = torch.flatten(g_truth)

    # s_map values at fixation locations
    s_th = s_vec[g_vec > 0]

    num_fixations = s_th.shape[0]
    num_pixels = s_vec.shape[0]

    all_threshes, _ = torch.sort(s_th, descending=True)

    tp = torch.zeros(all_threshes.shape[0] + 2, 1)
    fp = torch.zeros(all_threshes.shape[0] + 2, 1)

    tp[0] = fp[0] = 0
    tp[-1] = fp[-1] = 1

    for i in range(0, num_fixations):
        thresh = all_threshes[i]
        # total number of sal map values above threshold
        above_thresh = (s_th >= thresh).sum()
        # ratio sal map values at fixation locations above threshold
        tp[i + 1] = i / num_fixations
        # ratio other sal map values above threshold
        fp[i + 1] = (above_thresh - i) / (num_pixels - num_fixations)

    return torch.trapz(fp, tp)


def shuffled(saliency_map, g_truth, other_map, num_split=100, step_size=0.1):
    """
    This measures how well the saliencyMap of an image predicts the ground
    truth human fixations on the image.
        :param saliency_map: the saliency map, 4D tensor: (batch, 1, h, w)
        :param g_truth: the human fixation map (binary matrix), 4D tensor: (batch, 1, h, w)
        :param other_map: is a binary fixation map (like g_truth) by taking the union of
        fixations from M other random images (Borji uses M=10)
        :param num_split: number of random splits
        :param step_size: sweeping through saliency map
        :return: score
    """

    s_map = saliency_map.clone().detach()

    # make the saliencyMap the size of the image of fixationMap
    if s_map.shape[2:] != g_truth.shape[2:]:
        import torch.nn.functional as nnf
        s_map[2:] = nnf.interpolate(s_map[2:], size=g_truth.shape[2:], mode='bicubic', align_corners=False)

    # normalize saliency map
    _, s_mins = s_map.view(s_map.shape[0], s_map.shape[1], -1).min(dim=2)
    _, s_maxs = s_map.view(s_map.shape[0], s_map.shape[1], -1).max(dim=2)

    for i in range(0, s_map.shape[0]):
        s_map[i, :, :, :] -= s_mins[i]
        s_map[i, :, :, :] /= s_maxs[i]
    # vector of s_map
    s_vec = torch.flatten(s_map)
    g_vec = torch.flatten(g_truth)

    # s_map values at fixation locations
    s_th = s_vec[g_vec > 0]

    num_fixations = s_th.shape[0]

    # find fixation locations on other images
    ind = numpy.argwhere(other_map.numpy())

    num_fixations_oth = min(num_fixations, len(ind))
    rand_fix = torch.zeros(num_fixations_oth, num_split)

    for i in range(0, num_split):
        # randomize choice of fixation locations
        rand_ind = ind[torch.randint(low=0, high=len(ind), size=(1, len(ind)))]
        # sal map values at random fixation locations of other random image
        rand_fix[:, i] = s_vec[rand_ind[0, num_fixations_oth-1]]

    # calculate AUC per random split (set of random locations)
    auc = []

    for i in range(0, num_split):
        cur_fix = rand_fix[:, i]
        h_bound = max(torch.max(s_th), torch.max(cur_fix))
        all_threshes = torch.flip(torch.FloatTensor([i for i in arange(0, h_bound.item(), step_size)]), dims=[0])
        tp = torch.zeros(all_threshes.shape[0] + 2, 1)
        fp = torch.zeros(all_threshes.shape[0] + 2, 1)

        tp[0] = fp[0] = 0
        tp[-1] = fp[-1] = 1

        for j in range(0, all_threshes.shape[0]):
            thresh = all_threshes[j]
            tp[j + 1] = (s_th >= thresh).sum() / num_fixations
            fp[j + 1] = (cur_fix >= thresh).sum() / num_fixations_oth
        # still, im not sure about this line:
        auc.append(torch.trapz(fp, tp))

    result = torch.stack(auc, dim=0)
    score = torch.mean(result)
    return score

# TODO: i need to change all s_vec = torch.flatten(s_map) to s_vec with size of batch for each Image and each map @makbn
# TODO: i need to change  @makbn
# TODO: this code has some problem please dont merge @shupei

