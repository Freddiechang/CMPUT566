import torch

def nss(prediction, ground_truth):
        """
        Compute NSS score.
            :param ground_truth : ground-truth fixation points(binary map)  3D tensor: (batch, h, w)
            :param prediction : predicted saliency map, 3D tensor: (batch, h, w)
            :return score: NSS score: tensor of (batch)
        """
        sal_map = prediction - prediction.mean(dim=(1, 2)).view(prediction.shape[0], 1, 1)
        sal_map = sal_map / prediction.std(dim=(1, 2), unbiased=True).view(prediction.shape[0], 1, 1)
        sal_map = sal_map * (ground_truth > 0)

        return sal_map.sum(dim=(1, 2)) / ground_truth.count_nonzero(dim=(1, 2))

def vcorrcoef(x, y):
    """
    Compute correlation coefficients.
        :param x,y : 2D tensor: (batch, d)
        :return score: tensor: (batch)
    """
    x_sub_mean = x - x.mean(dim=1).view(x.shape[0], 1)
    y_sub_mean = y - y.mean(dim=1).view(y.shape[0], 1)
    r_num = (x_sub_mean * y_sub_mean).sum(dim=1)
    r_den = ((x_sub_mean**2).sum(dim=1) * (y_sub_mean**2).sum(dim=1)).sqrt()
    return r_num / r_den

def cc(prediction, ground_truth):
    """
    Compute CC score.
        :param prediction : predicted saliency map, 3D tensor: (batch, h, w)
        :param ground_truth : ground-truth saliency map, 3D tensor: (batch, h, w)
        :return score: CC score, tensor: (batch)
    """
    sal_map = prediction - prediction.mean(dim=(1, 2)).view(prediction.shape[0], 1, 1)
    sal_map = sal_map / prediction.std(dim=(1, 2), unbiased=True).view(prediction.shape[0], 1, 1)

    fix_map = ground_truth - ground_truth.mean(dim=(1, 2)).view(ground_truth.shape[0], 1, 1)
    fix_map = fix_map / ground_truth.std(dim=(1, 2), unbiased=True).view(ground_truth.shape[0], 1, 1)

    return vcorrcoef(sal_map.view(sal_map.shape[0], -1), fix_map.view(fix_map.shape[0], -1))