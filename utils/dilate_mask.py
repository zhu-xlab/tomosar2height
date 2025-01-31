from scipy import ndimage


def dilate_mask(mask_in, iterations=1):
    """
    Dilates a binary mask.

    Args:
        mask_in (np.array): Binary mask to be dilated.
        iterations (int): Number of dilation iterations.

    Returns:
        np.array: Dilated binary mask.
    """
    return ndimage.morphology.binary_dilation(mask_in, iterations=iterations)
