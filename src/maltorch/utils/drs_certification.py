import math


def calculate_delta(chunk_size: int, manipulations: dict) -> int:
    """
    Calculate the total delta for the certification procedure.

    Parameters
    ----------
    chunk_size : int
        The size of the chunk.
    manipulations : dict
        The manipulations dictionary. It contains the number of bytes patched, appended, and injected.

    Returns
    -------
    int
        The total delta.
    """
    if manipulations["num_bytes_patched"] is not None:
        patch_delta = sum(
            [math.ceil(adv_content / chunk_size) + 1 for adv_content in manipulations["num_bytes_patched"]])
    else:
        patch_delta = 0

    if manipulations["num_bytes_appended"] is not None:
        append_delta = sum(
            [math.ceil(adv_content / chunk_size) + 1 for adv_content in manipulations["num_bytes_appended"]])
    else:
        append_delta = 0

    if manipulations["num_bytes_injected"] is not None:
        inject_delta = sum(
            [math.ceil(adv_content / chunk_size) + 1 for adv_content in manipulations["num_bytes_injected"]])
    else:
        inject_delta = 0

    total_delta = (2 * patch_delta) + append_delta + inject_delta
    return total_delta

def certify_EXEmple(labels: list, chunk_size: int, manipulations: dict) -> bool:
    """
    Certify whether the final classification is robust to patch and injection manipulations based on the following work:
    Daniel Gibert, Luca Demetrio, Giulio Zizzo, Quan Le, Jordi Planes, Battista Biggio
    Certified Adversarial Robustness of Machine Learning-based Malware Detectors via (De) Randomized Smoothing
    ArXiv'24

    Parameters
    ----------
    labels : list
        The list of labels.
    chunk_size : int
        The size of the chunk.
    manipulations : dict
        The manipulations dictionary. It contains the number of bytes patched, appended, and injected.

    Returns
    -------
    bool
        Certification guarantee
    """
    total_delta = calculate_delta(chunk_size, manipulations)
    num_top_class = sum([1 for label in labels if label >= 0.5])
    num_second_class = sum([1 for label in labels if label < 0.5])

    if num_top_class > num_second_class + total_delta:
        is_certified = True
    else:
        is_certified = False
    return is_certified

