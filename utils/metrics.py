from typing import Sequence

def nash_sutcliffe_efficiency(observed: Sequence[float], simulated: Sequence[float]) -> float:
    """
    Calculate the Nash Sutcliffe Efficiency (NSE).

    Args:
        observed (Sequence[float]): An iterable of observed values.
        simulated (Sequence[float]): An iterable of simulated values.

    Returns:
        float: The NSE value computed as:
            1 - (sum((observed - simulated)^2) / sum((observed - mean(observed))^2)).

    Raises:
        ValueError: If lengths of observed and simulated differ or if the denominator is zero.
    """
    if len(observed) != len(simulated):
        raise ValueError("The length of observed and simulated data must be the same.")
    
    mean_obs = sum(observed) / len(observed)
    numerator = sum((o - s) ** 2 for o, s in zip(observed, simulated))
    denominator = sum((o - mean_obs) ** 2 for o in observed)
    
    if denominator == 0:
        raise ValueError("Denominator in NSE calculation is zero; check observed values.")
    
    return 1 - (numerator / denominator)
