import numpy as np 

def uniform_FDP_upper_bound(pvals, n = 100, m = 100, N=100, delta=0.05, beta=0.5, lower=0.01, upper=0.99,
    setting='cpvals', seed= 1, method = 'MC-HC', precision = 1e-8, randomize = False, boost = True):
    """
    Computes and uniform FDP upper bounds under various settings. 

    Parameters:
    - pvals: p values
    - n (int): Number of calibration data points.
    - m (int): Number of test data points.
    - N (int): Number of Monte Carlo replications.
    - delta (float): Confidence level.
    - beta (float): Scaling exponent for the truncated Higher Criticism statistic.
    - lower (float): Lower truncation threshold.
    - upper (float): Upper truncation threshold.
    - setting (str): Setting for p-value sampling: 'iid' (iid U(0,1)) or 'cpvals' (oracle conformal p-values). 
    - seed (int): Seed for reproducibility.hod: 'MC-HC', 'Marginal MC', 'MC-THC', 'MC-KS', 'MC-BJ'
    - precision: precision when using MC-BJ method
    - randomize: whether to use the seed

    Returns:
    - A function from [0,1] to [0,1]
    """
    pvals = np.sort(pvals)
    ecdf_upper = uniform_ecdf_upper_bound(n = n, m = m, N = N, delta = delta, beta = beta, lower = lower,
                                          upper = upper, setting= setting, seed = seed, method = method,
                                        precision = precision, randomize = randomize)
    def f(x):
        # Convert inputs to a NumPy array for vectorized operations
        x = np.asarray(x)
        # Compute ECDF upper bounds for all x
        if boost:
            # For each x, keep track of "the largest pval <= x"
            max_p_under_x = np.zeros(x.size)
            # Start numerator with an array of 'm'
            numerator = np.full(x.size, fill_value=m, dtype=float)
            
            for i in range(len(pvals)):
                # Boolean array: where is pvals[i] <= x?
                mask = (pvals[i] <= x)  # shape: (n_x,)
                # Update max_p_under_x in the masked positions
                max_p_under_x[mask] = np.maximum(
                    max_p_under_x[mask], 
                    pvals[i]
                )
                # We must do np.minimum for array-wise "min"
                # but first compute the second argument:
                second_term = m * ecdf_upper(pvals[i]) - np.sum(pvals <= pvals[i])
                # 'second_term' is a scalar, so we do
                numerator[mask] = np.minimum(numerator[mask], second_term)
            
            add_terms = np.zeros(x.size)
            for i_x in range(x.size):
                add_terms[i_x] = np.sum(pvals <= max_p_under_x[i_x])
            
            numerator += add_terms
            
        else:
            numerator = m * ecdf_upper(x)
        # Compute the denominator (number of p-values ≤ x)
        denominator = np.sum(np.array(pvals)[:, None] <= x, axis=0)
        # Avoid division by zero
        denominator = np.where(denominator == 0, np.inf, denominator)
        return np.minimum(numerator / denominator,1)
    return f


def uniform_ecdf_upper_bound(
    n=100, m=50, N=100, delta=0.05, beta=0.5, lower=0.01, upper=0.99,
    setting='cpvals', seed= 1, method = 'MC-HC', precision = 1e-8, randomize = False
):
    """
    Computes and uniform ECDF upper bounds under various settings. 

    Parameters:
    - n (int): Number of calibration data points.
    - m (int): Number of test data points.
    - N (int): Number of Monte Carlo replications.
    - delta (float): Confidence level.
    - beta (float): Scaling exponent for the truncated Higher Criticism statistic.
    - lower (float): Lower truncation threshold.
    - upper (float): Upper truncation threshold.
    - setting (str): Setting for p-value sampling: 'iid' (iid U(0,1)) or 'cpvals' (oracle conformal p-values). 
    - seed (int): Seed for reproducibility.
    - method: 'MC-HC', 'Marginal MC', 'MC-THC', 'MC-KS', 'MC-BJ'
    - precision: precision when using MC-BJ method
    - randomize: whether to use the seed

    Returns:
    - A function from [0,1] to [0,1]
    """
    
    summary_stats = np.zeros(N)
    if not randomize:
        np.random.seed(seed)

    if method == 'MC-HC':
        for t in range(N):
            pvals_tem = sample_cpvals(n, m, settings=setting)
            summary_stats[t] = compute_M(pvals_tem)
        summary_quantile = custom_quantile(summary_stats, 1-delta)
        def f(x):
            return np.minimum(x + np.sqrt(x * (1 - x)) * summary_quantile,1)
        return f
        
    elif method == 'MC-THC':
        for t in range(N):
            pvals_tem = sample_cpvals(n, m, settings=setting)
            summary_stats[t] = compute_M(pvals_tem, lower = lower, upper = upper, beta = beta)
        summary_quantile = custom_quantile(summary_stats, 1-delta)
        f_lower_value = np.minimum(1, lower + np.power(lower * (1 - lower), beta) * summary_quantile
    )
        def f_vec(x):
            # Convert x to array (works even if x is scalar)
            x = np.asarray(x, dtype=float)
            
            # Prepare output array
            out = np.empty_like(x, dtype=float)
            
            # Boolean masks
            mask_lower = (x < lower)
            mask_upper = (x > upper)
            mask_else  = (~mask_lower) & (~mask_upper)  # everything in [lower, upper]

            # Region 1: x < lower => f(lower)
            out[mask_lower] = f_lower_value
            
            # Region 2: x > upper => 1
            out[mask_upper] = 1.0

            # Region 3: lower <= x <= upper => the formula
            out[mask_else] = np.minimum(
                1,
                x[mask_else] + np.power(
                    x[mask_else] * (1 - x[mask_else]), 
                    beta
                ) * summary_quantile
            )
            
            return out

        return f_vec

    elif method == 'MC-KS':
        for t in range(N):
            pvals_tem = sample_cpvals(n, m, settings=setting)
            summary_stats[t] = compute_KS(pvals_tem)
        summary_quantile = custom_quantile(summary_stats, 1-delta)
        return lambda x: np.minimum(x + summary_quantile,1)
        

    elif method == 'KS':
        summary_quantile = DKW_lambda(1000,1,m,n,delta)
        return lambda x: np.minimum(x + summary_quantile,1)
    

    elif method == 'MC-BJ':
        for t in range(N):
            pvals_tem = sample_cpvals(n, m, settings=setting)
            summary_stats[t] = BJ(pvals_tem) 
        summary_quantile = custom_quantile(summary_stats, 1-delta)
        lower_bound_on_pvals = solve_KL(np.arange(1,m//2+1)/m, summary_quantile, precision = precision)
        lb_array = np.asarray(lower_bound_on_pvals)
        def f_vec(x):
            
            # Convert x to a NumPy array for consistent processing
            x = np.asarray(x)
            
            # Use searchsorted to find the insertion indices
            idx = np.searchsorted(lb_array, x, side="left")
            
            # Compute idx/m
            ratio = idx / m
            
            # Use np.where to handle the condition where idx == len(lb_array)
            out = np.where(idx == len(lb_array), 1.0, ratio)
            return out

        return f_vec
        # def f(x):
        #     index = 0
        #     while index < len(lower_bound_on_pvals) and x >= lower_bound_on_pvals[index]:
        #         index += 1
        #     if index >= len(lower_bound_on_pvals):
        #         return 1
        #     else:
        #         return index/m
        # return f

    elif method == 'Marginal-MC':
        all_samples = np.zeros((N,m))
        for t in range(N):
            pvals_tem = sample_cpvals(n,m, settings = setting)
            all_samples[t] = pvals_tem
        def f_vec(x):
            # Convert x to array (handles scalar or array input)
            x = np.asarray(x, dtype=float)  # shape (X,) or ()

            # Reshape all_samples to (S, D, 1)
            samples_3d = all_samples[..., None]  # shape (S, D, 1)

            # Reshape x so it broadcasts along the (S, D) dimensions
            # Now x_3d is shape (1, 1, X) if x has shape (X,)
            x_3d = x[np.newaxis, np.newaxis, :]

            # Compare => shape (S, D, X)
            bool_mask = (samples_3d <= x_3d)

            # Average over axis=1 => shape (S, X)
            # i.e. for each row S, and each x in X, we get the fraction <= x
            row_means = np.mean(bool_mask, axis=1)  # shape (S, X)

            # We now want the (1 - delta) quantile across the S dimension => shape (X,)
            # axis=0 means "take the quantile along the S dimension"
            result = np.quantile(row_means, 1 - delta, axis=0)

            return result
        def f(x):
            return np.quantile(np.average(all_samples <= x, axis = 1), 1-delta)
        return f_vec

        
    

##############################################################################################################################
##############################################################################################################################


def sample_cpvals(n,m, settings = 'cpvals'):
    """
    Generates random p-values based on specified settings.

    Parameters:
    - n (int): Number of calibration points.
    - m (int): Number of test p-values to generate.
    - settings (str): Sampling strategy. Options are:
        - 'cpvals': Oracle conformal p values
        - 'iid': Independent and identically distributed uniform p-values.
    
    Returns:
    - numpy.ndarray: Array of m generated p-values.
    """
    if settings == 'cpvals':
        iid_uniform = np.random.rand(n)
        sorted_uniform = np.sort(iid_uniform)
        p = np.diff(np.concatenate(([0], sorted_uniform, [1])))
        generated_variables = (np.random.choice(np.arange(len(p)), size=m, p=p))/len(p) + np.random.rand(m)/len(p)
    if settings == 'iid':
        generated_variables = np.random.rand(m)

    return generated_variables


def compute_M(pvals, lower = 0, upper = 1, beta = 1/2):
    """
    Computes a summary statistic for the given p-values using a truncated Higher Criticism (HC) statistic.

    Parameters:
    - pvals (numpy.ndarray): Array of p-values.
    - lower (float): Lower threshold for truncation.
    - upper (float): Upper threshold for truncation.
    - beta (float): Power scaling parameter for normalization.
    
    Returns:
    - float: Maximum normalized statistic within the specified p-values.
    """
    m = len(pvals)
    sorted_pvals = np.sort(pvals)
    grid = np.arange(1, m+1)/m
    r1 = (grid - sorted_pvals) / np.power(sorted_pvals *(1-sorted_pvals), beta)
    l_index = sum(pvals <= lower)
    r_index = sum(pvals <= upper)
    ll = (l_index/m - lower)/np.power(lower*(1-lower), beta) if lower > 0 else 0
    return np.maximum(np.max(r1[l_index:r_index]), ll )


def compute_KS(pvals):
    """
    Computes the Dvoretzky-Kiefer-Wolfowitz (DKW) type constant.

    Parameters:
    - pvals (numpy.ndarray): Array of p-values.
    - lower (float): Lower truncation threshold.
    - upper (float): Upper truncation threshold.

    Returns:
    - float: Maximum deviation within the specified p-values.
    """
    m = len(pvals)
    sorted_pvals = np.sort(pvals)
    grid = np.arange(1, m+1)/m
    r1 = (grid - sorted_pvals) 
    return np.max(r1)

def KL_Bernoulli(p0,p1): #Computes the Kullback-Leibler (KL) divergence between two Bernoulli distributions.
    return p0 * np.log(p0/p1) + (1-p0) * np.log((1-p0)/(1-p1))

def BJ(pvals):
    """
    Computes the BJ (Berk-Jones) statistic using the KL divergence between empirical and expected distributions.

    Parameters:
    - pvals (numpy.ndarray): Array of p-values.

    Returns:
    - float: BJ statistic value.
    """
    m = len(pvals)
    sorted_pvals = np.sort(pvals)
    return m * np.max(KL_Bernoulli(sorted_pvals[0:m//2], np.arange(1,m//2+1)/m))


def solve_KL(pvals1, BJ, precision = 1e-8): 
    """
    Finds the solution of the KL-Bernoulli equation: KL(p, pval1) = BJ using a binary search approach.

    Parameters:
    - p1 (numpy.ndarray): Array of probabilities.
    - BJ (float): BJ statistic value.
    - precision (float): Desired precision for the solution.

    Returns:
    - numpy.ndarray: Solution for each p-value in the array.
    """
    m = len(pvals1)
    solution = np.zeros(m)
    for tt in range(m):
        upper_bound = pvals1[tt]
        lower_bound = 0
        while upper_bound - lower_bound > precision: # The solution is within the desired precision
            mid_point = (lower_bound + upper_bound) / 2
            if KL_Bernoulli(mid_point, pvals1[tt]) < BJ/m:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
            solution[tt] = (lower_bound + upper_bound) / 2
    return solution

# p(j) > t => F(t) <= (j-1)/m
def ecdf_upper_BJ(grid, p_lower,m):
    """
    Computes the upper bound of the empirical cumulative distribution function (ECDF).

    Parameters:
    - grid (numpy.ndarray): Grid of values to evaluate.
    - p_lower (numpy.ndarray): Array of lower bounds (typically p-values).
    - m (int): Number of p-values.

    Returns:
    - numpy.ndarray: Array representing the upper bound of the ECDF.
    """
    grid_len = len(grid)
    result = np.zeros(grid_len)
    index = 0
    for tt in range(grid_len):
        while index < len(p_lower) and grid[tt] >= p_lower[index]:
            index += 1
        if index >= len(p_lower):
            result[tt] = 1
        else:
            result[tt] = index/m
    return result

import numpy as np

def custom_quantile(x, q):
    """
    Computes the quantile of the given array x using the custom quantile definition:
    (1/(n+1)) * sum(delta(x_i)) + (1/(n+1)) * delta(infinity)
    
    Parameters:
    - x (numpy.ndarray): Array of values.
    - q (float): Quantile to compute (between 0 and 1).
    
    Returns:
    - float: The computed quantile value.
    """
    n = len(x)
    sorted_x = np.sort(x)
    index = q * (n + 1)

    if index <= n:
        return sorted_x[int(np.ceil(index)) - 1]
    else:
        return np.inf
    
## DKW in transductive paper
def DKW_tau(n,m):
    return m*n/(m+n)
def DKW_Psi(x, m, n, delta):
    return np.minimum(1, np.sqrt((np.log(1/delta) + np.log(1+np.sqrt(2*np.pi)*2*x*DKW_tau(n,m)/np.sqrt(n+m)))\
                                 /(2*DKW_tau(n,m))))
def DKW_lambda(r,x,m,n,delta):
    re = x
    for kk in range(r):
        re = DKW_Psi(re,m,n,delta)
    return re