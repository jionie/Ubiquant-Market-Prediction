from scipy.stats import pearsonr, spearmanr


def pearson_correlation(pred, label):
    correlation, p_value = pearsonr(pred.flatten(), label.flatten())
    return correlation


def spearman_correlation(pred, label):
    correlation, p_value = spearmanr(pred.flatten(), label.flatten())
    return correlation
