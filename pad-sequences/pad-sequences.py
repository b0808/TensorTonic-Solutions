import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    n = len(seqs)
    if max_len is None:
        max_len = 0
        for i in range(len(seqs)):
            max_len = max(max_len,len(seqs[i]))
    results = np.full((n,max_len),pad_value,dtype=int)

    for i in range(len(results)):
        len_seqs = len(seqs[i])
        results[i][:len_seqs] = seqs[i][:max_len]
    
    return results