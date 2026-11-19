"""Metadata-only target-cell allocation partitions; models stay unchanged."""
import pandas as pd

def partition(labels):
    groups={}
    for i,k in enumerate(labels):groups.setdefault(k,[]).append(i)
    return {tuple(g) for g in groups.values()}

def metadata_labels(d,panels):
    coarse=list(zip(d.year,d.treatment))
    target=[tuple(int(p.g[i]) if p.eligible[i] else -1 for p in panels) for i in range(len(d))]
    equivalent=partition(coarse)==partition(target)
    changed=d.copy()
    if not equivalent:
        # Allocation metadata only. p.F, p.g, p.K and all scientific targets were
        # already constructed from the original data and are never rebuilt.
        changed['year']=pd.factorize(pd.Series(target,dtype=object),sort=False)[0]
        changed['treatment']=0
    return changed,equivalent,target
