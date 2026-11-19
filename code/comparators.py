"""Paid-pilot influence-function and LOO-residual allocation comparators.

Allocation retains paid pilot records. Final reconstruction is supplied by the
caller; reference values are received only through the charged callback.
"""
import numpy as np
import archive_model as model

DESIGNS=('a2_if_year','a4_loo_year','a4_loo_cell')


def matched_adaptive(d, panels, reveal, pilot, pilot_values, budget, design, rng):
    """Complete one exact-budget design after an already paid common pilot.

    The caller creates a fresh oracle for each design/budget, pays reveal(pilot)
    once, and passes its returned per-panel arrays as pilot_values. This function
    never reveals the pilot again. Return selected, full observed arrays, logs.
    The first log entry describes the caller's paid pilot for a complete ledger.

    a2_if_year: pilot + one final update.
    a4_loo_year/cell: pilot + up to three distinct cumulative completion waves,
        ceil(B0 + (B-B0)*w/3), w=1,2,3.
    All original targets are retained, including infeasible fine-cell designs;
    the metadata-only balanced fallback is logged when needed.
    """
    if design not in DESIGNS:
        raise ValueError(f'Unsupported matched comparator: {design}')
    if any(c=='y' or c.startswith('y_') for c in d.columns):
        raise ValueError('Unpaid reference columns may not enter allocation')
    N=len(d); pilot=np.asarray(pilot,dtype=int)
    if pilot.ndim!=1 or len(set(pilot.tolist()))!=len(pilot):
        raise ValueError('Pilot must contain unique indices')
    if np.any(pilot<0) or np.any(pilot>=N):
        raise ValueError('Pilot index outside frame')
    if int(budget)!=budget or not len(pilot)<=budget<=N:
        raise ValueError('Budget must retain the entire paid pilot')
    budget=int(budget);B0=len(pilot)
    selected=pilot.tolist();mask=np.zeros(N,bool);mask[pilot]=True
    observed={p.name:np.full(N,np.nan) for p in panels}
    for p in panels:
        v=np.asarray(pilot_values[p.name],float)
        if v.shape!=(B0,):
            raise ValueError('pilot_values must contain one value per paid pilot index')
        observed[p.name][pilot]=v
        eligible_pilot=pilot[p.eligible[pilot]]
        if not np.isfinite(observed[p.name][eligible_pilot]).all():
            raise ValueError('Pilot has missing eligible reference measurements')
    kind='if' if design=='a2_if_year' else 'loo'
    partition='year_treatment' if design.endswith('_cell') else 'year'
    groups=model.design.strata(d,partition)
    Nh=np.asarray([len(g) for g in groups],int)
    previous=np.asarray([mask[g].sum() for g in groups],int)
    prior=model.influence_variance(panels,{p.name:p.x for p in panels},np.arange(N),groups)
    prior=np.nan_to_num(prior,nan=0.)
    logs=[dict(wave=1,cumulative=B0,new=B0,reason='common_paid_srs_pilot',
               floor_fallback=0,sparse_strata_using_x_prior=0,
               loo_identity_count=0,loo_direct_refit_count=0,
               counts=previous.tolist(),new_indices=pilot.tolist(),
               reveal_performed_by='caller')]
    remaining_waves=1 if kind=='if' else 3
    cumulative=sorted(set(int(np.ceil(B0+(budget-B0)*w/remaining_waves))
                          for w in range(1,remaining_waves+1)))
    for total in cumulative:
        if total<=len(selected):
            continue
        values={};loo_fallback=0;loo_refits=0
        for p in panels:
            v=np.zeros(N)
            ix=np.asarray(selected,int);ix=ix[p.eligible[ix]]
            if kind=='if':
                v[ix]=observed[p.name][ix]
            else:
                _,_,res,ix,lf,lr=model.fit_predict(p,selected,observed[p.name],
                                                    'year_affine',loo=True)
                v[ix]=res;loo_fallback+=lf;loo_refits+=lr
            values[p.name]=v
        current=model.influence_variance(panels,values,selected,groups)
        usable=np.isfinite(current);variance=prior.copy();variance[usable]=current[usable]
        sparse=int((~usable&(Nh>1)).sum())
        allocation=model.allocate_a_optimal(Nh,Nh.astype(float)**2*variance,total,previous)
        counts=allocation.total
        if allocation.fallback:
            counts=model.balanced_counts(Nh,previous,total,rng)
        new=[]
        for group,extra in zip(groups,counts-previous):
            available=group[~mask[group]]
            if extra:
                new.extend(rng.choice(available,int(extra),replace=False).tolist())
        new=np.asarray(new,int)
        if len(new)!=total-len(selected) or mask[new].any() or len(np.unique(new))!=len(new):
            raise AssertionError('Matched allocation violates exact unique-record cost')
        paid=reveal(new)
        for p in panels:
            v=np.asarray(paid[p.name],float)
            if v.shape!=(len(new),):
                raise ValueError('reveal must return arrays aligned to new_indices')
            observed[p.name][new]=v
        selected.extend(new.tolist());mask[new]=True;previous=counts
        logs.append(dict(wave=len(logs)+1,cumulative=total,new=len(new),
                         floor_fallback=int(allocation.fallback),reason=allocation.reason,
                         sparse_strata_using_x_prior=sparse,
                         loo_identity_count=loo_fallback,loo_direct_refit_count=loo_refits,
                         counts=counts.tolist(),new_indices=new.tolist(),
                         reveal_performed_by='comparator'))
    selected=np.asarray(selected,int)
    if len(selected)!=budget or mask.sum()!=budget or len(np.unique(selected))!=budget:
        raise AssertionError('Final exact budget failed')
    if not np.array_equal(selected[:B0],pilot):
        raise AssertionError('Paid pilot was not retained')
    for p in panels:
        if np.isfinite(observed[p.name][~mask]).any():
            raise AssertionError('Unpaid reference values leaked into comparator')
    return selected,observed,logs
