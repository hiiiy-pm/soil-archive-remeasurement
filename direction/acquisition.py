"""Outcome-adaptive one-assay sign loss and matched sampling objectives."""
from trials import np
from experiment import conditional
from scipy.special import ndtr, owens_t

MODEL_RULES = ['sign_error','sign_variance','contrast_mse','all_contrast_mse','conversion_mse','concentration_mse']

def sign_gain(z, r, variance_loss=False):
    """Expected reduction under a unit Gaussian. z is |mean|/SD, r absolute correlation."""
    r = np.clip(r,0.,1.)
    if variance_loss:
        q = r*r
        slope = np.sqrt((1-q)/(1+q))
        gain = 2*(owens_t(z,1.)-owens_t(z,slope))
    else:
        slope = np.divide(np.sqrt((1-r)*(1+r)),r,out=np.full_like(r,np.inf),where=r>0)
        gain = ndtr(-z)-2*owens_t(z,slope)
    return np.maximum(0.,gain)

class PosteriorState:
    def __init__(self, channel, targets, ratio, pilot, paid, rule):
        self.channel = channel
        self.rule = rule
        self.targets = targets
        selected = np.flatnonzero(np.isin(channel.rows,pilot))
        lookup = dict(zip(pilot,paid))
        values = np.array([lookup[channel.rows[j]] for j in selected])-channel.x[selected]
        if len(selected)==0:
            raise ValueError('Unknown-mean calibration requires one paid reference in a channel.')
        cov = channel.kernel+ratio*np.eye(len(channel.x))
        update, residual = conditional(cov,selected,values,True)
        residual[selected,:] = 0
        residual[:,selected] = 0
        self.R = residual
        self.pred = channel.x+update
        self.pred[selected] = channel.x[selected]+values
        block = cov[np.ix_(selected,selected)]
        sol = np.linalg.solve(block,np.column_stack([values,np.ones(len(values))]))
        mean = sol[:,0].sum()/sol[:,1].sum()
        variance = (values-mean)@(sol[:,0]-mean*sol[:,1])/(len(values)-1) if len(values)>1 else channel.x.var()
        self.scale = max(variance,np.finfo(float).eps*max(channel.x.var(),np.finfo(float).tiny))
        self.available = np.ones(len(channel.x),bool)
        self.available[selected] = False
        primary = targets.primary
        if rule=='all_contrast_mse':
            self.C = targets.C
            self.weights = targets.all_mse_weights
        else:
            self.C = targets.C[primary]
            self.weights = (targets.primary_weights if rule in ['sign_error','sign_variance']
                            else targets.mse_weights)[primary]
        self.mu = self.C@self.pred
        self.cross = self.C@residual
        self.var = np.einsum('ij,ij->i',self.cross,self.C)
        self.dependencies = self.C!=0
        self.unobserved = self.dependencies[:,self.available].sum(axis=1)
        if rule=='conversion_mse':
            prior_cross = self.C@channel.kernel
            solved = np.linalg.solve(block,cov[selected,:])
            q = sol[:,1]
            self.cross = prior_cross-prior_cross[:,selected]@solved
            self.cross += np.outer(self.C.sum(axis=1)-prior_cross[:,selected]@q,1-solved.sum(axis=0))/q.sum()

    def gains(self):
        ids = np.flatnonzero(self.available)
        den = self.R.diagonal()[ids]
        assert np.all(den>0)
        if self.rule=='concentration_mse':
            gains = self.scale*(self.targets.record_weights @ (self.R[:,ids]**2))/den
        elif self.rule in ['sign_error','sign_variance']:
            use = self.unobserved>0
            assert np.all(self.var[use]>0)
            variance = self.var[use]
            z = np.abs(self.mu[use])/np.sqrt(self.scale*variance)
            correlations = np.abs(self.cross[np.ix_(use,ids)])/np.sqrt(variance[:,None]*den)
            gains = self.weights[use] @ sign_gain(z[:,None],correlations,self.rule=='sign_variance')
        else:
            gains = self.scale*(self.weights @ (self.cross[:,ids]**2))/den
        return self.channel.rows[ids],gains

    def update(self,local,value):
        col = self.R[:,local].copy()
        den = col[local]
        assert self.available[local] and den>0
        contrast_col = self.cross[:,local].copy()
        innovation = value-self.pred[local]
        self.pred += col*(innovation/den)
        self.pred[local] = value
        self.mu += contrast_col*(innovation/den)
        self.var -= contrast_col**2/den
        self.R -= np.outer(col,col/den)
        self.cross -= np.outer(contrast_col,col/den)
        self.R[local,:] = 0
        self.R[:,local] = 0
        self.available[local] = False
        self.unobserved -= self.dependencies[:,local]

def select_path(frame, channels, targets, ratio, pilot, initial, reveal, panels, rule, seed):
    states = [PosteriorState(c,t,ratio,pilot,initial[panels[c.panel_ids[0]].name],rule)
              for c,t in zip(channels,targets)]
    local_indices = [{v:j for j,v in enumerate(c.rows)} for c in channels]
    rng = np.random.default_rng(seed)
    priority = np.empty(len(frame),int)
    priority[rng.permutation(len(frame))] = np.arange(len(frame))
    available = np.ones(len(frame),bool)
    available[pilot] = False
    path = list(pilot)
    while available.any():
        gain = np.zeros(len(frame))
        for state in states:
            rows,values = state.gains()
            gain[rows] += values
        ids = np.flatnonzero(available)
        maximum = gain[ids].max()
        tied = ids[np.isclose(gain[ids],maximum,rtol=1e-12,atol=1e-15)]
        choice = tied[np.argmin(priority[tied])]
        got = reveal(np.array([choice]))
        for c,local,state in zip(channels,local_indices,states):
            if choice in local:
                state.update(local[choice],got[panels[c.panel_ids[0]].name][0])
        available[choice] = False
        path.append(choice)
    return np.asarray(path)
