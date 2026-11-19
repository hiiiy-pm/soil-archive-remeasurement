"""Construct two untouched evaluation frames from published records at their recorded precision."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'spectral'))
from experiment import original, discrepancy_kernel

@dataclass
class Channel:
    name: str
    rows: np.ndarray
    x: np.ndarray
    kernel: np.ndarray
    panel_ids: list

def load_trial(name):
    sources = ROOT/'data/source'
    if name == 'tcse':
        fields = ['Woodwell.ID','Trial','Year','U_depth (cm)','L_depth','System','Block.treatment',
                  'plot.study','soc.observed','mean.soc.eml.woodwell']
        raw = pd.read_csv(sources/'Spectral_SOC/measurements_and_spectra.csv',usecols=fields,
                          dtype=str,keep_default_na=False)
        d = raw[raw.Trial.eq('Lincoln-TCSE')].reset_index(drop=True)
        assert len(d)==239 and d['Woodwell.ID'].is_unique
        frame = pd.DataFrame(dict(year=d.Year.astype(int),treatment=d['Block.treatment']+' / '+d.System,
                                  depth=d['U_depth (cm)']+'-'+d.L_depth,plot=d['plot.study'],
                                  record_id=d['Woodwell.ID']))
        exact = [Fraction(v)*10 for v in d['soc.observed']]
        x = d['mean.soc.eml.woodwell'].astype(float).to_numpy()*10
        inputs = [('SOC',np.arange(len(d)),x,exact)]
    elif name == 'ultuna':
        table = pd.read_csv(sources/'Ultuna/RAM56_data.tsv',sep='\t',dtype=str,keep_default_na=False)
        ids = ['year','plotID','treatment_code','component']
        subset = table[table.variable.isin(['P_AL','P_HCl','K_AL','K_HCl'])]
        assert not subset.duplicated(ids+['variable']).any()
        assert set(subset.unit)=={'mg/kg'}
        wide = subset.pivot(index=ids,columns='variable',values='value').replace('NA',np.nan)
        masks = {a:wide[[a+'_AL',a+'_HCl']].notna().all(axis=1) for a in ['P','K']}
        wide = wide.loc[masks['P']|masks['K']].reset_index()
        assert set(wide.component)=={'soil_0_20_cm'} and 'NA' not in set(wide.plotID)
        frame = pd.DataFrame(dict(year=wide.year.astype(int),treatment=wide.treatment_code,
                                  depth=wide.component,plot=wide.plotID,record_id=wide.year+'-'+wide.plotID))
        assert frame.record_id.is_unique
        inputs = []
        for a in ['P','K']:
            rows = np.flatnonzero(wide[[a+'_AL',a+'_HCl']].notna().all(axis=1))
            inputs.append((a,rows,wide.iloc[rows][a+'_AL'].astype(float).to_numpy(),
                           [Fraction(v) for v in wide.iloc[rows][a+'_HCl']]))
    else:
        raise KeyError(name)
    channels, panels, responses, fractions = [], [], {}, {}
    domain_count = sum(frame.iloc[i].depth.nunique() for _,i,_,_ in inputs)
    for element,rows,x,exact in inputs:
        fullx = np.zeros(len(frame))
        fullx[rows] = x
        y = np.full(len(frame),np.nan)
        y[rows] = [float(v) for v in exact]
        exact_map = dict(zip(rows,exact))
        eligible = np.zeros(len(frame),bool)
        eligible[rows] = True
        panel_ids = []
        for depth in sorted(frame.iloc[rows].depth.unique()):
            panel_ids.append(len(panels))
            label = element+'_'+depth
            p = original.engine.make_panel(label,frame,fullx,eligible & frame.depth.eq(depth).to_numpy(),
                                           'g/kg' if name=='tcse' else 'mg/kg',domain_count)
            panels.append(p)
            responses[label] = y.copy()
            fractions[label] = exact_map
        channels.append(Channel(element,rows,x,discrepancy_kernel(frame.iloc[rows],x),panel_ids))
    return frame, panels, channels, responses, fractions

class TargetSet:
    def __init__(self, panels, channel):
        records = channel.rows
        self.columns = []
        self.groups = []
        self.definitions = []
        self.primary_weights = []
        self.mse_weights = []
        self.all_mse_weights = []
        self.record_weights = np.zeros(len(records))
        rows = []
        for pi in channel.panel_ids:
            p = panels[pi]
            eligible = p.eligible[records]
            aggregation = np.zeros((len(p.n),len(records)))
            g = p.g[records[eligible]]
            aggregation[g,np.flatnonzero(eligible)] = 1/p.n[g]
            self.record_weights[eligible] += 1/(len(panels)*len(p.rows)*p.scale)
            n_primary = sum(f!='interaction' for f in p.K)
            for family,(idx,coef) in p.K.items():
                matrix = np.einsum('ijk,ij->ik',aggregation[idx],coef)
                start = len(rows)
                rows.extend(matrix)
                self.groups.append((pi,p.name,family,start,len(rows),p.scale))
                self.definitions.extend([(pi,idx[j],coef[j]) for j in range(len(idx))])
                w = 0. if family=='interaction' else 1/(len(panels)*n_primary*len(idx))
                self.primary_weights.extend([w]*len(idx))
                self.mse_weights.extend([w/p.scale]*len(idx))
                self.all_mse_weights.extend([1/(len(panels)*len(p.K)*len(idx)*p.scale)]*len(idx))
        self.C = np.array(rows)
        self.primary_weights = np.array(self.primary_weights)
        self.mse_weights = np.array(self.mse_weights)
        self.all_mse_weights = np.array(self.all_mse_weights)
        self.primary = self.primary_weights > 0

def exact_targets(panels, target, fractions):
    means = {}
    for pi in {g[0] for g in target.groups}:
        p = panels[pi]
        raw = fractions[p.name]
        means[pi] = [sum((raw[i] for i in p.rows[p.g[p.rows]==j]),Fraction())/int(p.n[j])
                     for j in range(len(p.n))]
    return [sum((int(c)*means[pi][int(j)] for j,c in zip(idx,coef)),Fraction())
            for pi,idx,coef in target.definitions]
