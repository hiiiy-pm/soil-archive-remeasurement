"""Half-budget descriptive example for farmyard manure (J) and the cropped unamended control (B)."""
from trials import ROOT, load_trial, np, pd
from experiment import conditional
import json

def run():
    folder=ROOT/'direction/results/ultuna'
    frame,panels,channels,references,exact=load_trial('ultuna')
    config=json.loads((folder/'config.json').read_text())
    bi=config['budgets'].index(50)
    pairs=[((year,'J'),(year,'B')) for year in [1979,1989]]+[
        ((1989,t),(1979,t)) for t in ['B','J']]
    rows=[]
    for ch in channels:
        p=panels[ch.panel_ids[0]]
        a=frame.iloc[ch.rows]
        groups={k:np.flatnonzero((a.year.eq(k[0])&a.treatment.eq(k[1])).to_numpy())
                for k in {g for pair in pairs for g in pair}}
        truths={k:sum(exact[p.name][ch.rows[i]] for i in ids)/len(ids) for k,ids in groups.items()}
        V=ch.kernel+np.eye(len(ch.rows))
        for path in sorted(folder.glob('draw_*.npz')):
            with np.load(path) as saved:
                mask=np.unpackbits(saved['selections'][:,bi],axis=-1,count=len(frame)).astype(bool)
            for design in ['contrast_mse','sign_error','sign_variance']:
                paid=np.flatnonzero(mask[config['designs'].index(design),ch.rows])
                got=references[p.name][ch.rows[paid]]
                pred=ch.x+conditional(V,paid,got-ch.x[paid])
                pred[paid]=got
                for left,right in pairs:
                    truth=truths[left]-truths[right]
                    estimate=pred[groups[left]].mean()-pred[groups[right]].mean()
                    rows.append(dict(element=ch.name,comparison=f'{left[0]} {left[1]} minus {right[0]} {right[1]}',
                                     design=design,truth=float(truth),exact=str(truth),estimated=estimate,
                                     squared_error=(estimate-float(truth))**2,wrong=np.sign(estimate)!=np.sign(float(truth))))
    d=pd.DataFrame(rows)
    out=d.groupby(['element','comparison','design','truth','exact'],as_index=False).agg(
        mean_estimate=('estimated','mean'),rmse=('squared_error',lambda v:np.sqrt(v.mean())),direction_error=('wrong','mean'))
    out.to_csv(folder/'manure_control.csv',index=False)
    print(out.round(3).to_string(index=False))

if __name__=='__main__': run()
