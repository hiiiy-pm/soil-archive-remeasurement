"""Compare repair of all 34 Lanna treatment signs on the unchanged paid paths."""
import argparse
import json
from pathlib import Path

from diagnose_rankings import FOLDER
from multiperiod_trial import load_trial, SITE
from ranking_run import OUT, POLICIES, sources
from trials import TargetSet, np, pd
from experiment import conditional


def run(destination):
    item=pd.read_csv(destination/"contrast_losses.csv")
    item=item[item.family.eq("treatment")].reset_index(drop=True)
    actual=item.reference.to_numpy()
    assert len(item)==34 and np.all(actual!=0)
    frame,panels,(channel,),references,_=load_trial()
    _,bench,fixed=sources(SITE)
    cfg=json.loads((bench/"config.json").read_text())
    target=TargetSet(panels,channel)
    contrasts=np.concatenate([target.C[a:b] for _,_,fam,a,b,_ in target.groups if fam=="treatment"])
    y=references[panels[0].name]
    np.testing.assert_allclose(contrasts@y,actual)
    kernel=channel.kernel+cfg["ratio"]*np.eye(len(frame))
    periods=cfg["budgets"]
    labels=["joint_maximin", *POLICIES]
    losses=np.zeros((len(labels),cfg["replicates"],len(periods),len(item)),dtype=np.int8)
    for rep in range(cfg["replicates"]):
        filename=f"draw_{rep:03d}.npz"
        with np.load(fixed/filename) as old, np.load(OUT/SITE/filename) as new:
            np.testing.assert_array_equal(old["pilot"],new["pilot"])
            assert tuple(new["policies"])==POLICIES
            paths=np.concatenate([old["paths"][:1],new["paths"]])
            scores=np.concatenate([old["scores"][:1],new["scores"]])
        for di,path in enumerate(paths):
            for j,budget in enumerate(periods):
                paid=np.sort(path[:budget])
                estimate=channel.x+conditional(kernel,paid,y[paid]-channel.x[paid])
                estimate[paid]=y[paid]
                estimated=(contrasts@np.column_stack([estimate,estimate]))[:,0]
                unpaid=np.ones(len(y),bool)
                unpaid[paid]=False
                resolved=~contrasts[:,unpaid].any(axis=1)
                estimated[resolved]=actual[resolved]
                losses[di,rep,j]=np.sign(estimated)!=np.sign(actual)
                for name,group in item.groupby("domain"):
                    ci=cfg["coordinates"].index([name,"treatment"])
                    rate=losses[di,rep,j,group.index].mean()
                    np.testing.assert_allclose(rate,scores[di,j,0,ci,cfg["metrics"].index("direction")])
    assert not losses[:,:,-1].any()
    mean=losses[:,:,:-1].mean(axis=(1,2))
    np.testing.assert_allclose(mean[0],item.joint_maximin_error,atol=1e-14)
    published=pd.read_csv(OUT/"performance.csv",dtype={"budget":str})
    published=published[(published.site==SITE)&(published.model=="kriging")&(published.budget=="noncensus")
                        &(published.group=="treatment")&(published.metric=="direction")].set_index("design")
    for k,name in enumerate(labels):
        item[name+"_error"]=mean[k]
        np.testing.assert_allclose(item.weight_in_family@mean[k],published.loc[name,"mean"],atol=1e-14)
    a,b=losses[0,:,:-1],losses[1,:,:-1]
    item["old_wrong_new_correct"]=((a==1)&(b==0)).sum(axis=(0,1))
    item["old_correct_new_wrong"]=((a==0)&(b==1)).sum(axis=(0,1))
    item["repair"]=mean[0]-mean[1]
    item["repair_contribution"]=item.weight_in_family*item.repair
    cols=["domain","first_year","treatment_pair","small","reference_exact","reference","weight_in_family",
          *[name+"_error" for name in labels],"sign_error_error","old_wrong_new_correct","old_correct_new_wrong",
          "repair","repair_contribution"]
    item[cols].to_csv(destination/"repair_by_item.csv",index=False)
    groups=[]
    for group,partition in [("all",[("all",item)]),("field",item.groupby("domain")),
                             ("pair",item.groupby("treatment_pair")),("small",item.groupby("small"))]:
        for label,part in partition:
            w=part.weight_in_family
            row=dict(group=group,category=label,contrasts=len(part),weight=w.sum())
            for name in [*labels,"sign_error"]:
                row[name+"_error"]=np.average(part[name+"_error"],weights=w)
            row["repair_contribution"]=part.repair_contribution.sum()
            for field in ["old_wrong_new_correct","old_correct_new_wrong"]:
                row[field]=part[field].sum()
            groups.append(row)
    group=pd.DataFrame(groups)
    group.to_csv(destination/"repair_by_group.csv",index=False)
    print(group.to_string(index=False))
    print(item.query('first_year==2001 and treatment_pair=="P minus P6"')[cols].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=FOLDER)
    args=parser.parse_args()
    run(args.output)
