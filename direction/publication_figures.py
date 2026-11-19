"""Generate the integrated article figures from the archived outputs."""
from trials import ROOT, load_trial, TargetSet, np, pd
from acquisition import PosteriorState, sign_gain
from study import SEEDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(ROOT/'code'))
import generate_figures as historical

TEAL='#007477'
PLUM='#88456d'
BLUE='#346ca5'
AMBER='#ba710c'
GRAY='#737f85'
COLORS={'sign_error':TEAL,'sign_variance':PLUM,'contrast_mse':BLUE,'best_mse':AMBER,'cell_balance':GRAY}
LABELS={'sign_error':'Sign error','sign_variance':'Sign variance','contrast_mse':'Contrast MSE',
        'best_mse':'Best magnitude baseline','cell_balance':'Balanced sampling'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':10,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,folder,name):
    for ext in ['pdf','eps','png']:
        fig.savefig(folder/(name+'.'+ext),dpi=200,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

INK = '#26363C'
AXIS = '#56656B'
GUIDE = '#DCE2E3'
SEA = '#146C70'
VIOLET = '#75466D'
RUST = '#93451C'


def style_panel(ax, letter, heading):
    ax.set_title(f'{letter}   {heading}', loc='left', fontsize=10,
                 weight='bold', color=INK, pad=7)
    ax.tick_params(axis='both', labelsize=8.7, length=3, colors=AXIS, pad=3)
    ax.spines[['bottom', 'left']].set(color=AXIS, linewidth=.7)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GUIDE, linewidth=.55)


def workflow(folder):
    fig, (ax, field) = plt.subplots(1, 2, figsize=(7.15, 3.05))
    fig.subplots_adjust(left=.088, right=.985, bottom=.21, top=.87, wspace=.42)
    style_panel(ax, 'a', 'Sign-error reduction')
    style_panel(field, 'b', 'Candidate assays at Ultuna')
    z = np.linspace(0, 2.6, 280)[:, None]
    for correlation, color, line in [(.8, SEA, '-'), (.5, VIOLET, '--'), (.25, RUST, ':')]:
        ax.plot(z[:, 0], 100 * sign_gain(z, np.full_like(z, correlation))[:, 0],
                color=color, ls=line, label=f'{correlation:.2f}', lw=1.65)
    ax.legend(title='Assay–contrast correlation', frameon=False, loc='upper right',
              fontsize=8.7, title_fontsize=8.7, labelcolor=INK, handlelength=2.5,
              borderaxespad=.15, handletextpad=.65)
    ax.set(xlabel='Absolute contrast mean / posterior SD',
           ylabel='Expected sign-error reduction (pp)', xlim=(0, 2.6), ylim=(0, 32),
           yticks=[0, 5, 10, 15, 20, 25, 30])
    frame, panels, channels, responses, _ = load_trial('ultuna')
    with np.load(ROOT/'direction/joint_results/ultuna/draw_000.npz') as draw:
        pilot = draw['pilot']
        winner = draw['paths'][0, len(pilot)]
    normalized = []
    for criterion in ['contrast_mse', 'sign_variance']:
        gain = np.zeros(len(frame))
        for channel in channels:
            target = TargetSet(panels, channel)
            state = PosteriorState(channel, target, 1., pilot,
                                   responses[panels[channel.panel_ids[0]].name][pilot], criterion)
            rows, values = state.gains()
            gain[rows] += values
        normalized.append(gain/gain.max())
    x, y = normalized
    candidates = np.setdiff1d(np.arange(len(frame)), pilot)
    assert np.isclose(min(x[winner], y[winner]), np.minimum(x, y)[candidates].max())
    magnitude = x.argmax()
    field.plot([0, 1], [0, 1], color='#687A7E', ls=':', lw=1.0, zorder=1)
    field.scatter(x[candidates], y[candidates], s=18, facecolor='#D9E2E2',
                  edgecolor='#74888B', linewidth=.5, zorder=2)
    for idx, marker, color in [(winner, 'D', SEA), (magnitude, 's', RUST)]:
        field.plot(x[idx], y[idx], marker=marker, ls='', ms=6.8, zorder=4,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=.65)
    for idx, label, loc, color in [
        (winner, 'Joint choice\nPlot 22, 1979', (.25, .96), SEA),
        (magnitude, 'Magnitude maximum', (.32, .11), RUST),
    ]:
        field.annotate(label, (x[idx], y[idx]), xytext=loc, textcoords='axes fraction',
                       color=color, fontsize=8.7, weight='bold', va='top',
                       arrowprops=dict(arrowstyle='-', color=color, lw=.85, shrinkB=3.8),
                       zorder=5)
    field.set(xlim=(-.035, 1.06), ylim=(-.035, 1.07),
              xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1],
              xlabel='Fraction of greatest magnitude gain',
              ylabel='Fraction of greatest sign-variance gain')
    for panel in (ax, field):
        panel.xaxis.label.set(color=INK, fontsize=9)
        panel.yaxis.label.set(color=INK, fontsize=9)
    save(fig, folder, 'Fig1')

def transfer(folder):
    data=pd.read_csv(ROOT/'spectral/external_scores.csv')
    data=data[data.budget=='noncensus']
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.85))
    comparison=[('beltsville','concentration','external_concentration'),
                ('lincoln','maximin','external_conversion'),
                ('pendleton','retained_contrast','retained_contrast')]
    for ax,metric,k,letter,title in zip(axes,['scaled_mse','direction'],[1,100],'ab',
                                          ['Difference in scaled MSE','Directional error difference (pp)']):
        for i,(site,old,matched) in enumerate(comparison):
            for offset,rule,model,color,mark in [(0.20,'cell_balance','external_kriging',TEAL,'o'),
                    (0.,matched,'external_kriging',PLUM,'D'),(-.20,old,'original_ridge',GRAY,'s')]:
                g=data[(data.site==site)&(data.selection==rule)&(data.reconstruction==model)&(data.metric==metric)].iloc[0]
                ax.hlines(2-i+offset,k*g.lower,k*g.upper,color=color,lw=1.4)
                ax.plot(k*g.paired_difference,2-i+offset,marker=mark,color=color,ms=4)
        ax.axvline(0,color='#42545b',lw=.7);ax.set_yticks([0,1,2],['Pendleton','Lincoln-REAP','Beltsville'])
        ax.set_ylim(-.45,2.42);ax.set_xlabel(title);ax.grid(axis='x',color='#e3e8e8',linewidth=.6)
        ax.set_axisbelow(True);ax.set_title(letter+'   '+('Comparison magnitude' if k==1 else 'Comparison direction'),loc='left',fontweight='bold')
    handles=[Line2D([],[],color=c,marker=m,ls='',label=n) for c,m,n in
             [(TEAL,'o','Balanced, same calibration'),(PLUM,'D','Best other MSE, same calibration'),(GRAY,'s','Best MSE, ridge benchmark')]]
    fig.legend(handles=handles,loc='lower center',frameon=False,ncol=2,columnspacing=1.2,bbox_to_anchor=(.5,-.10))
    fig.subplots_adjust(left=.12,right=.99,wspace=.45,bottom=.27)
    save(fig,folder,'FigS8')

def direction(folder):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.36))
    fig.subplots_adjust(left=.095, right=.985, bottom=.15, top=.94, wspace=.33, hspace=.89)
    fig2,curves=plt.subplots(2,2,figsize=(7,5.0))
    styles = [
        ('sign_error', 'o', SEA, 'Direct sign'),
        ('sign_variance', 'D', VIOLET, 'Sign variance'),
        ('contrast_mse', '^', '#315D90', 'Magnitude'),
        ('best_mse', 's', RUST, 'Lowest-MSE baseline'),
        ('cell_balance', 'X', '#404C55', 'Balanced sampling'),
    ]
    outcomes = [
        ('time', SEA, 'o', '-', '#D7E7E8', 'Temporal differences'),
        ('treatment', VIOLET, 's', '--', '#ECE1E8', 'Treatment differences'),
    ]
    bottom_bounds = []
    for col,(site,site_label,best) in enumerate([('tcse','Lincoln-TCSE','concentration_mse'),
                                                 ('ultuna','Ultuna','concentration')]):
        table=pd.read_csv(ROOT/'direction/results'/site/'paired_summary.csv')
        table=table[table.model=='kriging']
        nz=table[(table.group=='primary')&(table.budget=='noncensus')]
        p=nz.pivot(index='design',columns='metric',values='mean')
        ax = axs[0, col]
        style_panel(ax, 'ab'[col], site_label)
        highlighted = [best if name == 'best_mse' else name for name, _, _, _ in styles]
        other = p.drop(index=highlighted)
        ax.scatter(other.scaled_mse, 100 * other.direction, s=19, facecolor='white',
                   edgecolor='#77878D', linewidth=.75, zorder=2)
        for key, (_, marker, color, _) in zip(highlighted, styles):
            r = p.loc[key]
            ax.plot(r.scaled_mse, 100 * r.direction, marker=marker, ms=6.2, ls='',
                    markerfacecolor=color, markeredgecolor='white', markeredgewidth=.55)
        ax.set(xlabel='Scaled comparison MSE', ylabel='Directional error (%)')
        ax = axs[1, col]
        style_panel(ax, 'cd'[col], site_label)
        ax.axhline(0, color='#58646A', lw=.85, zorder=2)
        for name, color, marker, line, fill, _ in outcomes:
            z = table[(table.group == name) & (table.metric == 'direction') &
                      (table.design == 'contrast_mse') & table.budget.str.isdigit()]
            b = z.budget.astype(int).to_numpy()
            xx = 100 * b / b[-1]
            ax.fill_between(xx, 100 * z.lower, 100 * z.upper, color=fill, linewidth=0)
            ax.plot(xx, 100 * z.sign_minus_comparator, color=color, marker=marker,
                    ls=line, ms=3.6, lw=1.45, zorder=3)
            bottom_bounds.extend([100 * z.lower.min(), 100 * z.upper.max()])
        ax.set(xlabel='Assayed records (%)', ylabel='Direct sign − magnitude (pp)',
               xticks=[10, 30, 50, 70, 100])
        for row,(metric,scale,label) in enumerate([('direction',100,'Directional error (%)'),('scaled_mse',1,'Scaled comparison MSE')]):
            ax=curves[row,col]
            for name,mark in [('sign_error','o'),('sign_variance','D'),('contrast_mse','^'),('best_mse','s'),('cell_balance','x')]:
                z=table[(table.group=='primary')&(table.metric==metric)&(table.design==(best if name=='best_mse' else name))&table.budget.str.isdigit()]
                b=z.budget.astype(int).to_numpy()
                ax.plot(100*b/b[-1],scale*z['mean'],color=COLORS[name],lw=1.4,ms=3,marker=mark)
            ax.set_ylim(bottom=0);ax.set_xticks([10,30,50,70,100]);ax.set_ylabel(label);ax.grid(axis='y',color='#e3e9ea',lw=.55)
            if row==0:ax.set_title(site_label,loc='left',fontweight='bold')
            else:ax.set_xlabel('Assayed records (%)')
    handles=[Line2D([],[],color=COLORS[n],ls='',marker=m,label=LABELS[n]) for n,m in
             [('sign_error','o'),('sign_variance','D'),('contrast_mse','^'),('best_mse','s'),('cell_balance','X')]]
    magnitude_handles = [
        Line2D([], [], ls='', marker=mark, color=color, markersize=6, label=label)
        for _, mark, color, label in styles
    ]
    magnitude_handles.append(Line2D([], [], ls='', marker='o', markerfacecolor='white',
                                    markeredgecolor='#77878D', markersize=4.8,
                                    label='Other benchmarks'))
    fig.legend(handles=magnitude_handles, loc='center', bbox_to_anchor=(.535, .522),
               ncol=3, frameon=False, columnspacing=1.55, handlelength=1.1,
               handletextpad=.38, labelspacing=.42, fontsize=8.5, labelcolor=INK)
    diff_handles = [Line2D([], [], color=color, marker=mark, ls=line, lw=1.45,
                           markersize=4, label=label)
                    for _, color, mark, line, _, label in outcomes]
    fig.legend(handles=diff_handles, loc='lower center', bbox_to_anchor=(.53, .006),
               ncol=2, frameon=False, columnspacing=1.8, handlelength=2.6,
               handletextpad=.55, fontsize=8.5, labelcolor=INK)
    lo, hi = min(bottom_bounds), max(bottom_bounds)
    pad = (hi - lo) * .045
    for ax in axs[1]:
        ax.set_ylim(lo - pad, max(4, hi + pad))
        ax.set_yticks([-12, -8, -4, 0, 4])
    for ax in axs.flat:
        ax.xaxis.label.set(color=INK, fontsize=9)
        ax.yaxis.label.set(color=INK, fontsize=9)
    fig2.legend(handles=handles,loc='lower center',ncol=3,frameon=False,columnspacing=1.1,
                handletextpad=.25,fontsize=8)
    fig2.subplots_adjust(left=.09,right=.975,wspace=.34,top=.94,bottom=.195,hspace=.64)
    save(fig,folder,'Fig2');save(fig2,folder,'FigS4')

def run(out):
    out.mkdir(exist_ok=True,parents=True)
    workflow(out);transfer(out);direction(out)
    data = {n: historical.load(n) for n in historical.NAMES}
    historical.TITLES[-1] = 'Lincoln-REAP SOC'
    historical.fig_s7(out, data)
    historical.fig_s1(out, data)
    historical.fig_s2(out)
    historical.fig_s3(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'artwork')
    run(ap.parse_args().output)
