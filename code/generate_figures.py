"""Generate figures from the archived allocation results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

H=Path(__file__).resolve().parents[1]
NAMES=['muencheberg','weoc','wen','poxc','rothamsted','woburn','beltsville','lincoln']
TITLES=['Müncheberg SOC','Storage · WEOC','Storage · WEN','Storage · POXC','Rothamsted cations','Woburn cations','Beltsville SOC','Lincoln SOC']
LABELS={'retained_contrast':'Retained contrast','random':'Random','cell_balance':'Year–treatment balance','target_cell_balance':'Target-cell balance','maximin':'Historical maximin','conversion':'Conversion contrast','concentration':'Concentration','active_sampling':'Active-sampling adapter','a2_if_year':'Two-wave IF / year','a4_loo_year':'Four-wave LOO / year','a4_loo_cell':'Four-wave LOO / year–treatment','a4_loo_targetcell':'Four-wave LOO / target cell'}
COLORS={'retained_contrast':'#007D80','random':'#6B7280','target_cell_balance':'#D99A2B','maximin':'#6A58A0','conversion':'#BD5B49','concentration':'#3D7CA3','a4_loo_year':'#748B4D','active_sampling':'#AC7093'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#6F7881','axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def load(n):
    c=json.loads((H/'results'/n/'config.json').read_text())
    fs=sorted((H/'results'/n).glob('draw_*.npz'));assert len(fs)==200
    a=np.stack([np.load(f)['scores'] for f in fs]);panels=list(dict.fromkeys(p for p,f in c['coordinates']))
    j=np.mean([a[:,:,:,:,[i for i,(p,f) in enumerate(c['coordinates']) if p==q],:].mean(axis=4) for q in panels],axis=0)
    return c,a,j
def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
    fig.savefig(out/(name+'.eps'),bbox_inches='tight',pad_inches=.05)
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
def fig_s7(out, data):
    fig, axs = plt.subplots(2, 4, figsize=(6.85, 4.6), sharex=True)
    styles = {
        'retained_contrast': ('-', None),
        'target_cell_balance': ((0, (5, 2)), None),
        'maximin': ((0, (6, 2, 1, 2)), None),
        'concentration': ('-', 's'),
        'a4_loo_year': ((0, (3, 2)), '^'),
        'random': (':', None),
    }
    for k, (name, title, ax) in enumerate(zip(NAMES, TITLES, axs.flat)):
        config, _, scores = data[name]
        x = np.array(config['budgets']) / config['N'] * 100
        for design, (line, marker) in styles.items():
            idx = config['designs'].index(design)
            draws = scores[:, idx, :, 0, 4]
            mean = draws.mean(axis=0)
            focal = design == 'retained_contrast'
            ax.plot(x, mean, color=COLORS[design], linestyle=line, marker=marker,
                    markevery=(1, 3), markersize=2.6, markeredgewidth=.55,
                    markerfacecolor='white', linewidth=1.65 if focal else 1.0,
                    label=LABELS[design], zorder=5 if focal else 2)
            if focal:
                error = draws.std(axis=0, ddof=1) / np.sqrt(200)
                ax.fill_between(x, np.maximum(0, mean - 1.96 * error), mean + 1.96 * error,
                                color='#C7E6E0', lw=0, zorder=1)
        site, indicator = title.rsplit(' ', 1)
        site = site.removesuffix(' ·')
        ax.set_title(f'{chr(97+k)}  {site}', loc='left', fontsize=9.6,
                     color='#26363C', fontweight='bold', pad=13)
        ax.text(0, 1.012, indicator, transform=ax.transAxes, va='bottom',
                fontsize=8.2, color='#486169')
        ax.set_ylim(bottom=0)
        ax.set_xticks([10, 50, 100])
        ax.grid(axis='y', color='#E5EAED', lw=.5)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        if k % 4 == 0:
            ax.set_ylabel('History-scaled MSE')
        if k >= 4:
            ax.set_xlabel('Assayed records (%)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.015), handlelength=2.7, handletextpad=.6)
    fig.subplots_adjust(left=.075, right=.985, top=.93, bottom=.22, wspace=.48, hspace=.68)
    save(fig, out, 'FigS7')
def fig_s1(out,data):
    designs=['random','cell_balance','target_cell_balance','maximin','conversion','concentration','retained_contrast','active_sampling','a2_if_year','a4_loo_year','a4_loo_cell','a4_loo_targetcell']
    vals=[]
    for n in NAMES:
        c,a,j=data[n];m=j[:,:,:-1,0,4].mean(axis=(0,2));vals.append([m[c['designs'].index(d)]/m[c['designs'].index('random')] for d in designs])
    v=np.array(vals).T
    fig,ax=plt.subplots(figsize=(6.85,4.35));im=ax.imshow(np.log2(v),cmap='BrBG_r',norm=TwoSlopeNorm(vcenter=0,vmin=-.6,vmax=.6),aspect='auto')
    ax.set_yticks(range(len(designs)),[LABELS[d] for d in designs]);ax.set_xticks(range(8),['Müncheberg','WEOC','WEN','POXC','Rothamsted','Woburn','Beltsville','Lincoln'],rotation=40,ha='right')
    for i in range(len(designs)):
        for j in range(8):
            rgb = np.asarray(im.to_rgba(np.log2(v[i, j]))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
            black_contrast = (luminance + 0.05) / 0.05
            white_contrast = 1.05 / (luminance + 0.05)
            text_color = 'black' if black_contrast >= white_contrast else 'white'
            ax.text(j, i, f'{v[i, j]:.2f}', ha='center', va='center', fontsize=8, color=text_color)
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,len(designs),1),minor=True);ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',bottom=False,left=False)
    ax.axhline(5.5,color='#214D54',lw=1.1);ax.axhline(6.5,color='#214D54',lw=1.1)
    cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.8,ticks=np.log2([.7,1,1.4]));cb.ax.set_yticklabels(['0.70','1.00','1.40']);cb.set_label('MSE / random MSE')
    fig.subplots_adjust(left=.31,bottom=.22,right=.92,top=.92);save(fig,out,'FigS1')
def fig_s2(out):
    t=pd.read_csv(H/'diagnostics/retention_effects.csv');t=t[(t.method=='pilot_ridge')&(t.metric=='historical_scaled_mse')]
    fig,axs=plt.subplots(2,4,figsize=(6.85,4.4))
    show=['retained_contrast','conversion','random']
    styles = {
        'retained_contrast': dict(marker='o', linestyle='-', markersize=3.5),
        'conversion': dict(marker='s', linestyle='--', markersize=3.5, markerfacecolor='none'),
        'random': dict(marker='^', linestyle=':', markersize=4.5, markerfacecolor='none'),
    }
    for k,(n,title,ax) in enumerate(zip(NAMES,TITLES,axs.flat)):
        z=t[t.universe==n]
        for design in show:
            r=z[z.design==design].iloc[0]
            ax.plot([0,1],[r.pure,r.retained],color=COLORS[design],lw=1.4,label=LABELS[design],**styles[design])
        ax.set_xticks([0,1],['Pure','Retained']);ax.set_xlim(-.18,1.18);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#E5EAED',lw=.5)
        site, analyte = title.replace(' · ', ' ').rsplit(' ', 1)
        ax.set_title(f'{chr(97 + k)}  {site}', loc='left', fontweight='bold', pad=16)
        ax.annotate(analyte, xy=(0, 1), xycoords='axes fraction',
                    xytext=(0, 3), textcoords='offset points',
                    ha='left', va='bottom', fontsize=8, color='#354D55')
        if k%4==0:ax.set_ylabel('Mean non-census MSE')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.08,right=.995,top=.91,bottom=.17,wspace=.42,hspace=.64);save(fig,out,'FigS2')
def fig_s3(out):
    t=pd.read_csv(H/'diagnostics/error_decomposition.csv')
    show=['retained_contrast','concentration','maximin','target_cell_balance']
    fig,axs=plt.subplots(1,2,figsize=(6.85,2.75))
    plotted = t[t.universe.isin(['beltsville', 'lincoln']) & t.design.isin(show)]
    common_limit = plotted.history_scaled_mse.max() * 1.2
    for k,(n,title,ax) in enumerate(zip(['beltsville','lincoln'],['Beltsville · 28 / 56 records','Lincoln · 90 / 179 records'],axs)):
        z=t[t.universe==n].set_index('design').loc[show];yy=np.arange(len(show))
        ax.barh(yy,z.squared_mean_error,color='#607F9B',height=.58,label='Squared mean error')
        ax.barh(yy,z.allocation_variance,left=z.squared_mean_error,color='#A3C5C1',height=.58,label='Allocation variance')
        for y,v in zip(yy,z.history_scaled_mse):ax.text(v+.003,y,f'{v:.3f}',va='center',fontsize=7.5)
        ax.set_yticks(yy,[LABELS[x] for x in show]);ax.invert_yaxis();ax.set_xlim(0,common_limit);ax.set_xlabel('History-scaled contrast MSE')
        ax.set_title(chr(97+k)+'  '+title,loc='left',fontweight='bold',pad=10)
        ax.grid(axis='x',color='#E5EAED',lw=.5);ax.set_axisbelow(True)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.02))
    fig.subplots_adjust(left=.19,right=.985,top=.86,bottom=.27,wspace=.85);save(fig,out,'FigS3')
