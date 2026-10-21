"""Diagnose clock failures and test finite-trigger segmentation locally only."""

import json
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from validate import OUT, CHANNELS, curve, width, stamp, write_json
from Q_Walk.qwalk_t0_data import (
    _read_mcp_events, assign_scope_segments, reconstruct_reference_delta_t,
    robust_centered_line,
)
from Q_Walk.qwalk_t0_models import robust_sigma


def main():
    data=pd.read_csv(OUT/'matched_hits.csv.gz')
    selected=pd.read_csv(OUT/'selected_events.csv.gz')
    one=data.loc[data.channelID.eq(137)]
    diagnostic=[]
    for (file_id,seg),g in one.groupby(['file_id','segment']):
        diagnostic.append(dict(file_id=file_id,segment=int(seg),n=len(g),
            clock_line_slope=float(g.ch192_line_slope.iloc[0]),
            clock_line_scaled_mad_ps=robust_sigma(g.ch192_alignment_residual),
            large_signal_residuals=int((g.delta_t_ref_ps.abs()>1e6).sum())))
    initial=pd.DataFrame(diagnostic)
    initial.to_csv(OUT/'clock_group_audit.csv',index=False)
    suspicious=initial.loc[initial.clock_line_scaled_mad_ps>1e6]
    resets=[]; shifts=[]; mappings={}
    for file_id in suspicious.file_id.unique():
        path=f'/eos/user/l/lichengz/MTD/TB2025/sync_out_4399_example/{file_id}.root'
        with uproot.open(path) as root:
            mcp=_read_mcp_events(root['MCP'])
            hits=ak.to_dataframe(root['data'].arrays(['channelID','time'],library='ak')).reset_index()
        clean=mcp.loc[np.isfinite(mcp.mcp_trigger_time)].copy()
        clean['finite_segment']=assign_scope_segments(clean.mcp_trigger_time.to_numpy())
        mappings[file_id]=dict(zip(clean.entry,clean.finite_segment))
        positions=np.flatnonzero(np.diff(clean.mcp_trigger_time)<0)+1
        for i in positions:
            a,b=clean.iloc[i-1],clean.iloc[i]
            if a.segment==b.segment:
                resets.append(dict(file_id=file_id,previous_entry=int(a.entry),next_entry=int(b.entry),
                    previous_trigger_ps=float(a.mcp_trigger_time),next_trigger_ps=float(b.mcp_trigger_time),
                    old_segment=int(a.segment),new_segment=int(b.finite_segment),
                    intervening_nonfinite=int(mcp.entry.between(a.entry,b.entry,inclusive='neither').sum())))
        trig=hits.loc[hits.channelID.eq(192),['entry','time']]
        trig=trig.loc[~trig.entry.duplicated(keep=False)].set_index('entry').time
        mcp['ch192']=mcp.entry.map(trig)
        # Diagnostic: shift scope timestamps by whole MCP rows, without changing any analysis input.
        for seg in suspicious.loc[suspicious.file_id.eq(file_id),'segment']:
            group=mcp.loc[mcp.segment.eq(seg)].copy()
            for shift in (-2,-1,0,1,2):
                x=group.mcp_trigger_time.shift(shift).to_numpy()
                y=group.ch192.to_numpy()
                _,res,mask,slope=robust_centered_line(x,y)
                shifts.append(dict(file_id=file_id,segment=int(seg),mcp_row_shift=shift,
                    finite_pairs=int(np.sum(np.isfinite(x)&np.isfinite(y))),
                    clock_line_slope=slope,scaled_mad_ps=robust_sigma(res),inliers=int(mask.sum())))
        print('Clock audit',file_id,flush=True)
    write_json('missed_reset_evidence.json',resets)
    pd.DataFrame(shifts).to_csv(OUT/'mcp_row_shift_diagnostic.csv',index=False)
    fixed=data.loc[np.isfinite(data.mcp_trigger_time)].copy()
    for file_id,mapping in mappings.items():
        mask=fixed.file_id.eq(file_id)
        fixed.loc[mask,'segment']=fixed.loc[mask,'entry'].map(mapping)
    fixed=reconstruct_reference_delta_t(fixed)
    rows=[]
    for (file_id,seg),g in fixed.loc[fixed.channelID.eq(137)].groupby(['file_id','segment']):
        rows.append(dict(file_id=file_id,segment=int(seg),clock_line_scaled_mad_ps=robust_sigma(g.ch192_alignment_residual),
                         slope=float(g.ch192_line_slope.iloc[0]),n=len(g)))
    groups=pd.DataFrame(rows)
    groups.to_csv(OUT/'finite_segment_clock_groups.csv',index=False)
    l=fixed.loc[fixed.channelID.eq(137)].set_index('event_id').loc[selected.event_id]
    r=fixed.loc[fixed.channelID.eq(150)].set_index('event_id').loc[selected.event_id]
    result=selected.copy()
    result['finite_segment']=l.segment.to_numpy()
    result['R_finite_segments']=(l.delta_t_ref_ps.to_numpy()+r.delta_t_ref_ps.to_numpy()
        -curve(137,result.Q137.to_numpy())-curve(150,result.Q150.to_numpy()))/2
    scales={(row.file_id,row.segment):row.clock_line_scaled_mad_ps for row in groups.itertuples()}
    result['clock_group_scale_ps']=[scales[(fi,sg)] for fi,sg in zip(result.file_id,result.finite_segment)]
    result.to_csv(OUT/'finite_segmentation_diagnostic.csv.gz',index=False)
    summary=dict(missed_resets=resets,original_selected=len(result),
        original_R=width(result.R_corrected),finite_segmentation_R=width(result.R_finite_segments),
        clock_quality_sensitivity={})
    for threshold in (1e3,1e4,1e5,1e6):
        good=result.clock_group_scale_ps<threshold
        summary['clock_quality_sensitivity'][str(threshold)]=dict(n=int(good.sum()),
            D=width(result.loc[good,'D_corrected']),R=width(result.loc[good,'R_finite_segments']))
    good=result.clock_group_scale_ps<1e6
    summary['rejected_groups']=groups.loc[groups.clock_line_scaled_mad_ps>=1e6].to_dict('records')
    summary['finite_segmentation_large_residuals']=int((result.R_finite_segments.abs()>1e6).sum())
    summary['original_large_residuals']=int((result.R_corrected.abs()>1e6).sum())
    write_json('clock_audit_report.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    with PdfPages(OUT/'clock_audit.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        fig.subplots_adjust(top=.83,bottom=.22,wspace=.3)
        stamp(fig,'Clock validation: failures hidden by a core-only plot')
        for field,label in [('R_corrected','Original segmentation'),('R_finite_segments','Finite-trigger segmentation')]:
            vals=np.maximum(abs(result[field]),1)
            axes[0].hist(np.log10(vals),bins=np.linspace(0,13,53),histtype='step',label=label)
        axes[0].set(xlabel='log10(|bar - mapped MCP residual| / ps)',ylabel='Events',yscale='log')
        axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
        axes[1].hist(result.loc[good,'R_finite_segments'],bins=np.linspace(-400,400,81),histtype='step',label=f'Clock-quality subset: N={int(good.sum())}')
        axes[1].set(xlabel='Bar - mapped MCP residual [ps]',ylabel='Events / 10 ps')
        w=summary['clock_quality_sensitivity']['1000000.0']['R']
        axes[1].set_title(f"Core sigma {w['sigma_ps']:.2f} ps; full sample SD {w['rms_about_mean_ps']:.2f} ps")
        axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
        fig.text(.08,.06,'Local diagnostic only: segment finite scope triggers across NaN gaps, then reject entire clock groups with\nline-residual scaled MAD >= 1 microsecond. This uses clock signals, not the SiPM timing residual.\nFrozen walk curves are unchanged. No production fix or final resolution claim is made.',fontsize=10)
        pdf.savefig(fig);fig.savefig(OUT/'clock_failure_diagnostic.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        fig.subplots_adjust(top=.84,bottom=.23)
        stamp(fig,'Diagnostic of a possible one-event matching offset')
        shiftframe=pd.DataFrame(shifts)
        for (fi,sg),g in shiftframe.groupby(['file_id','segment']):
            ax.plot(g.mcp_row_shift,g.scaled_mad_ps,'o-',label=f'{fi}, segment {sg}')
        ax.set(xlabel='Scope timestamp row shift (positive = preceding MCP row)',ylabel='Scaled MAD of trigger-line residual [ps]',yscale='log',xticks=[-2,-1,0,1,2])
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
        fig.text(.08,.07,'Row shifts are a diagnostic, not an applied correction. A large improvement at a one-row shift warrants\nan audit of event association upstream of clock calibration. This does not establish a safe remapping rule.',fontsize=10)
        pdf.savefig(fig);fig.savefig(OUT/'matching_offset_diagnostic.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
