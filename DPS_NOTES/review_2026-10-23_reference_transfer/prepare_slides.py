"""Render slide-sized closure summaries from saved numerical results."""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.special import ndtr

from run import OUT, COLORS, export, histogram


def main():
    report = json.loads((OUT / "report.json").read_text())
    events = pd.read_csv(OUT / "closure_events.csv.gz")
    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
    table = histogram(ax, events, report['stats'], ['pair_raw', 'pair_direct', 'pair_transfer'],
                      ['Both raw', 'Both calibrated by MCP', 'Ref by MCP; DUT by ref'],
                      list(COLORS.values()), r'$t_{\rm DUT}-t_{\rm ref}$ [ps], centered')
    table.to_csv(OUT / 'pair_closure_histogram.csv', index=False)
    ax.set_title('Module under test minus reference bar', fontsize=13)
    fig.suptitle(f"CMS Preliminary Phase 2 | T1=3, T2=60 DAC | Same {len(events):,} events", fontsize=14)
    export(fig, 'closure_timing')

    # Scale the observable, fitted parameters and bin widths together; no refit.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    estimates = {}
    for ax, title, keys, labels, colors, name in (
        (axes[0], 'Reference bar', ['ref_lr_raw', 'ref_lr_corrected'],
         ['Before', 'MCP calibrated'], [COLORS['raw'], COLORS['direct']], 'ref'),
        (axes[1], 'Module under test', ['dut_lr_raw', 'dut_lr_direct', 'dut_lr_transfer'],
         ['Before', 'Direct MCP', 'Via reference'], list(COLORS.values()), 'dut')):
        edges = np.arange(-200, 205, 5)
        table = {'bin_low_ps': edges[:-1], 'bin_high_ps': edges[1:]}
        for key, label, color in zip(keys, labels, colors):
            s, ci = report['stats'][key], report['intervals'][key]
            x = events[key].to_numpy()/2
            median = s['median_ps']/2
            lo, hi, mu, sigma = [s[k]/2 for k in ('fit_low_ps', 'fit_high_ps', 'fit_mean_ps', 'sigma_ps')]
            table[key] = np.histogram(x-median, edges)[0]
            ax.hist(x-median, bins=edges, histtype='step', lw=1.6, color=color,
                    label=f"{label}: {sigma:.1f} ps")
            grid = np.linspace(lo, hi, 300)
            norm = ndtr((hi-mu)/sigma)-ndtr((lo-mu)/sigma)
            count = np.sum((x >= lo) & (x <= hi))
            y = count*5/(norm*sigma*np.sqrt(2*np.pi))*np.exp(-.5*((grid-mu)/sigma)**2)
            ax.plot(grid-median, y, color=color, ls='--', lw=1)
            estimates[key] = dict(sigma_bar_estimate_ps=sigma, p16_ps=ci['p16']/2,
                                  p84_ps=ci['p84']/2, conditional_bootstrap_std_ps=ci['std_ps']/2)
        pd.DataFrame(table).to_csv(OUT / f'{name}_half_difference_histogram.csv', index=False)
        ax.set(title=title, xlim=(-175, 175), xlabel=r'$(t_L-t_R)/2$ [ps], centered', ylabel='Events / 5 ps')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=.2)
    fig.suptitle(f"CMS Preliminary Phase 2 | T1=3, T2=60 DAC | Same {len(events):,} events", fontsize=14)
    export(fig, 'bar_resolution_check')
    estimates['combined_transfer_estimate_ps'] = report['lr_core_proxy_ps']['transfer']
    estimates['interpretation'] = 'Half-difference core width estimates bar-average resolution only for negligible within-bar covariance and position spread. Quadrature also assumes negligible inter-bar covariance.'
    (OUT / 'bar_resolution_estimates.json').write_text(json.dumps(estimates, indent=2)+'\n')

    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    fig.subplots_adjust(left=.01, right=.99, top=.88, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                   facecolor='white', edgecolor=color, linewidth=2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=13, linespacing=1.4)

    def arrow(start, end, color):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=color, lw=2))

    box(.02, .34, .19, .29, 'Clock-aligned\nMCP pulse time', COLORS['direct'])
    box(.34, .63, .25, .29, 'Reference bar\nCalibrate left and right', COLORS['transfer'])
    box(.71, .63, .27, .29, 'Module under test\nCalibrate against reference', COLORS['transfer'])
    box(.71, .04, .27, .29, 'Module under test\nCalibrate directly with MCP', COLORS['direct'])
    arrow((.21, .59), (.34, .76), COLORS['transfer'])
    arrow((.59, .77), (.71, .77), COLORS['transfer'])
    ax.text(.65, .91, r'$t_{\rm ref}$', ha='center', fontsize=15, color=COLORS['transfer'])
    arrow((.21, .38), (.71, .19), COLORS['direct'])
    ax.text(.45, .13, 'Direct calibration for closure', ha='center', color=COLORS['direct'], fontsize=12)
    ax.annotate('', xy=(.845, .62), xytext=(.845, .34),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.4, linestyle='--'))
    ax.text(.825, .48, 'Compare on\nthe same events', ha='right', va='center', fontsize=11)
    fig.suptitle('CMS Preliminary Phase 2 | Calibration transfer and closure test', fontsize=14)
    export(fig, 'calibration_paths')

    s = report['stats']
    ci = report['intervals']
    values = {
        'TransferEvents': str(report['evaluation_events']),
        'TransferBefore': f"{s['pair_raw']['sigma_ps']:.1f}",
        'TransferRefOnly': f"{s['pair_ref_only']['sigma_ps']:.1f}",
        'TransferDirect': f"{s['pair_direct']['sigma_ps']:.1f}",
        'TransferViaRef': f"{s['pair_transfer']['sigma_ps']:.1f}",
        'TransferDirectLow': f"{ci['pair_direct']['p16']:.1f}",
        'TransferDirectHigh': f"{ci['pair_direct']['p84']:.1f}",
        'TransferViaLow': f"{ci['pair_transfer']['p16']:.1f}",
        'TransferViaHigh': f"{ci['pair_transfer']['p84']:.1f}",
        'TransferDelta': f"{s['pair_transfer']['sigma_ps']-s['pair_direct']['sigma_ps']:.2f}",
        'TransferDeltaLow': f"{ci['pair_transfer_minus_direct']['p16']:.2f}",
        'TransferDeltaHigh': f"{ci['pair_transfer_minus_direct']['p84']:.2f}",
        'TransferLRProxy': f"{report['lr_core_proxy_ps']['transfer']:.1f}",
        'TransferRefBarEstimate': f"{s['ref_lr_corrected']['sigma_ps']/2:.1f}",
        'TransferDUTBarEstimate': f"{s['dut_lr_transfer']['sigma_ps']/2:.1f}",
    }
    text = '\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in values.items())+'\n'
    (OUT / 'closure_values.tex').write_text(text)


if __name__ == '__main__':
    main()
