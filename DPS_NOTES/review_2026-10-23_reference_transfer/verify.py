"""Check calibration-transfer algebra and the saved common-event results."""
import hashlib
import json

import numpy as np
import pandas as pd

from run import OUT, OLD, CACHE, CHANNELS, _quality_mask, frozen_fit, save


def main():
    r = json.loads((OUT / 'report.json').read_text())
    old = json.loads((OLD / 'report.json').read_text())
    source = pd.read_csv(OUT / 'selected_input.csv.gz')
    events = pd.read_csv(OUT / 'closure_events.csv.gz').set_index('event_id')
    assert source.event_id.nunique() == r['calibration_events'] == 1512
    assert len(events) == r['evaluation_events'] == 1456
    assert not events.index.duplicated().any()
    assert not source.duplicated(['event_id', 'channelID']).any()
    assert source.step1.nunique() == source.step2.nunique() == 1
    q = source.pivot(index='event_id', columns='channelID', values='energy')
    t = source.pivot(index='event_id', columns='channelID', values='time')
    m = source.pivot(index='event_id', columns='channelID', values='delta_t_ref_ps')
    ul = frozen_fit(137, q[137].to_numpy(), old)
    ur = frozen_fit(150, q[150].to_numpy(), old)
    ref = (m[137]+m[150]-ul-ur)/2
    shift = np.linspace(-10000, 10000, len(q))
    shifted = m.subtract(shift, axis=0)
    shifted_ref = (shifted[137]+shifted[150]-ul-ur)/2
    errors = []
    for ch in (234, 243):
        internal = ((t[ch]-t[137])+(t[ch]-t[150])+ul+ur)/2
        errors.append(float(np.max(np.abs(internal-(m[ch]-ref)))))
        assert errors[-1] < .001
        assert np.allclose(internal, shifted[ch]-shifted_ref, atol=.001, rtol=0)
        for mode in ('direct', 'transfer'):
            p = pd.read_csv(OUT / f'channel_{ch}_{mode}_profile.csv')
            fit = json.loads((OUT / f'channel_{ch}_{mode}_fit.json').read_text())
            x = 1/events[f'q_{ch}_adc']
            assert x.between(fit['inv_q_min'], fit['inv_q_max']).all()
            occupied = np.zeros(len(x), dtype=bool)
            for row in p.itertuples():
                occupied |= x.between(row.inv_q_bin_low, row.inv_q_bin_high).to_numpy()
            assert occupied.all()
            coefficients = np.polyfit(p.inv_q_mean, p.delta_t_ref_mean_ps, 2, w=1/p.delta_t_ref_sem_ps)
            grid = np.linspace(fit['inv_q_min'], fit['inv_q_max'], 100)
            assert np.allclose(np.polyval(coefficients, grid), np.polyval(fit['coefficients_ps'], grid), atol=1e-7, rtol=0)
            expected = m.loc[events.index, ch]-np.polyval(fit['coefficients_ps'], x)
            assert np.allclose(events[f'dut_{ch}_{mode}_mcp'], expected, atol=1e-7, rtol=0)
    boot = pd.read_csv(OUT / 'paired_block_bootstrap.csv')
    assert len(boot) == 400
    for name, ci in r['intervals'].items():
        finite = boot[name].dropna()
        assert len(finite) == ci['successful'] >= 380
        assert np.allclose(np.percentile(finite, [16, 84]), [ci['p16'], ci['p84']], atol=1e-8)
    assert np.allclose(boot.pair_transfer-boot.pair_direct, boot.pair_transfer_minus_direct)
    assert abs(r['original_1461_event_widths_ps']['after']-64.69952052230771) < 1e-6
    checks = json.loads((OUT/'verification.json').read_text())
    checks.update(mcp_shift_invariance=True, maximum_internal_target_discrepancy_ps=max(errors),
                  independent_polynomial_refit=True, saved_corrections_recomputed=True,
                  bootstrap_intervals_recomputed=True)
    if CACHE.exists():
        cache = pd.read_csv(CACHE)
        cache = cache.loc[cache.step1.eq(old['step1']) & cache.step2.eq(old['step2'])]
        sets = []
        for ch in CHANNELS:
            c = old['channels'][str(ch)]
            f = cache.loc[cache.channelID.eq(ch)]
            keep = _quality_mask(f) & f.energy.ge(c['q_cut_adc'])
            if ch in (137, 150):
                keep &= f.energy.between(c['q_min_adc'], c['q_max_adc'])
            sets.append(set(f.loc[keep, 'event_id']))
        assert set.intersection(*sets) == set(source.event_id)
        a = cache.set_index(['event_id', 'channelID']).loc[source.set_index(['event_id', 'channelID']).index]
        b = source.set_index(['event_id', 'channelID'])
        for col in ('energy', 'time', 'delta_t_ref_ps'):
            assert np.allclose(a[col], b[col], atol=1e-8, rtol=0)
        assert (a.clock_line_scaled_mad_ps <= 1000).all()
        checks['cache_selection_reproduced'] = True
        checks['source_cache_sha256'] = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    save('verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
