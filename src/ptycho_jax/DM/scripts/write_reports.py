"""Regenerate implementation phase reports from measured checks and completed runs."""
from pathlib import Path
import json,re,shutil,xml.etree.ElementTree as ET


def write_reports():
    root=Path(__file__).parents[1];project=root.parents[2]
    reports=root/'reports'
    checks=json.loads((reports/'measurements.json').read_text())
    tree=ET.parse(reports/'tests.xml')
    tests=tree.findall('.//testcase')
    total=len(tests);failures=sum(bool(t.findall('failure') or t.findall('error')) for t in tests)
    run=project/'results/nanomax_dm_ad_comparison'
    results=json.loads((run/'summary.json').read_text())
    rates=json.loads((run/'original_ad_rates_summary.json').read_text())
    bench=project/'results/dm_benchmark'
    notes={
      0:'The reference audit and complex128 golden archive are complete. General full complex AD matrices cannot map back to the restricted nonnegative diagonal DM weights and are explicitly rejected.',
      1:'Stages use the exact existing object-patch shift, rather than a shifted-probe approximation. Value and gradient tests cover all fitted blocks. Production arrays remain float32/complex64.',
      2:'Crop origins and fractions are fixed arrays. Enlarged patches are validated before compilation; no persistent per-pixel index arrays are stored. On-demand broadcast scatter is used, with an independent scan implementation available.',
      3:'Hermitian adjoints are independently checked with linear_transpose. The fractional object normal is non-diagonal; its exact diagonal is used only as a preconditioner. The probe normal is diagonal for all positions.',
      4:'A shared multimode amplitude rescaling preserves masked Fourier values. Zero predicted amplitudes use a deterministic first-channel seed. This singular projection choice is documented and no noise relaxation is implemented.',
      5:'Object solves are proximal CG (direct diagonal for integers); probe and nonnegative amplitude solves are analytical. Frozen blocks remain bit-identical. The guide’s assertion that 20 alternating sweeps must recover all bilinear factors to 1e-6 is not certified; block stationarity and measured nonlinear convergence are reported instead.',
      6:'Compensated object/mode gauges preserve fields. Inactive zero-weight modes freeze and are logged. Optional constant-weight mode canonicalization and the non-exact per-shot rotation heuristic are not implemented or claimed as tested. Production DM orthogonalization is off.',
      7:'The supplied probe is the first QR direction; higher modes are seeded smooth perturbations. Masked counts are calibrated exactly. Fixed-weight synthetic A/B instead use known generating amplitudes so the specified inverse problem is physically correct.',
      8:'Beta=1 and general-beta paths, stateless second overlap, traced schedules, NumPy reference, scan blocks and donation are checked. Production logs every 10 iterations and writes parameter checkpoints every 50.',
      9:'M1, update norm, current overlap-model MSE/R, mode-weight statistics and inactive modes are recorded each iteration. Raw DM iterates are not mistaken for the fitted overlap model at handoff.',
      10:'A–F and frozen-weight controls were run on 100 shots of 64x64 frames, 200 DM iterations plus 50 AD epochs from warm and naive starts. Targets are unchanged and limitations are explicit below.',
      11:'The last overlap factors transfer exactly to the full complex AD matrix parameterization with balanced scale. Positions are fixed injected arrays and absent from optimizer leaves. Both nominal and +30% NanoMAX cases and equal-AD-budget controls were executed.',
      12:'Full versus chunked steps and schedule traces are tested. Four virtual CPU devices reproduce single-device results. Production NanoMAX uses one GPU because 169 shots are not divisible by two; no measurements are silently dropped. Profiling-operation percentage attribution is not implemented.',
    }
    numerical={
      0:[('Reference intensity',checks['reference_intensity_relative_error'],'1e-12')],
      1:[('Reference intensity',checks['reference_intensity_relative_error'],'1e-12')],
      3:[(k,v,'1e-12') for k,v in checks.items() if 'dot_' in k or 'transpose_' in k],
      4:[(k,v,'1e-12') for k,v in checks.items() if k.startswith('PF_')],
      5:[(k,v,'1e-8') for k,v in checks.items() if 'gradient_ratio' in k],
      6:[('Gauge field error',checks['gauge_field_relative_error'],'1e-12')],
      9:[('MSE handoff error',checks['handoff_MSE_relative_error'],'1e-12')],
      11:[('MSE handoff error',checks['handoff_MSE_relative_error'],'1e-12')],
    }
    for phase in range(13):
        selected=[t for t in tests if phase in [int(x) for x in re.findall(r'C(\d+)_',t.attrib['name'])]]
        text=[f'# Phase {phase:02d} implementation report','',notes[phase],'',
              f'Current package suite: {total} tests, {failures} failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).','',
              '| Implemented check/test | Status |','| --- | --- |']
        for t in selected:
            status='FAIL' if t.findall('failure') or t.findall('error') else 'PASS'
            text.append(f"| `{t.attrib['name']}` | {status} |")
        if not selected:text.append('| Supporting integrated checks are recorded in the other phase reports | See notes |')
        if phase in numerical:
            text+=['','| Measured quantity | Relative value | Requirement |','| --- | --- | --- |']
            for name,value,tol in numerical[phase]:text.append(f'| {name} | {value:.5g} | < {tol} |')
        if phase==10:
            ladder=(project/'results/dm_validation_ladder_final/ladder_report.md').read_text()
            text+=['',ladder,'',
                'A/B meet the R targets and object errors <1e-2 after using the specified known fixed amplitudes. '
                'C meets its R target but coherence overlap ≈0.88 misses 0.95. D meets its R target but not every matched-mode weight correlation exceeds 0.8. '
                'E remains above 1.1× the generating-parameter noise floor after 200 iterations. F per-shot weights improve both fit and object error versus frozen mean weights. '
                'A/B are already near noiseless truth; a fixed-rate 50-epoch AD trial slightly perturbs their approximately 1e-6 MSE rather than improving it. '
                'Translation alignment is deferred. Synthetic generator seeds are deterministic; GPU scatter-reduction order can cause small run-to-run floating-point changes in nonlinear DM.']
        if phase==11:
            text+=['','| NanoMAX probe defocus | DM MSE | Handoff relative error | Conservative AD final | Original-rate AD final |','| --- | --- | --- | --- | --- |']
            for a,b in zip(results,rates):text.append(f"| {a['initialization_defocus_m']*1e3:.3f} mm | {a['dm_mse']:.6g} | {a['handoff_relative_difference']:.4g} | {a['dm_ad']['final_loss']:.6g} | {b['warm']['final_loss']:.6g} |")
            text+=['','The +30% case remains visually poor and high-error; lower loss does not establish reliable specimen recovery. '
                    'AD controls share rates/epochs but DM→AD has additional DM compute.']
        if phase==12:
            text+=['',(bench/'benchmark.md').read_text(),'',
                'Four-device relative errors: `'+json.dumps(json.loads((bench/'sharding.json').read_text())['relative_errors'])+'`.','',
                'The 169×2×512×512 complex64 iterate is 676 MiB. Sampled production process usage was about 9428 MiB, including cached compiled workspaces. '
                'This sample is not a measured peak. Full/chunked variants retain the full iterate; the current chunking only bounds backprojection intermediates. '
                'The alternate diagonal-versus-CG probe benchmark in the guide is inapplicable here: the exact-compatible object-shift model has a diagonal probe solve and a CG object solve.']
        text+=['','All departures and numerical settings are documented in [decisions](../docs/decisions.md).','']
        (reports/f'phase_{phase:02d}.md').write_text('\n'.join(text))
    shutil.copytree(project/'results/dm_validation_ladder_final',run/'validation_ladder',dirs_exist_ok=True)
    shutil.copytree(bench,run/'benchmark',dirs_exist_ok=True)
    shutil.copytree(reports,run/'implementation_reports',dirs_exist_ok=True)
    print(f'Wrote 13 phase reports from {total} tests; copied validation results into {run}.')

if __name__=='__main__':write_reports()
