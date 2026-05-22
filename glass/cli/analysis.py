import glob
import os
import warnings

import click


@click.command(
    "plot_loss",
    help="""
Plot training loss curves from TensorBoard event files.

MODELS_DIR is the directory containing model subdirectories (each with version*/events* files).
Defaults to ./models/.

EXAMPLES:

  # Plot all models in ./models/
  glass plot_loss

  # Plot models in a specific directory
  glass plot_loss /path/to/demo_Si/models

  # Custom output file and y-axis range
  glass plot_loss --output my_plot.pdf --ylim 0.1 2.0

  # Only plot specific models
  glass plot_loss --model Si_1.5_2.5_3.5 --model Si_2.0_3.0

""",
)
@click.argument("models_dir", type=click.Path(exists=True), default="./models/")
@click.option(
    "--model",
    "model_filter",
    multiple=True,
    help="Plot only these model(s). Can be repeated. Default: all.",
)
@click.option(
    "--output",
    type=str,
    default="score_LC_all.pdf",
    show_default=True,
    help="Output PDF filename.",
)
@click.option(
    "--ylim",
    type=(float, float),
    default=(0.2, 3.2),
    show_default=True,
    help="Y-axis limits: MIN MAX.",
)
@click.option(
    "--step",
    type=int,
    default=20,
    show_default=True,
    help="Downsample factor for plotted points.",
)
@click.option(
    "--figsize",
    type=(float, float),
    default=(10.0, 6.0),
    show_default=True,
    help="Figure size: W H.",
)
def plot_loss(models_dir, model_filter, output, ylim, step, figsize):
    """Plot loss curves from TensorBoard logs in MODELS_DIR."""
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import seaborn as sns
    from tensorboard.backend.event_processing import event_accumulator

    warnings.filterwarnings("ignore")

    params = {
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
        "axes.linewidth": 1.5,
    }
    pylab.rcParams.update(params)
    sns.set_context("talk")

    def get_lc(model_path):
        events = sorted(glob.glob(os.path.join(model_path, "version*", "events*")))
        epoch, loss = [], []
        for event_file in events:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            try:
                epoch += [int(x.value) for x in ea.Scalars("epoch")]
                loss += [x.value for x in ea.Scalars("train_loss")]
            except KeyError:
                pass
        return {"epoch": epoch, "loss": loss}

    all_models = sorted(
        f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))
    )
    models = (
        [m for m in all_models if m in model_filter] if model_filter else all_models
    )

    if not models:
        raise click.ClickException(f"No model directories found in {models_dir}")

    click.echo(f"Found models: {models}")

    colors = sns.color_palette("deep")
    plt.figure(figsize=figsize)
    for i, model in enumerate(models):
        data = get_lc(os.path.join(models_dir, model))
        if not data["epoch"]:
            click.echo(f"  Warning: no loss data found for {model}, skipping.")
            continue
        plt.plot(
            data["epoch"][::step],
            data["loss"][::step],
            ".",
            label=model,
            color=colors[i % len(colors)],
        )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Saved plot to {output}")


@click.command(
    "write_spec_feature",
    help="""
Compute and write structural/spectral features for denoised or reference structures.

Features computed per structure: PDF, ADF, XRD, ND, EXAFS, XANES.
Each structure's features are saved as a single JSON file in OUTDIR.

MODES:

  denoise   -- reads *_final.xyz from denoise_logs/{denoise_tag}/{system}-*/init_*/
  reference -- reads {system}_*.xyz from a user-specified --atoms-path

EXAMPLES:

  # Denoise mode
  glass write_spec_feature --mode denoise --system Si \\
      --denoise-tag "unconditional/Si-1.5_2.5_3.5" \\
      --exafs-model ./models/Si_exafs.ckpt \\
      --xanes-model ./models/Si_xanes.ckpt

  # Reference mode
  glass write_spec_feature --mode reference --system Si \\
      --atoms-path /path/to/reference/amorph_Si_216 \\
      --exafs-model ./models/Si_exafs.ckpt \\
      --xanes-model ./models/Si_xanes.ckpt \\
      --outdir results/reference
""",
)
@click.option(
    "--mode",
    type=click.Choice(["denoise", "reference"]),
    default="denoise",
    show_default=True,
    help="Input source mode.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--denoise-tag",
    type=str,
    default="*",
    show_default=True,
    help="Glob tag under denoise_logs/ for denoise mode.",
)
@click.option(
    "--denoise-root",
    type=str,
    default="denoise_logs",
    show_default=True,
    help="Root directory for denoised outputs.",
)
@click.option(
    "--atoms-path",
    type=str,
    default=None,
    help="Directory with {system}_*.xyz files (reference mode only).",
)
@click.option(
    "--outdir",
    type=str,
    default="results",
    show_default=True,
    help="Output directory for the combined JSON file.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: {mode}_{system}_spectra.json.",
)
@click.option(
    "--exafs-model",
    type=str,
    required=True,
    help="Path to EXAFS LitSpecNet checkpoint.",
)
@click.option(
    "--xanes-model",
    type=str,
    required=True,
    help="Path to XANES LitSpecNet checkpoint.",
)
@click.option(
    "--qmin",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum q value for XRD/ND.",
)
@click.option(
    "--qmax",
    type=float,
    default=20.0,
    show_default=True,
    help="Maximum q value for XRD/ND.",
)
@click.option(
    "--qstep",
    type=float,
    default=0.1,
    show_default=True,
    help="Q step size for XRD/ND.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Device for spectral model inference.",
)
def write_spec_feature(
    mode,
    system,
    denoise_tag,
    denoise_root,
    atoms_path,
    outdir,
    output,
    exafs_model,
    xanes_model,
    qmin,
    qmax,
    qstep,
    device,
):
    """Compute PDF, ADF, XRD, ND, EXAFS, XANES and write to JSON."""
    import json
    import numpy as np
    import torch
    from ase.io import read
    from ase.data import chemical_symbols
    from collections import defaultdict
    from glass.lit.modules import DifferentiableRDF, DifferentiableADF, LitSpecNet
    from glass.utils.atoms import initialize_atoms
    from glass.nn import periodic_radius_graph
    from debyecalculator import DebyeCalculator

    q_vals = [qmin, qmax, qstep]

    def _compute_iq(pos, species, Z_list):
        import warnings

        pos_np = pos.detach().cpu().numpy()
        species_indices = species.argmax(dim=1).detach().cpu().numpy()
        elements = np.array(Z_list)[species_indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            calc = DebyeCalculator(
                qmin=q_vals[0],
                qmax=q_vals[1],
                qstep=q_vals[2],
                qdamp=0.04,
                rmin=0,
                rmax=20,
                rstep=0.01,
                rthres=0.0,
                biso=1.5,
                device=device,
                rad_type="xray",
            )
            q, iq_xrd = calc.iq((elements, pos_np))
            calc.update_parameters(rad_type="neutron")
            _, iq_nd = calc.iq((elements, pos_np))
        return q, iq_xrd, iq_nd

    def _get_spec(spec_net, pos, species, cell, atomic_numbers, cutoff=5):
        edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        ys = spec_net.ema_model(species, edge_index, edge_attr)
        species_indices = torch.argmax(species, dim=1).tolist()
        element_indices = defaultdict(list)
        for i, idx in enumerate(species_indices):
            element_indices[atomic_numbers.get(idx, f"elem_{idx}")].append(i)
        return {
            elem: ys[torch.tensor(idxs, device=ys.device)].mean(dim=0).tolist()
            for elem, idxs in element_indices.items()
        }

    def _write_into_dict(spec_type, x_values, y_types, y_values, atomic_numbers):
        spec_dict = {"bins": x_values.tolist()}
        if spec_type == "PDF":
            for idx, (a, b) in enumerate(y_types):
                spec_dict[f"{atomic_numbers[a]}-{atomic_numbers[b]}"] = y_values[
                    idx
                ].tolist()
        elif spec_type == "ADF":
            for idx, (a, b, c) in enumerate(y_types):
                spec_dict[
                    f"{atomic_numbers[a]}-{atomic_numbers[b]}-{atomic_numbers[c]}"
                ] = y_values[idx].tolist()
        return spec_dict

    def _process(atoms_file):
        atoms = read(atoms_file, "-1")
        Z_list, species, pos, cell = initialize_atoms(atoms)
        atomic_numbers = {i: chemical_symbols[Z] for i, Z in enumerate(Z_list)}

        rdf_model = DifferentiableRDF(cutoff=8.0, bin_size=100, sigma=0.15)
        rdf_bins, rdf_hist, pair_types = rdf_model(pos, species, cell)

        adf_model = DifferentiableADF(
            cutoff=3.0,
            angle_bins=100,
            angle_range=[0, np.pi],
            sigma=0.1,
            normalize=False,
        )
        adf_bins, adf_hist, triplet_types = adf_model(pos, species, cell)

        q, iq_xrd, iq_nd = _compute_iq(pos, species, Z_list)

        exafs_net = LitSpecNet.load_from_checkpoint(exafs_model)
        exafs_net.ema_model.to(device)
        exafs = _get_spec(exafs_net, pos, species, cell, atomic_numbers)

        xanes_net = LitSpecNet.load_from_checkpoint(xanes_model)
        xanes_net.ema_model.to(device)
        xanes = _get_spec(xanes_net, pos, species, cell, atomic_numbers)

        return {
            "PDF": _write_into_dict(
                "PDF", rdf_bins, pair_types, rdf_hist, atomic_numbers
            ),
            "ADF": _write_into_dict(
                "ADF", adf_bins, triplet_types, adf_hist, atomic_numbers
            ),
            "XRD": {"q": q.tolist(), "xrd": iq_xrd.tolist()},
            "ND": {"q": q.tolist(), "nd": iq_nd.tolist()},
            "EXAFS": exafs,
            "XANES": xanes,
        }

    # --- collect xyz files ---
    if mode == "denoise":
        search = os.path.join(denoise_root, denoise_tag, f"{system}_*", "*_final.xyz")
        click.echo(f"Searching: {search}")
        xyz_files = sorted(glob.glob(search))
    else:
        if not atoms_path:
            raise click.ClickException("--atoms-path is required for reference mode.")
        search = os.path.join(atoms_path, f"{system}_*.xyz")
        click.echo(f"Searching: {search}")
        xyz_files = sorted(glob.glob(search))

    if not xyz_files:
        raise click.ClickException(f"No .xyz files found under: {search}")

    click.echo(f"Found {len(xyz_files)} structure(s).")
    os.makedirs(outdir, exist_ok=True)

    if output:
        out_filename = output
    elif mode == "denoise":
        tag = denoise_tag.replace("/", "_")
        out_filename = f"denoise_{system}_{tag}_spectra.json"
    else:
        out_filename = f"reference_{system}_spectra.json"
    out_path = os.path.join(outdir, out_filename)

    combined = {}
    for xyz_file in xyz_files:
        if mode == "denoise":
            # e.g. denoise_logs/unconditional/Si-1.5_2.5_3.5/Si_2.0_01/00_final.xyz
            parts = xyz_file.replace("_final.xyz", "").split(os.sep)
            run_id = parts[-1]  # 00
            struct_id = parts[-2]  # Si_2.0_01
            model_id = parts[-3]  # Si-1.5_2.5_3.5
            key = f"{model_id}/{struct_id}/{run_id}"
        else:
            key = os.path.basename(xyz_file).replace(".xyz", "")

        combined[key] = _process(xyz_file)
        click.echo(f"  {key} done")

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    click.echo(f"Saved: {out_path}")


@click.command(
    "calc_metrics",
    help="""
Compute error and diversity metrics comparing denoised structures to reference.

Reads feature JSON files produced by `write_spec_feature`, compares denoised spectra
against a reference master JSON, and saves results to a single JSON per denoise folder.

Output structure per group (denoise_label x ref_label):
  error  -- mean normalized error vs reference
  score  -- error - ref_div (lower is better)

EXAMPLES:

  glass calc_metrics \\
      --ref-master-json final_data_dir/a-Si_ref_stats.json \\
      --system Si \\
      --denoise-folder unconditional \\
      --denoise-label 1.5_2.5_3.5 \\
      --ref-label 1.5 --ref-label 2.0 --ref-label 2.5 --ref-label 3.0 --ref-label 3.5 \\
      --outdir final_data_dir

  # Multiple denoise densities
  glass calc_metrics \\
      --ref-master-json final_data_dir/a-Si_ref_stats.json \\
      --system Si \\
      --denoise-folder unconditional \\
      --denoise-label 1.5 --denoise-label 2.5 --denoise-label 3.5 \\
      --ref-label 1.5 --ref-label 2.0 --ref-label 2.5 \\
      --outdir final_data_dir
""",
)
@click.option(
    "--denoise-json",
    type=str,
    required=True,
    help="Combined denoise spectra JSON from write_spec_feature.",
)
@click.option(
    "--ref-master-json",
    type=str,
    required=True,
    help="Reference master stats JSON from build_ref_stats.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--denoise-label",
    "denoise_label_list",
    multiple=True,
    required=True,
    help="Label(s) identifying the denoised model/condition. Can be repeated.",
)
@click.option(
    "--ref-label",
    "ref_label_list",
    multiple=True,
    required=True,
    help="Label(s) identifying the reference condition. Can be repeated.",
)
@click.option(
    "--outdir",
    type=str,
    default="final_data_dir",
    show_default=True,
    help="Output directory for metrics JSON.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: a-{system}_metrics.json.",
)
@click.option(
    "--spectrum-types",
    "spectrum_types",
    multiple=True,
    default=("XRD", "ND", "XANES", "EXAFS", "PDF", "ADF"),
    show_default=True,
    help="Spectrum types to evaluate. Can be repeated.",
)
@click.option(
    "--exafs-slice",
    type=(int, int),
    default=(40, 280),
    show_default=True,
    help="EXAFS index slice: START END.",
)
@click.option(
    "--xrd-nd-npts",
    type=int,
    default=100,
    show_default=True,
    help="Number of points to use for XRD/ND spectra.",
)
def calc_metrics(
    denoise_json,
    ref_master_json,
    system,
    denoise_label_list,
    ref_label_list,
    outdir,
    output,
    spectrum_types,
    exafs_slice,
    xrd_nd_npts,
):
    """Compute error and diversity metrics for denoised vs reference structures."""
    import json
    import numpy as np

    def _is_axis_key(spec_type, key, idx):
        if spec_type in ["PDF", "ADF"] and key == "bins":
            return True
        if spec_type in ["XRD", "ND"] and key == "q":
            return True
        if spec_type not in ["EXAFS", "XANES"] and idx == 0:
            return True
        return False

    def _trim(spec_type, values):
        arr = np.asarray(values, dtype=float)
        if spec_type == "EXAFS":
            arr = arr[exafs_slice[0] : exafs_slice[1]]
        if spec_type in ["XRD", "ND"]:
            arr = arr[:xrd_nd_npts]
        return arr.tolist()

    def _extract_spectra(entry, spec_type):
        """Extract trimmed spectra vectors from a single JSON entry."""
        spectra_by_key = {}
        if spec_type not in entry:
            return spectra_by_key
        for idx, (key, values) in enumerate(entry[spec_type].items()):
            if _is_axis_key(spec_type, key, idx):
                continue
            if spec_type in ["EXAFS", "XRD", "ND"]:
                values = _trim(spec_type, values)
            spectra_by_key[key] = values
        return spectra_by_key

    def _mean_normalized_error(pred, ref, norm_factor):
        pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
        err = np.abs(pred - ref).mean()
        return float(err / norm_factor) if norm_factor > 0 else float(err)

    with open(denoise_json) as f:
        all_denoise = json.load(f)
    with open(ref_master_json) as f:
        ref_master = json.load(f)

    os.makedirs(outdir, exist_ok=True)
    out_filename = output or f"a-{system}_metrics.json"
    out_path = os.path.join(outdir, out_filename)

    out = {
        "meta": {
            "system": system,
            "denoise_json": denoise_json,
            "ref_master_json": ref_master_json,
            "denoise_label_list": list(denoise_label_list),
            "ref_label_list": list(ref_label_list),
            "spectrum_types": list(spectrum_types),
            "exafs_slice": list(exafs_slice),
            "xrd_nd_npts": xrd_nd_npts,
        },
        "groups": {},
    }

    for denoise_label in denoise_label_list:
        click.echo(f"\n=== denoise_label: {denoise_label} ===")

        # keys: "{system}-{denoise_label}/{struct_id}/{run_id}"
        model_prefix = f"{system}-{denoise_label}/"
        model_entries = {
            k: v for k, v in all_denoise.items() if k.startswith(model_prefix)
        }
        click.echo(f"  Found {len(model_entries)} runs for {model_prefix}")

        for ref_label in ref_label_list:
            group_key = f"denoise_label={denoise_label} ref_label={ref_label}"
            ref_block = ref_master["groups"][ref_label]["stats"]

            # filter entries matching this ref_label in the struct part
            # key format: {system}-{denoise_label}/{system}_{ref_label}_{idx}/{run_id}
            struct_prefix = f"{system}_{ref_label}_"
            matching = {
                k: v
                for k, v in model_entries.items()
                if os.path.basename(os.path.dirname(k)).startswith(struct_prefix)
            }

            click.echo(f"  ref_label={ref_label}: {len(matching)} run(s) found")

            # group by struct_id -> list of run entries
            structs = {}
            for k, v in matching.items():
                parts = k.split("/")
                struct_id, run_id = parts[-2], parts[-1]
                structs.setdefault(struct_id, {})[run_id] = v

            grp = {
                "samples": {
                    struct_id: {"runs": runs} for struct_id, runs in structs.items()
                },
                "stats": {},
            }

            # aggregate stats per spectrum type
            for spec_type in spectrum_types:
                mean_ref_all = ref_block[spec_type]["mean_ref_spec"]
                ref_div = ref_block[spec_type]["ref_div"]
                norm_factor = ref_block[spec_type]["norm_factor"]

                per_key_sample_means = {}
                n_samples_with_runs, n_runs_total = 0, 0

                for struct_id, runs in structs.items():
                    run_vecs_by_key = {}
                    for run_id, entry in runs.items():
                        spec = _extract_spectra(entry, spec_type)
                        for key, vec in spec.items():
                            run_vecs_by_key.setdefault(key, []).append(vec)
                    if not run_vecs_by_key:
                        continue
                    n_samples_with_runs += 1
                    n_runs_total += len(runs)
                    for key, vecs in run_vecs_by_key.items():
                        per_sample_mean = np.mean(np.asarray(vecs, dtype=float), axis=0)
                        per_key_sample_means.setdefault(key, []).append(per_sample_mean)

                error_values, score_values = {}, {}
                for key, vecs in per_key_sample_means.items():
                    if key not in mean_ref_all:
                        continue
                    mean_denoise = np.mean(np.asarray(vecs, dtype=float), axis=0)
                    err = _mean_normalized_error(
                        mean_denoise, mean_ref_all[key], norm_factor
                    )
                    div = float(ref_div.get(key, 0.0))
                    error_values[key] = float(err)
                    score_values[key] = float(err - div)

                grp["stats"][spec_type] = {
                    "norm_factor": float(norm_factor),
                    "ref_div": {k: float(v) for k, v in ref_div.items()},
                    "n_samples_with_runs": int(n_samples_with_runs),
                    "n_runs": int(n_runs_total),
                    "n_keys": int(len(per_key_sample_means)),
                    "error": error_values,
                    "score": score_values,
                }
                avg_error = (
                    float(np.mean(list(error_values.values())))
                    if error_values
                    else float("nan")
                )
                avg_div = (
                    float(np.mean(list(ref_div.values()))) if ref_div else float("nan")
                )
                avg_score = (
                    float(np.mean(list(score_values.values())))
                    if score_values
                    else float("nan")
                )
                click.echo(
                    f"    {spec_type}: n_keys={len(per_key_sample_means)} "
                    f"n_samples={n_samples_with_runs} n_runs={n_runs_total} "
                    f"norm={norm_factor:.4g} | "
                    f"avg_error={avg_error:.4f}  avg_div={avg_div:.4f}  avg_score={avg_score:.4f}"
                )
            out["groups"][group_key] = grp

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    click.echo(f"\nSaved: {out_path}")


@click.command(
    "build_ref_stats",
    help="""
Build the reference master stats JSON from a reference spectra JSON.

Reads the single JSON produced by `write_spec_feature --mode reference`,
groups structures by a regex-extracted value (e.g. density, temperature),
and computes per-group norm_factor, ref_div, and mean_ref_spec for each
spectrum type. Output is used as input to `calc_metrics`.

EXAMPLES:

  glass build_ref_stats \\
      --input results/reference_Si_spectra.json \\
      --system Si \\
      --atoms-path /path/to/reference/amorph_Si_216 \\
      --outdir final_data_dir

  # Custom grouping (e.g. by temperature instead of density)
  glass build_ref_stats \\
      --input results/reference_Si_spectra.json \\
      --group-var temperature \\
      --group-regex "_T(\\\\d+)_" \\
      --outdir final_data_dir
""",
)
@click.option(
    "--input",
    "input_json",
    type=str,
    required=True,
    help="Reference spectra JSON from write_spec_feature --mode reference.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--atoms-path",
    type=str,
    default=None,
    help="Directory with reference .xyz files for storing atoms info (optional).",
)
@click.option(
    "--outdir",
    type=str,
    default="final_data_dir",
    show_default=True,
    help="Output directory.",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output JSON filename. Default: a-{system}_ref_stats.json.",
)
@click.option(
    "--group-var",
    type=str,
    default="density",
    show_default=True,
    help="Name of the grouping variable (used in output metadata).",
)
@click.option(
    "--group-regex",
    type=str,
    default=r"_(\d+(?:\.\d+)?)_",
    show_default=True,
    help="Regex to extract group value from structure key.",
)
@click.option(
    "--spectrum-types",
    "spectrum_types",
    multiple=True,
    default=("XRD", "ND", "XANES", "EXAFS", "PDF", "ADF"),
    show_default=True,
    help="Spectrum types to process. Can be repeated.",
)
@click.option(
    "--exafs-slice",
    type=(int, int),
    default=(40, 280),
    show_default=True,
    help="EXAFS index slice: START END.",
)
@click.option(
    "--xrd-nd-npts",
    type=int,
    default=100,
    show_default=True,
    help="Number of points to use for XRD/ND.",
)
def build_ref_stats(
    input_json,
    system,
    atoms_path,
    outdir,
    output,
    group_var,
    group_regex,
    spectrum_types,
    exafs_slice,
    xrd_nd_npts,
):
    """Build reference master stats JSON from a reference spectra JSON."""
    import json
    import re
    import numpy as np
    from ase.io import read

    def _is_axis_key(spec_type, key, idx):
        if spec_type in ["PDF", "ADF"] and key == "bins":
            return True
        if spec_type in ["XRD", "ND"] and key == "q":
            return True
        if spec_type not in ["EXAFS", "XANES"] and idx == 0:
            return True
        return False

    def _trim(spec_type, values):
        arr = np.asarray(values, dtype=float)
        if spec_type == "EXAFS":
            arr = arr[exafs_slice[0] : exafs_slice[1]]
        if spec_type in ["XRD", "ND"]:
            arr = arr[:xrd_nd_npts]
        return arr.tolist()

    def _load_spectra(group_data, spec_type):
        spectra_by_key = {}
        for skey, sdata in group_data.items():
            if spec_type not in sdata:
                continue
            for idx, (key, values) in enumerate(sdata[spec_type].items()):
                if _is_axis_key(spec_type, key, idx):
                    continue
                if spec_type in ["EXAFS", "XRD", "ND"]:
                    values = _trim(spec_type, values)
                spectra_by_key.setdefault(key, []).append(values)
        return spectra_by_key

    def _norm_factor(ref_dict):
        all_spectra = [v for vlist in ref_dict.values() for v in vlist]
        if not all_spectra:
            return 0.0
        return float(np.max(np.abs(np.asarray(all_spectra, dtype=float))))

    def _diversity(spectra_list, norm):
        arr = np.asarray(spectra_list, dtype=float)
        std = float(arr.std(axis=0).mean())
        return std / norm if norm > 0 else std

    with open(input_json) as f:
        all_data = json.load(f)

    # group keys by regex-extracted value
    groups = {}
    for skey in all_data:
        m = re.search(group_regex, skey)
        if not m:
            click.echo(f"  [skip] could not parse {group_var} from: {skey}")
            continue
        gv = m.group(1)
        groups.setdefault(gv, []).append(skey)

    if not groups:
        raise click.ClickException(
            f"No groups found. Check --group-regex against your structure keys."
        )

    os.makedirs(outdir, exist_ok=True)
    out_filename = output or f"a-{system}_ref_stats.json"
    out_path = os.path.join(outdir, out_filename)

    out = {
        "meta": {
            "system": system,
            "group_var": group_var,
            "group_regex": group_regex,
            "input_json": input_json,
            "spectrum_types": list(spectrum_types),
            "exafs_slice": list(exafs_slice),
            "xrd_nd_npts": xrd_nd_npts,
        },
        "groups": {},
    }

    for gv in sorted(
        groups, key=lambda x: float(x) if re.fullmatch(r"\d+(\.\d+)?", x) else x
    ):
        keys = groups[gv]
        click.echo(f"\n[{group_var}={gv}] n_structures={len(keys)}")

        group_data = {k: all_data[k] for k in keys}

        grp = {
            "meta": {group_var: gv, "n_structures": len(keys)},
            "samples": {},
            "stats": {},
        }

        # ingest samples
        for skey in keys:
            entry = {
                "spectra": {
                    st: all_data[skey][st]
                    for st in spectrum_types
                    if st in all_data[skey]
                }
            }
            if atoms_path:
                xyz_file = os.path.join(atoms_path, f"{skey}.xyz")
                if os.path.exists(xyz_file):
                    atoms = read(xyz_file, "-1")
                    entry["atoms"] = {
                        "numbers": atoms.get_atomic_numbers().tolist(),
                        "positions": atoms.get_positions().tolist(),
                        "cell": atoms.get_cell().tolist(),
                        "pbc": atoms.get_pbc().tolist(),
                    }
            grp["samples"][skey] = entry

        # compute stats per spectrum type
        for spec_type in spectrum_types:
            ref_spec = _load_spectra(group_data, spec_type)
            norm = _norm_factor(ref_spec)
            ref_div = {k: float(_diversity(v, norm)) for k, v in ref_spec.items()}
            mean_ref = {
                k: np.mean(np.asarray(v, dtype=float), axis=0).tolist()
                for k, v in ref_spec.items()
                if v
            }
            grp["stats"][spec_type] = {
                "mean_ref_spec": mean_ref,
                "norm_factor": float(norm),
                "ref_div": ref_div,
                "n_keys": len(ref_spec),
            }
            avg_div = (
                float(np.mean(list(ref_div.values()))) if ref_div else float("nan")
            )
            click.echo(
                f"  {spec_type}: n_keys={len(ref_spec)} norm={norm:.4g} | "
                f"avg_div={avg_div:.4f}"
            )

        out["groups"][gv] = grp

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    click.echo(f"\nSaved: {out_path}")