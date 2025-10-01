#!/usr/bin/env python3
"""
API_MG5-kin.py
-----------------------
End-to-end pipeline:
  OpenAI API -> MG5 command file -> run MG5 -> parse σ -> locate LHE ->
  rapidity analysis (particle/system) -> map (y,M) -> (x1,x2,Q2) with smearing ->
  Parquet + PNG histogram + 2D (x1_det,x2_det) smearing plot

Examples
--------
# e+ e- -> mu+ mu- at 40 TeV CM (20 TeV/beam), analyze muons (system rapidity)
python agentic_mg5_pipeline.py \
  --mg5-path /Users/tjhobbs/Documents/computing/MG5_aMC_v3_5_9/bin/mg5_aMC \
  --model-name gpt-4o \
  --import-model sm \
  --process "e+ e- > mu+ mu-" \
  --outtag EE_to_mumu_20TeV \
  --nevents 2000 --ebeam-gev 20000 --iseed 12345 \
  --pdg "13,-13" --mode system --plot

# pp -> e+ e- at 13 TeV, analyze per-particle electrons (leading-pT only)
python agentic_mg5_pipeline.py \
  --mg5-path /path/to/mg5_aMC \
  --model-name gpt-4o \
  --import-model sm \
  --process "p p > e+ e-" \
  --outtag DY_13TeV \
  --nevents 5000 --ebeam-gev 6500 \
  --pdg "11,-11" --mode particle --one-per-event --plot \
  --momentum-order pxpypzE

Dependencies
------------
pip install openai numpy pyarrow pylhe matplotlib
"""

import os, re, sys, math, time, argparse, subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import matplotlib.pyplot as plt
from openai import OpenAI

# ==================== MG5 + API helpers ====================

_XS_LINE = re.compile(
    r"Cross-?section\s*:\s*([0-9.+\-Ee]+)\s*\+\-\s*([0-9.+\-Ee]+)\s*pb",
    re.IGNORECASE,
)

def get_mg5_commands_via_api(import_model: str, process: str, outtag: str,
                             nevents: int, ebeam_gev: float, iseed: int,
                             model_name: str) -> str:
    """Use OpenAI to emit MG5 CLI commands only."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[ERROR] OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    prompt = f"""You are generating commands for the MadGraph5_aMC@NLO CLI.
Output ONLY valid MG5 CLI commands (no markdown, no shell).
Do exactly:
1) import model {import_model}
2) generate {process}
3) output {outtag}
4) launch
   - set run_card nevents {nevents}
   - set run_card ebeam1 {ebeam_gev}
   - set run_card ebeam2 {ebeam_gev}
   - set run_card iseed {iseed}
5) launch
"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Only return valid MG5 CLI commands. No markdown, no shell."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        sys.exit("[ERROR] Empty MG5 command response from API.")
    return content

def run_cmd_live(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    print(f"[INFO] Running: {' '.join(cmd)}\n[INFO] CWD: {cwd}\n")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        proc.wait()
    return proc.returncode, "".join(lines)

def parse_all_xs(text: str) -> List[Tuple[float,float]]:
    vals = []
    for m in _XS_LINE.finditer(text):
        try:
            vals.append((float(m.group(1)), float(m.group(2))))
        except ValueError:
            pass
    return vals

def summarize_xs(xs_list: List[Tuple[float,float]]) -> None:
    if not xs_list:
        print("[WARN] No cross-section lines parsed.")
        return
    arr = np.array(xs_list, dtype=float)
    final_xs, final_err = xs_list[-1]
    rel = (final_err/abs(final_xs)*100.0) if final_xs != 0 else float("inf")
    print("\n================= CROSS-SECTION SUMMARY =================")
    print(f"Reports parsed       : {len(xs_list)}")
    print(f"Final σ              : {final_xs:.6e} pb")
    print(f"Final Δσ (int. err)  : {final_err:.6e} pb ({rel:.3f} %)")
    print(f"Mean σ (all reports) : {np.mean(arr[:,0]):.6e} pb")
    print(f"Std  σ (all reports) : {np.std(arr[:,0], ddof=1) if len(arr)>1 else 0.0:.6e} pb")
    print(f"Min/Max σ            : {np.min(arr[:,0]):.6e} / {np.max(arr[:,0]):.6e} pb")
    print("==========================================================\n")

def find_latest_lhe(outdir: Path) -> Optional[Path]:
    events_dir = outdir / "Events"
    if not events_dir.exists():
        return None
    runs = sorted([p for p in events_dir.glob("run_*") if p.is_dir()])
    if not runs:
        return None
    latest = runs[-1]
    cands = list(latest.glob("*.lhe.gz")) + list(latest.glob("*.lhe"))
    return cands[0] if cands else None

# ==================== Analysis utilities ====================

def smear_fractional(val: float, frac_sigma: float,
                     rng=np.random, clip_min=None, clip_max=None) -> float:
    """Gaussian smear with fractional sigma given; optional clipping."""
    sigma = abs(frac_sigma) * abs(val)
    smeared = rng.normal(val, sigma)
    if clip_min is not None or clip_max is not None:
        smeared = float(np.clip(smeared, clip_min, clip_max))
    return smeared

def smear_x_with_edges(x: float, rel_sigma_x: float,
                       alpha: float, beta: float,
                       rng=np.random, clip_min=1e-12, clip_max=1.0) -> float:
    """
    Edge-aware fractional resolution:
      frac_sigma_x(x) = rel_sigma_x * (1 + alpha * (2*|x-0.5|)^beta)
      -> larger relative smearing near x≈0 or 1; unchanged at x=0.5.
    """
    # Keep x in (0,1) for the extremeness measure only (avoid NaN if over-clipped upstream)
    x_for_shape = float(np.clip(x, 1e-15, 1.0-1e-15))
    extremeness = 2.0 * abs(x_for_shape - 0.5)     # in [0,1]
    frac_sigma  = rel_sigma_x * (1.0 + alpha * (extremeness ** beta))
    return smear_fractional(x, frac_sigma, rng=rng, clip_min=clip_min, clip_max=clip_max)

def safe_rapidity(E: float, pz: float) -> Optional[float]:
    num, den = (E + pz), (E - pz)
    if num <= 0 or den <= 0:
        return None
    return 0.5 * math.log(num / den)

def inv_mass(E: float, px: float, py: float, pz: float) -> float:
    m2 = E*E - (px*px + py*py + pz*pz)
    return math.sqrt(max(m2, 0.0))

def pt(px: float, py: float) -> float:
    return math.hypot(px, py)

def parse_pdg_list(s: str) -> List[int]:
    return [int(tok) for tok in s.split(",") if tok.strip()]

def unpack_momentum_from_tuple(mom_tuple, order: str):
    if order == "Epxpypz":
        E, px, py, pz = mom_tuple[:4]
        return float(E), float(px), float(py), float(pz)
    elif order == "pxpypzE":
        px, py, pz, E = mom_tuple[:4]
        return float(E), float(px), float(py), float(pz)
    else:
        raise ValueError("Unknown momentum order")

# --- Robust pylhe accessors ---
def _get_pid(p):
    if hasattr(p, "pid"):
        return p.pid
    if hasattr(p, "id"):
        return p.id
    raise AttributeError("Particle has neither .pid nor .id")

def _get_status(p):
    for name in ("status", "istatus", "istate"):
        if hasattr(p, name):
            return getattr(p, name)
    raise AttributeError("Particle has no status attribute")

def _get_momentum(p, order_hint: str):
    if hasattr(p, "momentum"):
        mom = p.momentum
        try:
            return unpack_momentum_from_tuple(mom, order_hint)
        except Exception:
            pass
    Ek = ("E", "e"); Pxk = ("px", "p_x"); Pyk = ("py", "p_y"); Pzk = ("pz", "p_z")
    for En in Ek:
        if hasattr(p, En):
            for Pxn in Pxk:
                if hasattr(p, Pxn):
                    for Pyn in Pyk:
                        if hasattr(p, Pyn):
                            for Pzn in Pzk:
                                if hasattr(p, Pzn):
                                    E  = float(getattr(p, En))
                                    px = float(getattr(p, Pxn))
                                    py = float(getattr(p, Pyn))
                                    pz = float(getattr(p, Pzn))
                                    return E, px, py, pz
    raise AttributeError("LHEParticle has no momentum tuple or (e/E,px,py,pz) components")

def _iter_events_with_pylhe(lhe_path: Path):
    import pylhe
    try:
        return pylhe.read_lhe_with_attributes(str(lhe_path))
    except Exception:
        return pylhe.read_lhe(str(lhe_path))

def iter_selected_particles(lhe_path: Path, pdgs: List[int], status: int, mom_order: str):
    pdg_set = set(pdgs)
    for ev_idx, event in enumerate(_iter_events_with_pylhe(lhe_path)):
        for p in event.particles:
            if _get_status(p) != status:
                continue
            pid = _get_pid(p)
            if pid in pdg_set:
                E, px, py, pz = _get_momentum(p, mom_order)
                yield ev_idx, (E, px, py, pz), pid

def collect_particle_mode(lhe_path: Path, pdgs: List[int], status: int,
                          one_per_event: bool, mom_order: str) -> List[dict]:
    rows = []
    bucket = {}
    for ev_idx, (E, px, py, pz), pid in iter_selected_particles(lhe_path, pdgs, status, mom_order):
        y = safe_rapidity(E, pz)
        if y is None or not math.isfinite(y):
            continue
        _pt = pt(px, py)
        M   = inv_mass(E, px, py, pz)
        row = dict(event=ev_idx, mode="particle", pdg=pid,
                   y=y, M=M, E=E, px=px, py=py, pz=pz, pt=_pt, n_constituents=1)
        rows.append(row)
        if one_per_event:
            bucket.setdefault(ev_idx, []).append((_pt, row))
    if one_per_event:
        kept = []
        for ev_idx, items in bucket.items():
            items.sort(key=lambda t: t[0], reverse=True)
            kept.append(items[0][1])
        return kept
    return rows

def collect_system_mode(lhe_path: Path, pdgs: List[int], status: int, mom_order: str) -> List[dict]:
    rows = []
    pdg_set = set(pdgs)
    for ev_idx, event in enumerate(_iter_events_with_pylhe(lhe_path)):
        E_sum = px_sum = py_sum = pz_sum = 0.0
        n = 0
        for p in event.particles:
            if _get_status(p) != status:
                continue
            pid = _get_pid(p)
            if pid in pdg_set:
                E, px, py, pz = _get_momentum(p, mom_order)
                E_sum  += E;  px_sum += px;  py_sum += py;  pz_sum += pz
                n += 1
        if n == 0:
            continue
        y = safe_rapidity(E_sum, pz_sum)
        if y is None or not math.isfinite(y):
            continue
        M = inv_mass(E_sum, px_sum, py_sum, pz_sum)
        rows.append(dict(
            event=ev_idx, mode="system", pdg=None,
            y=y, M=M, E=E_sum, px=px_sum, py=py_sum, pz=pz_sum, pt=pt(px_sum, py_sum), n_constituents=n
        ))
    return rows

def map_to_xxQ2(y: float, M: float, sqrt_s: float):
    M = max(M, 1e-9)
    sqrt_s = max(sqrt_s, 1e-6)
    xfac = M / sqrt_s
    return xfac * math.exp(+y), xfac * math.exp(-y), M*M

def plot_hist(yvals: np.ndarray, out_png: Path, bins: int, title: str):
    if yvals.size == 0:
        print("[WARN] No rapidities to plot.")
        return
    plt.figure()
    plt.hist(yvals, bins=bins)
    plt.xlabel("Rapidity y")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved histogram: {out_png}")

def plot_x_scatter(x1_true: np.ndarray, x2_true: np.ndarray,
                   x1_det:  np.ndarray, x2_det:  np.ndarray,
                   out_png: Path, title: str):
    x1_true = np.asarray(x1_true, dtype=float)
    x2_true = np.asarray(x2_true, dtype=float)
    x1_det  = np.asarray(x1_det,  dtype=float)
    x2_det  = np.asarray(x2_det,  dtype=float)

    eps = 1e-12
    frac1 = np.abs(x1_det - x1_true) / np.clip(np.abs(x1_true), eps, None)
    frac2 = np.abs(x2_det - x2_true) / np.clip(np.abs(x2_true), eps, None)
    smear_frac = 0.5 * (frac1 + frac2)

    good = np.isfinite(x1_det) & np.isfinite(x2_det) & (x1_det > 0) & (x2_det > 0) & np.isfinite(smear_frac)
    x = x1_det[good]; y = x2_det[good]; c = smear_frac[good]
    if x.size == 0:
        print("[WARN] No finite positive (x1_det,x2_det) to plot.")
        return

    plt.figure()
    plt.scatter(x, y, c=c, s=6, alpha=0.8)
    cb = plt.colorbar(); cb.set_label("avg fractional smear")

    plt.xscale("log"); plt.yscale("log")
    xmin = max(np.min(x)*0.9, 1e-12); xmax = min(np.max(x)*1.1, 1.0)
    ymin = max(np.min(y)*0.9, 1e-12); ymax = min(np.max(y)*1.1, 1.0)
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)

    plt.xlabel(r"$x_1^{\mathrm{det}}$")
    plt.ylabel(r"$x_2^{\mathrm{det}}$")
    plt.title(title + " — smeared $(x_1,x_2)$")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved scatter: {out_png}")

# ==================== Main pipeline ====================

def main():
    ap = argparse.ArgumentParser(description="API->MG5->LHE->rapidity analysis pipeline")
    # MG5/LLM config
    ap.add_argument("--mg5-path", required=True, help="Path to mg5_aMC executable")
    ap.add_argument("--model-name", default="gpt-4o", help="OpenAI model to use")
    ap.add_argument("--import-model", default="sm", help="MG5 'import model <...>'")
    ap.add_argument("--process", required=True, help='MG5 process, e.g. "e+ e- > mu+ mu-"')
    ap.add_argument("--outtag", required=True, help="MG5 output directory name")
    ap.add_argument("--nevents", type=int, default=10000, help="Number of events")
    ap.add_argument("--ebeam-gev", type=float, required=True, help="Beam energy in GeV")
    ap.add_argument("--iseed", type=int, default=12345, help="Random seed")

    # Analysis config
    ap.add_argument("--pdg", type=str, required=True,
                    help="Comma-separated PDGs to select (e.g., '11,-11,13,-13')")
    ap.add_argument("--mode", choices={"particle","system"}, default="system",
                    help="particle: per selected particle; system: sum selected particles per event")
    ap.add_argument("--status", type=int, default=1, help="LHE status (default 1=final state)")
    ap.add_argument("--one-per-event", action="store_true",
                    help="(particle mode) keep leading-pT only")
    ap.add_argument("--momentum-order", choices={"Epxpypz","pxpypzE"}, default="Epxpypz",
                    help="Match your pylhe momentum tuple ordering")

    # Smearing controls
    ap.add_argument("--rel-sigma-x", type=float, default=0.02, help="Baseline fractional σ(x) at x=0.5")
    ap.add_argument("--x-smear-alpha", type=float, default=2.0, help="Edge boost for σ(x): factor grows by (1+alpha) at edges")
    ap.add_argument("--x-smear-beta",  type=float, default=2.0, help="Sharpness of edge growth (power on extremeness)")
    ap.add_argument("--rel-sigma-q2", type=float, default=0.05, help="Relative σ(Q²) detector resolution")

    ap.add_argument("--fixed-mass", type=float, default=None, help="Fallback mass (GeV) if needed")
    ap.add_argument("--default-mass", type=float, default=None, help="Default mass (GeV) if computed is NaN")
    ap.add_argument("-o", "--out-parquet", default=None,
                    help="Parquet output (default: <outtag>_xq2.parquet)")

    # Plotting
    ap.add_argument("--plot", action="store_true", help="Save plots (rapidity histogram + x1/x2 smearing)")
    ap.add_argument("--plot-file", default=None, help="Rapidity PNG path (default: <outtag>_rapidity_hist.png)")
    ap.add_argument("--bins", type=int, default=60, help="Histogram bins")

    args = ap.parse_args()

    mg5_path = Path(args.mg5_path)
    if not mg5_path.exists():
        sys.exit(f"[ERROR] MG5 not found at: {mg5_path}")

    # Compose MG5 commands via API
    print("[INFO] Requesting MG5 command script from OpenAI…")
    mg5_cmds = get_mg5_commands_via_api(
        args.import_model, args.process, args.outtag,
        args.nevents, args.ebeam_gev, args.iseed, args.model_name
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    mg5_file = Path.cwd() / f"auto_{args.outtag}_{ts}.mg5"
    mg5_file.write_text(mg5_cmds)
    print(f"[INFO] MG5 script written to: {mg5_file}")

    # Run MG5
    rc, combined = run_cmd_live([str(mg5_path), str(mg5_file)], Path.cwd())
    (Path.cwd() / f"mg5_{args.outtag}_{ts}.log").write_text(combined)
    if rc != 0:
        sys.exit(f"[ERROR] MG5 exited with code {rc}")

    # XS summary
    xs_list = parse_all_xs(combined)
    summarize_xs(xs_list)

    # Locate LHE
    outdir = Path.cwd() / args.outtag
    lhe = find_latest_lhe(outdir)
    if not lhe:
        sys.exit(f"[ERROR] Could not find LHE under {outdir}/Events/run_*")
    print(f"[INFO] Found LHE: {lhe}")

    # sqrt(s) in GeV
    sqrt_s = 2.0 * float(args.ebeam_gev)

    # ---- Rapidities & (x1,x2,Q2) mapping ----
    pdgs = parse_pdg_list(args.pdg)
    if not pdgs:
        sys.exit("Please provide non-empty --pdg list.")
    if args.mode == "particle":
        rows = collect_particle_mode(lhe, pdgs, args.status, args.one_per_event, args.momentum_order)
    else:
        rows = collect_system_mode(lhe, pdgs, args.status, args.momentum_order)

    if not rows:
        sys.exit("No matching particles/systems found – aborting.")

    # Build arrays and map
    y_arr, M_arr = [], []
    for r in rows:
        y_arr.append(float(r["y"]))
        mval = float(r["M"]) if np.isfinite(r["M"]) else (args.fixed_mass or args.default_mass or 0.0)
        M_arr.append(mval)
    y_arr = np.asarray(y_arr, dtype=float)
    M_arr = np.asarray(M_arr, dtype=float)

    x1_t = []; x2_t = []; Q2_t = []
    x1_s = []; x2_s = []; Q2_s = []
    for y, M in zip(y_arr, M_arr):
        x1, x2, Q2 = map_to_xxQ2(y, M, sqrt_s)
        x1_t.append(x1); x2_t.append(x2); Q2_t.append(Q2)
        # x-dependent fractional smearing (edge-aware)
        x1_s.append(smear_x_with_edges(x1, args.rel_sigma_x, args.x_smear_alpha, args.x_smear_beta,
                                       clip_min=1e-9, clip_max=1.0))
        x2_s.append(smear_x_with_edges(x2, args.rel_sigma_x, args.x_smear_alpha, args.x_smear_beta,
                                       clip_min=1e-9, clip_max=1.0))
        # Q2 smearing kept simple (constant fractional)
        Q2_s.append(smear_fractional(Q2, args.rel_sigma_q2, clip_min=1e-9))

    # Write Parquet
    out_parquet = args.out_parquet or f"{args.outtag}_xq2.parquet"
    table = pa.Table.from_pydict(dict(
        event          = [r["event"] for r in rows],
        mode           = [r["mode"]  for r in rows],
        pdg            = [r["pdg"]   for r in rows],
        y              = y_arr,
        M              = M_arr,
        E              = [r["E"]     for r in rows],
        px             = [r["px"]    for r in rows],
        py             = [r["py"]    for r in rows],
        pz             = [r["pz"]    for r in rows],
        pt             = [r["pt"]    for r in rows],
        n_constituents = [r["n_constituents"] for r in rows],
        x1_true        = x1_t,  x2_true = x2_t,  Q2_true = Q2_t,
        x1_det         = x1_s,  x2_det  = x2_s,  Q2_det  = Q2_s
    ))
    pq.write_table(table, out_parquet)
    print(f"[✓] {table.num_rows} rows  →  {out_parquet}")

    # Optional plotting
    if args.plot:
        # Rapidity histogram
        png_hist = args.plot_file or f"{args.outtag}_rapidity_hist.png"
        title = f"Rapidity distribution ({args.mode}, PDG={args.pdg})"
        plot_hist(y_arr, Path(png_hist), args.bins, title)

        # (x1_det, x2_det) smearing scatter with color = avg fractional smearing
        x_scatter_png = Path(f"{args.outtag}_x1x2_smear.png")
        plot_x_scatter(
            np.asarray(x1_t), np.asarray(x2_t),
            np.asarray(x1_s), np.asarray(x2_s),
            x_scatter_png,
            f"{args.outtag}"
        )

    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()
