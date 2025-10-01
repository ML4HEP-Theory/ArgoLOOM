#!/usr/bin/env python3
"""
toolkit_class.py
----------------
CLASS pipeline:
  • Build .ini (ΛCDM + sterile ncdm)
  • Run CLASS
  • Robustly find *_cl*.dat
  • Parse columns, compute D_ell = l(l+1)C_l/(2π)
  • Plots: auto-spectra (TT/EE/BB/ϕϕ) and cross-spectra (TE/Tϕ/Eϕ/…)
    - auto:  log x, log y
    - cross: log x, linear y (symmetric)

Outputs (with --plot):
  outdir/<root>_Dell_auto.png
  outdir/<root>_Dell_cross.png
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# ====================== Utilities ======================

def resolve_class_binary(class_path_str: str) -> Path:
    p = Path(class_path_str).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for cand in ("class", "bin/class", "build/class"):
            q = (p / cand).resolve()
            if q.exists() and q.is_file():
                return q
    raise SystemExit(f"[ERROR] CLASS executable not found. Got: {p}\n"
                     f"Hint: use --class-path /path/to/class_public/class")

def run_cmd_live(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    print(f"[INFO] Running: {' '.join(str(x) for x in cmd)}")
    print(f"[INFO] CWD: {cwd}\n")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
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

def _csv(s: Optional[str]) -> List[str]:
    return [t.strip() for t in s.split(",")] if s else []

def _maybe_join_csv(name: str, vals: List[str]) -> Optional[str]:
    if not vals:
        return None
    return f"{name} = {','.join(vals)}"

# ====================== INI builder ======================

def build_class_ini(args) -> str:
    lines = []
    # Base ΛCDM
    lines += [
        f"h = {args.h}",
        f"omega_b = {args.ombh2}",
        f"omega_cdm = {args.omch2}",
        f"A_s = {args.As}",
        f"n_s = {args.ns}",
        f"tau_reio = {args.tau_reio}",
    ]
    # Radiation
    if args.N_ur is not None:
        lines.append(f"N_ur = {args.N_ur}")
    # Sterile (ncdm)
    if args.N_ncdm and args.N_ncdm > 0:
        lines.append(f"N_ncdm = {args.N_ncdm}")
        m_list = _csv(args.m_ncdm)
        T_list = _csv(args.T_ncdm)
        g_list = _csv(args.deg_ncdm)
        if m_list and len(m_list) != args.N_ncdm:
            sys.exit("[ERROR] --m-ncdm length must match --N-ncdm")
        if T_list and len(T_list) != args.N_ncdm:
            sys.exit("[ERROR] --T-ncdm length must match --N-ncdm")
        if g_list and len(g_list) != args.N_ncdm:
            sys.exit("[ERROR] --deg-ncdm length must match --N-ncdm")
        if m_list: lines.append(_maybe_join_csv("m_ncdm", m_list))
        if T_list: lines.append(_maybe_join_csv("T_ncdm", T_list))
        if g_list: lines.append(_maybe_join_csv("deg_ncdm", g_list))
    # Output / accuracy
    outdir_prefix = Path(args.outdir).as_posix().rstrip("/")
    root_prefixed = f"{outdir_prefix}/{args.outtag}" if outdir_prefix else args.outtag
    lines += [
        f"output = {args.output}",          # e.g. tCl,pCl,lCl
        f"lensing = {args.lensing}",        # yes/no
        f"l_max_scalars = {args.lmax}",
        f"root = {root_prefixed}",
    ]
    if args.verbose:
        lines += ["write parameters = yes", "write warnings = yes"]
    return "\n".join(x for x in lines if x)

# ====================== Output discovery ======================

def find_cl_file(root: str, workdir: Path, outdir: Path) -> Path:
    search_dirs = []
    if outdir:
        search_dirs.append(outdir)
        search_dirs.append(outdir / "output")
    search_dirs.append(workdir)
    search_dirs.append(workdir / "output")
    patterns = [
        f"{root}_cl.dat",
        f"{root}_*cl*.dat",
        f"{root}*cl*.dat",
        f"{root}_*Cls*.dat",
        f"{root}*Cls*.dat",
        f"{root}_*.dat",
    ]
    candidates: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            candidates.extend(d.glob(pat))
    if not candidates:
        print(f"[DEBUG] No C_ell file matched. Scanned dirs:")
        for d in search_dirs:
            try:
                entries = sorted(p.name for p in d.iterdir())
                print(f"  - {d} ({len(entries)} items)")
            except Exception:
                print(f"  - {d} (unreadable)")
        raise SystemExit(f"[ERROR] Could not find C_ell output for root='{root}'")
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ====================== Parsing & plotting ======================

def parse_cl_with_headers(path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    header_cols: List[str] = []
    with path.open("r") as fh:
        for line in fh:
            if line.startswith("#"):
                header_cols = line[1:].strip().split()
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    data: Dict[str, np.ndarray] = {}
    ncols = arr.shape[1]
    if header_cols and len(header_cols) == ncols:
        for i, name in enumerate(header_cols):
            data[name] = arr[:, i]
        cols_used = header_cols
    else:
        cols_used = ["ell"] + [f"col{i}" for i in range(2, ncols + 1)]
        for i, name in enumerate(cols_used):
            data[name] = arr[:, i]
    return data, cols_used

def _norm_key(name: str) -> str:
    """Normalize a column name for matching (lowercase, drop underscores)."""
    return name.lower().replace("_", "")

def split_auto_cross(columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Classify columns into auto vs cross spectra based on name.
    Auto: TT, EE, BB, phiphi (or 'pp').
    Cross: TE, Tphi, Ephi, TB, EB, etc.
    """
    autos, crosses = [], []
    for c in columns:
        k = _norm_key(c)
        if k in {"l", "ell", "multipole"}:
            continue
        # mappings
        if k in {"tt", "ee", "bb", "phiphi", "pp"}:
            autos.append(c)
        elif any(k == s for s in ("te", "tb", "eb", "tphi", "ephi", "phiT".lower(), "phie".lower())) or \
             any(s in k for s in ("te", "tb", "eb", "tphi", "ephi")):
            crosses.append(c)
        else:
            # Heuristic: if name contains only one of {t,e,b,phi}, treat as auto; if multiple, cross
            tokens = {"t": "t" in k, "e": "e" in k, "b": "b" in k, "phi": "phi" in k or "pp" in k}
            if sum(bool(v) for v in tokens.values()) <= 1:
                autos.append(c)
            else:
                crosses.append(c)
    return autos, crosses

def compute_Dell(ell: np.ndarray, Cl: np.ndarray) -> np.ndarray:
    """D_ell = ell(ell+1) Cl / (2π)."""
    factor = ell * (ell + 1.0) / (2.0 * np.pi)
    return factor * Cl

def _finite_positive(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr) & (arr > 0)]

def plot_auto_spectra(ell: np.ndarray, data: Dict[str, np.ndarray],
                      autos: List[str], title: str, out_png: Path):
    """
    Auto-spectra: plot D_ell on log–log axes. Auto-scale y from finite positives.
    """
    if not autos:
        return
    plt.figure()
    ymins, ymaxs = [], []
    for c in autos:
        Dell = compute_Dell(ell, data[c])
        pos = _finite_positive(Dell)
        if pos.size == 0:
            continue
        ymins.append(np.min(pos))
        ymaxs.append(np.max(pos))
        plt.plot(ell, Dell, label=c.upper())
    if not ymins:
        plt.close()
        return
    # Log axes with padding
    ymin = max(min(ymins) * 0.7, 1e-20)
    ymax = max(ymaxs) * 1.4
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(max(1.0, np.min(ell[ell > 0]) * 0.9), np.max(ell) * 1.05)
    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell \equiv \ell(\ell+1)C_\ell/2\pi$")
    plt.title(title + " — auto spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_png}")

def plot_cross_spectra(ell: np.ndarray, data: Dict[str, np.ndarray],
                       crosses: List[str], title: str, out_png: Path):
    """
    Cross-spectra: plot D_ell with log x, linear y; symmetric y-limits around 0.
    """
    if not crosses:
        return
    plt.figure()
    max_abs = 0.0
    for c in crosses:
        Dell = compute_Dell(ell, data[c])
        good = np.isfinite(Dell)
        if not np.any(good):
            continue
        max_abs = max(max_abs, np.max(np.abs(Dell[good])))
        plt.plot(ell[good], Dell[good], label=c.upper())
    if max_abs == 0.0:
        plt.close()
        return
    plt.xscale("log")
    plt.xlim(max(1.0, np.min(ell[ell > 0]) * 0.9), np.max(ell) * 1.05)
    plt.ylim(-1.2 * max_abs, 1.2 * max_abs)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell \equiv \ell(\ell+1)C_\ell/2\pi$")
    plt.title(title + " — cross spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {out_png}")

# ====================== Main ======================

def main():
    p = argparse.ArgumentParser(description="Agentic CLASS pipeline (C_ell with sterile-neutrino options).")
    p.add_argument("--class-path", required=True, help="Path to CLASS 'class' executable OR class_public directory")
    p.add_argument("--outtag", required=True, help="CLASS 'root' (file prefix) for outputs (e.g., 'sterile_demo')")
    p.add_argument("--outdir", default=".", help="Directory for CLASS outputs (used to prefix 'root')")
    p.add_argument("--workdir", default=".", help="Working directory where .ini is written and CLASS is run")

    # Base ΛCDM
    p.add_argument("--h", type=float, required=True, help="Reduced Hubble parameter h")
    p.add_argument("--ombh2", type=float, required=True, help="Ω_b h^2")
    p.add_argument("--omch2", type=float, required=True, help="Ω_c h^2")
    p.add_argument("--As", type=float, required=True, help="Primordial scalar amplitude A_s")
    p.add_argument("--ns", type=float, required=True, help="n_s")
    p.add_argument("--tau-reio", type=float, required=True, dest="tau_reio", help="τ_reio")

    # Radiation & sterile neutrinos
    p.add_argument("--N-ur", type=float, default=None, help="Effective number of ultra-relativistic species (beyond photons)")
    p.add_argument("--N-ncdm", type=int, default=0, help="Number of non-cold DM (sterile) species")
    p.add_argument("--m-ncdm", type=str, default=None, help="Comma-separated masses (eV) for each ncdm species")
    p.add_argument("--T-ncdm", type=str, default=None, help="Comma-separated temperature ratios T_i/T_cmb for ncdm")
    p.add_argument("--deg-ncdm", type=str, default=None, help="Comma-separated degeneracy factors for ncdm")

    # Output controls
    p.add_argument("--lensing", choices=("yes", "no"), default="yes", help="Enable CMB lensing in spectra")
    p.add_argument("--lmax", type=int, default=3000, help="l_max_scalars")
    p.add_argument("--output", default="tCl,pCl,lCl", help="CLASS 'output' string (default: tCl,pCl,lCl)")

    # QoL
    p.add_argument("--plot", action="store_true", help="Make spectra plots")
    p.add_argument("--verbose", action="store_true", help="Ask CLASS to echo parameters/warnings to files")

    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    class_path = resolve_class_binary(args.class_path)

    ini_text = build_class_ini(args)
    ini_path = workdir / f"{args.outtag}.ini"
    ini_path.write_text(ini_text)
    print(f"[INFO] Wrote CLASS ini: {ini_path}")

    rc, combined = run_cmd_live([str(class_path), str(ini_path)], workdir)
    (workdir / f"{args.outtag}.log").write_text(combined)
    if rc != 0:
        sys.exit(f"[ERROR] CLASS exited with code {rc}")

    # Find C_ell output (robust search)
    cl_path = find_cl_file(args.outtag, workdir, outdir)
    print(f"[INFO] Found Cl file: {cl_path}")

    data, cols = parse_cl_with_headers(cl_path)
    print(f"[INFO] Columns: {', '.join(cols)}")

    if args.plot:
        # Identify ell column (tolerant)
        ell = None
        for name in ("ell", "l", "multipole"):
            if name in data:
                ell = data[name]
                break
        if ell is None:
            ell = next(iter(data.values()))  # first column fallback
        # Make sure ell is float and >= 2 for D_ell; keep as-is for plotting
        ell = np.asarray(ell, dtype=float)

        # Choose columns to plot
        all_cols = [c for c in data.keys() if c not in ("ell", "l", "multipole")]
        autos, crosses = split_auto_cross(all_cols)

        # Plot auto and cross
        auto_png  = outdir / f"{args.outtag}_Dell_auto.png"
        cross_png = outdir / f"{args.outtag}_Dell_cross.png"
        title = f"CLASS power spectra: {args.outtag}"

        plot_auto_spectra(ell, data, autos, title, auto_png)
        plot_cross_spectra(ell, data, crosses, title, cross_png)

    print("\n[INFO] Done.")
    print(f"[INFO] Root prefix : {args.outtag}")
    print(f"[INFO] INI         : {ini_path}")
    print(f"[INFO] Cl data     : {cl_path}")
    if args.plot:
        print(f"[INFO] Plots       :")
        print(f"  - {outdir / f'{args.outtag}_Dell_auto.png'}")
        print(f"  - {outdir / f'{args.outtag}_Dell_cross.png'}")

if __name__ == "__main__":
    main()
