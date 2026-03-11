#!/usr/bin/env python3
"""
DEVSIM RNPU pipeline (planar + etched-like proxy):
  - 2D drift-diffusion IV sweep
  - d(log10|I|)/d(log10|V|) slope extraction
  - optional stability and band-diagram outputs

Output directories:
  - configurable from CLI via --output-dir and --figures-dir
  - by default created next to this script:
      <script_dir>/output
      <script_dir>/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import utils as u

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc

try:
    from devsim import *  # noqa: F401,F403
except Exception as exc:
    raise SystemExit("DEVSIM is not importable in this environment") from exc

try:
    from devsim.python_packages.model_create import *  # noqa: F401,F403
    from devsim.python_packages.simple_physics import *  # noqa: F401,F403
except Exception:
    from model_create import *  # type: ignore # noqa: F401,F403
    from simple_physics import *  # type: ignore # noqa: F401,F403


# Runtime-configurable output locations (set in main()).
OUT_DIR = Path(".")
FIG_DIR = Path(".")
DEVICE = "Simple2D"
REGION = "Si"
LEFT = "left"
RIGHT = "right"


def default_output_dirs() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    return script_dir / "output", script_dir / "figures"


def _si_bandgap_ev(temp_k: float) -> float:
    # Varshni relation for silicon bandgap.
    alpha = 4.73e-4
    beta = 636.0
    return 1.17 - alpha * (temp_k * temp_k) / (temp_k + beta)


def _bias_token(v: float) -> str:
    s = f"{v:.6g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return s


def apply_gaussian_trap_block(
    *,
    device: str,
    region: str,
    width_cm: float,
    metal_width_cm: float | None,
    ea_mev: float,
    nga1_cm3: float,
    nga2_cm3: float,
    y1_um: float,
    y2_um: float,
    sigma1_um: float,
    sigma2_um: float,
    lateral_factor: float,
    trap_strength: float,
) -> None:
    """
    Controlled approximation of a two-Gaussian trap block:
      - builds spatial trap weight from two Gaussian components,
      - applies Ea-dependent scaling to local SRH lifetimes,
      - overrides USRH/Gn/Gp node models used by DD equations.
    """
    x0_cm = 0.5 * width_cm
    sx_ref_cm = metal_width_cm if metal_width_cm is not None else width_cm
    sx_cm = max(abs(lateral_factor) * sx_ref_cm, 1e-8)
    y1_cm = max(0.0, y1_um * 1e-4)
    y2_cm = max(0.0, y2_um * 1e-4)
    sy1_cm = max(abs(sigma1_um) * 1e-4, 1e-8)
    sy2_cm = max(abs(sigma2_um) * 1e-4, 1e-8)
    nref = 1e14
    a1 = max(0.0, nga1_cm3 / nref)
    a2 = max(0.0, nga2_cm3 / nref)
    ea_ev = max(0.0, ea_mev * 1e-3)
    # 250 meV -> factor ~1; 500 meV -> factor ~(1+alpha)
    alpha = 1.0
    e_factor = 1.0 + alpha * ((ea_ev - 0.25) / 0.25)

    g1 = (
        f"exp(-0.5*pow((x-{x0_cm:.8e})/{sx_cm:.8e},2))"
        f"*exp(-0.5*pow((y-{y1_cm:.8e})/{sy1_cm:.8e},2))"
    )
    g2 = (
        f"exp(-0.5*pow((x-{x0_cm:.8e})/{sx_cm:.8e},2))"
        f"*exp(-0.5*pow((y-{y2_cm:.8e})/{sy2_cm:.8e},2))"
    )
    CreateNodeModel(device, region, "TrapG1", g1)
    CreateNodeModel(device, region, "TrapG2", g2)
    CreateNodeModel(device, region, "TrapWeight", f"{a1:.8e}*TrapG1 + {a2:.8e}*TrapG2")
    CreateNodeModel(device, region, "TrapEnergyFactor", f"{e_factor:.8e}")
    CreateNodeModel(
        device,
        region,
        "TrapLifetimeScale",
        f"1.0 + {max(trap_strength, 0.0):.8e}*TrapWeight*TrapEnergyFactor",
    )
    CreateNodeModel(device, region, "taun_eff", "taun/TrapLifetimeScale")
    CreateNodeModel(device, region, "taup_eff", "taup/TrapLifetimeScale")

    # Override default SRH models with trap-modified lifetime models.
    usrh = "(Electrons*Holes - n_i^2)/(taup_eff*(Electrons + n1) + taun_eff*(Holes + p1))"
    CreateNodeModel(device, region, "USRH", usrh)
    for var in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "USRH", usrh, var)
    gn = "-ElectronCharge * USRH"
    gp = "+ElectronCharge * USRH"
    CreateNodeModel(device, region, "Gn", gn)
    CreateNodeModel(device, region, "Gp", gp)
    for var in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "Gn", gn, var)
        CreateNodeModelDerivative(device, region, "Gp", gp, var)


def solve_dc() -> bool:
    attempts = (
        (1e10, 1e-10, 60),
        (1e10, 1e-8, 100),
        (1e8, 1e-6, 160),
        (1e8, 1e-5, 260),
        (1e6, 1e-4, 400),
    )
    for ae, re, mi in attempts:
        try:
            solve(type="dc", absolute_error=ae, relative_error=re, maximum_iterations=mi)
            return True
        except Exception:
            pass
    return False


def robust_move_bias(
    bias_name: str,
    v_from: float,
    v_to: float,
    max_step: float,
    min_step: float,
    fallback_retries: int,
    anchor_bias: float,
) -> float:
    """
    Bias move with recovery path for stagnation/divergence:
      1) try normal move
      2) on failure, re-anchor near reference bias and retry with reduced max_step
    """
    try:
        return move_bias(bias_name, v_from, v_to, max_step=max_step, min_step=min_step)
    except Exception:
        pass

    last_exc: Exception | None = None
    for k in range(max(0, fallback_retries)):
        retry_step = max(min_step * 10.0, max_step / (2.0 ** (k + 1)))
        print(f"FALLBACK_RETRY {k+1} target={v_to:.12g} retry_step={retry_step:.6g}")
        try:
            # Recovery anchor (typically 0 V + contact offset)
            set_parameter(device=DEVICE, name=bias_name, value=anchor_bias)
            if not solve_dc():
                continue
            return move_bias(bias_name, anchor_bias, v_to, max_step=retry_step, min_step=min_step)
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Convergence fallback failed at target {v_to:.6g} V")


def build_device(
    temp_k: float,
    na: float,
    nd: float,
    taun_s: float,
    taup_s: float,
    width_cm: float,
    depth_cm: float,
    junction_x_cm: float,
    doping_mode: str,
    implant_species: str,
    implant_dose_cm2: float,
    implant_peak_um: float,
    implant_sigma_um: float,
    implant_lateral_factor: float,
    metal_width_cm: float | None,
    include_oxide: bool,
    native_oxide_thickness_um: float,
    include_sidewall_oxide: bool,
    sidewall_oxide_thickness_um: float,
    sidewall_oxide_height_um: float,
    include_metal_stack: bool,
    ti_thickness_um: float,
    pd_thickness_um: float,
    enable_traps: bool,
    trap_ea_mev: float,
    trap_nga1_cm3: float,
    trap_nga2_cm3: float,
    trap_y1_um: float,
    trap_y2_um: float,
    trap_sigma1_um: float,
    trap_sigma2_um: float,
    trap_lateral_factor: float,
    trap_strength: float,
) -> None:
    # ####### PROCESS / GEOMETRY SETUP (Silvaco-oriented structure proxy) #######
    create_2d_mesh(mesh="m")
    tside_cm = max(0.0, sidewall_oxide_thickness_um * 1e-4) if include_sidewall_oxide else 0.0
    hside_cm = max(0.0, sidewall_oxide_height_um * 1e-4) if include_sidewall_oxide else 0.0
    tti_cm = max(0.0, ti_thickness_um * 1e-4) if include_metal_stack else 0.0
    tpd_cm = max(0.0, pd_thickness_um * 1e-4) if include_metal_stack else 0.0
    pad_cm = max(0.1 * width_cm, 5e-6, 1.5 * tside_cm)
    x_left = -pad_cm
    x_si_l = 0.0
    x_si_r = width_cm
    x_right = width_cm + pad_cm
    # ---------- X-direction mesh (lateral) ----------
    # local refinement near contacts, junction, and implant center.
    x_lines: list[tuple[float, float]] = [
        (x_left, max(pad_cm / 20.0, 1e-8)),
        (x_si_l, max(width_cm / 200.0, 1e-8)),
        (junction_x_cm, max(width_cm / 400.0, 5e-9)),
        (x_si_r, max(width_cm / 200.0, 1e-8)),
        (x_right, max(pad_cm / 20.0, 1e-8)),
    ]
    if tside_cm > 0.0:
        x_lines.append((x_si_l - tside_cm, max(tside_cm / 5.0, 1e-9)))
        x_lines.append((x_si_r + tside_cm, max(tside_cm / 5.0, 1e-9)))
    # Keep local refinement bounded to avoid OOM on large 2D windows.
    local_dx = max(min(width_cm / 400.0, 2e-6), 1e-8)
    for xc in (x_si_l, junction_x_cm, x_si_r):
        for d in (5e-8, 2e-7, 1e-6):
            for sgn in (-1.0, 1.0):
                xp = xc + sgn * d
                if x_si_l < xp < x_si_r:
                    x_lines.append((xp, local_dx))
    if metal_width_cm is not None:
        x_impl = 0.5 * width_cm
        sx_ref = max(metal_width_cm * implant_lateral_factor, 1e-8)
        for mul in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
            xp = x_impl + mul * sx_ref
            if x_si_l < xp < x_si_r:
                x_lines.append((xp, max(local_dx * 0.8, 1e-8)))
    for pos, ps in sorted(x_lines, key=lambda t: t[0]):
        add_2d_mesh_line(mesh="m", dir="x", pos=pos, ps=ps)
    if include_metal_stack and metal_width_cm is not None:
        mleft = max(x_si_l, min(x_si_r, metal_width_cm))
        mright = min(x_si_r, max(x_si_l, x_si_r - metal_width_cm))
        add_2d_mesh_line(mesh="m", dir="x", pos=mleft, ps=max(metal_width_cm / 40.0, 1e-8))
        add_2d_mesh_line(mesh="m", dir="x", pos=mright, ps=max(metal_width_cm / 40.0, 1e-8))

    # ---------- Y-direction mesh (vertical) ----------
    # refined near surface, implant peak/tail, and optional stack layers.
    y_lines: list[tuple[float, float]] = [
        (0.0, max(depth_cm / 150.0, 1e-8)),
        (0.5 * depth_cm, max(depth_cm / 100.0, 2e-8)),
        (depth_cm, max(depth_cm / 80.0, 2e-8)),
    ]
    tox_cm = max(0.0, native_oxide_thickness_um * 1e-4) if include_oxide else 0.0
    if tox_cm > 0.0:
        y_lines.append((-tox_cm, max(tox_cm / 4.0, 1e-9)))
    if tti_cm > 0.0:
        y_lines.append((-(tox_cm + tti_cm), max(tti_cm / 3.0, 1e-9)))
    if tpd_cm > 0.0:
        y_lines.append((-(tox_cm + tti_cm + tpd_cm), max(tpd_cm / 8.0, 1e-9)))
    if hside_cm > 0.0:
        y_lines.append((hside_cm, max(hside_cm / 20.0, 1e-9)))
    sy_cm = max(implant_sigma_um * 1e-4, 1e-8)
    y0_cm = max(0.0, min(depth_cm, implant_peak_um * 1e-4))
    fine_dy = max(min(depth_cm / 300.0, 2e-6), 1e-8)
    for yp in (5e-8, 2e-7, 1e-6, 5e-6):
        if 0.0 < yp < depth_cm:
            y_lines.append((yp, fine_dy))
    for mul in (0.5, 1.0, 2.0, 3.0, 5.0):
        for sgn in (-1.0, 1.0):
            yp = y0_cm + sgn * mul * sy_cm
            if 0.0 < yp < depth_cm:
                y_lines.append((yp, max(fine_dy * 0.8, 1e-8)))
    for pos, ps in sorted(y_lines, key=lambda t: t[0]):
        add_2d_mesh_line(mesh="m", dir="y", pos=pos, ps=ps)

    # ---------- Regions ----------
    # silicon bulk + optional oxide/metal proxy regions.
    oxide_regions: list[str] = []
    add_2d_region(mesh="m", material="Si", region=REGION, xl=x_si_l, xh=x_si_r, yl=0.0, yh=depth_cm)
    if tside_cm > 0.0 and hside_cm > 0.0:
        # Split gas around sidewall-oxide strips to avoid overlapping regions.
        add_2d_region(mesh="m", material="gas", region="gas_l_far", xl=x_left, xh=x_si_l - tside_cm, yl=0.0, yh=depth_cm)
        add_2d_region(mesh="m", material="gas", region="gas_l_low", xl=x_si_l - tside_cm, xh=x_si_l, yl=hside_cm, yh=depth_cm)
        add_2d_region(mesh="m", material="gas", region="gas_r_far", xl=x_si_r + tside_cm, xh=x_right, yl=0.0, yh=depth_cm)
        add_2d_region(mesh="m", material="gas", region="gas_r_low", xl=x_si_r, xh=x_si_r + tside_cm, yl=hside_cm, yh=depth_cm)
        add_2d_region(
            mesh="m",
            material="oxide",
            region="OxideLeft",
            xl=x_si_l - tside_cm,
            xh=x_si_l,
            yl=0.0,
            yh=min(hside_cm, depth_cm),
        )
        add_2d_region(
            mesh="m",
            material="oxide",
            region="OxideRight",
            xl=x_si_r,
            xh=x_si_r + tside_cm,
            yl=0.0,
            yh=min(hside_cm, depth_cm),
        )
        oxide_regions.extend(["OxideLeft", "OxideRight"])
    else:
        add_2d_region(mesh="m", material="gas", region="gas_l", xl=x_left, xh=x_si_l, yl=0.0, yh=depth_cm)
        add_2d_region(mesh="m", material="gas", region="gas_r", xl=x_si_r, xh=x_right, yl=0.0, yh=depth_cm)
    if tox_cm > 0.0:
        add_2d_region(mesh="m", material="oxide", region="OxideTop", xl=x_si_l, xh=x_si_r, yl=-tox_cm, yh=0.0)
        oxide_regions.append("OxideTop")

    # ---------- Metal stack ----------
    # optional explicit Ti/Pd layers over source/drain windows.
    if include_metal_stack and metal_width_cm is not None and (tti_cm > 0.0 or tpd_cm > 0.0):
        mleft = max(x_si_l, min(x_si_r, metal_width_cm))
        mright = min(x_si_r, max(x_si_l, x_si_r - metal_width_cm))
        y_top = -tox_cm
        if tti_cm > 0.0:
            add_2d_region(
                mesh="m",
                material="titanium",
                region="TiLeft",
                xl=x_si_l,
                xh=mleft,
                yl=y_top - tti_cm,
                yh=y_top,
            )
            add_2d_region(
                mesh="m",
                material="titanium",
                region="TiRight",
                xl=mright,
                xh=x_si_r,
                yl=y_top - tti_cm,
                yh=y_top,
            )
            y_top = y_top - tti_cm
        if tpd_cm > 0.0:
            add_2d_region(
                mesh="m",
                material="palladium",
                region="PdLeft",
                xl=x_si_l,
                xh=mleft,
                yl=y_top - tpd_cm,
                yh=y_top,
            )
            add_2d_region(
                mesh="m",
                material="palladium",
                region="PdRight",
                xl=mright,
                xh=x_si_r,
                yl=y_top - tpd_cm,
                yh=y_top,
            )

    # ---------- Contacts ----------
    # contacts are placed on silicon sidewalls for robust DEVSIM contact detection.
    bloat = max(1e-10, min(width_cm, depth_cm) * 1e-4)
    add_2d_contact(
        mesh="m",
        name=LEFT,
        region=REGION,
        material="metal",
        xl=x_si_l,
        xh=x_si_l,
        yl=0.0,
        yh=depth_cm,
        bloat=bloat,
    )
    add_2d_contact(
        mesh="m",
        name=RIGHT,
        region=REGION,
        material="metal",
        xl=x_si_r,
        xh=x_si_r,
        yl=0.0,
        yh=depth_cm,
        bloat=bloat,
    )
    finalize_mesh(mesh="m")
    create_device(mesh="m", device=DEVICE)
    contacts = set(get_contact_list(device=DEVICE))
    if LEFT not in contacts or RIGHT not in contacts:
        raise RuntimeError(f"2D contact creation failed. Contacts: {sorted(contacts)}")

    # ####### DEVICE PHYSICS #######
    SetSiliconParameters(DEVICE, REGION, temp_k)
    set_parameter(device=DEVICE, region=REGION, name="taun", value=taun_s)
    set_parameter(device=DEVICE, region=REGION, name="taup", value=taup_s)
    for oxide_region in oxide_regions:
        SetOxideParameters(DEVICE, oxide_region, temp_k)

    # ---------- Doping ----------
    if doping_mode == "abrupt":
        CreateNodeModel(DEVICE, REGION, "Acceptors", f"{na:.8e}*step({junction_x_cm:.8e}-x)")
        CreateNodeModel(DEVICE, REGION, "Donors", f"{nd:.8e}*step(x-{junction_x_cm:.8e})")
    else:
        # Silvaco-like Gaussian implant profile parameters (converted to cm units).
        sy_cm = max(implant_sigma_um * 1e-4, 1e-8)
        sx_base_cm = metal_width_cm if metal_width_cm is not None else width_cm
        sx_cm = max((sx_base_cm * implant_lateral_factor), 1e-8)
        y0_cm = implant_peak_um * 1e-4
        x0_cm = 0.5 * width_cm
        peak_cm3 = implant_dose_cm2 / (math.sqrt(2.0 * math.pi) * sy_cm)
        implant_expr = (
            f"{peak_cm3:.8e}"
            f"*exp(-0.5*pow((x-{x0_cm:.8e})/{sx_cm:.8e},2))"
            f"*exp(-0.5*pow((y-{y0_cm:.8e})/{sy_cm:.8e},2))"
        )
        if implant_species == "boron":
            CreateNodeModel(DEVICE, REGION, "Acceptors", f"{na:.8e} + ({implant_expr})")
            CreateNodeModel(DEVICE, REGION, "Donors", f"{nd:.8e}")
        else:
            CreateNodeModel(DEVICE, REGION, "Acceptors", f"{na:.8e}")
            CreateNodeModel(DEVICE, REGION, "Donors", f"{nd:.8e} + ({implant_expr})")
    CreateNodeModel(DEVICE, REGION, "NetDoping", "Donors-Acceptors")

    # ---------- Initialization ----------
    CreateSolution(DEVICE, REGION, "Potential")
    CreateSiliconPotentialOnly(DEVICE, REGION)
    for oxide_region in oxide_regions:
        CreateSolution(DEVICE, oxide_region, "Potential")
        CreateOxidePotentialOnly(DEVICE, oxide_region)
    if oxide_regions:
        for interface_name in get_interface_list(device=DEVICE):
            try:
                CreateSiliconOxideInterface(DEVICE, interface_name)
            except Exception:
                # Ignore non Si/oxide interfaces.
                pass
    for c in get_contact_list(device=DEVICE):
        set_parameter(device=DEVICE, name=GetContactBiasName(c), value=0.0)
        CreateSiliconPotentialOnlyContact(DEVICE, REGION, c)
    if not solve_dc():
        raise RuntimeError("Poisson initialization failed")

    # ---------- Drift-diffusion solve ----------
    CreateSolution(DEVICE, REGION, "Electrons")
    CreateSolution(DEVICE, REGION, "Holes")
    set_node_values(device=DEVICE, region=REGION, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=DEVICE, region=REGION, name="Holes", init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(DEVICE, REGION)
    if enable_traps:
        apply_gaussian_trap_block(
            device=DEVICE,
            region=REGION,
            width_cm=width_cm,
            metal_width_cm=metal_width_cm,
            ea_mev=trap_ea_mev,
            nga1_cm3=trap_nga1_cm3,
            nga2_cm3=trap_nga2_cm3,
            y1_um=trap_y1_um,
            y2_um=trap_y2_um,
            sigma1_um=trap_sigma1_um,
            sigma2_um=trap_sigma2_um,
            lateral_factor=trap_lateral_factor,
            trap_strength=trap_strength,
        )
    for c in get_contact_list(device=DEVICE):
        CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, c)
    if not solve_dc():
        raise RuntimeError("DD initialization failed")


def current_at(contact: str) -> float:
    ie = get_contact_current(device=DEVICE, contact=contact, equation="ElectronContinuityEquation")
    ih = get_contact_current(device=DEVICE, contact=contact, equation="HoleContinuityEquation")
    return ie + ih


def move_bias(bias_name: str, v_from: float, v_to: float, max_step: float, min_step: float) -> float:
    v_now = v_from
    while abs(v_to - v_now) > 1e-15:
        dv = v_to - v_now
        trial = v_now + math.copysign(min(abs(dv), max_step), dv)
        set_parameter(device=DEVICE, name=bias_name, value=trial)
        if solve_dc():
            v_now = trial
            continue
        set_parameter(device=DEVICE, name=bias_name, value=v_now)
        h = 0.5 * abs(trial - v_now)
        ok = False
        while h >= min_step:
            trial = v_now + math.copysign(min(abs(v_to - v_now), h), dv)
            set_parameter(device=DEVICE, name=bias_name, value=trial)
            if solve_dc():
                v_now = trial
                ok = True
                break
            set_parameter(device=DEVICE, name=bias_name, value=v_now)
            h *= 0.5
        if not ok:
            raise RuntimeError(f"Convergence failure between {v_now:.6g} and {v_to:.6g}")
    return v_now


def _seq(a: float, b: float, step: float) -> list[float]:
    vals: list[float] = []
    if step <= 0:
        raise ValueError("step must be > 0")
    if a <= b:
        v = a
        while v <= b + 1e-15:
            vals.append(round(v, 12))
            v += step
    else:
        v = a
        while v >= b - 1e-15:
            vals.append(round(v, 12))
            v -= step
    return vals


def sparse_key_targets(vmin: float, vmax: float) -> tuple[list[float], list[float]]:
    """
    Return sparse reverse/forward bias targets using only key checkpoints.
    This preserves the staged measurement concept while minimizing solve count.
    """
    if vmin >= 0.0 or vmax <= 0.0:
        raise ValueError("Need vmin < 0 < vmax")

    # Moderate-density logarithmic checkpoints: still light enough for low-memory runs.
    key_neg = [-1e-3, -3e-3, -1e-2, -3e-2, -1e-1, -3e-1, -5e-1, -1.0, -1.5, -3.0, -10.0]
    key_pos = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 1.0, 1.5, 3.0, 10.0]

    rev = [v for v in key_neg if v >= vmin - 1e-15]
    fwd = [v for v in key_pos if v <= vmax + 1e-15]

    if vmin < 0.0 and (not rev or abs(rev[-1] - vmin) > 1e-12):
        rev.append(vmin)
    if vmax > 0.0 and (not fwd or abs(fwd[-1] - vmax) > 1e-12):
        fwd.append(vmax)

    rev = list(dict.fromkeys(round(v, 12) for v in rev))
    fwd = list(dict.fromkeys(round(v, 12) for v in fwd))
    return rev, fwd


def sparse_targets_one_sided(vmin: float, vmax: float) -> list[float]:
    if vmin < vmax <= 0.0:
        key = [-1e-3, -3e-3, -1e-2, -3e-2, -1e-1, -3e-1, -5e-1, -1.0, -1.5, -3.0, -10.0]
        vals = [v for v in key if vmin - 1e-15 <= v <= vmax + 1e-15]
    elif 0.0 <= vmin < vmax:
        key = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 1.0, 1.5, 3.0, 10.0]
        vals = [v for v in key if vmin - 1e-15 <= v <= vmax + 1e-15]
    else:
        raise ValueError("One-sided range must not cross 0")
    bounds = [round(vmin, 12), round(vmax, 12)]
    vals = list(dict.fromkeys([*bounds, *(round(v, 12) for v in vals)]))
    vals.sort()
    return vals


def make_silvaco_like_targets(vmin: float, vmax: float) -> tuple[list[float], list[float]]:
    """
    Return (reverse_targets, forward_targets), both excluding 0 V.
    Uses a Silvaco-like piecewise schedule mirrored around 0 V.
    """
    if vmin >= 0.0 or vmax <= 0.0:
        raise ValueError("Need vmin < 0 < vmax")

    rev: list[float] = []
    if vmin < 0.0:
        rev_sched = [
            (-0.001, 0.001),
            (-0.01, 0.001),
            (-0.1, 0.001),
            (-1.0, 0.001),
            (-10.0, 0.01),
        ]
        vprev = 0.0
        for vend, step in rev_sched:
            if vmin > vprev and vmin > vend:
                continue
            seg_end = max(vmin, vend)
            if seg_end >= vprev:
                continue
            pts = _seq(vprev - min(step, abs(vprev - seg_end)), seg_end, step)
            rev.extend([v for v in pts if v < 0.0 and v >= vmin - 1e-15])
            vprev = seg_end
            if abs(vprev - vmin) < 1e-15:
                break
        if vprev > vmin:
            step = 0.01 if abs(vprev) >= 1.0 else (0.001 if abs(vprev) <= 1.0 else 0.01)
            pts = _seq(vprev - min(step, abs(vprev - vmin)), vmin, step)
            rev.extend([v for v in pts if v < 0.0])

    fwd: list[float] = []
    if vmax > 0.0:
        fwd_sched = [
            (0.01, 0.001),
            (0.1, 0.001),
            (1.0, 0.001),
            (10.0, 0.01),
        ]
        vprev = 0.0
        for vend, step in fwd_sched:
            if vmax < vprev and vmax < vend:
                continue
            seg_end = min(vmax, vend)
            if seg_end <= vprev:
                continue
            pts = _seq(vprev + min(step, abs(seg_end - vprev)), seg_end, step)
            fwd.extend([v for v in pts if v > 0.0 and v <= vmax + 1e-15])
            vprev = seg_end
            if abs(vprev - vmax) < 1e-15:
                break
        if vprev < vmax:
            step = 0.01 if vprev >= 1.0 else 0.001
            pts = _seq(vprev + min(step, abs(vmax - vprev)), vmax, step)
            fwd.extend([v for v in pts if v > 0.0])

    # Deduplicate while preserving order
    rev = list(dict.fromkeys(rev))
    fwd = list(dict.fromkeys(fwd))
    return rev, fwd


def make_silvaco_short_targets(vmin: float, vmax: float) -> tuple[list[float], list[float]]:
    """
    Silvaco-style staged sweep without the 1.01->10 V segment.
    Stages:
      reverse: -1e-3 -> -1e-2, -1.1e-2 -> -1e-1, -1.01e-1 -> -1.0 (step 1e-3)
      forward:  1e-3 ->  1e-2,  1.1e-2 ->  1e-1,  1.01e-1 ->  1.0 (step 1e-3)
    Reverse sweep is mirrored to the positive magnitude when allowed by vmin.
    """
    if vmin >= 0.0 or vmax <= 0.0:
        raise ValueError("Need vmin < 0 < vmax")

    rev: list[float] = []
    rev_target = min(abs(vmin), abs(vmax))
    rev_sched = [
        (-1e-3, -1e-2, 1e-3),
        (-1.1e-2, -1e-1, 1e-3),
        (-1.01e-1, -1.0, 1e-3),
    ]
    for a, b, h in rev_sched:
        if rev_target < abs(a):
            break
        start = -min(abs(a), rev_target)
        end = -min(abs(b), rev_target)
        if start >= end:
            rev.extend([v for v in _seq(start, end, h) if v < 0.0 and v >= vmin - 1e-15])
    if not rev and vmin < 0.0:
        rev = [round(vmin, 12)]

    fwd: list[float] = []
    fwd_sched = [
        (1e-3, 1e-2, 1e-3),
        (1.1e-2, 1e-1, 1e-3),
        (1.01e-1, 1.0, 1e-3),
    ]
    for a, b, h in fwd_sched:
        if vmax < a:
            break
        start = max(a, 0.0 if not fwd else a)
        end = min(b, vmax)
        if end >= start:
            fwd.extend([v for v in _seq(start, end, h) if v > 0.0])
    if not fwd and vmax > 0.0:
        fwd = [round(vmax, 12)]

    rev = list(dict.fromkeys(round(v, 12) for v in rev))
    fwd = list(dict.fromkeys(round(v, 12) for v in fwd))
    return rev, fwd


def make_points_between(vstart: float, vend: float) -> list[float]:
    if abs(vend - vstart) < 1e-15:
        return [round(vstart, 12)]
    lo = min(vstart, vend)
    hi = max(vstart, vend)
    segs = [
        (lo, min(hi, -1.0), 0.1),
        (max(lo, -1.0), min(hi, -0.1), 0.01),
        (max(lo, -0.1), min(hi, 0.1), 0.001),
        (max(lo, 0.1), min(hi, 1.0), 0.001),
        (max(lo, 1.0), hi, 0.01),
    ]
    vals: list[float] = []
    for a, b, h in segs:
        if a > b:
            continue
        v = a
        while v <= b + 1e-15:
            vals.append(round(v, 12))
            v += h
    vals = list(dict.fromkeys(vals))
    if vstart > vend:
        vals.reverse()
    return vals


def run_iv(
    vmin: float,
    vmax: float,
    max_step: float,
    min_step: float,
    sweep_mode: str = "adaptive",
    contact_offset_v: float = 0.0,
    fallback_retries: int = 3,
) -> list[tuple[float, float]]:
    bname = GetContactBiasName(RIGHT)
    set_parameter(device=DEVICE, name=bname, value=contact_offset_v)
    if not solve_dc():
        raise RuntimeError("Cannot converge at 0V")
    out: list[tuple[float, float]] = []

    if vmin < 0.0 < vmax:
        out.append((0.0, current_at(RIGHT)))

        # Reverse branch first: 0 -> vmin (helps keep the branch interpretation explicit).
        vprev = contact_offset_v
        if sweep_mode == "sparse":
            rev_targets, fwd_targets = sparse_key_targets(vmin, vmax)
        elif sweep_mode == "silvaco_short":
            rev_targets, fwd_targets = make_silvaco_short_targets(vmin, vmax)
        elif sweep_mode == "silvaco":
            rev_targets, fwd_targets = make_silvaco_like_targets(vmin, vmax)
        else:
            rev_targets = [v for v in make_points_between(0.0, vmin) if v < 0.0]
            fwd_targets = [v for v in make_points_between(0.0, vmax) if v > 0.0]

        for v in rev_targets:
            print(f"BIAS_TARGET {v:.12g}")
            step = max_step
            if abs(v) <= 0.1:
                step = min(step, 0.001)
            elif abs(v) <= 1.0:
                step = min(step, 0.005)
            veff = v + contact_offset_v
            vprev = robust_move_bias(
                bname,
                vprev,
                veff,
                max_step=step,
                min_step=min_step,
                fallback_retries=fallback_retries,
                anchor_bias=contact_offset_v,
            )
            i_now = current_at(RIGHT)
            out.append((v, i_now))
            print(f"BIAS_DONE {v:.12g} {i_now:.8e}")

        # Re-initialize at 0 V before forward branch (Silvaco-style staged sweep behavior).
        vprev = robust_move_bias(
            bname,
            vprev,
            contact_offset_v,
            max_step=min(max_step, 0.01),
            min_step=min_step,
            fallback_retries=fallback_retries,
            anchor_bias=contact_offset_v,
        )
        set_parameter(device=DEVICE, name=bname, value=contact_offset_v)
        if not solve_dc():
            raise RuntimeError("Cannot re-converge at 0V before forward sweep")

        vprev = contact_offset_v
        for v in fwd_targets:
            print(f"BIAS_TARGET {v:.12g}")
            step = max_step
            if v <= 0.1:
                step = min(step, 0.001)
            elif v <= 1.0:
                step = min(step, 0.002)
            elif v <= 10.0:
                step = min(step, 0.01)
            veff = v + contact_offset_v
            vprev = robust_move_bias(
                bname,
                vprev,
                veff,
                max_step=step,
                min_step=min_step,
                fallback_retries=fallback_retries,
                anchor_bias=contact_offset_v,
            )
            i_now = current_at(RIGHT)
            out.append((v, i_now))
            print(f"BIAS_DONE {v:.12g} {i_now:.8e}")
    else:
        if abs(vmax - vmin) < 1e-15:
            targets = [round(vmin, 12)]
        elif sweep_mode == "sparse":
            targets = sparse_targets_one_sided(vmin, vmax)
        else:
            hi = max(abs(vmin), abs(vmax))
            if vmin >= 0.0:
                targets = [v for v in make_points_between(0.0, hi) if vmin <= v <= vmax]
            else:
                targets = [v for v in make_points_between(0.0, -hi) if vmin <= v <= vmax]
        vprev = contact_offset_v
        for v in targets:
            print(f"BIAS_TARGET {v:.12g}")
            step = max_step
            av = abs(v)
            if av <= 0.1:
                step = min(step, 0.001)
            elif av <= 1.0:
                step = min(step, 0.005 if v < 0.0 else 0.002)
            elif av <= 10.0:
                step = min(step, 0.01)
            veff = v + contact_offset_v
            vprev = robust_move_bias(
                bname,
                vprev,
                veff,
                max_step=step,
                min_step=min_step,
                fallback_retries=fallback_retries,
                anchor_bias=contact_offset_v,
            )
            i_now = current_at(RIGHT)
            out.append((v, i_now))
            print(f"BIAS_DONE {v:.12g} {i_now:.8e}")

    out.sort(key=lambda t: t[0])
    # Drop accidental duplicate voltage points (keep the later one).
    dedup: dict[float, float] = {}
    for v, i in out:
        dedup[round(v, 12)] = i
    return sorted(dedup.items(), key=lambda t: t[0])


def slope_data(iv: list[tuple[float, float]]) -> list[tuple[float, float, str]]:
    arr = np.array(iv, dtype=float)
    v = arr[:, 0]
    i = np.abs(arr[:, 1])
    out: list[tuple[float, float, str]] = []
    pos = v > 1e-8
    if np.count_nonzero(pos) >= 3:
        vp = v[pos]
        ip = i[pos]
        sp = np.gradient(np.log10(ip), np.log10(vp))
        out.extend((float(x), float(y), "forward") for x, y in zip(vp, sp))
    neg = v < -1e-8
    if np.count_nonzero(neg) >= 3:
        vn = v[neg]
        inn = i[neg]
        sn = np.gradient(np.log10(inn), np.log10(np.abs(vn)))
        out.extend((float(x), float(y), "reverse") for x, y in zip(vn, sn))
    return out


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def maybe_reset_devsim() -> None:
    try:
        reset_devsim()
    except Exception:
        pass


def figure_tag(run_number: int) -> str:
    return f"{datetime.now().strftime('%B%d')}_run{run_number}"


def validate_planar_review_args(args: argparse.Namespace) -> None:
    # Scientific-review guardrails for planar workflow.
    if args.geometry_mode != "silvaco_window":
        raise SystemExit("--planar-review requires --geometry-mode silvaco_window")
    if args.doping_mode != "gaussian_implant":
        raise SystemExit("--planar-review requires --doping-mode gaussian_implant")
    if args.implant_species != "boron":
        raise SystemExit("--planar-review requires --implant-species boron")
    if not (args.nd > 0.0 and args.na > 0.0):
        raise SystemExit("--planar-review requires both --nd > 0 and --na > 0")
    if args.vmin >= args.vmax:
        raise SystemExit("--planar-review requires vmin < vmax")


def write_run_manifest(
    args: argparse.Namespace,
    prefix: str,
    *,
    metal_width_nm: float | None,
    contact_spacing_um: float | None,
    depth_um: float | None,
    temp_k: float,
    vmin: float,
    vmax: float,
    status: str,
    message: str = "",
) -> Path:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prefix": prefix,
        "output_dir": str(OUT_DIR),
        "figures_dir": str(FIG_DIR),
        "status": status,
        "message": message,
        "mode_planar_review": bool(args.planar_review),
        "geometry_mode": args.geometry_mode,
        "metal_width_nm": metal_width_nm,
        "contact_spacing_um": contact_spacing_um,
        "depth_um": depth_um,
        "temp_k": temp_k,
        "vmin": vmin,
        "vmax": vmax,
        "sweep_mode": args.sweep_mode,
        "doping_mode": args.doping_mode,
        "include_oxide": bool(args.include_oxide),
        "native_oxide_thickness_um": args.native_oxide_thickness_um,
        "include_sidewall_oxide": bool(args.include_sidewall_oxide),
        "sidewall_oxide_thickness_um": args.sidewall_oxide_thickness_um,
        "sidewall_oxide_height_um": args.sidewall_oxide_height_um,
        "include_metal_stack": bool(args.include_metal_stack),
        "ti_thickness_um": args.ti_thickness_um,
        "pd_thickness_um": args.pd_thickness_um,
        "implant_species": args.implant_species,
        "implant_dose_cm2": args.implant_dose_cm2,
        "implant_peak_um": args.implant_peak_um,
        "implant_sigma_um": args.implant_sigma_um,
        "implant_lateral_factor": args.implant_lateral_factor,
        "enable_traps": bool(args.enable_traps),
        "trap_ea_mev": args.trap_ea_mev,
        "trap_nga1_cm3": args.trap_nga1_cm3,
        "trap_nga2_cm3": args.trap_nga2_cm3,
        "trap_y1_um": args.trap_y1_um,
        "trap_y2_um": args.trap_y2_um,
        "trap_sigma1_um": args.trap_sigma1_um,
        "trap_sigma2_um": args.trap_sigma2_um,
        "trap_lateral_factor": args.trap_lateral_factor,
        "trap_strength": args.trap_strength,
        "nd_cm3": args.nd,
        "na_cm3": args.na,
        "taun_s": args.taun_s,
        "taup_s": args.taup_s,
        "contact_mode": args.contact_mode,
        "wf_ev": args.wf_ev,
        "wf_boron_ev": args.wf_boron_ev,
        "wf_arsenic_ev": args.wf_arsenic_ev,
        "chi_si_ev": args.chi_si_ev,
        "eg_si_ev": args.eg_si_ev,
        "schottky_ref_barrier_ev": args.schottky_ref_barrier_ev,
        "max_step": args.max_step,
        "min_step": args.min_step,
        "fallback_retries": args.fallback_retries,
    }
    path = OUT_DIR / f"{prefix}_manifest.json"
    u.write_json(path, manifest)
    return path


def plot_iv(path: Path, iv: list[tuple[float, float]], xmin: float, xmax: float) -> None:
    arr = np.array(iv, dtype=float)
    v = arr[:, 0]
    i = arr[:, 1]
    vis = (v >= xmin) & (v <= xmax)
    v = v[vis]
    i = i[vis]
    m = float(np.max(np.abs(i))) if i.size else 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(v, i, marker="o", markersize=2, linewidth=1.0, color="C0")
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=max(1e-18, 1e-6 * m))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("I (A/cm), signed")
    ax.set_title(f"Minimal 2D IV ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_slope(path: Path, rows: list[tuple[float, float, str]], xmin: float, xmax: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if rows:
        pts = np.array([(abs(v), s) for v, s, _ in rows if xmin <= v <= xmax and abs(v) > 1e-12], dtype=float)
        if pts.size:
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2, linewidth=1.0, color="C0")
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    x_low = max(min(abs(xmin), abs(xmax), 1.0), 1e-6)
    x_high = max(abs(xmin), abs(xmax), x_low * 10.0)
    ax.set_xscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("|V| (V), log10 scale")
    ax.set_ylabel("d(log10 I)/d(log10 V)")
    ax.set_title(f"Slope vs |V| ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined_iv(path: Path, cases: list[tuple[float, list[tuple[float, float]]]], xmin: float, xmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    max_abs_i = 1.0
    for width_nm, iv in cases:
        arr = np.array(iv, dtype=float)
        v = arr[:, 0]
        i = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        v = v[vis]
        i = i[vis]
        if i.size:
            max_abs_i = max(max_abs_i, float(np.max(np.abs(i))))
            ax.plot(v, i, marker="o", markersize=2, linewidth=1.0, label=f"{width_nm:g} nm")
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=max(1e-18, 1e-6 * max_abs_i))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("I (A/cm), signed")
    ax.set_title(f"Combined IV vs Metal Width ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="Width")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined_slope(
    path: Path,
    cases: list[tuple[float, list[tuple[float, float, str]]]],
    xmin: float,
    xmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for width_nm, rows in cases:
        pts = np.array([(abs(v), s) for v, s, _ in rows if xmin <= v <= xmax and abs(v) > 1e-12], dtype=float)
        if pts.size:
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2, linewidth=1.0, label=f"{width_nm:g} nm")
    x_low = max(min(abs(xmin), abs(xmax), 1.0), 1e-6)
    x_high = max(abs(xmin), abs(xmax), x_low * 10.0)
    ax.set_xscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("|V| (V), log10 scale")
    ax.set_ylabel("d(log10 I)/d(log10 V)")
    ax.set_title(f"Combined Slope vs Metal Width ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="Width")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined_iv_labeled(
    path: Path,
    cases: list[tuple[str, list[tuple[float, float]]]],
    xmin: float,
    xmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    # If labels are in "<species>_<width>nm" form, enforce:
    #   color -> width, linestyle -> species
    parsed: list[tuple[str, str | None, float | None, list[tuple[float, float]]]] = []
    width_vals: list[float] = []
    for label, iv in cases:
        m = re.match(r"^(boron|arsenic)_([0-9]+(?:\.[0-9]+)?)nm$", label)
        if m:
            species = m.group(1)
            width_nm = float(m.group(2))
            width_vals.append(width_nm)
            parsed.append((label, species, width_nm, iv))
        else:
            parsed.append((label, None, None, iv))
    width_to_color: dict[float, str] = {}
    if width_vals:
        cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        for i, w in enumerate(sorted(set(width_vals))):
            width_to_color[w] = cmap[i % len(cmap)]
    species_to_style = {"boron": "-", "arsenic": "--"}

    max_abs_i = 1.0
    for label, species, width_nm, iv in parsed:
        arr = np.array(iv, dtype=float)
        v = arr[:, 0]
        i = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        v = v[vis]
        i = i[vis]
        if i.size:
            max_abs_i = max(max_abs_i, float(np.max(np.abs(i))))
            kwargs: dict[str, object] = {"label": label, "marker": "o", "markersize": 2, "linewidth": 1.0}
            if width_nm is not None and width_nm in width_to_color:
                kwargs["color"] = width_to_color[width_nm]
            if species is not None and species in species_to_style:
                kwargs["linestyle"] = species_to_style[species]
            ax.plot(v, i, **kwargs)
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=max(1e-18, 1e-6 * max_abs_i))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("I (A/cm), signed")
    ax.set_title(f"Combined IV ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined_slope_labeled(
    path: Path,
    cases: list[tuple[str, list[tuple[float, float, str]]]],
    xmin: float,
    xmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    parsed: list[tuple[str, str | None, float | None, list[tuple[float, float, str]]]] = []
    width_vals: list[float] = []
    for label, rows in cases:
        m = re.match(r"^(boron|arsenic)_([0-9]+(?:\.[0-9]+)?)nm$", label)
        if m:
            species = m.group(1)
            width_nm = float(m.group(2))
            width_vals.append(width_nm)
            parsed.append((label, species, width_nm, rows))
        else:
            parsed.append((label, None, None, rows))
    width_to_color: dict[float, str] = {}
    if width_vals:
        cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        for i, w in enumerate(sorted(set(width_vals))):
            width_to_color[w] = cmap[i % len(cmap)]
    species_to_style = {"boron": "-", "arsenic": "--"}

    for label, species, width_nm, rows in parsed:
        pts = np.array([(abs(v), s) for v, s, _ in rows if xmin <= v <= xmax and abs(v) > 1e-12], dtype=float)
        if pts.size:
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            kwargs: dict[str, object] = {"label": label, "marker": "o", "markersize": 2, "linewidth": 1.0}
            if width_nm is not None and width_nm in width_to_color:
                kwargs["color"] = width_to_color[width_nm]
            if species is not None and species in species_to_style:
                kwargs["linestyle"] = species_to_style[species]
            ax.plot(pts[:, 0], pts[:, 1], **kwargs)
    x_low = max(min(abs(xmin), abs(xmax), 1.0), 1e-6)
    x_high = max(abs(xmin), abs(xmax), x_low * 10.0)
    ax.set_xscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("|V| (V), log10 scale")
    ax.set_ylabel("d(log10 I)/d(log10 V)")
    ax.set_title(f"Combined Slope ({xmin:g}..{xmax:g} V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def export_band_diagram(prefix: str, temp_k: float, bias_v: float, run_number: int) -> tuple[Path, Path]:
    x = np.array(get_node_model_values(device=DEVICE, region=REGION, name="x"), dtype=float)
    y = np.array(get_node_model_values(device=DEVICE, region=REGION, name="y"), dtype=float)
    pot = np.array(get_node_model_values(device=DEVICE, region=REGION, name="Potential"), dtype=float)
    if x.size == 0 or y.size == 0 or pot.size == 0:
        raise RuntimeError("Cannot export band diagram: empty node models")

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-30)
    y_tol = max(0.01 * y_span, 5e-9)
    mask = np.abs(y - y_min) <= y_tol
    if np.count_nonzero(mask) < 3:
        # fallback to shallowest 5% nodes
        y_cut = y_min + 0.05 * y_span
        mask = y <= y_cut
    xs = x[mask]
    ys = y[mask]
    ps = pot[mask]
    if xs.size < 3:
        raise RuntimeError("Cannot export band diagram: not enough points on cutline")

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    ps = ps[order]

    # Proxy absolute bands in eV from electrostatic potential:
    # Ec = chi - q*phi, Ev = Ec - Eg(T)
    chi_si_ev = 4.05
    eg_ev = _si_bandgap_ev(temp_k)
    ec = chi_si_ev - ps
    ev = ec - eg_ev

    tok = u.bias_token(bias_v)
    csv_path = OUT_DIR / f"{prefix}_band_{tok}V.csv"
    png_path = FIG_DIR / f"{u.figure_tag(run_number)}_{prefix}_band_{tok}V.png"

    u.write_csv(
        csv_path,
        ["x_cm", "y_cm", "Potential_V", "Ec_eV", "Ev_eV"],
        list(zip(xs.tolist(), ys.tolist(), ps.tolist(), ec.tolist(), ev.tolist())),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x_um = xs * 1e4
    ax.plot(x_um, ec, linewidth=1.2, label="Ec")
    ax.plot(x_um, ev, linewidth=1.2, label="Ev")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Band Diagram at V={bias_v:g} V (surface cut)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return csv_path, png_path


def parse_elapsed_seconds(raw: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    # /usr/bin/time -v format examples: "0:12.34" or "1:02:03"
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            mm = float(parts[0])
            ss = float(parts[1])
            return 60.0 * mm + ss
        if len(parts) == 3:
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])
            return 3600.0 * hh + 60.0 * mm + ss
    except Exception:
        return 0.0
    return 0.0


def parse_stability_log(stdout_text: str) -> tuple[list[dict[str, float | int | str]], dict[float, list[tuple[int, float]]]]:
    points: list[dict[str, float | int | str]] = []
    curves: dict[float, list[tuple[int, float]]] = {}
    current_v: float | None = None
    current_iters = 0
    fallback_count = 0
    rel_start: float | None = None
    rel_end: float | None = None
    abs_start: float | None = None
    abs_end: float | None = None
    current_curve: list[tuple[int, float]] = []
    iter_index: int | None = None

    rel_abs_re = re.compile(r'RelError:\s*([0-9eE+\-.]+)\s+AbsError:\s*([0-9eE+\-.]+)')

    def flush_point(converged: int) -> None:
        nonlocal current_v, current_iters, fallback_count, rel_start, rel_end, abs_start, abs_end, current_curve
        if current_v is None:
            return
        rel_reduction = (rel_start / rel_end) if (rel_start is not None and rel_end not in (None, 0.0)) else math.nan
        abs_reduction = (abs_start / abs_end) if (abs_start is not None and abs_end not in (None, 0.0)) else math.nan
        points.append(
            {
                "voltage_V": current_v,
                "converged": converged,
                "newton_iterations": current_iters,
                "fallback_retries": fallback_count,
                "rel_error_start": rel_start if rel_start is not None else math.nan,
                "rel_error_end": rel_end if rel_end is not None else math.nan,
                "abs_error_start": abs_start if abs_start is not None else math.nan,
                "abs_error_end": abs_end if abs_end is not None else math.nan,
                "rel_reduction": rel_reduction,
                "abs_reduction": abs_reduction,
            }
        )
        curves[current_v] = current_curve[:]
        current_v = None
        current_iters = 0
        fallback_count = 0
        rel_start = None
        rel_end = None
        abs_start = None
        abs_end = None
        current_curve = []

    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("BIAS_TARGET "):
            # close previous unfinished target as non-converged
            flush_point(converged=0)
            try:
                current_v = float(line.split()[1])
            except Exception:
                current_v = None
            continue
        if line.startswith("FALLBACK_RETRY "):
            fallback_count += 1
            continue
        if line.startswith("Iteration:"):
            if current_v is not None:
                current_iters += 1
                try:
                    iter_index = int(line.split(":", 1)[1].strip())
                except Exception:
                    iter_index = None
            continue
        if line.startswith("Device:") and current_v is not None:
            m = rel_abs_re.search(line)
            if m:
                rel = float(m.group(1))
                ab = float(m.group(2))
                if rel_start is None:
                    rel_start = rel
                rel_end = rel
                if abs_start is None:
                    abs_start = ab
                abs_end = ab
                if iter_index is not None:
                    current_curve.append((iter_index, rel))
            continue
        if line.startswith("BIAS_DONE "):
            flush_point(converged=1)

    flush_point(converged=0)
    return points, curves


def parse_time_metrics(stderr_text: str) -> tuple[float, float]:
    elapsed_s = 0.0
    peak_rss_kb = 0.0
    for line in stderr_text.splitlines():
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            _, val = line.split(":", 1)
            elapsed_s = parse_elapsed_seconds(val.strip())
        elif line.startswith("Maximum resident set size (kbytes)"):
            _, val = line.split(":", 1)
            try:
                peak_rss_kb = float(val.strip())
            except Exception:
                peak_rss_kb = 0.0
    return elapsed_s, peak_rss_kb


def run_stability_batch(args: argparse.Namespace) -> int:
    widths = args.stability_widths_nm or args.batch_metal_widths_nm or [args.metal_width_nm]
    widths = [float(w) for w in widths]
    all_points: list[dict[str, float | int | str]] = []
    summary_rows: list[tuple[float, int, int, float, float, int, float, float, str, str]] = []
    curve_bank: dict[float, dict[float, list[tuple[int, float]]]] = {}
    ok_cases_iv: list[tuple[float, list[tuple[float, float]]]] = []
    ok_cases_sl: list[tuple[float, list[tuple[float, float, str]]]] = []

    for width_nm in widths:
        case_prefix = f"{args.prefix}_{int(round(width_nm))}nm"
        cmd = [
            "/usr/bin/time",
            "-v",
            sys.executable,
            str(Path(__file__).resolve()),
            "--stability-child",
            "--prefix",
            case_prefix,
            "--run-number",
            str(args.run_number),
            "--output-dir",
            str(OUT_DIR),
            "--figures-dir",
            str(FIG_DIR),
            "--temp-k",
            str(args.temp_k),
            "--taun-s",
            str(args.taun_s),
            "--taup-s",
            str(args.taup_s),
            "--contact-mode",
            args.contact_mode,
            "--wf-ev",
            str(args.wf_ev),
            "--wf-neutral-ev",
            str(args.wf_neutral_ev),
            "--contact-proxy-gain",
            str(args.contact_proxy_gain),
            "--na",
            str(args.na),
            "--nd",
            str(args.nd),
            "--doping-mode",
            args.doping_mode,
            *(["--include-oxide"] if args.include_oxide else []),
            "--native-oxide-thickness-um",
            str(args.native_oxide_thickness_um),
            *(["--include-sidewall-oxide"] if args.include_sidewall_oxide else []),
            "--sidewall-oxide-thickness-um",
            str(args.sidewall_oxide_thickness_um),
            "--sidewall-oxide-height-um",
            str(args.sidewall_oxide_height_um),
            *(["--include-metal-stack"] if args.include_metal_stack else []),
            "--ti-thickness-um",
            str(args.ti_thickness_um),
            "--pd-thickness-um",
            str(args.pd_thickness_um),
            "--implant-species",
            args.implant_species,
            "--implant-dose-cm2",
            str(args.implant_dose_cm2),
            "--implant-peak-um",
            str(args.implant_peak_um),
            "--implant-sigma-um",
            str(args.implant_sigma_um),
            "--arsenic-dose-cm2",
            str(args.arsenic_dose_cm2),
            "--arsenic-peak-um",
            str(args.arsenic_peak_um),
            "--arsenic-sigma-um",
            str(args.arsenic_sigma_um),
            "--implant-parameter-mode",
            args.implant_parameter_mode,
            "--implant-lateral-factor",
            str(args.implant_lateral_factor),
            *(["--enable-traps"] if args.enable_traps else []),
            "--trap-ea-mev",
            str(args.trap_ea_mev),
            "--trap-nga1-cm3",
            str(args.trap_nga1_cm3),
            "--trap-nga2-cm3",
            str(args.trap_nga2_cm3),
            "--trap-y1-um",
            str(args.trap_y1_um),
            "--trap-y2-um",
            str(args.trap_y2_um),
            "--trap-sigma1-um",
            str(args.trap_sigma1_um),
            "--trap-sigma2-um",
            str(args.trap_sigma2_um),
            "--trap-lateral-factor",
            str(args.trap_lateral_factor),
            "--trap-strength",
            str(args.trap_strength),
            "--geometry-mode",
            args.geometry_mode,
            "--metal-width-nm",
            str(width_nm),
            "--contact-spacing-um",
            str(args.contact_spacing_um),
            "--depth-um",
            str(args.depth_um),
            "--vmin",
            str(args.vmin),
            "--vmax",
            str(args.vmax),
            "--sweep-mode",
            args.sweep_mode,
            "--max-step",
            str(args.max_step),
            "--min-step",
            str(args.min_step),
            "--fallback-retries",
            str(args.fallback_retries),
        ]
        if args.geometry_mode == "simple":
            cmd.extend(["--width-cm", str(args.width_cm), "--depth-cm", str(args.depth_cm), "--junction-x-cm", str(args.junction_x_cm)])

        print(f"[STABILITY] running width={width_nm:g} nm")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        _ = time.time() - t0

        log_path = OUT_DIR / f"{case_prefix}_stability_stdout.log"
        time_path = OUT_DIR / f"{case_prefix}_stability_time.log"
        log_path.write_text(proc.stdout, encoding="utf-8")
        time_path.write_text(proc.stderr, encoding="utf-8")

        elapsed_s, peak_rss_kb = u.parse_time_metrics(proc.stderr)
        points, curves = u.parse_stability_log(proc.stdout)
        curve_bank[width_nm] = curves
        for row in points:
            row["width_nm"] = width_nm
            all_points.append(row)

        npts = len(points)
        nconv = sum(int(r["converged"]) for r in points)
        iters = [int(r["newton_iterations"]) for r in points]
        mean_iters = float(np.mean(iters)) if iters else 0.0
        max_iters = float(np.max(iters)) if iters else 0.0
        fb_total = int(sum(int(r["fallback_retries"]) for r in points))
        status = "ok" if proc.returncode == 0 else "fail"
        summary_rows.append(
            (
                width_nm,
                npts,
                nconv,
                mean_iters,
                max_iters,
                fb_total,
                elapsed_s,
                peak_rss_kb,
                status,
                f"rc={proc.returncode}",
            )
        )
        print(f"[STABILITY] width={width_nm:g} nm status={status} points={npts} converged={nconv} elapsed={elapsed_s:.1f}s rss={peak_rss_kb:.0f}KB")
        if status == "ok":
            iv_csv = OUT_DIR / f"{case_prefix}_iv.csv"
            sl_csv = OUT_DIR / f"{case_prefix}_slope.csv"
            if iv_csv.exists() and sl_csv.exists():
                try:
                    with iv_csv.open() as f:
                        rr = csv.DictReader(f)
                        iv = [(float(x["voltage_V"]), float(x["current_A_per_cm"])) for x in rr]
                    with sl_csv.open() as f:
                        rr = csv.DictReader(f)
                        sl = [
                            (
                                float(x["voltage_V"]),
                                float(x["dlog10I_dlog10V"]),
                                str(x.get("branch", "")),
                            )
                            for x in rr
                        ]
                    ok_cases_iv.append((width_nm, iv))
                    ok_cases_sl.append((width_nm, sl))
                except Exception:
                    pass

    # CSV exports
    pts_path = OUT_DIR / f"{args.prefix}_stability_points.csv"
    sum_path = OUT_DIR / f"{args.prefix}_stability_summary.csv"
    u.write_csv(
        pts_path,
        [
            "width_nm",
            "voltage_V",
            "converged",
            "newton_iterations",
            "fallback_retries",
            "rel_error_start",
            "rel_error_end",
            "abs_error_start",
            "abs_error_end",
            "rel_reduction",
            "abs_reduction",
        ],
        [
            (
                r["width_nm"],
                r["voltage_V"],
                r["converged"],
                r["newton_iterations"],
                r["fallback_retries"],
                r["rel_error_start"],
                r["rel_error_end"],
                r["abs_error_start"],
                r["abs_error_end"],
                r["rel_reduction"],
                r["abs_reduction"],
            )
            for r in all_points
        ],
    )
    u.write_csv(
        sum_path,
        [
            "width_nm",
            "points_total",
            "points_converged",
            "mean_iterations",
            "max_iterations",
            "fallback_total",
            "elapsed_s",
            "peak_rss_kb",
            "status",
            "message",
        ],
        summary_rows,
    )

    # Plot 1: iterations vs voltage
    iter_png = FIG_DIR / f"{u.figure_tag(args.run_number)}_{args.prefix}_stability_iters_vs_v.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for w in sorted(set(float(r["width_nm"]) for r in all_points)):
        rows = [r for r in all_points if float(r["width_nm"]) == w and int(r["converged"]) == 1]
        if not rows:
            continue
        rows.sort(key=lambda z: float(z["voltage_V"]))
        vv = [float(r["voltage_V"]) for r in rows]
        ii = [float(r["newton_iterations"]) for r in rows]
        ax.plot(vv, ii, marker="o", markersize=2, linewidth=1.0, label=f"{w:g} nm")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("Newton iterations per bias point")
    ax.set_title("Convergence Effort vs Voltage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="Width")
    fig.tight_layout()
    fig.savefig(iter_png, dpi=220)
    plt.close(fig)

    # Plot 2: relative error curves (representative biases) per width
    for width_nm in widths:
        curves = curve_bank.get(width_nm, {})
        if not curves:
            continue
        preferred = [0.01, 0.1, 0.5, 1.0]
        avail = sorted(curves.keys(), key=lambda x: abs(x))
        chosen: list[float] = []
        for p in preferred:
            pos = [v for v in avail if v > 0]
            if not pos:
                continue
            best = min(pos, key=lambda v: abs(v - p))
            if best not in chosen:
                chosen.append(best)
        if not chosen:
            chosen = avail[: min(4, len(avail))]

        fig, ax = plt.subplots(figsize=(8, 5))
        for v in chosen:
            pts = curves.get(v, [])
            if not pts:
                continue
            xs = [k for k, _ in pts]
            ys = [max(y, 1e-300) for _, y in pts]
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.0, label=f"V={v:g} V")
        ax.set_yscale("log")
        ax.set_xlabel("Newton iteration index")
        ax.set_ylabel("RelError (Device)")
        ax.set_title(f"Residual Curves, {width_nm:g} nm")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        outp = FIG_DIR / f"{u.figure_tag(args.run_number)}_{args.prefix}_error_curves_{int(round(width_nm))}nm.png"
        fig.savefig(outp, dpi=220)
        plt.close(fig)

    # Plot 3: runtime and peak RSS (report utility)
    util_png = FIG_DIR / f"{u.figure_tag(args.run_number)}_{args.prefix}_stability_runtime_rss.png"
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ws = [r[0] for r in summary_rows]
    tt = [r[6] for r in summary_rows]
    rr = [r[7] for r in summary_rows]
    ax1.plot(ws, tt, marker="o", linewidth=1.0, color="C0", label="elapsed_s")
    ax1.set_xlabel("Width (nm)")
    ax1.set_ylabel("Elapsed time (s)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(ws, rr, marker="s", linewidth=1.0, color="C1", label="peak_rss_kb")
    ax2.set_ylabel("Peak RSS (KB)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Runtime and Memory vs Width")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(util_png, dpi=220)
    plt.close(fig)

    # Combined IV/slope from the same single-pass runs (no second solve pass).
    if ok_cases_iv:
        comb_iv = FIG_DIR / f"{u.figure_tag(args.run_number)}_{args.prefix}_combined_iv.png"
        comb_sl = FIG_DIR / f"{u.figure_tag(args.run_number)}_{args.prefix}_combined_slope.png"
        u.plot_combined_iv(comb_iv, ok_cases_iv, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope(comb_sl, ok_cases_sl, xmin=args.vmin, xmax=args.vmax)
    else:
        comb_iv = None
        comb_sl = None

    print("Stability artifacts:")
    print(pts_path)
    print(sum_path)
    print(iter_png)
    print(util_png)
    if comb_iv is not None and comb_sl is not None:
        print(comb_iv)
        print(comb_sl)
    return 0 if any(r[8] == "ok" for r in summary_rows) else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEVSIM RNPU 2D pipeline (planar + etched-like proxy)")
    p.add_argument("--prefix", default="minimal_base", help="Output file prefix")
    p.add_argument(
        "--preset",
        choices=("none", "etched_like"),
        default="none",
        help="Optional parameter preset. etched_like applies Silvaco-oriented etched proxy defaults.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV/JSON/log output files. Default: <script_dir>/output",
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory for PNG figure files. Default: <script_dir>/figures",
    )
    p.add_argument(
        "--planar-review",
        action="store_true",
        help="Enable strict planar scientific-review mode with additional parameter validation and run manifests.",
    )
    p.add_argument("--run-number", type=int, default=1, help="Daily run counter used in figure filenames")
    p.add_argument("--temp-k", type=float, default=300.0)
    p.add_argument("--taun-s", type=float, default=1e-7, help="Electron lifetime [s] (Silvaco-like baseline)")
    p.add_argument("--taup-s", type=float, default=1e-7, help="Hole lifetime [s] (Silvaco-like baseline)")
    p.add_argument(
        "--contact-mode",
        choices=("neutral", "hole", "electron", "spec_ohmic", "schottky_approx"),
        default="neutral",
        help="Contact injection proxy mode. neutral=0 offset; hole/electron apply wf-derived effective bias shift; spec_ohmic uses boron/arsenic workfunction presets; schottky_approx uses barrier-based proxy.",
    )
    p.add_argument("--wf-ev", type=float, default=5.25, help="Workfunction used by contact proxy [eV].")
    p.add_argument("--wf-neutral-ev", type=float, default=4.61, help="Neutral reference workfunction [eV].")
    p.add_argument("--wf-boron-ev", type=float, default=5.25, help="Boron-case metal workfunction [eV] for spec_ohmic mode.")
    p.add_argument("--wf-arsenic-ev", type=float, default=4.17, help="Arsenic-case metal workfunction [eV] for spec_ohmic mode.")
    p.add_argument("--chi-si-ev", type=float, default=4.05, help="Silicon electron affinity [eV] for schottky_approx mode.")
    p.add_argument("--eg-si-ev", type=float, default=1.12, help="Silicon bandgap [eV] for schottky_approx mode.")
    p.add_argument(
        "--schottky-ref-barrier-ev",
        type=float,
        default=0.30,
        help="Reference Schottky barrier [eV] mapped to zero proxy offset in schottky_approx mode.",
    )
    p.add_argument(
        "--contact-proxy-gain",
        type=float,
        default=0.2,
        help="Converts wf delta [eV] to effective bias offset [V] for contact proxy.",
    )
    p.add_argument("--na", type=float, default=0.0)
    p.add_argument("--nd", type=float, default=1e15)
    p.add_argument("--doping-mode", choices=("abrupt", "gaussian_implant"), default="gaussian_implant")
    p.add_argument("--include-oxide", action="store_true", help="Include explicit top oxide region with Si/Oxide interface equations.")
    p.add_argument(
        "--native-oxide-thickness-um",
        type=float,
        default=0.003,
        help="Top oxide thickness [um] used when --include-oxide is enabled.",
    )
    p.add_argument(
        "--include-sidewall-oxide",
        action="store_true",
        help="Include explicit sidewall oxide strips near left/right Si boundaries.",
    )
    p.add_argument(
        "--sidewall-oxide-thickness-um",
        type=float,
        default=0.003,
        help="Sidewall oxide thickness [um] used when --include-sidewall-oxide is enabled.",
    )
    p.add_argument(
        "--sidewall-oxide-height-um",
        type=float,
        default=0.07,
        help="Sidewall oxide vertical extent into Si [um].",
    )
    p.add_argument(
        "--include-metal-stack",
        action="store_true",
        help="Include explicit Ti/Pd metal stack geometry above contact windows.",
    )
    p.add_argument("--ti-thickness-um", type=float, default=0.001, help="Titanium thickness [um] for --include-metal-stack.")
    p.add_argument("--pd-thickness-um", type=float, default=0.025, help="Palladium thickness [um] for --include-metal-stack.")
    p.add_argument("--implant-species", choices=("boron", "arsenic"), default="boron")
    p.add_argument("--implant-dose-cm2", type=float, default=6e13)
    p.add_argument("--implant-peak-um", type=float, default=2.927699e-03)
    p.add_argument("--implant-sigma-um", type=float, default=2.766469e-02)
    p.add_argument("--arsenic-dose-cm2", type=float, default=6.5e13)
    p.add_argument("--arsenic-peak-um", type=float, default=1.317234e-03)
    p.add_argument("--arsenic-sigma-um", type=float, default=2.7181546e-02)
    p.add_argument(
        "--implant-parameter-mode",
        choices=("fixed", "deck_by_species"),
        default="deck_by_species",
        help="fixed: use implant-dose/peak/sigma for any species; deck_by_species: boron/arsenic deck presets.",
    )
    p.add_argument("--implant-lateral-factor", type=float, default=1.0 / 6.0)
    p.add_argument("--enable-traps", action="store_true", help="Enable two-Gaussian trap approximation block (SRH lifetime scaling).")
    p.add_argument("--trap-ea-mev", type=float, default=400.0, help="Trap activation energy [meV].")
    p.add_argument("--trap-nga1-cm3", type=float, default=2e14, help="Trap Gaussian #1 amplitude [cm^-3].")
    p.add_argument("--trap-nga2-cm3", type=float, default=2e14, help="Trap Gaussian #2 amplitude [cm^-3].")
    p.add_argument("--trap-y1-um", type=float, default=0.02, help="Trap Gaussian #1 center depth [um].")
    p.add_argument("--trap-y2-um", type=float, default=0.12, help="Trap Gaussian #2 center depth [um].")
    p.add_argument("--trap-sigma1-um", type=float, default=0.03, help="Trap Gaussian #1 vertical sigma [um].")
    p.add_argument("--trap-sigma2-um", type=float, default=0.08, help="Trap Gaussian #2 vertical sigma [um].")
    p.add_argument("--trap-lateral-factor", type=float, default=1.0, help="Trap lateral sigma factor relative to metal width.")
    p.add_argument("--trap-strength", type=float, default=0.6, help="Multiplier mapping trap weight to lifetime reduction.")
    p.add_argument(
        "--batch-trap-ea-mev",
        type=float,
        nargs="*",
        default=None,
        help="Optional batch list of trap activation energies [meV]. Runs all values and generates combined plots.",
    )
    p.add_argument(
        "--geometry-mode",
        choices=("simple", "silvaco_window"),
        default="simple",
        help="simple: explicit width/depth/junction args; silvaco_window: derive geometry from metal/channel/depth",
    )
    p.add_argument("--width-cm", type=float, default=1e-5, help="2D device width along x [cm] (simple mode)")
    p.add_argument("--depth-cm", type=float, default=1e-5, help="2D device depth along y [cm] (simple mode)")
    p.add_argument("--junction-x-cm", type=float, default=5e-6, help="PN junction x-position [cm] (simple mode)")
    p.add_argument("--metal-width-nm", type=float, default=500.0, help="Metal width [nm] (silvaco_window mode)")
    p.add_argument(
        "--contact-spacing-um",
        type=float,
        default=0.5,
        help="Explicit contact spacing [um] in silvaco_window mode (primary geometry control).",
    )
    p.add_argument(
        "--batch-metal-widths-nm",
        type=float,
        nargs="*",
        default=None,
        help="Optional batch list of metal widths [nm]. Runs all widths and generates combined plots.",
    )
    p.add_argument(
        "--batch-contact-spacings-um",
        type=float,
        nargs="*",
        default=None,
        help="Optional batch list of contact spacings [um]. Runs all spacings and generates combined plots.",
    )
    p.add_argument(
        "--batch-voltage-windows",
        type=str,
        nargs="*",
        default=None,
        help="Optional batch list of voltage windows as vmin:vmax, e.g. 0.2:0.4 0.4:0.6",
    )
    p.add_argument(
        "--batch-implant-species",
        type=str,
        nargs="*",
        choices=("boron", "arsenic"),
        default=None,
        help="Optional batch list of implant species for A/B comparison (e.g. boron arsenic).",
    )
    p.add_argument(
        "--channel-length-um",
        type=float,
        default=None,
        help="Backward-compatible alias for --contact-spacing-um. If provided, it overrides contact-spacing.",
    )
    p.add_argument("--depth-um", type=float, default=2.0, help="Silicon depth [um] (silvaco_window mode)")
    p.add_argument("--vmin", type=float, default=-10.0)
    p.add_argument("--vmax", type=float, default=10.0)
    p.add_argument(
        "--batch-temps-k",
        type=float,
        nargs="*",
        default=None,
        help="Optional batch list of temperatures [K]. Runs all temperatures and generates combined plots.",
    )
    p.add_argument(
        "--sweep-mode",
        choices=("adaptive", "silvaco", "silvaco_short", "sparse"),
        default="silvaco_short",
        help="adaptive: generic segmented sweep; silvaco: dense to 10V; silvaco_short: staged sweep up to 1V; sparse: key checkpoints only",
    )
    p.add_argument("--max-step", type=float, default=0.002)
    p.add_argument("--min-step", type=float, default=1e-8)
    p.add_argument("--fallback-retries", type=int, default=3, help="Recovery retries when bias step stagnates/diverges")
    p.add_argument(
        "--save-band-diagram",
        dest="save_band_diagram",
        action="store_true",
        default=True,
        help="Export 1D surface band diagram (CSV+PNG). Enabled by default.",
    )
    p.add_argument(
        "--no-save-band-diagram",
        dest="save_band_diagram",
        action="store_false",
        help="Disable band diagram export.",
    )
    p.add_argument(
        "--band-bias-v",
        type=float,
        default=None,
        help="Bias [V] at which to save band diagram. Default: vmax of the active run.",
    )
    p.add_argument(
        "--stability-analysis",
        action="store_true",
        help="Run multi-width stability analysis in one launch (collect logs, CSV metrics, and PNGs).",
    )
    p.add_argument(
        "--run-and-stability",
        action="store_true",
        help="Run normal simulation first, then run stability analysis automatically with the same arguments.",
    )
    p.add_argument(
        "--stability-widths-nm",
        type=float,
        nargs="*",
        default=None,
        help="Width list [nm] for --stability-analysis; defaults to batch widths or current width.",
    )
    p.add_argument(
        "--stability-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def solve_one_case(
    args: argparse.Namespace,
    prefix: str,
    metal_width_nm: float | None,
    contact_spacing_um: float | None = None,
    implant_species_override: str | None = None,
    implant_dose_override: float | None = None,
    implant_peak_override: float | None = None,
    implant_sigma_override: float | None = None,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
    temp_k_override: float | None = None,
    trap_ea_override_mev: float | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, str]], Path, Path]:
    maybe_reset_devsim()

    if args.geometry_mode == "silvaco_window":
        effective_metal_width_nm = args.metal_width_nm if metal_width_nm is None else metal_width_nm
        metal_w_um = effective_metal_width_nm * 1e-3
        spacing_base_um = args.channel_length_um if args.channel_length_um is not None else args.contact_spacing_um
        spacing_um = spacing_base_um if contact_spacing_um is None else contact_spacing_um
        width_cm = (2.0 * metal_w_um + spacing_um) * 1e-4
        depth_cm = args.depth_um * 1e-4
        junction_x_cm = metal_w_um * 1e-4
        metal_width_cm = metal_w_um * 1e-4
    else:
        width_cm = args.width_cm
        depth_cm = args.depth_cm
        junction_x_cm = args.junction_x_cm
        metal_width_cm = None

    eff_temp = args.temp_k if temp_k_override is None else temp_k_override
    eff_vmin = args.vmin if vmin_override is None else vmin_override
    eff_vmax = args.vmax if vmax_override is None else vmax_override
    eff_spacing = None
    eff_depth_um = None
    eff_width_nm = metal_width_nm
    if args.geometry_mode == "silvaco_window":
        eff_spacing = spacing_um
        eff_depth_um = args.depth_um
    try:
        build_device(
        temp_k=args.temp_k if temp_k_override is None else temp_k_override,
        na=args.na,
        nd=args.nd,
        taun_s=args.taun_s,
        taup_s=args.taup_s,
        width_cm=width_cm,
        depth_cm=depth_cm,
        junction_x_cm=junction_x_cm,
        doping_mode=args.doping_mode,
        implant_species=args.implant_species if implant_species_override is None else implant_species_override,
        implant_dose_cm2=args.implant_dose_cm2 if implant_dose_override is None else implant_dose_override,
        implant_peak_um=args.implant_peak_um if implant_peak_override is None else implant_peak_override,
        implant_sigma_um=args.implant_sigma_um if implant_sigma_override is None else implant_sigma_override,
        implant_lateral_factor=args.implant_lateral_factor,
        metal_width_cm=metal_width_cm,
        include_oxide=args.include_oxide,
        native_oxide_thickness_um=args.native_oxide_thickness_um,
        include_sidewall_oxide=args.include_sidewall_oxide,
        sidewall_oxide_thickness_um=args.sidewall_oxide_thickness_um,
        sidewall_oxide_height_um=args.sidewall_oxide_height_um,
        include_metal_stack=args.include_metal_stack,
        ti_thickness_um=args.ti_thickness_um,
        pd_thickness_um=args.pd_thickness_um,
        enable_traps=args.enable_traps,
        trap_ea_mev=args.trap_ea_mev if trap_ea_override_mev is None else trap_ea_override_mev,
        trap_nga1_cm3=args.trap_nga1_cm3,
        trap_nga2_cm3=args.trap_nga2_cm3,
        trap_y1_um=args.trap_y1_um,
        trap_y2_um=args.trap_y2_um,
        trap_sigma1_um=args.trap_sigma1_um,
        trap_sigma2_um=args.trap_sigma2_um,
        trap_lateral_factor=args.trap_lateral_factor,
        trap_strength=args.trap_strength,
        )
        iv = run_iv(
        vmin=eff_vmin,
        vmax=eff_vmax,
        max_step=args.max_step,
        min_step=args.min_step,
        sweep_mode=args.sweep_mode,
        contact_offset_v=contact_offset_from_args(args),
        fallback_retries=args.fallback_retries,
        )
        sl = slope_data(iv)
    except Exception as exc:
        if args.planar_review:
            write_run_manifest(
                args,
                prefix,
                metal_width_nm=eff_width_nm,
                contact_spacing_um=eff_spacing,
                depth_um=eff_depth_um,
                temp_k=eff_temp,
                vmin=eff_vmin,
                vmax=eff_vmax,
                status="fail",
                message=str(exc),
            )
        raise

    iv_csv = OUT_DIR / f"{prefix}_iv.csv"
    sl_csv = OUT_DIR / f"{prefix}_slope.csv"
    fig_prefix = f"{u.figure_tag(args.run_number)}_{prefix}"
    iv_png = FIG_DIR / f"{fig_prefix}_iv.png"
    sl_png = FIG_DIR / f"{fig_prefix}_slope.png"
    u.write_csv(iv_csv, ["voltage_V", "current_A_per_cm"], iv)
    u.write_csv(sl_csv, ["voltage_V", "dlog10I_dlog10V", "branch"], sl)
    u.plot_iv(iv_png, iv, xmin=eff_vmin, xmax=eff_vmax)
    u.plot_slope(sl_png, sl, xmin=eff_vmin, xmax=eff_vmax)

    if args.save_band_diagram:
        v_target = (
            args.band_bias_v
            if args.band_bias_v is not None
            else eff_vmax
        )
        bname = GetContactBiasName(RIGHT)
        try:
            # prefer robust stepping to target bias for reliable band extraction.
            v_now = float(get_parameter(device=DEVICE, name=bname))
            robust_move_bias(
                bname,
                v_now,
                v_target + contact_offset_from_args(args),
                max_step=max(0.001, min(args.max_step, 0.05)),
                min_step=args.min_step,
                fallback_retries=args.fallback_retries,
                anchor_bias=contact_offset_from_args(args),
            )
        except Exception:
            set_parameter(device=DEVICE, name=bname, value=v_target + contact_offset_from_args(args))
            solve_dc()
        band_csv, band_png = export_band_diagram(
            prefix=prefix,
            temp_k=args.temp_k if temp_k_override is None else temp_k_override,
            bias_v=v_target,
            run_number=args.run_number,
        )
        print(band_csv)
        print(band_png)
    if args.planar_review:
        manifest_path = write_run_manifest(
            args,
            prefix,
            metal_width_nm=eff_width_nm,
            contact_spacing_um=eff_spacing,
            depth_um=eff_depth_um,
            temp_k=eff_temp,
            vmin=eff_vmin,
            vmax=eff_vmax,
            status="ok",
            message="",
        )
        print(manifest_path)
    return iv, sl, iv_png, sl_png


def contact_offset_from_args(args: argparse.Namespace) -> float:
    if args.contact_mode == "neutral":
        return 0.0
    if args.contact_mode == "spec_ohmic":
        if args.implant_species == "arsenic":
            delta = args.wf_arsenic_ev - args.wf_neutral_ev
            return -args.contact_proxy_gain * delta
        delta = args.wf_boron_ev - args.wf_neutral_ev
        return args.contact_proxy_gain * delta
    if args.contact_mode == "schottky_approx":
        # Barrier-oriented proxy:
        #   electron barrier: PhiBn = Wm - chi
        #   hole barrier:     PhiBp = Eg - PhiBn
        # We map lower barrier to higher effective injection via positive proxy offset.
        wf = args.wf_arsenic_ev if args.implant_species == "arsenic" else args.wf_boron_ev
        phi_bn = max(0.0, wf - args.chi_si_ev)
        phi_bp = max(0.0, args.eg_si_ev - phi_bn)
        barrier = phi_bn if args.implant_species == "arsenic" else phi_bp
        return args.contact_proxy_gain * (args.schottky_ref_barrier_ev - barrier)
    delta = args.wf_ev - args.wf_neutral_ev
    offset = args.contact_proxy_gain * delta
    if args.contact_mode == "electron":
        offset = -offset
    return offset


def implant_params_for_species(args: argparse.Namespace, species: str) -> tuple[float, float, float]:
    if args.implant_parameter_mode == "fixed":
        return args.implant_dose_cm2, args.implant_peak_um, args.implant_sigma_um
    if species == "arsenic":
        return args.arsenic_dose_cm2, args.arsenic_peak_um, args.arsenic_sigma_um
    return args.implant_dose_cm2, args.implant_peak_um, args.implant_sigma_um


def apply_cli_preset(args: argparse.Namespace) -> None:
    if args.preset != "etched_like":
        return
    # Silvaco-etched-oriented proxy defaults for DEVSIM.
    args.geometry_mode = "silvaco_window"
    args.metal_width_nm = 200.0
    args.contact_spacing_um = 1.2
    args.depth_um = 4.0
    args.include_oxide = True
    args.native_oxide_thickness_um = 0.003
    args.include_sidewall_oxide = True
    args.sidewall_oxide_thickness_um = 0.003
    args.sidewall_oxide_height_um = 0.067
    args.include_metal_stack = True
    args.ti_thickness_um = 0.001
    args.pd_thickness_um = 0.025
    args.doping_mode = "gaussian_implant"
    args.implant_species = "boron"
    args.implant_parameter_mode = "deck_by_species"
    args.implant_lateral_factor = 1.0 / 6.0
    args.contact_mode = "spec_ohmic"
    args.wf_boron_ev = 5.25
    args.wf_arsenic_ev = 4.17
    args.taun_s = 1e-7
    args.taup_s = 1e-7


def main() -> int:
    global OUT_DIR, FIG_DIR
    args = parse_args()
    apply_cli_preset(args)

    default_out, default_fig = default_output_dirs()
    OUT_DIR = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_out
    FIG_DIR = Path(args.figures_dir).expanduser().resolve() if args.figures_dir else default_fig

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.planar_review:
        validate_planar_review_args(args)

    if args.run_and_stability and (not args.stability_child) and (not args.stability_analysis):
        # Single-pass behavior: stability mode already solves each case and writes
        # regular IV/slope CSV+PNG from child runs, so no second full solve pass is needed.
        args.stability_analysis = True

    if args.stability_analysis and not args.stability_child:
        return run_stability_batch(args)

    if args.batch_voltage_windows and (args.batch_metal_widths_nm or args.batch_contact_spacings_um or args.batch_implant_species):
        raise SystemExit("Use --batch-voltage-windows only as a standalone batch mode")
    if args.batch_implant_species and args.batch_contact_spacings_um:
        raise SystemExit("For implant A/B mode, keep spacing fixed via --contact-spacing-um (no spacing batch)")
    if args.batch_temps_k and (args.batch_metal_widths_nm or args.batch_contact_spacings_um or args.batch_implant_species or args.batch_voltage_windows):
        raise SystemExit("Use --batch-temps-k only as a standalone batch mode")
    if args.batch_trap_ea_mev and (
        args.batch_metal_widths_nm
        or args.batch_contact_spacings_um
        or args.batch_implant_species
        or args.batch_voltage_windows
        or args.batch_temps_k
    ):
        raise SystemExit("Use --batch-trap-ea-mev only as a standalone batch mode")

    if args.batch_temps_k:
        temp_cases: list[tuple[str, list[tuple[float, float]]]] = []
        temp_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[float, str, str]] = []
        for t in args.batch_temps_k:
            label = f"{t:g}K"
            case_prefix = f"{args.prefix}_{label}"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(
                    args,
                    prefix=case_prefix,
                    metal_width_nm=None,
                    temp_k_override=t,
                )
                temp_cases.append((label, iv))
                temp_slopes.append((label, sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((t, "ok", ""))
                print(f"[OK] {label}")
            except Exception as exc:
                summary_rows.append((t, "fail", str(exc)))
                print(f"[FAIL] {label}: {exc}")

        if not temp_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["temp_k", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["temp_k", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, temp_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, temp_slopes, xmin=args.vmin, xmax=args.vmax)
        if len(temp_cases) == len(args.batch_temps_k):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(temp_cases)} / {len(args.batch_temps_k)}")
    elif args.batch_trap_ea_mev:
        if not args.enable_traps:
            raise SystemExit("--batch-trap-ea-mev requires --enable-traps")
        ea_cases: list[tuple[str, list[tuple[float, float]]]] = []
        ea_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[float, str, str]] = []
        for ea in args.batch_trap_ea_mev:
            label = f"{ea:g}meV"
            case_prefix = f"{args.prefix}_{label}"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(
                    args,
                    prefix=case_prefix,
                    metal_width_nm=None,
                    trap_ea_override_mev=ea,
                )
                ea_cases.append((label, iv))
                ea_slopes.append((label, sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((ea, "ok", ""))
                print(f"[OK] {label}")
            except Exception as exc:
                summary_rows.append((ea, "fail", str(exc)))
                print(f"[FAIL] {label}: {exc}")

        if not ea_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["trap_ea_mev", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["trap_ea_mev", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, ea_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, ea_slopes, xmin=args.vmin, xmax=args.vmax)
        if len(ea_cases) == len(args.batch_trap_ea_mev):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(ea_cases)} / {len(args.batch_trap_ea_mev)}")
    elif args.batch_implant_species and args.batch_metal_widths_nm:
        if args.geometry_mode != "silvaco_window":
            raise SystemExit("Implant species x width grid requires --geometry-mode silvaco_window")
        grid_cases: list[tuple[str, list[tuple[float, float]]]] = []
        grid_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[str, float, str, str]] = []
        for species in args.batch_implant_species:
            dose, peak, sigma = implant_params_for_species(args, species)
            for width_nm in args.batch_metal_widths_nm:
                label = f"{species}_{width_nm:g}nm"
                case_prefix = f"{args.prefix}_{label}"
                try:
                    iv, sl, iv_png, sl_png = solve_one_case(
                        args,
                        prefix=case_prefix,
                        metal_width_nm=width_nm,
                        implant_species_override=species,
                        implant_dose_override=dose,
                        implant_peak_override=peak,
                        implant_sigma_override=sigma,
                    )
                    grid_cases.append((label, iv))
                    grid_slopes.append((label, sl))
                    individual_figs.extend((iv_png, sl_png))
                    summary_rows.append((species, width_nm, "ok", ""))
                    print(f"[OK] {label}")
                except Exception as exc:
                    summary_rows.append((species, width_nm, "fail", str(exc)))
                    print(f"[FAIL] {label}: {exc}")

        if not grid_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["implant_species", "metal_width_nm", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["implant_species", "metal_width_nm", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, grid_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, grid_slopes, xmin=args.vmin, xmax=args.vmax)
        total = len(args.batch_implant_species) * len(args.batch_metal_widths_nm)
        if len(grid_cases) == total:
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(grid_cases)} / {total}")
    elif args.batch_implant_species:
        species_cases: list[tuple[str, list[tuple[float, float]]]] = []
        species_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[str, str, str]] = []
        for species in args.batch_implant_species:
            dose, peak, sigma = implant_params_for_species(args, species)
            case_prefix = f"{args.prefix}_{species}"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(
                    args,
                    prefix=case_prefix,
                    metal_width_nm=None,
                    implant_species_override=species,
                    implant_dose_override=dose,
                    implant_peak_override=peak,
                    implant_sigma_override=sigma,
                )
                species_cases.append((species, iv))
                species_slopes.append((species, sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((species, "ok", ""))
                print(f"[OK] {species}")
            except Exception as exc:
                summary_rows.append((species, "fail", str(exc)))
                print(f"[FAIL] {species}: {exc}")

        if not species_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["implant_species", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["implant_species", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, species_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, species_slopes, xmin=args.vmin, xmax=args.vmax)
        if len(species_cases) == len(args.batch_implant_species):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(species_cases)} / {len(args.batch_implant_species)}")
    elif args.batch_voltage_windows:
        window_cases: list[tuple[str, list[tuple[float, float]]]] = []
        window_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[str, str, str]] = []
        for token in args.batch_voltage_windows:
            try:
                vs = token.replace(",", ":").split(":")
                if len(vs) != 2:
                    raise ValueError("Expected vmin:vmax")
                vmin_case = float(vs[0])
                vmax_case = float(vs[1])
            except Exception as exc:
                raise SystemExit(f"Invalid voltage window '{token}': {exc}")
            label = f"{vmin_case:g}_to_{vmax_case:g}V"
            case_prefix = f"{args.prefix}_{label}"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(
                    args,
                    prefix=case_prefix,
                    metal_width_nm=None,
                    contact_spacing_um=None,
                    vmin_override=vmin_case,
                    vmax_override=vmax_case,
                )
                window_cases.append((f"{vmin_case:g}..{vmax_case:g} V", iv))
                window_slopes.append((f"{vmin_case:g}..{vmax_case:g} V", sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((label, "ok", ""))
                print(f"[OK] {label}")
            except Exception as exc:
                summary_rows.append((label, "fail", str(exc)))
                print(f"[FAIL] {label}: {exc}")

        if not window_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["voltage_window", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["voltage_window", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, window_cases, xmin=min(float(x.split(":")[0].replace(",", ":").split(":")[0]) for x in args.batch_voltage_windows), xmax=max(float(x.replace(",", ":").split(":")[1]) for x in args.batch_voltage_windows))
        u.plot_combined_slope_labeled(combined_sl, window_slopes, xmin=min(float(x.replace(",", ":").split(":")[0]) for x in args.batch_voltage_windows), xmax=max(float(x.replace(",", ":").split(":")[1]) for x in args.batch_voltage_windows))
        if len(window_cases) == len(args.batch_voltage_windows):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(window_cases)} / {len(args.batch_voltage_windows)}")
    elif args.batch_metal_widths_nm and args.batch_contact_spacings_um:
        if args.geometry_mode != "silvaco_window":
            raise SystemExit("Grid batch requires --geometry-mode silvaco_window")
        grid_cases: list[tuple[str, list[tuple[float, float]]]] = []
        grid_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[float, float, str, str]] = []
        for width_nm in args.batch_metal_widths_nm:
            for spacing_um in args.batch_contact_spacings_um:
                case_prefix = f"{args.prefix}_{int(round(width_nm))}nm_{spacing_um:g}um"
                label = f"{width_nm:g}nm/{spacing_um:g}um"
                try:
                    iv, sl, iv_png, sl_png = solve_one_case(
                        args,
                        prefix=case_prefix,
                        metal_width_nm=width_nm,
                        contact_spacing_um=spacing_um,
                    )
                    grid_cases.append((label, iv))
                    grid_slopes.append((label, sl))
                    individual_figs.extend((iv_png, sl_png))
                    summary_rows.append((width_nm, spacing_um, "ok", ""))
                    print(f"[OK] {label}")
                except Exception as exc:
                    summary_rows.append((width_nm, spacing_um, "fail", str(exc)))
                    print(f"[FAIL] {label}: {exc}")

        if not grid_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["metal_width_nm", "contact_spacing_um", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["metal_width_nm", "contact_spacing_um", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, grid_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, grid_slopes, xmin=args.vmin, xmax=args.vmax)
        total = len(args.batch_metal_widths_nm) * len(args.batch_contact_spacings_um)
        if len(grid_cases) == total:
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(grid_cases)} / {total}")
    elif args.batch_contact_spacings_um:
        if args.geometry_mode != "silvaco_window":
            raise SystemExit("--batch-contact-spacings-um requires --geometry-mode silvaco_window")
        spacing_cases: list[tuple[str, list[tuple[float, float]]]] = []
        spacing_slopes: list[tuple[str, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[float, str, str]] = []
        for spacing_um in args.batch_contact_spacings_um:
            case_prefix = f"{args.prefix}_{spacing_um:g}um"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(
                    args,
                    prefix=case_prefix,
                    metal_width_nm=None,
                    contact_spacing_um=spacing_um,
                )
                spacing_cases.append((f"{spacing_um:g} um", iv))
                spacing_slopes.append((f"{spacing_um:g} um", sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((spacing_um, "ok", ""))
                print(f"[OK] {spacing_um:g} um")
            except Exception as exc:
                summary_rows.append((spacing_um, "fail", str(exc)))
                print(f"[FAIL] {spacing_um:g} um: {exc}")

        if not spacing_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["contact_spacing_um", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["contact_spacing_um", "status", "message"], summary_rows)
        u.plot_combined_iv_labeled(combined_iv, spacing_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope_labeled(combined_sl, spacing_slopes, xmin=args.vmin, xmax=args.vmax)
        if len(spacing_cases) == len(args.batch_contact_spacings_um):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(spacing_cases)} / {len(args.batch_contact_spacings_um)}")
    elif args.batch_metal_widths_nm:
        if args.geometry_mode != "silvaco_window":
            raise SystemExit("--batch-metal-widths-nm requires --geometry-mode silvaco_window")
        ok_cases: list[tuple[float, list[tuple[float, float]]]] = []
        ok_slopes: list[tuple[float, list[tuple[float, float, str]]]] = []
        individual_figs: list[Path] = []
        summary_rows: list[tuple[float, str, str]] = []
        for width_nm in args.batch_metal_widths_nm:
            case_prefix = f"{args.prefix}_{int(round(width_nm))}nm"
            try:
                iv, sl, iv_png, sl_png = solve_one_case(args, prefix=case_prefix, metal_width_nm=width_nm)
                ok_cases.append((width_nm, iv))
                ok_slopes.append((width_nm, sl))
                individual_figs.extend((iv_png, sl_png))
                summary_rows.append((width_nm, "ok", ""))
                print(f"[OK] {width_nm:g} nm")
            except Exception as exc:
                summary_rows.append((width_nm, "fail", str(exc)))
                print(f"[FAIL] {width_nm:g} nm: {exc}")

        if not ok_cases:
            summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
            u.write_csv(summary_path, ["metal_width_nm", "status", "message"], summary_rows)
            print(f"No successful runs. Summary: {summary_path}")
            return 1

        combined_prefix = f"{u.figure_tag(args.run_number)}_{args.prefix}"
        combined_iv = FIG_DIR / f"{combined_prefix}_combined_iv.png"
        combined_sl = FIG_DIR / f"{combined_prefix}_combined_slope.png"
        summary_path = OUT_DIR / f"{args.prefix}_batch_summary.csv"
        u.write_csv(summary_path, ["metal_width_nm", "status", "message"], summary_rows)
        u.plot_combined_iv(combined_iv, ok_cases, xmin=args.vmin, xmax=args.vmax)
        u.plot_combined_slope(combined_sl, ok_slopes, xmin=args.vmin, xmax=args.vmax)
        if len(ok_cases) == len(args.batch_metal_widths_nm):
            for path in individual_figs:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print("Saved:")
        print(summary_path)
        print(combined_iv)
        print(combined_sl)
        print(f"Successful runs: {len(ok_cases)} / {len(args.batch_metal_widths_nm)}")
    else:
        iv, sl, iv_png, sl_png = solve_one_case(args, prefix=args.prefix, metal_width_nm=None)
        iv_csv = OUT_DIR / f"{args.prefix}_iv.csv"
        sl_csv = OUT_DIR / f"{args.prefix}_slope.csv"
        print("Saved:")
        print(iv_csv)
        print(sl_csv)
        print(iv_png)
        print(sl_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
