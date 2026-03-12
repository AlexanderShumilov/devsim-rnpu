# Figures

This folder currently contains a contact-physics comparison set for the same device setup:

- `ohmic_w800_300K`
- `neutral_w800_300K`

For this comparison, treat these parameters as fixed and equal:

- geometry mode: `silvaco_window`
- metal width: `800 nm`
- contact spacing: `1.2 um`
- silicon depth: `4.0 um`
- oxide stack: enabled (`native oxide 0.003 um`, sidewall oxide enabled)
- metal stack: enabled (`Ti 0.001 um`, `Pd 0.025 um`)
- doping model: `gaussian_implant`
- implant: `boron`, `deck_by_species`, lateral factor `1/6`
- background doping: `ND=1e15 cm^-3`, `NA=1e14 cm^-3`
- temperature: `300 K`
- lifetimes: `taun=taup=1e-7 s`
- sweep mode and range: `silvaco_short`, `-0.001 .. 0.1 V`

Only intended variable between the two figures:

- contact mode (`spec_ohmic` vs `neutral`)
