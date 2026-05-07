# RTL Compile Order Policy

This repo uses `hw/vivado/filelist.f` as the canonical SystemVerilog
source order for Vivado project creation, synthesis, and source-level
compile checks. The file is intentionally ordered; do not sort it
alphabetically and do not replace it with recursive globs.

## Required Order

`filelist.f` must list definitions before any source that depends on
them:

1. Include-only preprocessor definition headers are documented first for
   reference, but are not compiled as standalone sources. They are
   provided through include directories.
2. Packages compile before modules, interfaces, or other packages that
   import them.
3. Interfaces and library packages compile before modules that instantiate
   or import them.
4. Leaf modules compile before wrappers, subsystem tops, and the final
   top-level file.
5. Board-design packaging shims stay after the synthesis top unless the
   shim itself becomes the selected tool top.

The current package block is ordered by dependency:

```text
A_const_svh/*.svh             include-only headers
B_device_pkg/device_pkg.sv    base device constants
C_type_pkg/*.sv               dtype and memory packages
D_pipeline_pkg/*.sv           vector-core configuration package
E_obs_pkg/perf_counter_pkg.sv observability package
```

Any new package must be inserted before the first file that imports it.
Any new module must be inserted after all packages, interfaces, and
modules it imports or instantiates.

## Adding Sources

When adding a SystemVerilog source:

- Add it to `hw/vivado/filelist.f` in dependency order in the same change.
- Keep include-only `.svh` files out of the compile list; add their
  directories to the tool include path instead.
- Prefer moving the new file earlier in the list over adding local
  package imports inside unrelated files to mask an ordering issue.
- Keep `NPU_top.sv` last among RTL implementation files unless the top
  hierarchy changes.
- Keep `vivado/npu_core_wrapper.sv` after `NPU_top.sv`; it exists for BD
  packaging around the plain-signal AXI boundary and is not the synthesis
  top.

## Review Check

Before handing off a change that adds or reorders sources, run the
source compile check from `CLAUDE.md` or an equivalent Vivado `xvlog`
command that consumes `hw/vivado/filelist.f` directly:

```bash
cd hw
mkdir -p build/lint_check
cd build/lint_check
xvlog -sv -i ../../rtl/Constants/compilePriority_Order/A_const_svh \
          -i ../../rtl/NPU_Controller \
          -i ../../rtl/MEM_control/IO \
          -i ../../rtl/MAT_CORE \
        $(sed -n 's|^rtl/|../../rtl/|p' ../../vivado/filelist.f)
```

The expected handoff result is 0 compile errors. Warnings must be
recorded in the PR if they are pre-existing or intentionally deferred.
