# KV260 SD Card Reset Procedure

This runbook resets the KV260 boot media, then rebuilds the repo-local
firmware bundle expected by `xmutil` after first boot. It records setup
steps only; board smoke, model runtime, throughput, and timing evidence
still come from the existing evidence flow.

## Success Criteria

- The existing SD card state is preserved well enough to recover login and
  network access.
- A known-good Ubuntu or PetaLinux image is written to the microSD card.
- First boot reaches SSH or UART login.
- `pccx_npu_bd.bit.bin`, `pccx_npu_bd.dtbo`, and `shell.json` are generated
  under `sw/dtbo/build/pccx_npu_bd/`.
- When the user chooses to register the bundle on the board,
  `/lib/firmware/xilinx/pccx_npu_bd/` contains matching bundle filenames and
  `xmutil listapps` can discover the application entry.

## Before Wiping The Card

1. Photograph the current microSD card, carrier board, boot switches, cable
   layout, and any label on the card adapter.
2. Save a host-side copy of the current public SSH key and `known_hosts`
   entries used for the board. Private keys must stay outside the repository,
   issue tracker, screenshots, and logs.
3. If the current card still boots, collect read-only notes:

   ```bash
   uname -a
   cat /etc/os-release
   command -v xmutil || true
   xmutil listapps 2>&1 || true
   ls -la /lib/firmware/xilinx 2>/dev/null || true
   ip addr
   ```

4. If the card has files that only exist on the board, copy them to a private
   backup location before flashing. Do not put board dumps, keys, or generated
   bitstreams into this repository.

## Image Download Links

Use one route per reset.

| Route | When to use | Link |
| --- | --- | --- |
| Ubuntu Server 24.04 LTS | Default clean reset for current KV260/K26 work | [Ubuntu Kria images](https://ubuntu.com/download/amd) |
| Ubuntu Desktop 22.04 LTS | Desktop image when a local GUI is required | [Ubuntu Kria images](https://ubuntu.com/download/amd) |
| PetaLinux pre-built image | Legacy Kria application package flow or BSP parity checks | [KV260 PetaLinux SD image setup](https://xilinx.github.io/kria-apps-docs/kv260/2021.1/linux_boot/petalinux_2021.1/build/html/docs/sdcard.html) |
| PetaLinux tools / embedded downloads | Rebuilding a PetaLinux image or BSP locally | [embedded downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools.html) |
| Vitis / Vivado 2025.2 installer | Exporting the XSA and using `bootgen`, `xsct`, and Vitis | [tool downloads](https://www.xilinx.com/support/download.html) |

The PetaLinux starter-kit image path is useful for older application-package
flows. For new SD-card reset work, prefer the current Ubuntu image unless a
specific BSP regression needs to be reproduced.

## Flash With balenaEtcher

1. Install the flasher from <https://etcher.balena.io/>.
2. Insert the microSD card into the host.
3. Select `Flash from file` and choose the downloaded `.img`, `.wic`, or
   compressed image file.
4. Select the microSD card as the target. Verify the capacity and drive name
   before continuing.
5. Select `Flash` and wait for validation to complete.
6. Eject the card from the host before removing it.

## Flash With Win32 Disk Imager

1. Install the tool from <https://sourceforge.net/projects/win32diskimager/>.
2. If the image is compressed, extract it first so the selected file is the raw
   `.img` or `.wic` file.
3. Start the tool, choose the raw image, and select the removable drive letter
   for the microSD card.
4. Use `Write`, confirm the destructive prompt, and wait until the write
   finishes.
5. Eject the card from Windows before removing it.

## First Boot Checklist

1. Insert the card, connect UART and Ethernet, then power the board.
2. Wait for the first boot resize/setup path to settle before interrupting.
3. Confirm basic access:

   ```bash
   uname -a
   cat /etc/os-release
   ip addr
   command -v xmutil || true
   xmutil listapps 2>&1 || true
   ```

4. Install host utilities on the board only when needed:

   ```bash
   sudo apt update
   sudo apt install -y device-tree-compiler openssh-server rsync
   ```

5. Reinstall the saved public SSH key into the board user's
   `~/.ssh/authorized_keys` if SSH access is required.

## Vitis 2025.2 Hardware Platform Export

Run these steps on the development host after the Vivado project has an
implemented design and an intended `.bit` artifact.

1. Source the tool environment:

   ```bash
   source <Vitis_Installation_Directory>/settings64.sh
   ```

2. Export the hardware platform from Vivado with the bitstream included:

   ```tcl
   open_project <project>.xpr
   open_run impl_1
   write_hw_platform -fixed -include_bit -force \
     -file build/pccx_v002_kv260.xsa
   ```

   The GUI path is `File -> Export -> Export Hardware`, then select the
   include-bitstream option.

3. Generate a device-tree overlay source from the XSA when DTG is available:

   ```tcl
   hsi open_hw_design build/pccx_v002_kv260.xsa
   hsi set_repo_path <device-tree-xlnx>
   hsi create_sw_design device-tree -os device_tree -proc psu_cortexa53_0
   hsi set_property CONFIG.dt_overlay true [hsi::get_os]
   hsi generate_target -dir build/dt
   hsi close_hw_design [hsi current_hw_design]
   ```

4. Launch the unified IDE only if software workspace work is needed:

   ```bash
   vitis -w build/vitis-workspace
   ```

## Generate The Post-Reimage Platform Bundle

The repo now provides:

```bash
scripts/kv260/generate_sd_platform_bundle.sh
sw/dtbo/Makefile
```

Preferred flow with a DTG-generated `pl.dtsi`:

```bash
scripts/kv260/generate_sd_platform_bundle.sh \
  --bit build/pccx_v002_kv260.bit \
  --dts build/dt/pl.dtsi \
  --overlay pccx_npu_bd
```

Fallback flow when a DTG output is not available yet:

```bash
scripts/kv260/generate_sd_platform_bundle.sh \
  --bit build/pccx_v002_kv260.bit \
  --overlay pccx_npu_bd
```

The fallback overlay uses the compiled v002 AXI-Lite aperture
`0xA0000000` with a `0x00010000` generic-UIO window. Prefer the DTG
overlay once the XSA is available because it mirrors the actual Vivado
address map.

The existing deploy script can regenerate the bundle through the Makefile:

```bash
PCCX_BIT=build/pccx_v002_kv260.bit \
hw/scripts/deploy_to_kv260.sh --dry-run
```

## Register The Bundle On The Reimaged Board

Copy the generated directory to the board, then register it locally:

```bash
sudo mkdir -p /lib/firmware/xilinx/pccx_npu_bd
sudo install -m 0644 pccx_npu_bd.bit.bin /lib/firmware/xilinx/pccx_npu_bd/
sudo install -m 0644 pccx_npu_bd.dtbo /lib/firmware/xilinx/pccx_npu_bd/
sudo install -m 0644 shell.json /lib/firmware/xilinx/pccx_npu_bd/
xmutil listapps
```

If the repo and build tools are available on the board, the script can do the
same registration:

```bash
scripts/kv260/generate_sd_platform_bundle.sh \
  --bit build/pccx_v002_kv260.bit \
  --dts build/dt/pl.dtsi \
  --overlay pccx_npu_bd \
  --register /lib/firmware/xilinx
```

Loading the application changes board state. Run it only when the intended
bitstream and device-tree overlay are ready for that board session:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp pccx_npu_bd
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `bootgen not found` | Source the Vitis/Vivado `settings64.sh` on the host. |
| `dtc not found` | Install `device-tree-compiler`. |
| App missing from `xmutil listapps` | Confirm the directory is `/lib/firmware/xilinx/pccx_npu_bd/` and the files are named `pccx_npu_bd.bit.bin`, `pccx_npu_bd.dtbo`, and `shell.json`. |
| DTBO compile fails | Use the DTG-generated `pl.dtsi`; the fallback overlay assumes the standard `fpga_full` label exists in the boot image. |
| `/dev/uio*` is missing after load | Confirm the boot image enables `generic-uio` binding for `compatible = "generic-uio"` or use a DTG overlay with the board's intended driver binding. |
| SSH changed after reimage | Reinstall only the public key and update the host `known_hosts` entry deliberately. |

## References

- Ubuntu Kria image page: <https://ubuntu.com/download/amd>
- KV260 PetaLinux SD image setup: <https://xilinx.github.io/kria-apps-docs/kv260/2021.1/linux_boot/petalinux_2021.1/build/html/docs/sdcard.html>
- Kria on-target firmware layout: <https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/target.html>
- Vitis 2025.2 installation: <https://docs.amd.com/r/en-US/ug1742-vitis-release-notes/Installing-the-Vitis-Software-Platform>
- Vitis 2025.2 IDE launch: <https://docs.amd.com/r/en-US/ug1399-vitis-hls/Launching-the-Vitis-Unified-IDE>
