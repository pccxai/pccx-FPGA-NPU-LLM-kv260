# SPDX-License-Identifier: Apache-2.0

OVERLAY ?= pccx_npu_bd
BUILD ?= $(CURDIR)/build/$(OVERLAY)
BIT_SRC ?=
DTS_SRC ?=
GEN_SCRIPT ?= ../../scripts/kv260/generate_sd_platform_bundle.sh

.PHONY: all bundle bitbin dtbo shell-json clean

all: bundle

bundle bitbin dtbo shell-json:
	@test -n "$(BIT_SRC)" || { echo "BIT_SRC=<system.bit> is required"; exit 2; }
	"$(GEN_SCRIPT)" --bit "$(BIT_SRC)" --overlay "$(OVERLAY)" --build "$(BUILD)" $(if $(DTS_SRC),--dts "$(DTS_SRC)")

clean:
	rm -rf build
