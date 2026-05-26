# PCCX(TM) - reusable AI accelerator project.
# SPDX-FileCopyrightText: 2026 Hyun Woo Kim
# SPDX-License-Identifier: Apache-2.0

CROSS_COMPILE ?= aarch64-none-elf-
CC := $(CROSS_COMPILE)gcc
READELF ?= $(CROSS_COMPILE)readelf
SIZE ?= $(CROSS_COMPILE)size

BSP_ROOT ?=
BSP_INCLUDE ?= $(if $(BSP_ROOT),$(BSP_ROOT)/include,)
BSP_LIB_DIR ?= $(if $(BSP_ROOT),$(BSP_ROOT)/lib,)

BUILD_DIR ?= build
TARGET ?= gemma_inference
ELF := $(BUILD_DIR)/$(TARGET).elf
MAP := $(BUILD_DIR)/$(TARGET).map

SRCS := startup.S gemma_inference.c npu_driver/npu_driver.c
OBJS := $(addprefix $(BUILD_DIR)/,$(SRCS:=.o))
DEPS := $(OBJS:.o=.d)

CPPFLAGS := -I. -Inpu_driver
ifneq ($(BSP_INCLUDE),)
CPPFLAGS += -I$(BSP_INCLUDE)
endif

CFLAGS := -Wall -Wextra -Werror -Os -g \
          -mcpu=cortex-a53 \
          -ffreestanding -fno-builtin -fno-stack-protector \
          -ffunction-sections -fdata-sections \
          -MMD -MP
ASFLAGS := -mcpu=cortex-a53 -MMD -MP
LDARCHFLAGS := -mcpu=cortex-a53
LDFLAGS := -T linker.ld -nostartfiles -Wl,--gc-sections -Wl,-Map,$(MAP)
LDLIBS := -Wl,--start-group -lxil -lgcc -Wl,--end-group

ifneq ($(BSP_LIB_DIR),)
LDFLAGS += -L$(BSP_LIB_DIR)
endif

.PHONY: all check-toolchain check-bsp check-elf clean

all: check-toolchain check-bsp $(ELF)

check-toolchain:
	@command -v $(CC) >/dev/null || { \
		echo "error: $(CC) not found; install/source the Vitis standalone AArch64 toolchain" >&2; \
		exit 1; \
	}

check-bsp:
	@test -n "$(BSP_INCLUDE)" || { \
		echo "error: BSP_INCLUDE or BSP_ROOT is required for Vitis BSP headers" >&2; \
		exit 1; \
	}
	@test -f "$(BSP_INCLUDE)/xil_io.h" || { \
		echo "error: xil_io.h not found under BSP_INCLUDE=$(BSP_INCLUDE)" >&2; \
		exit 1; \
	}
	@test -n "$(BSP_LIB_DIR)" || { \
		echo "error: BSP_LIB_DIR or BSP_ROOT is required for libxil.a" >&2; \
		exit 1; \
	}
	@test -f "$(BSP_LIB_DIR)/libxil.a" || { \
		echo "error: libxil.a not found under BSP_LIB_DIR=$(BSP_LIB_DIR)" >&2; \
		exit 1; \
	}

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/npu_driver

$(BUILD_DIR)/%.S.o: %.S | $(BUILD_DIR)
	$(CC) $(ASFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(ELF): $(OBJS) linker.ld
	$(CC) $(LDARCHFLAGS) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@
	$(SIZE) $@

check-elf: $(ELF)
	@$(READELF) -h $(ELF) | grep -q 'Machine:[[:space:]]*AArch64'
	@$(READELF) -S $(ELF) | grep -q '\.text'
	@$(READELF) -S $(ELF) | grep -q '\.data'
	@$(READELF) -S $(ELF) | grep -q '\.pccx_weights'
	@$(READELF) -S $(ELF) | grep -q '\.pccx_activations'
	@$(READELF) -s $(ELF) | grep -q '__pccx_weight_buffer_start'
	@$(READELF) -s $(ELF) | grep -q '__pccx_activation_buffer_start'

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
