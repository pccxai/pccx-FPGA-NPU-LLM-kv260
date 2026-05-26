// PCCX(TM) - reusable AI accelerator project.
// SPDX-FileCopyrightText: 2026 Hyun Woo Kim
// SPDX-License-Identifier: Apache-2.0

    .section .text.startup, "ax"
    .align 7
    .global _start
    .type _start, %function

_start:
    ldr x0, =_stack_top
    mov sp, x0
    bl main

1:
    wfe
    b 1b
