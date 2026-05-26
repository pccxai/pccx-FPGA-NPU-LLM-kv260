/* PCCX(TM) - reusable AI accelerator project.
 * SPDX-FileCopyrightText: 2026 Hyun Woo Kim
 * SPDX-License-Identifier: Apache-2.0
 */

ENTRY(_start)

MEMORY
{
    DDR_CODE       (rwx) : ORIGIN = 0x00100000, LENGTH = 64M
    DDR_WEIGHTS    (rw)  : ORIGIN = 0x10000000, LENGTH = 256M
    DDR_ACTIVATION (rw)  : ORIGIN = 0x20000000, LENGTH = 128M
}

PROVIDE(__pccx_weight_buffer_start = ORIGIN(DDR_WEIGHTS));
PROVIDE(__pccx_weight_buffer_end = ORIGIN(DDR_WEIGHTS) + LENGTH(DDR_WEIGHTS));
PROVIDE(__pccx_activation_buffer_start = ORIGIN(DDR_ACTIVATION));
PROVIDE(__pccx_activation_buffer_end = ORIGIN(DDR_ACTIVATION) + LENGTH(DDR_ACTIVATION));

SECTIONS
{
    .text : ALIGN(64)
    {
        KEEP(*(.text.startup))
        *(.text*)
        *(.rodata*)
    } > DDR_CODE

    .data : ALIGN(64)
    {
        __data_start = .;
        *(.data*)
        __data_end = .;
    } > DDR_CODE

    .bss (NOLOAD) : ALIGN(64)
    {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > DDR_CODE

    .heap_stack (NOLOAD) : ALIGN(64)
    {
        __heap_start = .;
        . = . + 0x00100000;
        __heap_end = .;
        __stack_limit = .;
        . = . + 0x00100000;
        _stack_top = .;
    } > DDR_CODE

    .pccx_weights (NOLOAD) : ALIGN(64)
    {
        __pccx_weight_section_start = .;
        KEEP(*(.pccx_weights))
        . = ALIGN(64);
        __pccx_weight_section_end = .;
    } > DDR_WEIGHTS

    .pccx_activations (NOLOAD) : ALIGN(64)
    {
        __pccx_activation_section_start = .;
        KEEP(*(.pccx_activations))
        . = ALIGN(64);
        __pccx_activation_section_end = .;
    } > DDR_ACTIVATION

    /DISCARD/ :
    {
        *(.note*)
        *(.comment*)
    }
}
