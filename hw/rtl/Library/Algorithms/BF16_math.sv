`timescale 1ns / 1ps
`ifndef BF16_MATH_SV
`define BF16_MATH_SV

package bf16_math_pkg;

  /*─────────────────────────────────────────────
  BF16 struct
  [15]=sign  [14:7]=exp(8b)  [6:0]=mantissa(7b)
  hidden bit is implicit (not stored)
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic       sign;
    logic [7:0] exp;
    logic [6:0] mantissa;
  } bf16_t;

  /*─────────────────────────────────────────────
  Aligned output
  24-bit 2's complement
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic [7:0]  emax;
    logic [23:0] val;
  } bf16_aligned_t;

  /*─────────────────────────────────────────────
  cast raw 16-bit → bf16_t
  ─────────────────────────────────────────────*/
  function automatic bf16_t to_bf16(input logic [15:0] raw);
    return bf16_t'{sign: raw[15], exp: raw[14:7], mantissa: raw[6:0]};
  endfunction

  /*─────────────────────────────────────────────
  align one BF16 value to a given emax
  returns 24-bit 2's complement
  ─────────────────────────────────────────────*/
  function automatic logic [23:0] align_to_emax(input bf16_t val, input logic [7:0] emax);
    logic [ 7:0] diff;
    logic [22:0] mag;
    logic [23:0] result;

    diff   = emax - val.exp;
    mag    = ({1'b1, val.mantissa, 15'd0}) >> diff;
    result = val.sign ? (~{1'b0, mag} + 24'd1) : {1'b0, mag};
    return result;
  endfunction

  /*─────────────────────────────────────────────
  BF16 add: a + b as packed 16-bit values
  - aligns to the larger exponent
  - signed-adds the 24-bit aligned mantissas
  - renormalizes by counting the leading one
  - repacks to BF16
  First-pass implementation: no denormal / NaN / Inf handling; softmax
  uses normalized BF16 operands so the subtle corner cases don't fire
  on the autoregressive decode path. Used by CVO_top's sub-emax stage.
  ─────────────────────────────────────────────*/
  function automatic logic [15:0] bf16_add(input logic [15:0] a,
                                           input logic [15:0] b);
    bf16_t         av, bv;
    logic [7:0]    emax;
    logic [23:0]   aa, ba;
    logic signed [24:0] sum;
    logic               out_sign;
    logic [23:0]   mag;
    int            lead;
    logic [7:0]    out_exp;
    logic [6:0]    out_mant;

    av   = to_bf16(a);
    bv   = to_bf16(b);
    emax = (av.exp > bv.exp) ? av.exp : bv.exp;

    aa = align_to_emax(av, emax);
    ba = align_to_emax(bv, emax);
    sum = $signed({aa[23], aa}) + $signed({ba[23], ba});

    out_sign = sum[24];
    mag      = out_sign ? (~sum[23:0] + 24'd1) : sum[23:0];

    if (mag == 24'd0) return 16'd0;

    // Find the position of the leading 1 (MSB-first).
    lead = 23;
    while (lead > 0 && mag[lead] == 1'b0) lead = lead - 1;

    // Re-bias exponent. The mantissa's implicit leading-1 is at bit 15
    // before alignment; "lead - 15" is the net exponent correction.
    out_exp  = emax + 8'(lead - 15);

    // 7 mantissa bits immediately below the leading 1.
    if (lead >= 7)
      out_mant = mag[lead-1 -: 7];
    else
      out_mant = 7'(mag << (7 - lead));

    return {out_sign, out_exp, out_mant};
  endfunction

endpackage

`endif
