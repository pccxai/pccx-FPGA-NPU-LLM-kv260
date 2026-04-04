`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// Descending order
module Vdotm_accumulate #(
    parameter fmap_line_length = 32,
    parameter weight_batch = 32,
    parameter weight_column = 1024
) (
    input logic clk,
    input logic rst_n,
    input logic [`FIXED_MANT_WIDTH+2:0] IN_reduction_result,

    input logic IN_vaild,
    //idx range 1~1024
    input logic [11:0] index_of_result
);
  // fmap_line_length = Throughput per clk(N/32) = 32
  // weight_batch * weight_batch = weight_column / 2
  // [32] * [32] = [1024] * 2 = [2048]
  // vector (1, N) dot Matrix (N, M)
  // result (1, M)
  // Before we send result to ACP, we must type-cast FP32 to BF16(2Byte)


  logic [`FIXED_MANT_WIDTH+2:0] vdotm_result_vector[0:weight_batch-1][0:weight_batch-1];

  //0~31
  logic [6:0] currnt_index;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int batch_idx = 0; batch_idx < weight_batch; batch_idx++) begin
        for (int i = 0; i < weight_column; i++) begin
          vdotm_result_vector[batch_idx][i] <= '0;
        end
      end
    end else begin
      if (IN_vaild) begin
        vdotm_result_vector[currnt_index][index_of_result] <=
        vdotm_result_vector[currnt_index][index_of_result]
        + IN_reduction_result;
      end
    end
  end
endmodule
