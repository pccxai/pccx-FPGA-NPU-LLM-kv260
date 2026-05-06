`timescale 1ns / 1ps

import isa_pkg::*;
import vec_core_pkg::*;

// ===| Module: GEMV_lane_mask_decode |=========================================
// Purpose      : Translate the ISA GEMV parallel_lane field into per-lane
//                activation bits for the Vector Core.
// Policy       : parallel_lane == 0 selects all implemented GEMV lanes.
// =============================================================================
module GEMV_lane_mask_decode
  import isa_pkg::*;
  import vec_core_pkg::*;
#(
    parameter gemv_cfg_t param = VecCoreDefaultCfg
) (
    input  parallel_lane_t IN_parallel_lane,
    output logic           OUT_activated_lane[0:param.num_gemv_pipeline-1]
);

  always_comb begin
    for (int lane = 0; lane < param.num_gemv_pipeline; lane++) begin
      OUT_activated_lane[lane] = (IN_parallel_lane == '0) ? 1'b1 : IN_parallel_lane[lane];
    end
  end

endmodule
