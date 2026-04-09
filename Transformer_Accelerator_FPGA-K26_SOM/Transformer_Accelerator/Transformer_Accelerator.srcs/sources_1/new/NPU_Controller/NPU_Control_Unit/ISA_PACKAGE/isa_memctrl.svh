package isa_memctrl;

  `define PORT_MOD_E_WRITE 1
  `define PORT_MOD_E_READ 0

  typedef enum logic [3:0] {
    data_to_host              = 4'h0,
    data_to_L2_cache          = 4'h1,
    data_to_L1_cache_stlc_in  = 4'h2,
    data_to_L1_cache_vdotm_in = 4'h3,
    data_to_fmap_shape        = 4'h4,
    data_to_weight_shape      = 4'h5
  } data_dest_e;

  typedef enum logic [3:0] {
    data_from_host               = 4'h0,
    data_from_L2_cache           = 4'h1,
    data_from_L1_cache_stlc_res  = 4'h2,
    data_from_L1_cache_vdotm_res = 4'h3
  } data_source_e;

  typedef enum logic [7:0] {
    from_host_to_L2 = {data_from_host, data_to_L2_cache},

    from_host_to_fmap_shape   = {data_from_host, data_to_fmap_shape},
    from_host_to_weight_shape = {data_from_host, data_to_weight_shape},

    from_L2_to_host = {data_from_L2_cache, data_to_host},

    from_L2_to_L1_stlc  = {data_from_L2_cache, data_to_L1_cache_stlc_in},
    from_L2_to_L1_vdotm = {data_from_L2_cache, data_to_L1_cache_vdotm_in},

    from_vdotm_res_to_L2 = {data_from_L1_cache_vdotm_res, data_to_L2_cache},
    from_stlc_res_to_L2  = {data_from_L1_cache_stlc_res, data_to_L2_cache}
  } data_route_e;

  typedef struct packed {
    data_route_e data_dest;

    dest_addr_t dest_addr;
    src_addr_t  src_addr;

    ptr_addr_t shape_ptr_addr;

    async_e async;
  } memory_control_uop_t;


  // mem dispatcher.sv
  typedef enum logic {
    NPU_U_OP_WIDTH = 33,
    ACP_U_OP_WIDTH = 33
  } npu_acp_u_op_width_e;

  typedef struct packed {
    logic        acp_write_en_wire;
    logic [16:0] acp_base_addr_wire;
    logic [16:0] acp_end_addr;
  } acp_uop_t;  //33 bit == [32:0]

  typedef struct packed {
    logic        npu_write_en_wire;
    logic [16:0] npu_base_addr_wire;
    logic [16:0] npu_end_addr;
  } npu_uop_t;  //33 bit == [32:0]


  /*
  function automatic [3:0] option_data_to_host(
    input data_dest_e dest,
    input data_addr_t dest_addr
    );
    case(dest)
      data_from_L2_cache: begin
        mem_uop.src_addr
      end
    default: begin end
    endcase
  endfunction


  acp_rx_start
  acp_write_en

  function automatic [3:0] option_data_to_L2_cache(
    input data_dest_e dest,
    input data_addr_t dest_addr
    );
    case(dest)
      data_from_host: begin
        option_data_to_L2_cache = dest_addr;
      end
      data_from_L1_cache_stlc_res : begin

      end
      data_from_L1_cache_vdotm_res: begin

      end

    default: begin end
    endcase
  endfunction


  function automatic [3:0] option_data_to_L1_cache_stlc_res(input data_dest_e dest);
    case(dest)
    :begin
    end
    default: begin end
    endcase
  endfunction


  function automatic [3:0] option_data_to_L1_cache_vdotm_res(input data_dest_e dest);
    case(dest)
    :begin
    end
    default: begin end
    endcase
  endfunction
*/

  /*
  function automatic [3:0] option_data_from_host(input data_dest_e dest);
    case(dest)
      data_to_L2_cache: begin
        mem_uop.dest_addr
      end
    default: begin end
    endcase
  endfunction


  function automatic [3:0] option_data_from_L2_cache(input data_dest_e dest);
    case(dest)
    :begin
    end
    default: begin end
    endcase
  endfunction


  function automatic [3:0] option_data_from_L1_cache_stlc_res(input data_dest_e dest);
    case(dest)
    :begin
    end
    default: begin end
    endcase
  endfunction


  function automatic [3:0] option_data_from_L1_cache_vdotm_res(input data_dest_e dest);
    case(dest)
    :begin
    end
    default: begin end
    endcase
  endfunction

*/
endpackage
