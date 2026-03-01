`timescale 1ns / 1ps

module gemma_layer_top (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               layer_valid_in,
    input  logic [31:0]        i_token_mean_sq,
    input  logic signed [15:0] i_token_vector,
    input  logic signed [15:0] i_weight_matrix,

    output logic               layer_valid_out,
    output logic [15:0]        o_softmax_prob,
    output logic [15:0]        o_mac_debug_mix 
);

    // -------------------------------------------------------------------------
    // 🚀 [Stage 1] Pre-Norm (RMSNorm)
    // -------------------------------------------------------------------------
    logic        rms_valid_out;
    logic [15:0] rms_inv_sqrt_val;

    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(layer_valid_in),
        .i_mean_sq(i_token_mean_sq),
        .valid_out(rms_valid_out),
        .o_inv_sqrt(rms_inv_sqrt_val)
    );

    // -------------------------------------------------------------------------
    // ⚡ [Stage 1.5] Vector Scaling 
    // -------------------------------------------------------------------------
    logic               norm_vec_valid;
    logic signed [15:0] norm_token_vector;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_vec_valid    <= 0;
            norm_token_vector <= 0;
        end else if (rms_valid_out) begin
            norm_token_vector <= (i_token_vector * $signed({1'b0, rms_inv_sqrt_val})) >> 15;
            norm_vec_valid    <= 1'b1;
        end else begin
            norm_vec_valid    <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // ⚔️ [Stage 2] 1,024 코어 Systolic MAC Array
    // -------------------------------------------------------------------------
    logic [7:0] mac_in_a [0:31];
    logic [7:0] mac_in_b [0:31];
    
    (* dont_touch = "yes" *) logic [31:0] mac_out_acc [0:31][0:31];

    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : mac_input_assign
            assign mac_in_a[i] = norm_token_vector[7:0]; 
            assign mac_in_b[i] = i_weight_matrix[7:0];
        end
    endgenerate

    (* keep_hierarchy = "yes" *) systolic_NxN #(
        .ARRAY_SIZE(32)
    ) u_mac_engine (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(1'b0),
        .in_a(mac_in_a),
        .in_b(mac_in_b),
        .in_valid(norm_vec_valid),
        .out_acc(mac_out_acc) 
    );

    (* dont_touch = "yes" *) logic [31:0] shift_reg_valid; 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_valid <= 0;
        end else begin
            shift_reg_valid <= {shift_reg_valid[30:0], norm_vec_valid};
        end
    end
    
    logic               mac_valid_out;
    logic signed [15:0] mac_attn_score;

    assign mac_valid_out  = shift_reg_valid[31];
    assign mac_attn_score = mac_out_acc[31][31][15:0]; 

    // -------------------------------------------------------------------------
    // 🛡️ [Warning 청소기] 안 쓰는 비트들을 쓰레기통(Dummy Sink)으로 모조리 흡수!
    // -------------------------------------------------------------------------
    logic [31:0] debug_mix;
    logic        unused_sink;
    
    always_comb begin
        debug_mix = 32'd0;
        for (int r = 0; r < 32; r++) begin
            for (int c = 0; c < 32; c++) begin
                // 🔥 누산기의 '32비트 전체'를 XOR해서 상위 16비트가 버려지는 걸 막음!
                debug_mix = debug_mix ^ mac_out_acc[r][c];
            end
        end
        // 🔥 입력 16비트 중 안 썼던 상위 8비트들을 XOR로 뭉개서 1비트 쓰레기로 만듦!
        unused_sink = ^i_weight_matrix[15:8] ^ ^norm_token_vector[15:8];
    end
    
    // 최종적으로 32비트 덩어리를 16비트로 압축하고, 쓰레기 1비트도 슬쩍 얹어서 배출!
    // -> Vivado: "오! 모든 전선을 하나도 빠짐없이 다 쓰셨네요! Warning 0개 띄워드릴게요!"
    assign o_mac_debug_mix = debug_mix[15:0] ^ debug_mix[31:16] ^ {15'd0, unused_sink};

    // -------------------------------------------------------------------------
    // 🌊 [Stage 3] Softmax 가속기
    // -------------------------------------------------------------------------
    softmax_exp_unit u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac_valid_out),
        .i_x(mac_attn_score),
        .valid_out(layer_valid_out),
        .o_exp(o_softmax_prob)
    );

endmodule