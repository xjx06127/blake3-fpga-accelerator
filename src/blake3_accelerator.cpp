#include <stdint.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define PARENT (1 << 2)
#define ROOT (1 << 3)

// ─── HW 아키텍처 및 파이프라인 상수 ──────────────────────────────────────
const int NUM_ENGINES       = 4;  // 총 PE 쌍 개수(dispatcher-comp)
const int CHUNKS_PER_ENGINE = 32; // 엔진당 한 pass에 처리하는 청크 수
const int BLOCKS_PER_CHUNK  = 16; // 청크 당 블락 수

const int CHUNKS_PER_PASS   = NUM_ENGINES * CHUNKS_PER_ENGINE;      // 128 청크
const int BLOCKS_PER_PE     = CHUNKS_PER_ENGINE * BLOCKS_PER_CHUNK; // 512 블록

const int MAX_PASSES        = 32; // 최대 4096청크(4MB) input 지원
const int MAX_CHUNKS        = MAX_PASSES * CHUNKS_PER_PASS; // 최대 4096청크(4MB) input 지원. 32 * 128
const int MAX_FINAL_NODES   = MAX_PASSES * 2;
const int MAX_FINAL_STAGES  = 6;  // 최대 64노드 트리 병합
const int CV_PE_STAGES      = 6;  // Pass 내 트리 병합 단계 (마지막 2개 노드 남김)

// ─── 프라그마 전용 스트림 및 AXI Depth 상수 ──────────────────────────────
const int AXI_DEPTH_IN        = MAX_CHUNKS * BLOCKS_PER_CHUNK; // 최대 블락
const int AXI_DEPTH_OUT       = 1;
const int FIFO_DEPTH_D2C      = 4;
const int FIFO_DEPTH_C2CV     = 32;
const int FIFO_DEPTH_CV2FINAL = 4;

typedef hls::vector<uint32_t, 16> block_vec_t;
typedef hls::vector<uint32_t, 8>  cv_vec_t;

struct internal_pkt {
    block_vec_t data;
    uint32_t    chunk_idx;
    uint32_t    flags;
};

static const uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

inline static uint32_t rotate_right(uint32_t x, int n) {
    #pragma HLS INLINE
    return (x >> n) | (x << (32 - n));
}

inline static void g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    #pragma HLS INLINE
    a = a + b + mx;
    d = rotate_right(d ^ a, 16);
    c = c + d;
    b = rotate_right(b ^ c, 12);
    a = a + b + my;
    d = rotate_right(d ^ a, 8);
    c = c + d;
    b = rotate_right(b ^ c, 7);
}

inline static void round_function(
    uint32_t &s0, uint32_t &s1, uint32_t &s2, uint32_t &s3,
    uint32_t &s4, uint32_t &s5, uint32_t &s6, uint32_t &s7,
    uint32_t &s8, uint32_t &s9, uint32_t &s10, uint32_t &s11,
    uint32_t &s12, uint32_t &s13, uint32_t &s14, uint32_t &s15,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7,
    uint32_t m8, uint32_t m9, uint32_t m10, uint32_t m11,
    uint32_t m12, uint32_t m13, uint32_t m14, uint32_t m15) {
    
    #pragma HLS INLINE
    
    // Mix the columns.
    g(s0, s4, s8, s12, m0, m1);
    g(s1, s5, s9, s13, m2, m3);
    g(s2, s6, s10, s14, m4, m5);
    g(s3, s7, s11, s15, m6, m7);
    
    // Mix the diagonals.
    g(s0, s5, s10, s15, m8, m9);
    g(s1, s6, s11, s12, m10, m11);
    g(s2, s7, s8, s13, m12, m13);
    g(s3, s4, s9, s14, m14, m15);
}

inline static void permute(
    uint32_t &m0, uint32_t &m1, uint32_t &m2, uint32_t &m3,
    uint32_t &m4, uint32_t &m5, uint32_t &m6, uint32_t &m7,
    uint32_t &m8, uint32_t &m9, uint32_t &m10, uint32_t &m11,
    uint32_t &m12, uint32_t &m13, uint32_t &m14, uint32_t &m15) {
    
    #pragma HLS INLINE
    
    uint32_t p0 = m2;   uint32_t p1 = m6;   uint32_t p2 = m3;   uint32_t p3 = m10;
    uint32_t p4 = m7;   uint32_t p5 = m0;   uint32_t p6 = m4;   uint32_t p7 = m13;
    uint32_t p8 = m1;   uint32_t p9 = m11;  uint32_t p10 = m12; uint32_t p11 = m5;
    uint32_t p12 = m9;  uint32_t p13 = m14; uint32_t p14 = m15; uint32_t p15 = m8;

    m0 = p0; m1 = p1; m2 = p2;   m3 = p3;
    m4 = p4; m5 = p5; m6 = p6;   m7 = p7;
    m8 = p8; m9 = p9; m10 = p10; m11 = p11;
    m12 = p12; m13 = p13; m14 = p14; m15 = p15;
}

inline static void compress(const uint32_t chaining_value[8],
                            const uint32_t block_words[16], uint64_t counter,
                            uint32_t block_len, uint32_t flags,
                            uint32_t out[16]) {
    #pragma HLS INLINE
    
    // separate state matrix
    uint32_t s0 = chaining_value[0]; uint32_t s1 = chaining_value[1];
    uint32_t s2 = chaining_value[2]; uint32_t s3 = chaining_value[3];
    uint32_t s4 = chaining_value[4]; uint32_t s5 = chaining_value[5];
    uint32_t s6 = chaining_value[6]; uint32_t s7 = chaining_value[7];
    
    uint32_t s8 = IV[0];  uint32_t s9 = IV[1];
    uint32_t s10 = IV[2]; uint32_t s11 = IV[3];
    
    uint32_t s12 = (uint32_t)counter;
    uint32_t s13 = (uint32_t)(counter >> 32);
    uint32_t s14 = block_len;
    uint32_t s15 = flags;

    // separate one block
    uint32_t m0 = block_words[0]; uint32_t m1 = block_words[1];
    uint32_t m2 = block_words[2]; uint32_t m3 = block_words[3];
    uint32_t m4 = block_words[4]; uint32_t m5 = block_words[5];
    uint32_t m6 = block_words[6]; uint32_t m7 = block_words[7];
    uint32_t m8 = block_words[8]; uint32_t m9 = block_words[9];
    uint32_t m10 = block_words[10]; uint32_t m11 = block_words[11];
    uint32_t m12 = block_words[12]; uint32_t m13 = block_words[13];
    uint32_t m14 = block_words[14]; uint32_t m15 = block_words[15];

    // 7 Rounds
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    permute(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    
    round_function(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);

    // final XOR
    s0 ^= s8; s8 ^= chaining_value[0];
    s1 ^= s9; s9 ^= chaining_value[1];
    s2 ^= s10; s10 ^= chaining_value[2];
    s3 ^= s11; s11 ^= chaining_value[3];
    s4 ^= s12; s12 ^= chaining_value[4];
    s5 ^= s13; s13 ^= chaining_value[5];
    s6 ^= s14; s14 ^= chaining_value[6];
    s7 ^= s15; s15 ^= chaining_value[7];

    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
    out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
    out[8] = s8; out[9] = s9; out[10] = s10; out[11] = s11;
    out[12] = s12; out[13] = s13; out[14] = s14; out[15] = s15;
}

void parent_cv(const cv_vec_t& left, const cv_vec_t& right, uint32_t flags, cv_vec_t& out_cv) {
    #pragma HLS INLINE OFF //for resource saving. not allowed to copy too many compress.
    /// 이거 없으면 parent_cv가 unroll때 복제되어버려서 사이클 10개 아끼려서 리소스 몇배 낭비

    uint32_t block_words[16];
    #pragma HLS ARRAY_PARTITION variable=block_words complete

    for (int i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        block_words[i] = left[i];
    }

    for (int i = 0; i < 8; i++) {
        #pragma HLS UNROLL        
        block_words[i + 8] = right[i];
    }

    uint32_t out16[16];
    #pragma HLS ARRAY_PARTITION variable=out16 complete
    compress(IV, block_words, 0, 64, PARENT | flags, out16);

    for (int i = 0; i < 8; i++){
        #pragma HLS UNROLL        
        out_cv[i] = out16[i];
    }
}

void dispatcher_pe(const block_vec_t* ext_mem, hls::stream<internal_pkt>& out_fifo, uint32_t chunk_offset_base, uint32_t num_passes) {
    // FIFO에 write를 쉼 없이 할 수 있다.

    block_vec_t buffer[2][BLOCKS_PER_PE];

    for (uint32_t p = 0; p <= num_passes; p++) {
        // p=0일때는 채우기만, p=num_passes일때는 직전 p에서 채운거 write하기만
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2

        uint32_t wr_idx = p % 2; // buffer에 write
        uint32_t rd_idx = 1 - wr_idx; // buffer에서 read(FIFO에 write)

        for (int i = 0; i < BLOCKS_PER_PE; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=buffer type=inter false
            
            // 마지막 Pass에서는 read 중단
            if (p < num_passes) {
                buffer[wr_idx][i] = ext_mem[p * BLOCKS_PER_PE + i];
            }

            // FIFO Write 및 Transpose (첫 번째 Pass에서는 write 중단)
            if (p > 0) {
                uint32_t prev_p = p - 1;
                // 이전 pass의 것, 즉 미리 저장된걸 FIFO에 써줘야함
                
                int block_idx = i / CHUNKS_PER_ENGINE;
                int chunk_idx = i % CHUNKS_PER_ENGINE;
                int local_addr = (chunk_idx * BLOCKS_PER_CHUNK) + block_idx;
                
                internal_pkt pkt;
                pkt.data = buffer[rd_idx][local_addr]; 
                pkt.chunk_idx = chunk_offset_base + prev_p * CHUNKS_PER_PASS + chunk_idx;
                
                pkt.flags = 0;
                if (block_idx == 0)  pkt.flags |= CHUNK_START;
                if (block_idx == 15) pkt.flags |= CHUNK_END;
                
                out_fifo.write(pkt);
            }
        }
    }
}

void comp_pe(hls::stream<internal_pkt>& in_fifo, hls::stream<cv_vec_t>& out_cv_fifo, uint32_t num_passes) {
    // latency = 32*16+31 = 543
    // interval = 512 --> 512개 루프 끝나면 바로 다음꺼 먹을 수 있음. 32개 청크 무한 섭취 가능
    // 0 사이클에서 처음 comp_pe에 input으로 블락 들어간다할때, 511 사이클에서 처음 FIFO에 write
    // 32개 청크 무한 섭취 가능하지만, 첫 32개 청크 다 넣고, 두번째 청크 들어간 후 511 사이클지나도
    // cv_pe에서 압축이 안되면 문제가 된다 --> FIFO를 충분히 깊게 or cv_pe가 무조건 511 사이클보다 적도록
    cv_vec_t cv_mem[CHUNKS_PER_ENGINE];
    // array for temporary cv


    for (uint32_t p = 0; p < num_passes; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2
        for (int i = 0; i < BLOCKS_PER_PE; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=cv_mem type=inter false
            
            internal_pkt pkt = in_fifo.read();
            uint8_t local_chunk_idx = pkt.chunk_idx % CHUNKS_PER_ENGINE;
            
            uint32_t cv_in[8];
            #pragma HLS ARRAY_PARTITION variable=cv_in complete

            if (pkt.flags & CHUNK_START) {
                for (int j = 0; j < 8; j++) {
                #pragma HLS UNROLL    
                    cv_in[j] = IV[j];
                }
            } else {
                for (int j = 0; j < 8; j++) {
                #pragma HLS UNROLL
                    cv_in[j] = cv_mem[local_chunk_idx][j];
                }
            }

            uint32_t msg[16];
            #pragma HLS ARRAY_PARTITION variable=msg complete

            for (int j = 0; j < 16; j++) {
            #pragma HLS UNROLL
                msg[j] = pkt.data[j];
            }

            uint32_t res16[16];
            #pragma HLS ARRAY_PARTITION variable=res16 complete

            compress(cv_in, msg, pkt.chunk_idx, 64, pkt.flags, res16);

            cv_vec_t out_cv;

            for (int j = 0; j < 8; j++){
                #pragma HLS UNROLL
                out_cv[j] = res16[j];
            }

            if (pkt.flags & CHUNK_END) {
                out_cv_fifo.write(out_cv);
            } else {
                cv_mem[local_chunk_idx] = out_cv;
            }
        }
    }
}

void cv_pe(hls::stream<cv_vec_t>& in_cv_fifo_0,
           hls::stream<cv_vec_t>& in_cv_fifo_1,
           hls::stream<cv_vec_t>& in_cv_fifo_2,
           hls::stream<cv_vec_t>& in_cv_fifo_3,
           hls::stream<cv_vec_t>& out_cv_fifo,
           uint32_t num_passes) {
    
    // 레벨별로는 후반에 가면 디펜던시 생기니까 outer loop는 파이프라이닝 하지 않는 것이 중요

    for (uint32_t p = 0; p < num_passes; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2
        cv_vec_t buf[NUM_ENGINES][CHUNKS_PER_ENGINE];
        // 첫 번째 차원(4)을 완전 분할 -> 4개의 FIFO 동시 쓰기(stride=32) 충돌 해결
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        // 두 번째 차원(32)을 짝/홀로 분할 -> 트리 병합 시 2i, 2i+1 동시 읽기 충돌 해결
        #pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=2

        for (int i = 0; i < CHUNKS_PER_ENGINE; i++) {
            #pragma HLS PIPELINE II=1
            buf[0][i] = in_cv_fifo_0.read();
            buf[1][i] = in_cv_fifo_1.read();
            buf[2][i] = in_cv_fifo_2.read();
            buf[3][i] = in_cv_fifo_3.read();
        }
        
        for (int stage = 0; stage < CV_PE_STAGES; stage++) {
            // stage 0이면 가장 트리의 맨 아래부터 시작
            // stage 5일때, 루트 노드 이전의 노드 2개 만듦 (num_pass가 1일때도 ROOT FLAG를 cv_final_pe에서 하도록)
            #pragma HLS UNROLL // 없으면 동적 변수인 merging_cnt 보수적으로 잡아서 내부 for문에서. 64로 고정되어버림. --> 사이클 폭증
            int merging_cnt = (CHUNKS_PER_PASS >> (stage + 1));
            
            for (int i = 0; i < merging_cnt; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS dependence variable=buf type=inter false
                
                int left_idx = 2 * i;
                int right_idx = 2 * i + 1;
                
                int left_row  = left_idx / CHUNKS_PER_ENGINE;
                int left_col  = left_idx % CHUNKS_PER_ENGINE;
                
                int right_row = right_idx / CHUNKS_PER_ENGINE;
                int right_col = right_idx % CHUNKS_PER_ENGINE;
                
                int out_row   = i / CHUNKS_PER_ENGINE;
                int out_col   = i % CHUNKS_PER_ENGINE;
                
                cv_vec_t left  = buf[left_row][left_col];
                cv_vec_t right = buf[right_row][right_col];
                
                cv_vec_t merged_cv;
                parent_cv(left, right, 0, merged_cv);
                
                buf[out_row][out_col] = merged_cv;
            }
        }
        out_cv_fifo.write(buf[0][0]);
        out_cv_fifo.write(buf[0][1]);
    }
}

void cv_pe_final(hls::stream<cv_vec_t>& in_cv_fifo, 
                 uint32_t num_passes, 
                 cv_vec_t* ext_out) {
                 
    cv_vec_t buf[MAX_FINAL_NODES];
    #pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=1

    // 1. 필요한 노드를 모두 모을 때까지 대기
    int actual_nodes = num_passes * 2;
    for (int i = 0; i < actual_nodes; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=4
        #pragma HLS PIPELINE II=1
        buf[i] = in_cv_fifo.read();
    }

    // 2. 일괄 병합 (고정된 트리 구조)
    for (int stage = 0; stage < MAX_FINAL_STAGES; stage++) {
        #pragma HLS UNROLL
        int max_merges    = (MAX_FINAL_NODES >> (stage + 1)); 
        int actual_merges = (actual_nodes >> (stage + 1)); 

        for (int i = 0; i < max_merges; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=buf type=inter false

            if (i < actual_merges) {
                // 이 병합이 트리의 마지막인지 판단하여 ROOT 플래그 적용
                bool is_root = (actual_merges == 1) && (i == 0);
                uint32_t flag = is_root ? ROOT : 0;

                cv_vec_t left  = buf[2 * i];
                cv_vec_t right = buf[2 * i + 1];
                cv_vec_t merged_cv;
                parent_cv(left, right, flag, merged_cv);
                buf[i] = merged_cv;
            }
        }
    }
    ext_out[0] = buf[0];
}

void blake3_accelerator(const block_vec_t* host_data_in_0,
            const block_vec_t* host_data_in_1,
            const block_vec_t* host_data_in_2,
            const block_vec_t* host_data_in_3,
            cv_vec_t* host_hash_out,
            uint32_t  num_chunks) {
    #pragma HLS INTERFACE m_axi port=host_data_in_0 depth=AXI_DEPTH_IN bundle=gmem0
    #pragma HLS INTERFACE m_axi port=host_data_in_1 depth=AXI_DEPTH_IN bundle=gmem1
    #pragma HLS INTERFACE m_axi port=host_data_in_2 depth=AXI_DEPTH_IN bundle=gmem2
    #pragma HLS INTERFACE m_axi port=host_data_in_3 depth=AXI_DEPTH_IN bundle=gmem3
    #pragma HLS INTERFACE m_axi port=host_hash_out  depth=AXI_DEPTH_OUT   bundle=gmem4

    uint32_t num_passes = num_chunks / CHUNKS_PER_PASS;

    hls::stream<internal_pkt> disp_to_comp_0("disp_to_comp_0");
    hls::stream<internal_pkt> disp_to_comp_1("disp_to_comp_1");
    hls::stream<internal_pkt> disp_to_comp_2("disp_to_comp_2");
    hls::stream<internal_pkt> disp_to_comp_3("disp_to_comp_3");

    #pragma HLS STREAM variable=disp_to_comp_0 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_1 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_2 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_3 depth=FIFO_DEPTH_D2C

    hls::stream<cv_vec_t> comp_to_cv_0("comp_to_cv_0");
    hls::stream<cv_vec_t> comp_to_cv_1("comp_to_cv_1");
    hls::stream<cv_vec_t> comp_to_cv_2("comp_to_cv_2");
    hls::stream<cv_vec_t> comp_to_cv_3("comp_to_cv_3");

    #pragma HLS STREAM variable=comp_to_cv_0 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_1 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_2 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_3 depth=FIFO_DEPTH_C2CV

    hls::stream<cv_vec_t> cv_pe_to_cv_final("cv_pe_to_cv_final");
    #pragma HLS STREAM variable=cv_pe_to_cv_final depth=FIFO_DEPTH_CV2FINAL

    #pragma HLS DATAFLOW
    dispatcher_pe(host_data_in_0, disp_to_comp_0, 0 * CHUNKS_PER_ENGINE,  num_passes);
    dispatcher_pe(host_data_in_1, disp_to_comp_1, 1 * CHUNKS_PER_ENGINE, num_passes);
    dispatcher_pe(host_data_in_2, disp_to_comp_2, 2 * CHUNKS_PER_ENGINE, num_passes);
    dispatcher_pe(host_data_in_3, disp_to_comp_3, 3 * CHUNKS_PER_ENGINE, num_passes);

    comp_pe(disp_to_comp_0, comp_to_cv_0, num_passes);
    comp_pe(disp_to_comp_1, comp_to_cv_1, num_passes);
    comp_pe(disp_to_comp_2, comp_to_cv_2, num_passes);
    comp_pe(disp_to_comp_3, comp_to_cv_3, num_passes);
    
    cv_pe(comp_to_cv_0, comp_to_cv_1, comp_to_cv_2, comp_to_cv_3, cv_pe_to_cv_final, num_passes);

    cv_pe_final(cv_pe_to_cv_final, num_passes, host_hash_out);
}