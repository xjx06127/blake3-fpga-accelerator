#include <stdint.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define PARENT (1 << 2)
#define ROOT (1 << 3)

// ─── HW 아키텍처 및 파이프라인 상수 ──────────────────────────────────────
const int NUM_ENGINES       = 8;  // 총 PE 쌍 개수(dispatcher-comp) 
const int CHUNKS_PER_ENGINE = 32; // 엔진당 한 pass에 처리하는 청크 수
const int BLOCKS_PER_CHUNK  = 16; // 청크 당 블락 수 

const int CHUNKS_PER_PASS   = NUM_ENGINES * CHUNKS_PER_ENGINE;      // 256 청크
const int BLOCKS_PER_PE     = CHUNKS_PER_ENGINE * BLOCKS_PER_CHUNK; // 512 블록

// S > 32 일 때 pass간에 쌓아둘 32청크 서브트리 루트의 최대 개수.
// S <= CHUNKS_PER_ENGINE * MAX_SUB = 32 * 32 = 1024 청크(1MiB) 까지 지원.
const int MAX_SUB = 32;
// ─── FIFO depth 상수 ──────────────────────────────
const int FIFO_DEPTH_D2C      = 4;
const int FIFO_DEPTH_C2CV     = 16;

// const int FIFO_DEPTH_CV2FINAL = 32;
// // 기존엔 pass당 2개만 흘렀지만, configurable ver.는 S이 작을수록 pass당 128/M개가
// //        한꺼번에 쏟아진다 (S=4면 32개). 얕으면 cv_pe가 stall → BW 손실.
// // cv2final에서 나중에 혹시 512 cycle안에 처리 못하면 FIFO 터질수도

typedef hls::vector<uint32_t, 16> block_vec_t; // 각 원소가 한 워드 -> 한 개의 block
typedef hls::vector<uint32_t, 8>  cv_vec_t; // 벡터 전체가 한 CV

struct internal_pkt {
    block_vec_t data;
    uint64_t    chunk_idx; // 전역 청크 번호
    uint32_t    flags;
};

struct cv_pair_t{
    cv_vec_t l; // 짝수 청크
    cv_vec_t r; // 홀수 청크
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

// 나중에 실제 핑퐁말고 그냥 dispatcher와 fpga 상에서 성능비교해보자
void dispatcher_pe(const block_vec_t* ext_mem, hls::stream<internal_pkt>& out_fifo, uint64_t chunk_offset_base, uint64_t num_passes) {
    // FIFO에 write를 쉼 없이 할 수 있다.

    block_vec_t buffer[2][BLOCKS_PER_PE];

    for (uint64_t p = 0; p <= num_passes; p++) {
        // p=0일때는 채우기만, p=num_passes일때는 직전 p에서 채운거 write하기만

        uint32_t wr_idx = p % 2; // buffer에 write
        uint32_t rd_idx = 1 - wr_idx; // buffer에서 read(FIFO에 write)

        for (int i = 0; i < BLOCKS_PER_PE; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=buffer type=inter false
            
            // p == num_passes일 경우엔 read 중단
            if (p < num_passes) {
                buffer[wr_idx][i] = ext_mem[p * BLOCKS_PER_PE + i];
            }

            // FIFO Write 및 Transpose (첫 번째 Pass에서는 write 중단)
            if (p > 0) {
                uint64_t prev_p = p - 1;
                // 이전 pass의 것, 즉 미리 저장된걸 FIFO에 써줘야함
                
                int block_idx = i / CHUNKS_PER_ENGINE;
                int chunk_idx = i % CHUNKS_PER_ENGINE;
                int local_addr = (chunk_idx * BLOCKS_PER_CHUNK) + block_idx;
                
                internal_pkt pkt;
                if(rd_idx==0){
                    pkt.data = buffer[0][local_addr];
                }
                else{
                    pkt.data = buffer[1][local_addr];
                }
                // read만 분기하는 것이 자원소모량, negative slack 측면에서 제일 우수

                // pkt.chunk_idx = chunk_offset_base + prev_p * CHUNKS_PER_PASS + chunk_idx;
                // // 전역 청크 번호를 그대로 실어보낸다.
                // // 세그먼트 내 counter로의 변환(마스킹)은 comp_pe에서 수행.

                // block 파티셔닝을 위해 엔진이 맡은 청크단위로 건너뛰기
                pkt.chunk_idx = chunk_offset_base + prev_p * CHUNKS_PER_ENGINE + chunk_idx;
                
                pkt.flags = 0;
                if (block_idx == 0)  pkt.flags |= CHUNK_START;
                if (block_idx == 15) pkt.flags |= CHUNK_END;
                
                out_fifo.write(pkt);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  comp_pe  [변경점 : counter 마스킹 한 줄]
//
//  BLAKE3는 세그먼트가 바뀌면 chunk counter가 0부터 다시 시작해야 한다.
//    전역 청크번호:  0 1 2 3 | 4 5 6 7 | 8 9 10 11 ...
//    S=4일 때 counter: 0 1 2 3 | 0 1 2 3 | 0 1  2  3 ...   ← 리셋
//  S가 2의 거듭제곱이므로 마스크 한 번이면 끝
//
//  주의: cv_mem 슬롯 주소(local_chunk_idx)는 전역 청크번호 % 32 그대로 써야 한다.
//        세그먼트 counter와 혼동하지 말 것.
// ══════════════════════════════════════════════════════════════════════════ 
void comp_pe(hls::stream<internal_pkt>& in_fifo, hls::stream<cv_pair_t>& out_cv_fifo,  
    uint64_t num_passes, uint32_t chunks_per_segment) {
    // latency = 32*16+31 = 543
    // interval = 512 --> 512개 루프 끝나면 바로 다음꺼 먹을 수 있음. 32개 청크 무한 섭취 가능
    // 0 사이클에서 처음 comp_pe에 input으로 블락 들어간다할때, 511 사이클에서 처음 FIFO에 write
    // 32개 청크 무한 섭취 가능하지만, 첫 32개 청크 다 넣고, 두번째 청크 들어간 후 511 사이클지나도
    // cv_pe에서 압축이 안되면 문제가 된다 --> FIFO를 충분히 깊게 or cv_pe가 무조건 511 사이클보다 적도록
    cv_vec_t cv_mem[CHUNKS_PER_ENGINE];
    // array for temporary cv

    cv_vec_t pend; // pair로 만들어서 보내기 위함.

    const uint64_t counter_mask = (uint64_t)chunks_per_segment - 1; // M이 2^k라서 마스크로 대체

    for (uint64_t p = 0; p < num_passes; p++) {
        for (int i = 0; i < BLOCKS_PER_PE; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=cv_mem type=inter false
            
            internal_pkt pkt = in_fifo.read();
            uint8_t engine_chunk_idx = pkt.chunk_idx % CHUNKS_PER_ENGINE; // 0~31
            
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
                    cv_in[j] = cv_mem[engine_chunk_idx][j];
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

            // 변경점: 전역 청크번호 -> 세그먼트 내 counter
            // 하위 S비트 추출
            uint64_t seg_counter = pkt.chunk_idx & counter_mask;
            compress(cv_in, msg, seg_counter, 64, pkt.flags, res16);

            cv_vec_t out_cv;

            for (int j = 0; j < 8; j++){
                #pragma HLS UNROLL
                out_cv[j] = res16[j];
            }

            // if (pkt.flags & CHUNK_END) {
            //     out_cv_fifo.write(out_cv);
            // } else {
            //     cv_mem[engine_chunk_idx] = out_cv;
            // }

            if (pkt.flags & CHUNK_END) {
                // 짝수 청크면 보류, 홀수 청크면 pair로 write
                if ((pkt.chunk_idx & 1) == 0) {
                    pend = out_cv;
                } else {
                    cv_pair_t pr;
                    pr.l = pend;
                    pr.r = out_cv;
                    out_cv_fifo.write(pr);
                }
            } else {
                cv_mem[engine_chunk_idx] = out_cv;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  cv_pe
//
//  각 stage 공통 패턴:
//    span    = 이 stage가 만드는 노드가 덮는 청크 수 = 2^(k+1)
//    active  = (span <= S)   이 stage를 실행하는가
//    is_root = (span == S)   이 stage 결과가 곧 세그먼트 루트인가
//    active가 false면 이후 stage도 자동으로 멈춘다.
// ══════════════════════════════════════════════════════════════════════════
void cv_pe(hls::stream<cv_pair_t>& in_cv_fifo_0,
           hls::stream<cv_pair_t>& in_cv_fifo_1,
           hls::stream<cv_pair_t>& in_cv_fifo_2,
           hls::stream<cv_pair_t>& in_cv_fifo_3,
           hls::stream<cv_pair_t>& in_cv_fifo_4,
           hls::stream<cv_pair_t>& in_cv_fifo_5,
           hls::stream<cv_pair_t>& in_cv_fifo_6,
           hls::stream<cv_pair_t>& in_cv_fifo_7,
           cv_vec_t* ext_out,
           uint64_t num_chunks,
           uint64_t num_passes,
           uint32_t chunks_per_segment) {

    // S = 2^segment_log2  (4 <= S <= 1024)
    uint32_t segment_log2 = 0;
    for (int bit = 0; bit < 11; bit++) {
        #pragma HLS UNROLL
        if (chunks_per_segment & (1u << bit)) segment_log2 = bit;
    }
    // 엔진 하나가 소유하는 세그먼트 개수 = (num_chunks / S) / NUM_ENGINES
    const uint64_t segments_per_engine = (num_chunks >> segment_log2) / NUM_ENGINES;

    // 목표가 32 이하라면? -> 1 Pass 돌 때 완성품(해시)이 여러 개 쏟아진다.
    // 나눗셈(32 / S)을 피하기 위해 비트 시프트로 계산한다.
    const uint32_t hashes_per_pass_per_engine_log2 = (segment_log2 <= 5) ? (5 - segment_log2) : 0;
    const uint32_t hashes_per_pass_per_engine      = 1u << hashes_per_pass_per_engine_log2;

    // 목표가 32를 초과한다면? -> 1 Pass로는 택도 없으니 여러 번(Pass) 돌려서 재료를 누적해야 한다.
    // 한 패스 당 32 크기의 subtree가 완성된다
    // 나눗셈(S / 32)을 피하기 위해 비트 시프트로 계산한다.
    const uint32_t passes_needed_per_segment_log2 = (segment_log2 > 5) ? (segment_log2 - 5) : 0;
    const uint32_t passes_needed_per_segment      = 1u << passes_needed_per_segment_log2;

    // tree_A, B: 32 크기의 subtree를 만들기 위해 사용하는 배열
    cv_vec_t tree_A[CHUNKS_PER_PASS / 2];
    cv_vec_t tree_B[CHUNKS_PER_PASS / 4];

    // acc_A, B: S 크기의 subtree를 만들기 위해 사용하는 배열
    cv_vec_t acc_A[NUM_ENGINES * MAX_SUB];       // 256 : 엔진 e 블록 [e*passes_needed_per_segment, +passes_needed_per_segment)
                                                 //       = 엔진 e개 x 각 32개 (S 최대 1024청크=1MiB 기준)
                                                 // 엔진당 32 size subtree를 최대 32개 저장하니까.
    cv_vec_t acc_B[NUM_ENGINES * MAX_SUB / 2];   // 128
    // 파티션 없음. [2i],[2i+1] 두 read는 dual-port로 충분하고 write는 반대편
    // 버퍼로 나가므로 포트 충돌이 없다. 뱅크가 없으니 select mux도 없다.

    uint32_t subtrees_collected = 0;   // acc에 쌓인 서브트리 루트 개수 (4엔진 공통)
    uint64_t segments_emitted   = 0;   // 엔진당 지금까지 확정된 세그먼트 개수 (S > 32 경로)

    for (uint64_t p = 0; p < num_passes; p++) {
        
        // FIFO_0에서 가져오는건 II=2라는거 인지
        // FIFO_1~7은 II=1
        for (int i = 0; i < CHUNKS_PER_PASS / 2; i++) {
            #pragma HLS PIPELINE II=1
            cv_pair_t in_pair;
            if      (i < 1 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_0.read();
            else if (i < 2 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_1.read();
            else if (i < 3 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_2.read();
            else if (i < 4 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_3.read();
            else if (i < 5 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_4.read();
            else if (i < 6 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_5.read();
            else if (i < 7 * (CHUNKS_PER_ENGINE / 2)) in_pair = in_cv_fifo_6.read();
            else                                      in_pair = in_cv_fifo_7.read();
            cv_vec_t left = in_pair.l, right = in_pair.r, merged;
            parent_cv(left, right, 0, merged);
            tree_A[i] = merged;
        }

        // 아래 tree stage 4개와 acc sub-stage 5개가 공유하는 레벨 파라미터.
        //   merge_count   : 병합 횟수
        //   is_root : S 크기까지 subtree가 완성되었나
        int  merge_count;
        bool is_root;

        // ── Stage 1 (64 -> 32, span 4 chunks), A read / B write ──
        is_root = (chunks_per_segment == 4);
        merge_count   = (4 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 4) : 0;
        for (int i = 0; i < merge_count; i++) {
            #pragma HLS PIPELINE II=1
            cv_vec_t left = tree_A[2*i];
            cv_vec_t right = tree_A[2*i+1];
            cv_vec_t merged;
            parent_cv(left, right, is_root ? ROOT : 0, merged);
            tree_B[i] = merged;
        }

        // ── Stage 2 (32 -> 16, span 8 chunks), B read / A write ──
        is_root = (chunks_per_segment == 8);
        merge_count   = (8 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 8) : 0;
        for (int i = 0; i < merge_count; i++) {
            #pragma HLS PIPELINE II=1
            cv_vec_t left = tree_B[2*i];
            cv_vec_t right = tree_B[2*i+1];
            cv_vec_t merged;
            parent_cv(left, right, is_root ? ROOT : 0, merged);
            tree_A[i] = merged;
        }

        // ── Stage 3 (16 -> 8, span 16 chunks), A read / B write ──
        is_root = (chunks_per_segment == 16);
        merge_count   = (16 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 16) : 0;
        for (int i = 0; i < merge_count; i++) {
            #pragma HLS PIPELINE II=1
            cv_vec_t left = tree_A[2*i];
            cv_vec_t right = tree_A[2*i+1];
            cv_vec_t merged;
            parent_cv(left, right, is_root ? ROOT : 0, merged);
            tree_B[i] = merged;
        }

        // ── Stage 4 (8 -> 4, span 32 chunks), B read / A write ──
        //    S == 32 이면 여기가 세그먼트 루트.
        //    S >  32 이면 flags=0 인 32청크 서브트리 루트가 tree_A[0..3] 에 남는다.
        is_root = (chunks_per_segment == 32);
        merge_count   = (32 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 32) : 0;
        for (int i = 0; i < merge_count; i++) {
            #pragma HLS PIPELINE II=1
            cv_vec_t left = tree_B[2*i];
            cv_vec_t right = tree_B[2*i+1];
            cv_vec_t merged;
            parent_cv(left, right, is_root ? ROOT : 0, merged);
            tree_A[i] = merged;
        }

        if (chunks_per_segment <= CHUNKS_PER_ENGINE) {
            // ── (S <= 32) : pass 안에서 세그먼트가 닫혔다 ──
            //   루트 결과는 엔진 e 블록 [e*hashes_per_pass_per_engine, +hashes_per_pass_per_engine) 에 정렬돼 있다.
            //   엔진마다 목적지 주소가 연속이라 burst가 잡힌다.
            //   루트가 A인지 B인지는 stage 개수의 홀짝 = segment_log2의 홀짝.
            const bool is_root_in_tree_A = ((segment_log2 & 1) == 1);
            uint64_t out_addr = (uint64_t)p << hashes_per_pass_per_engine_log2;
            // 패스마다 앞에서 부터 채워짐. ex: p==0이면 0~3칸에, p==1이면 4~7칸에 각 엔진들이 채워넣음

            for (int engine = 0; engine < NUM_ENGINES; engine++) {
                uint32_t node_base = (uint32_t)engine << hashes_per_pass_per_engine_log2;
                for (uint32_t j = 0; j < hashes_per_pass_per_engine; j++) {
                    #pragma HLS PIPELINE II=1
                    ext_out[out_addr + j] = is_root_in_tree_A ? tree_A[node_base + j] : tree_B[node_base + j];
                }
                out_addr += segments_per_engine;
                // 엔진 단위로 block partitioning 이니까 한 엔진이 맡은 세그먼트들 만큼 건너뛰어야 다음 엔진 세그먼트들이 나옴
                // 0, 2, 4, 6
                // 1, 3, 5, 7 이런식으로 채워넣음
            }
        } else {
            // ── pass간 누적 (S > 32) ──
            //   pass마다 실행이 된다.
            //   엔진 e의 서브트리 루트를 acc_A[e*passes_needed_per_segment + subtrees_collected] 에 쌓는다.
            //   stride를 passes_needed_per_segment(2의 거듭제곱)로 두면 acc에서도 (2i, 2i+1)
            //   평면 스캔이 엔진 경계를 넘지 않는다. tree_A와 같은 논리.
            for (int engine = 0; engine < NUM_ENGINES; engine++) {
                #pragma HLS PIPELINE II=1
                acc_A[((uint32_t)engine << passes_needed_per_segment_log2) + subtrees_collected] = tree_A[engine];
            }
            subtrees_collected++;
            // 0, 4, 8, 12
            // 1, 5, 9, 13 이렇게 뛰엄뛰엄 되어서 결국 앞에는 engine 0, 그다음은 engine 1 이런 크기 32짜리 subtree(CV)가 엔진별로 모여서 저장

            if (subtrees_collected == passes_needed_per_segment) {
                // 마지막 pass에서만 실행됨
                subtrees_collected = 0;

                // sub-stage 0 : acc_A -> acc_B  (엔진당 passes_needed_per_segment/2 노드)
                is_root = (passes_needed_per_segment == 2);
                merge_count   = (passes_needed_per_segment >= 2) ? (int)(passes_needed_per_segment >> 1) * NUM_ENGINES : 0;
                for (int i = 0; i < merge_count; i++) {
                    #pragma HLS PIPELINE II=1
                    cv_vec_t left = acc_A[2*i];
                    cv_vec_t right = acc_A[2*i+1];
                    cv_vec_t merged;
                    parent_cv(left, right, is_root ? ROOT : 0, merged);
                    acc_B[i] = merged;
                }

                // sub-stage 1 : acc_B -> acc_A
                is_root = (passes_needed_per_segment == 4);
                merge_count = (passes_needed_per_segment >= 4) ? (int)(passes_needed_per_segment >> 2) * NUM_ENGINES : 0;
                for (int i = 0; i < merge_count; i++) {
                    #pragma HLS PIPELINE II=1
                    cv_vec_t left = acc_B[2*i];
                    cv_vec_t right = acc_B[2*i+1];
                    cv_vec_t merged;
                    parent_cv(left, right, is_root ? ROOT : 0, merged);
                    acc_A[i] = merged;
                }

                // sub-stage 2 : acc_A -> acc_B
                is_root = (passes_needed_per_segment == 8);
                merge_count = (passes_needed_per_segment >= 8) ? (int)(passes_needed_per_segment >> 3) * NUM_ENGINES : 0;
                for (int i = 0; i < merge_count; i++) {
                    #pragma HLS PIPELINE II=1
                    cv_vec_t left = acc_A[2*i];
                    cv_vec_t right = acc_A[2*i+1];
                    cv_vec_t merged;
                    parent_cv(left, right, is_root ? ROOT : 0, merged);
                    acc_B[i] = merged;
                }

                // sub-stage 3 : acc_B -> acc_A
                is_root = (passes_needed_per_segment == 16);
                merge_count = (passes_needed_per_segment >= 16) ? (int)(passes_needed_per_segment >> 4) * NUM_ENGINES : 0;
                for (int i = 0; i < merge_count; i++) {
                    #pragma HLS PIPELINE II=1
                    cv_vec_t left = acc_B[2*i];
                    cv_vec_t right = acc_B[2*i+1];
                    cv_vec_t merged; 
                    parent_cv(left, right, is_root ? ROOT : 0, merged);
                    acc_A[i] = merged;
                }

                // sub-stage 4 : acc_A -> acc_B  (MAX_SUB=32의 마지막 단)
                is_root = (passes_needed_per_segment == 32);
                merge_count = (passes_needed_per_segment >= 32) ? (int)(passes_needed_per_segment >> 5) * NUM_ENGINES : 0;
                for (int i = 0; i < merge_count; i++) {
                    #pragma HLS PIPELINE II=1
                    cv_vec_t left = acc_A[2*i];
                    cv_vec_t right = acc_A[2*i+1];
                    cv_vec_t merged;
                    parent_cv(left, right, is_root ? ROOT : 0, merged);
                    acc_B[i] = merged;
                }

                // ── hash write(S > 32) : 엔진당 세그먼트 1개씩, 총 4개 ──
                const bool root_in_acc_A = ((passes_needed_per_segment_log2 & 1) == 0);
                uint64_t out_addr = segments_emitted;
                for (int engine = 0; engine < NUM_ENGINES; engine++) {
                    #pragma HLS PIPELINE II=1
                    ext_out[out_addr] = root_in_acc_A ? acc_A[engine] : acc_B[engine];
                    out_addr += segments_per_engine;
                    // burst로 처리가 안됨.
                }
                segments_emitted++;
            }
        }
    }
    // 이전 버전
    // for (uint64_t p = 0; p < num_passes; p++) {
    //     #pragma HLS LOOP_TRIPCOUNT min=1 max=2

    //     cv_vec_t buf_A[CHUNKS_PER_PASS];      // 128
    //     cv_vec_t buf_B[CHUNKS_PER_PASS / 2];  // 64
    //     // BRAM의 포트가 2개인데, 압축하다보면 write와 read 2개 동시에 하는 경우가 있기에 이렇게 쪼갬

    //     for (int i = 0; i < CHUNKS_PER_PASS; i++) {
    //         #pragma HLS PIPELINE II=1
            
    //         if (i < CHUNKS_PER_ENGINE) {
    //             buf_A[i] = in_cv_fifo_0.read();
    //         } else if (i < 2 * CHUNKS_PER_ENGINE) {
    //             buf_A[i] = in_cv_fifo_1.read();
    //         } else if (i < 3 * CHUNKS_PER_ENGINE) {
    //             buf_A[i] = in_cv_fifo_2.read();
    //         } else {
    //             buf_A[i] = in_cv_fifo_3.read();
    //         }
    //     }

    //     // ── Stage 0 (128 -> 64), result node size: 2 chunks, A read / B write ──
    //     bool is_root = (chunks_per_segment == 2);
    //     int  n = (2 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 2) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=64 max=64
    //         cv_vec_t merged;
    //         parent_cv(buf_A[2*i], buf_A[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_B[i] = merged;
    //     }

    //     // ── Stage 1 (64 -> 32), result node size: 4 chunks , B read / A write ──
    //     is_root = (chunks_per_segment == 4);
    //     n = (4 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 4) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=32 max=32
    //         cv_vec_t merged;
    //         parent_cv(buf_B[2*i], buf_B[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_A[i] = merged;
    //     }

    //     // ── Stage 2 (32 -> 16), result node size: 8 chunks, A read / B write ──
    //     is_root = (chunks_per_segment == 8);
    //     n = (8 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 8) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=16 max=16
    //         cv_vec_t merged;
    //         parent_cv(buf_A[2*i], buf_A[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_B[i] = merged;
    //     }
 
    //     // ── Stage 3 (16 -> 8), result node size: 16 chunks, B read / A write ──
    //     is_root = (chunks_per_segment == 16);
    //     n = (16 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 16) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=8 max=8
    //         cv_vec_t merged;
    //         parent_cv(buf_B[2*i], buf_B[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_A[i] = merged;
    //     }

    //     // ── Stage 4 (8 -> 4), result node size: 32 chunks, A read / B write ──
    //     is_root = (chunks_per_segment == 32);
    //     n = (32 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 32) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=4 max=4
    //         cv_vec_t merged;
    //         parent_cv(buf_A[2*i], buf_A[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_B[i] = merged;
    //     }

    //     // ── Stage 5 (4 -> 2), result node size: 64 chunks, B read / A write ──
    //     is_root = (chunks_per_segment == 64);
    //     n = (64 <= chunks_per_segment) ? (CHUNKS_PER_PASS / 64) : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=2 max=2
    //         cv_vec_t merged;
    //         parent_cv(buf_B[2*i], buf_B[2*i+1], is_root ? ROOT : 0, merged);
    //         if (is_root) out_cv_fifo.write(merged);
    //         else         buf_A[i] = merged;
    //     }

    //     // ── Stage 6 (2 -> 1), result node size: 128 chunks, A read / 출력 ──  [v2 신규]
    //     //   S == 128 이면 이 노드가 세그먼트 루트 → ROOT
    //     //   S >  128 이면 flags=0인 128청크짜리 서브트리 루트를 뱉는다.
    //     //              (pass간 병합은 cv_pe_final 몫 — 2단계에서 구현)
    //     // 지금은 segment가 128KiB 까지 인것만 구현한 것
    //     is_root = (chunks_per_segment == 128);
    //     n = (128 <= chunks_per_segment) ? 1 : 0;
    //     for (int i = 0; i < n; i++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=1 max=1
    //         cv_vec_t merged;
    //         parent_cv(buf_A[2*i], buf_A[2*i+1], is_root ? ROOT : 0, merged);
    //         out_cv_fifo.write(merged);
    //     }
    // }
}

// // ══════════════════════════════════════════════════════════════════════════
// //  cv_pe_final 
// //
// //  S <= 128 이면 cv_pe가 이미 ROOT까지 찍어서 뱉으므로, 여기서는 그냥 받아 쓰기만 한다.
// //  기존 CV 스택(가변 길이 while 루프)은 전부 불필요.
// //
// //  [2단계 확장 지점] S > 128 지원 시
// //    passes_per_segment = S/128 개의 서브트리 루트를 모아 완전이진 병합 후 ROOT 부여.
// // ══════════════════════════════════════════════════════════════════════════
// void cv_final_pe(hls::stream<cv_vec_t>& in_cv_fifo, 
//                  uint64_t num_passes, 
//                  uint32_t chunks_per_segment,
//                  cv_vec_t* ext_out) {

//     // chunks_per_segment(S)는 항상 2의 거듭제곱(4~128)이므로 나눗셈 대신
//     // log2(S)를 우선순위 인코더로 구해 shift로 대체한다. (DSP divider 회피, LUT만 사용)
//     // const uint32_t hashes_per_pass = (uint32_t)CHUNKS_PER_PASS / chunks_per_segment;
//     uint32_t seg_log2 = 0;
//     for (int b = 0; b < 8; b++) {
//         #pragma HLS UNROLL
//         if (chunks_per_segment & (1u << b)) seg_log2 = b;
//     }
//     const uint32_t hashes_per_pass = (uint32_t)CHUNKS_PER_PASS >> seg_log2;



//     uint64_t k = 0;
//     for (uint64_t p = 0; p < num_passes; p++) {
//         #pragma HLS LOOP_TRIPCOUNT min=1 max=2
//         #pragma HLS LOOP_FLATTEN off
//         // flatten을 켜두면 outer(p)/inner(j) 인덱스를 합치는 과정에서
//         // hashes_per_pass(런타임 값)와의 곱셈이 생겨 DSP를 다시 잡아먹는다.
//         for (uint32_t j = 0; j < hashes_per_pass; j++) {
//             #pragma HLS PIPELINE II=1
//             #pragma HLS LOOP_TRIPCOUNT min=1 max=32
//             // chunks_per_segment가 4일때, 가장 많은 해쉬가 나오는데 그게 32개
//             ext_out[k] = in_cv_fifo.read();
//             k++;
//         }
//     }
// }

void blake3_accelerator(const block_vec_t* host_data_in_0,
            const block_vec_t* host_data_in_1,
            const block_vec_t* host_data_in_2,
            const block_vec_t* host_data_in_3,
            const block_vec_t* host_data_in_4,
            const block_vec_t* host_data_in_5,
            const block_vec_t* host_data_in_6,
            const block_vec_t* host_data_in_7,
            cv_vec_t* host_hash_out,
            uint64_t  num_chunks,
            uint32_t  chunks_per_segment) {
    #pragma HLS INTERFACE m_axi port=host_data_in_0 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=host_data_in_1 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=host_data_in_2 bundle=gmem2
    #pragma HLS INTERFACE m_axi port=host_data_in_3 bundle=gmem3
    #pragma HLS INTERFACE m_axi port=host_data_in_4 bundle=gmem4
    #pragma HLS INTERFACE m_axi port=host_data_in_5 bundle=gmem5
    #pragma HLS INTERFACE m_axi port=host_data_in_6 bundle=gmem6
    #pragma HLS INTERFACE m_axi port=host_data_in_7 bundle=gmem7
    #pragma HLS INTERFACE m_axi port=host_hash_out  bundle=gmem8

    uint64_t num_passes = num_chunks / CHUNKS_PER_PASS;
    uint64_t chunks_per_engine_total = num_chunks / NUM_ENGINES;

    hls::stream<internal_pkt> disp_to_comp_0("disp_to_comp_0");
    hls::stream<internal_pkt> disp_to_comp_1("disp_to_comp_1");
    hls::stream<internal_pkt> disp_to_comp_2("disp_to_comp_2");
    hls::stream<internal_pkt> disp_to_comp_3("disp_to_comp_3");
    hls::stream<internal_pkt> disp_to_comp_4("disp_to_comp_4");
    hls::stream<internal_pkt> disp_to_comp_5("disp_to_comp_5");
    hls::stream<internal_pkt> disp_to_comp_6("disp_to_comp_6");
    hls::stream<internal_pkt> disp_to_comp_7("disp_to_comp_7");

    #pragma HLS STREAM variable=disp_to_comp_0 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_1 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_2 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_3 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_4 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_5 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_6 depth=FIFO_DEPTH_D2C
    #pragma HLS STREAM variable=disp_to_comp_7 depth=FIFO_DEPTH_D2C

    hls::stream<cv_pair_t> comp_to_cv_0("comp_to_cv_0");
    hls::stream<cv_pair_t> comp_to_cv_1("comp_to_cv_1");
    hls::stream<cv_pair_t> comp_to_cv_2("comp_to_cv_2");
    hls::stream<cv_pair_t> comp_to_cv_3("comp_to_cv_3");
    hls::stream<cv_pair_t> comp_to_cv_4("comp_to_cv_4");
    hls::stream<cv_pair_t> comp_to_cv_5("comp_to_cv_5");
    hls::stream<cv_pair_t> comp_to_cv_6("comp_to_cv_6");
    hls::stream<cv_pair_t> comp_to_cv_7("comp_to_cv_7");

    #pragma HLS STREAM variable=comp_to_cv_0 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_1 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_2 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_3 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_4 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_5 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_6 depth=FIFO_DEPTH_C2CV
    #pragma HLS STREAM variable=comp_to_cv_7 depth=FIFO_DEPTH_C2CV

    #pragma HLS DATAFLOW
    dispatcher_pe(host_data_in_0, disp_to_comp_0, 0 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_1, disp_to_comp_1, 1 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_2, disp_to_comp_2, 2 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_3, disp_to_comp_3, 3 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_4, disp_to_comp_4, 4 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_5, disp_to_comp_5, 5 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_6, disp_to_comp_6, 6 * chunks_per_engine_total, num_passes);
    dispatcher_pe(host_data_in_7, disp_to_comp_7, 7 * chunks_per_engine_total, num_passes);
    // 여기서 2의 배수 연산을 하지 않는 곱셈에 대해서 1사이클씩 밀린다. 1사이클을 곱셈에 사용해서.
    // 나중에 chunks_per_engine_total을 없애버릴수도 있다. 일단은 놔두기

    comp_pe(disp_to_comp_0, comp_to_cv_0, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_1, comp_to_cv_1, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_2, comp_to_cv_2, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_3, comp_to_cv_3, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_4, comp_to_cv_4, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_5, comp_to_cv_5, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_6, comp_to_cv_6, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_7, comp_to_cv_7, num_passes, chunks_per_segment);

    cv_pe(comp_to_cv_0, comp_to_cv_1, comp_to_cv_2, comp_to_cv_3,
          comp_to_cv_4, comp_to_cv_5, comp_to_cv_6, comp_to_cv_7,
          host_hash_out, num_chunks, num_passes, chunks_per_segment);
} 

// void blake3_accelerator(const block_vec_t* host_data_in_0,
//             const block_vec_t* host_data_in_1,
//             const block_vec_t* host_data_in_2,
//             const block_vec_t* host_data_in_3,
//             cv_vec_t* host_hash_out,
//             uint64_t  num_chunks,
//             uint32_t  chunks_per_segment) {
//     #pragma HLS INTERFACE m_axi port=host_data_in_0 bundle=gmem0
//     #pragma HLS INTERFACE m_axi port=host_data_in_1 bundle=gmem1
//     #pragma HLS INTERFACE m_axi port=host_data_in_2 bundle=gmem2
//     #pragma HLS INTERFACE m_axi port=host_data_in_3 bundle=gmem3
//     #pragma HLS INTERFACE m_axi port=host_hash_out bundle=gmem4

//     uint64_t num_passes = num_chunks / CHUNKS_PER_PASS;

//     hls::stream<internal_pkt> disp_to_comp_0("disp_to_comp_0");
//     hls::stream<internal_pkt> disp_to_comp_1("disp_to_comp_1");
//     hls::stream<internal_pkt> disp_to_comp_2("disp_to_comp_2");
//     hls::stream<internal_pkt> disp_to_comp_3("disp_to_comp_3");

//     #pragma HLS STREAM variable=disp_to_comp_0 depth=FIFO_DEPTH_D2C
//     #pragma HLS STREAM variable=disp_to_comp_1 depth=FIFO_DEPTH_D2C
//     #pragma HLS STREAM variable=disp_to_comp_2 depth=FIFO_DEPTH_D2C
//     #pragma HLS STREAM variable=disp_to_comp_3 depth=FIFO_DEPTH_D2C

//     hls::stream<cv_vec_t> comp_to_cv_0("comp_to_cv_0");
//     hls::stream<cv_vec_t> comp_to_cv_1("comp_to_cv_1");
//     hls::stream<cv_vec_t> comp_to_cv_2("comp_to_cv_2");
//     hls::stream<cv_vec_t> comp_to_cv_3("comp_to_cv_3");

//     #pragma HLS STREAM variable=comp_to_cv_0 depth=FIFO_DEPTH_C2CV
//     #pragma HLS STREAM variable=comp_to_cv_1 depth=FIFO_DEPTH_C2CV
//     #pragma HLS STREAM variable=comp_to_cv_2 depth=FIFO_DEPTH_C2CV
//     #pragma HLS STREAM variable=comp_to_cv_3 depth=FIFO_DEPTH_C2CV

//     hls::stream<cv_vec_t> cv_pe_to_cv_final("cv_pe_to_cv_final");
//     #pragma HLS STREAM variable=cv_pe_to_cv_final depth=FIFO_DEPTH_CV2FINAL

//     #pragma HLS DATAFLOW
//     dispatcher_pe(host_data_in_0, disp_to_comp_0, 0 * CHUNKS_PER_ENGINE, num_passes);
//     dispatcher_pe(host_data_in_1, disp_to_comp_1, 1 * CHUNKS_PER_ENGINE, num_passes);
//     dispatcher_pe(host_data_in_2, disp_to_comp_2, 2 * CHUNKS_PER_ENGINE, num_passes);
//     dispatcher_pe(host_data_in_3, disp_to_comp_3, 3 * CHUNKS_PER_ENGINE, num_passes);

//     comp_pe(disp_to_comp_0, comp_to_cv_0, num_passes, chunks_per_segment);
//     comp_pe(disp_to_comp_1, comp_to_cv_1, num_passes, chunks_per_segment);
//     comp_pe(disp_to_comp_2, comp_to_cv_2, num_passes, chunks_per_segment);
//     comp_pe(disp_to_comp_3, comp_to_cv_3, num_passes, chunks_per_segment);

//     cv_pe(comp_to_cv_0, comp_to_cv_1, comp_to_cv_2, comp_to_cv_3,
//           cv_pe_to_cv_final, num_passes, chunks_per_segment);

//     cv_final_pe(cv_pe_to_cv_final, num_passes, chunks_per_segment, host_hash_out);
// } 
