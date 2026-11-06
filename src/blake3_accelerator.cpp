#include <stdint.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define PARENT (1 << 2)
#define ROOT (1 << 3)

// ─── HW 아키텍처 및 파이프라인 상수 ──────────────────────────────────────
const int NUM_ENGINES                = 8;  // 총 PE 쌍 개수(dispatcher-comp) 
const int CHUNKS_PER_ENGINE_PER_PASS = 32; // 엔진당 한 pass에 처리하는 청크 수
const int BLOCKS_PER_CHUNK           = 16; // 청크 당 블락 수 

const int CHUNKS_PER_PASS            = NUM_ENGINES * CHUNKS_PER_ENGINE_PER_PASS;      // 256 청크
const int BLOCKS_PER_PE              = CHUNKS_PER_ENGINE_PER_PASS * BLOCKS_PER_CHUNK; // 512 블록

// ─── relay 트리 상수 ──────────────────────────────
// comp_pe가 pass당 내보내는 cv_pair 개수
const int PAIRS_PER_ENGINE_PER_PASS = CHUNKS_PER_ENGINE_PER_PASS / 2;         // 16

// 4:1 relay가 pass당 중계하는 개수.
const int PAIRS_PER_RELAY4_PER_PASS = 4 * PAIRS_PER_ENGINE_PER_PASS; // 64

// ─── FIFO depth 상수 ──────────────────────────────
const int FIFO_DEPTH_D2C  = 4;   // disp_to_comp_0..7    : dispatcher_pe -> comp_pe
const int FIFO_DEPTH_C2R  = 16;  // comp_to_relay_0..7   : comp_pe       -> relay_4to1
const int FIFO_DEPTH_R2R  = 16;  // relay_a4/b4_to_r2    : relay_4to1    -> relay_2to1
const int FIFO_DEPTH_R2CV = 16;  // relay_r2_to_cv       : relay_2to1    -> cv_pe
// relay4의 두 번째 것이 relay2에 write할 때 16에서 막히지만 상관없다. relay 간 FIFO라서.
// 결국 cv_pe가 read stall 안 걸리고, comp_pe가 write stall 안 걸리기만 하면 된다. relay는 노상관.

typedef hls::vector<uint32_t, 16> block_vec_t; // 각 원소가 한 워드 -> 한 개의 block
typedef hls::vector<uint32_t, 8>  cv_vec_t; // 벡터 전체가 한 CV

struct internal_pkt {
    block_vec_t data;
    uint64_t    engine_local_chunk_idx; // 엔진 안 청크 번호
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
    /// 이거 없으면 parent_cv가 unroll때 복제되어버려서 사이클 10개 아끼려다 리소스 몇배 낭비

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

void dispatcher_pe(const block_vec_t* ext_mem, hls::stream<internal_pkt>& out_fifo, uint64_t num_passes) {

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
                
                int block_idx            = i / CHUNKS_PER_ENGINE_PER_PASS;
                int pass_local_chunk_idx = i % CHUNKS_PER_ENGINE_PER_PASS;
                int local_addr = (pass_local_chunk_idx * BLOCKS_PER_CHUNK) + block_idx;
                
                internal_pkt pkt;
                if(rd_idx==0){
                    pkt.data = buffer[0][local_addr];
                }
                else{
                    pkt.data = buffer[1][local_addr];
                }
                // read만 분기하는 것이 자원소모량, negative slack 측면에서 제일 우수

                // 엔진 안 청크 번호. 세그먼트 안 counter 로의 변환(마스킹)은 comp_pe 에서 한다
                pkt.engine_local_chunk_idx = prev_p * CHUNKS_PER_ENGINE_PER_PASS + pass_local_chunk_idx;
                
                pkt.flags = 0;
                if (block_idx == 0)  pkt.flags |= CHUNK_START;
                if (block_idx == 15) pkt.flags |= CHUNK_END;
                
                out_fifo.write(pkt);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  comp_pe : 블록을 압축해 청크 CV 쌍을 내보낸다
//
//  BLAKE3는 세그먼트가 바뀌면 chunk counter가 0부터 다시 시작해야 한다.
//    엔진 안 청크번호: 0 1 2 3 | 4 5 6 7 | 8 9 10 11 ...
//    S=4일 때 counter: 0 1 2 3 | 0 1 2 3 | 0 1  2  3 ...   ← 리셋
//  S가 2의 거듭제곱이므로 마스크 한 번이면 끝
//
//  주의: cv_mem 슬롯 주소(pass_local_chunk_idx)는 엔진 안 청크번호 % 32 그대로 써야 한다.
//        세그먼트 counter와 혼동하지 말 것.
// ══════════════════════════════════════════════════════════════════════════ 
void comp_pe(hls::stream<internal_pkt>& in_fifo, hls::stream<cv_pair_t>& out_cv_fifo,  
    uint64_t num_passes, uint32_t chunks_per_segment) {
    // latency = 32*16+31 = 543
    // interval = 512 --> 512개 루프 끝나면 바로 다음꺼 먹을 수 있음. 32개 청크 무한 섭취 가능
    // 0 사이클에서 처음 comp_pe에 input으로 블락 들어간다할때, 511 사이클에서 처음 FIFO에 write
    // 32개 청크 무한 섭취 가능하지만, 첫 32개 청크 다 넣고, 두번째 32개 청크 들어간 후 511 사이클지나도
    // cv_pe에서 압축이 안되면 문제가 된다 --> FIFO를 충분히 깊게 or cv_pe가 무조건 511 사이클보다 적도록
    cv_vec_t cv_mem[CHUNKS_PER_ENGINE_PER_PASS];
    // array for temporary cv

    cv_vec_t pend; // pair로 만들어서 보내기 위함.

    const uint64_t counter_mask = (uint64_t)chunks_per_segment - 1; // S가 2^k라서 마스크로 대체

    for (uint64_t p = 0; p < num_passes; p++) {
        for (int i = 0; i < BLOCKS_PER_PE; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=cv_mem type=inter false
            
            internal_pkt pkt = in_fifo.read();
            int pass_local_chunk_idx = pkt.engine_local_chunk_idx % CHUNKS_PER_ENGINE_PER_PASS; // 0~31
            
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
                    cv_in[j] = cv_mem[pass_local_chunk_idx][j];
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

            // 엔진 안 청크번호 -> 세그먼트 내부에서 몇 번째 청크냐(=counter)
            // 하위 log2(S) 비트 추출
            uint64_t seg_counter = pkt.engine_local_chunk_idx & counter_mask;
            compress(cv_in, msg, seg_counter, 64, pkt.flags, res16);

            cv_vec_t out_cv;

            for (int j = 0; j < 8; j++){
                #pragma HLS UNROLL
                out_cv[j] = res16[j];
            }

            if (pkt.flags & CHUNK_END) {
                // 짝수 청크면 보류, 홀수 청크면 pair로 write
                if ((pkt.engine_local_chunk_idx & 1) == 0) {
                    pend = out_cv;
                } else {
                    cv_pair_t pr;
                    pr.l = pend;
                    pr.r = out_cv;
                    out_cv_fifo.write(pr);
                }
            } else {
                cv_mem[pass_local_chunk_idx] = out_cv;
            }
        }
    }
}

void relay_4to1(hls::stream<cv_pair_t>& in_0,
                hls::stream<cv_pair_t>& in_1,
                hls::stream<cv_pair_t>& in_2,
                hls::stream<cv_pair_t>& in_3, 
                hls::stream<cv_pair_t>& out_fifo,
                uint64_t num_passes) {

    for (uint64_t p = 0; p < num_passes; p++) {
        #pragma HLS LOOP_FLATTEN off
        for (int i = 0; i < PAIRS_PER_RELAY4_PER_PASS; i++) {
            #pragma HLS PIPELINE II=1
            int sel = i / PAIRS_PER_ENGINE_PER_PASS; // 16개씩 끊어서 0,1,2,3

            cv_pair_t pr;
            if      (sel == 0) pr = in_0.read();
            else if (sel == 1) pr = in_1.read();
            else if (sel == 2) pr = in_2.read();
            else               pr = in_3.read(); 
            out_fifo.write(pr);
        }
    }
}

void relay_2to1(hls::stream<cv_pair_t>& in_0,
                hls::stream<cv_pair_t>& in_1,
                hls::stream<cv_pair_t>& out_fifo,
                uint64_t num_passes) {

    for (uint64_t p = 0; p < num_passes; p++) {
        #pragma HLS LOOP_FLATTEN off
        for (int i = 0; i < 2 * PAIRS_PER_RELAY4_PER_PASS; i++) {
            #pragma HLS PIPELINE II=1
            cv_pair_t pr;
            if (i < PAIRS_PER_RELAY4_PER_PASS) pr = in_0.read(); // 엔진 0~3
            else                               pr = in_1.read(); // 엔진 4~7
            out_fifo.write(pr);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  cv_pe : 청크 CV 쌍을 BLAKE3 트리로 합쳐 세그먼트 해시를 ext_out 에 쓴다
//
//  레벨 L 노드는 청크 2^(L+1) 개를 덮는다 (L = 0..9). 세그먼트 루트는 레벨 log2(S) - 1.
//
//  [규칙 하나] 레벨 L 노드는, 그 노드가 덮는 마지막 pass 가 q 이면 step q+L 에 병합한다.
//    표의 숫자가 step 이다. 같은 숫자끼리 한 step 에 돌고, 이 대각선이 wavefront 다.
//
//            q=0  q=1  q=2  q=3
//       L0    0    1    2    3
//       L1    1    2    3    4
//       L2    2    3    4    5
//
//    자식은 항상 지난 step 이전에 끝나 있으므로 한 step 안의 병합끼리는 서로 기다리지 않는다.
//
//  [순서] 한 step 은 위 레벨부터 돈다: [L9 .. L5][L4][L3][L2][L1][L0]
//    위 레벨이 자식 칸의 옛 값을 읽은 뒤에 아래 레벨이 그 칸을 덮어쓴다 (WAR 안전).
//    안쪽 루프가 step 마다 끝나서 지난 step 의 쓰기는 전부 완료돼 있다 (RAW 안전).
//    DEPENDENCE false 는 이 두 가지에서 나온다. 레벨 순서를 바꾸거나 step 을 이어 붙이면 깨진다.
//
//  [해시 쓰기] L0 는 자식을 FIFO 에서 받아서 tree 읽기 포트가 논다.
//    그 포트로 지난 step 의 루트를 읽어 ext_out 에 쓴다. 그래서 루트 칸은 step 홀짝으로 두 벌이다.
//    입력이 끝난 step 은 L0 자리에 해시 쓰기만 남는다.
//
//  [tree 배치] 자식 쌍은 항상 붙어 있어서 base + 2*node, base + 2*node + 1 로 한 번에 읽는다.
//    레벨 0~3 : tree[LV_BASE[L] + node]                  node = pass 안 노드 번호
//    레벨 4~8 : tree[LV_BASE[L] + 2e + (오른쪽 자식?)]   e = 엔진. step 마다 엔진당 노드 1개
//    루트     : tree[HASH_BASE + HASH_SLOTS*(p&1) + node]
// ══════════════════════════════════════════════════════════════════════════
void cv_pe(hls::stream<cv_pair_t>& in_cv_fifo,
           cv_vec_t* ext_out,
           uint64_t num_chunks,
           uint64_t num_passes,
           uint32_t chunks_per_segment) {

    const int CV_LEVELS  = 10;     // 레벨 0..9 (S=1024 의 루트가 레벨 9)
    const int PASS_LV    = 4;      // 엔진의 한 pass(32 청크)를 덮는 레벨. 이보다 위 레벨은 여러 pass 에 걸친다
    const int HASH_BASE  = 320;    // 해시 칸 [2][HASH_SLOTS]
    const int HASH_SLOTS = 64;     // 한 step 의 해시 최대 개수 (S=4)
    const int TREE_SLOTS = 456;    // HASH_BASE + 2 * HASH_SLOTS + 여유
    static const int LV_BASE[CV_LEVELS] = {0, 128, 192, 224, 240, 256, 272, 288, 304, 320};   // [9] 는 안 씀 (레벨 9 는 항상 루트)

    uint32_t seg_log2 = 0;
    for (int b = 0; b < 11; b++) {
        #pragma HLS UNROLL
        if (chunks_per_segment & (1u << b)) seg_log2 = b;
    }
    const uint32_t segs_per_engine = (uint32_t)((num_chunks >> seg_log2) / NUM_ENGINES);
    const int      root_lv   = (int)seg_log2 - 1;                                     // 세그먼트 루트 레벨
    const uint32_t hash_log2 = (seg_log2 <= PASS_LV + 1) ? PASS_LV + 1 - seg_log2 : 0;   // 루트가 나오는 step 의 엔진당 해시 수 = 2^hash_log2

    uint32_t engine_base[NUM_ENGINES];   // 엔진 e 의 해시 구역 시작 = e * segs_per_engine (block 배치)
    #pragma HLS ARRAY_PARTITION variable=engine_base complete
    for (int e = 0; e < NUM_ENGINES; e++) {
        #pragma HLS UNROLL
        engine_base[e] = e * segs_per_engine;
    }

    cv_vec_t tree[TREE_SLOTS];
    #pragma HLS BIND_STORAGE variable=tree type=RAM_1WNR impl=BRAM

    bool     hash_ready           = false;   // 지난 step 에 루트가 나왔나
    uint32_t engine_local_seg_idx = 0;       // 다음에 쓸 해시의 엔진 안 세그먼트 번호. 해시는 순서대로 나온다

    for (uint64_t p = 0; p < num_passes + root_lv + 1; p++) {   // p = step 번호. +1 은 마지막 루트를 쓰는 step
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32

        // ── 1. 계획: 규칙에 맞는 레벨을 찾아 레벨별 iteration 구간을 정한다 ──
        int  lv_start[CV_LEVELS];   // 레벨 L 구간의 시작 iteration
        bool is_right[CV_LEVELS];   // 레벨 L 이 이번에 만드는 노드가 부모의 오른쪽 자식인가
        #pragma HLS ARRAY_PARTITION variable=lv_start complete
        #pragma HLS ARRAY_PARTITION variable=is_right complete
        int  num_merge = 0;
        bool has_leaf  = false;     // 이번 step 에 FIFO 입력이 있나
        bool has_root  = false;     // 이번 step 에 루트가 나오나
        for (int L = CV_LEVELS - 1; L >= 0; L--) {
            #pragma HLS UNROLL
            const int  end_pass  = (int)p - L + 1;                             // 규칙에서 q + 1
            const int  span_log2 = (L > PASS_LV) ? L - PASS_LV : 0;            // 노드 하나가 덮는 pass 수 = 2^span_log2
            const bool in_range  = (end_pass >= 1) && (end_pass <= (int)num_passes);
            const bool span_done = (end_pass & ((1 << span_log2) - 1)) == 0;   // 노드가 덮는 마지막 pass 인가
            const bool active    = (L <= root_lv) && in_range && span_done;

            lv_start[L] = num_merge;
            is_right[L] = ((end_pass >> span_log2) & 1) == 0;                  // 노드 번호 (end_pass >> span_log2) - 1 이 홀수
            if (active) num_merge += (L <= PASS_LV) ? ((CHUNKS_PER_PASS / 2) >> L) : NUM_ENGINES;
            if (active && L == 0)       has_leaf = true;
            if (active && L == root_lv) has_root = true;
        }
        const int num_hash = hash_ready ? (NUM_ENGINES << hash_log2) : 0;
        const int num_iter = has_leaf ? num_merge : num_merge + num_hash;   // 입력이 끝난 step 은 L0 자리에 해시 쓰기만
        const int hash_rd_base = HASH_BASE + HASH_SLOTS * ((p - 1) & 1);    // 지난 step 의 루트 칸
        const int hash_wr_base = HASH_BASE + HASH_SLOTS * (p & 1);          // 이번 step 의 루트 칸

        // ── 2. 실행: 이번 step 의 노드를 위 레벨부터 쉬지 않고 발행한다 ──
        for (int i = 0; i < num_iter; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=264
            #pragma HLS DEPENDENCE variable=tree type=inter false

            // i 번째 iteration 이 맡은 노드 = 레벨 L 의 node 번
            int L = CV_LEVELS - 1;
            for (int k = CV_LEVELS - 1; k >= 0; k--) {
                #pragma HLS UNROLL
                if (i >= lv_start[k]) L = k;
            }
            const int  node    = i - lv_start[L];
            const bool is_leaf = (L == 0) && has_leaf;
            const bool is_hash = (L == 0) && (node < num_hash);
            const bool is_root = (L == root_lv);

            // tree 읽기: 병합이면 자식 쌍, L0 이면 지난 step 의 해시
            int rd_addr = 0;
            if (L != 0)       rd_addr = LV_BASE[L - 1] + 2 * node;
            else if (is_hash) rd_addr = hash_rd_base + node;
            const cv_vec_t tree_l = tree[rd_addr];
            const cv_vec_t tree_r = tree[rd_addr + 1];

            // FIFO 읽기 뒤에 해시 쓰기를 둔다. 순서를 바꾸면 FF 가 약 1K 늘어난다 (csynth 확인)
            cv_vec_t left = tree_l, right = tree_r, merged;
            if (is_leaf) {
                const cv_pair_t in_pair = in_cv_fifo.read();
                left  = in_pair.l;
                right = in_pair.r;
            }
            const int hash_engine = node >> hash_log2;               // 이 해시를 만든 엔진
            const int hash_in_eng = node & ((1 << hash_log2) - 1);   // 그 엔진의 이번 step 해시 중 몇 번째
            if (is_hash) ext_out[engine_base[hash_engine] + engine_local_seg_idx + hash_in_eng] = tree_l;

            // 병합. parent_cv 호출은 이 한 곳뿐이어야 compress 코어가 1개만 만들어진다
            parent_cv(left, right, is_root ? ROOT : 0, merged);

            const int wr_addr = is_root       ? hash_wr_base + node
                              : (L < PASS_LV) ? LV_BASE[L] + node
                              :                 LV_BASE[L] + 2 * node + (int)is_right[L];
            if (L != 0 || is_leaf) tree[wr_addr] = merged;   // 해시만 쓰는 iteration 은 저장할 결과가 없다
        }

        // ── 3. 뒷정리: 이번 step 에 쓴 해시만큼 번호를 넘기고, 나온 루트는 다음 step 이 쓴다 ──
        if (hash_ready) engine_local_seg_idx += 1u << hash_log2;
        hash_ready = has_root;
    }
}

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

    hls::stream<cv_pair_t> comp_to_relay_0("comp_to_relay_0");
    hls::stream<cv_pair_t> comp_to_relay_1("comp_to_relay_1");
    hls::stream<cv_pair_t> comp_to_relay_2("comp_to_relay_2");
    hls::stream<cv_pair_t> comp_to_relay_3("comp_to_relay_3");
    hls::stream<cv_pair_t> comp_to_relay_4("comp_to_relay_4");
    hls::stream<cv_pair_t> comp_to_relay_5("comp_to_relay_5");
    hls::stream<cv_pair_t> comp_to_relay_6("comp_to_relay_6");
    hls::stream<cv_pair_t> comp_to_relay_7("comp_to_relay_7");

    #pragma HLS STREAM variable=comp_to_relay_0 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_1 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_2 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_3 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_4 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_5 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_6 depth=FIFO_DEPTH_C2R
    #pragma HLS STREAM variable=comp_to_relay_7 depth=FIFO_DEPTH_C2R

    hls::stream<cv_pair_t> relay_a4_to_r2("relay_a4_to_r2");
    hls::stream<cv_pair_t> relay_b4_to_r2("relay_b4_to_r2");
    hls::stream<cv_pair_t> relay_r2_to_cv("relay_r2_to_cv");

    #pragma HLS STREAM variable=relay_a4_to_r2  depth=FIFO_DEPTH_R2R
    #pragma HLS STREAM variable=relay_b4_to_r2  depth=FIFO_DEPTH_R2R
    #pragma HLS STREAM variable=relay_r2_to_cv depth=FIFO_DEPTH_R2CV

    #pragma HLS DATAFLOW
    dispatcher_pe(host_data_in_0, disp_to_comp_0, num_passes);
    dispatcher_pe(host_data_in_1, disp_to_comp_1, num_passes);
    dispatcher_pe(host_data_in_2, disp_to_comp_2, num_passes);
    dispatcher_pe(host_data_in_3, disp_to_comp_3, num_passes); 
    dispatcher_pe(host_data_in_4, disp_to_comp_4, num_passes);
    dispatcher_pe(host_data_in_5, disp_to_comp_5, num_passes);
    dispatcher_pe(host_data_in_6, disp_to_comp_6, num_passes);
    dispatcher_pe(host_data_in_7, disp_to_comp_7, num_passes);

    comp_pe(disp_to_comp_0, comp_to_relay_0, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_1, comp_to_relay_1, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_2, comp_to_relay_2, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_3, comp_to_relay_3, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_4, comp_to_relay_4, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_5, comp_to_relay_5, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_6, comp_to_relay_6, num_passes, chunks_per_segment);
    comp_pe(disp_to_comp_7, comp_to_relay_7, num_passes, chunks_per_segment);

    relay_4to1(comp_to_relay_0, comp_to_relay_1, comp_to_relay_2, comp_to_relay_3,
               relay_a4_to_r2, num_passes);
    relay_4to1(comp_to_relay_4, comp_to_relay_5, comp_to_relay_6, comp_to_relay_7,
               relay_b4_to_r2, num_passes);
    relay_2to1(relay_a4_to_r2, relay_b4_to_r2, relay_r2_to_cv, num_passes);

    cv_pe(relay_r2_to_cv, host_hash_out, num_chunks, num_passes, chunks_per_segment);
}