#include <stdint.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define CHUNK_START 1 << 0
#define CHUNK_END 1 << 1
#define PARENT 1 << 2
#define ROOT 1 << 3

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
    #pragma HLS INLINE OFF //for resource saving. not allowed to copy too many compress

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

void dispatcher_pe(const block_vec_t* ext_mem, hls::stream<internal_pkt>& out_fifo) {

    block_vec_t local_buffer[512];

    for (int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        local_buffer[i] = ext_mem[i];
    }

    // for interleaving dispatching
    for (int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        
        int block_idx = i / 32; 
        int chunk_idx = i % 32;
        
        int local_addr = (chunk_idx * 16) + block_idx;
        
        internal_pkt pkt;
        pkt.data = local_buffer[local_addr]; 
        pkt.chunk_idx = chunk_idx;
        
        pkt.flags = 0;
        if (block_idx == 0)  pkt.flags |= CHUNK_START;
        if (block_idx == 15) pkt.flags |= CHUNK_END;
        
        out_fifo.write(pkt);
    }
}

void comp_pe(hls::stream<internal_pkt>& in_fifo, hls::stream<cv_vec_t>& out_cv_fifo) {
    cv_vec_t cv_mem[32];
    // array for temporary cv

    for (int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS dependence variable=cv_mem type=inter false
        
        internal_pkt pkt = in_fifo.read();
        uint8_t local_chunk_idx = pkt.chunk_idx % 32;
        
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
void cv_pe(hls::stream<cv_vec_t>& in_cv_fifo, cv_vec_t* ext_out) {

    cv_vec_t buf[2][32];
    #pragma HLS ARRAY_PARTITION variable=buf complete

    for (int i = 0; i < 32; i++) {
        #pragma HLS PIPELINE II=1
        buf[0][i] = in_cv_fifo.read();
    }

    // 5 level tree merging(32chunks need 31 merging, which means 5 level tree)
    for (int level = 0; level < 5; level++) {
        #pragma HLS UNROLL // for level parallelism
        
        int src = level % 2;
        int dst = 1 - src;
        int merges = 16 >> level; 
        
        for (int i = 0; i < merges; i++) {
            #pragma HLS PIPELINE II=1
            
            uint32_t flag = (merges == 1) ? ROOT : 0;
            cv_vec_t parent_out; 
            
            parent_cv(buf[src][2*i], buf[src][2*i+1], flag, parent_out);
            
            buf[dst][i] = parent_out; 
        }
    }

    ext_out[0] = buf[1][0];
}

extern "C"{
void blake3(const block_vec_t* host_data_in, cv_vec_t* host_hash_out) {
    #pragma HLS INTERFACE m_axi port=host_data_in  depth=512 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=host_hash_out depth=1 bundle=gmem1

    hls::stream<internal_pkt> disp_to_comp("disp_to_comp"); // probably should specify the depth cuz of deadlock   
    hls::stream<cv_vec_t> comp_to_cv("comp_to_cv");
    #pragma HLS STREAM variable=comp_to_cv depth=32

    #pragma HLS DATAFLOW
    dispatcher_pe(host_data_in, disp_to_comp);
    comp_pe(disp_to_comp, comp_to_cv);
    cv_pe(comp_to_cv, host_hash_out);
}
}