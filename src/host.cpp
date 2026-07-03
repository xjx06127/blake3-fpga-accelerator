/**
* Copyright (C) 2019-2021 Xilinx, Inc
* SPDX-License-Identifier: MIT
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "cmdlineparser.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

// XRT includes

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define CHUNK_START (1 << 0)
#define CHUNK_END   (1 << 1)
#define PARENT      (1 << 2)
#define ROOT        (1 << 3)

const int NUM_ENGINES = 4;  // 총 PE 쌍 개수(dispatcher-comp)

static const uint32_t SW_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

static const size_t MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13,
    1, 11, 12, 5, 9, 14, 15, 8
};

static inline uint32_t sw_rotate_right(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static inline void sw_g(uint32_t state[16], size_t a, size_t b, size_t c, size_t d, uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = sw_rotate_right(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = sw_rotate_right(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = sw_rotate_right(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = sw_rotate_right(state[b] ^ state[c], 7);
}

static inline void sw_round_function(uint32_t state[16], uint32_t m[16]) {
    sw_g(state, 0, 4, 8, 12, m[0], m[1]);
    sw_g(state, 1, 5, 9, 13, m[2], m[3]);
    sw_g(state, 2, 6, 10, 14, m[4], m[5]);
    sw_g(state, 3, 7, 11, 15, m[6], m[7]);
    
    sw_g(state, 0, 5, 10, 15, m[8], m[9]);
    sw_g(state, 1, 6, 11, 12, m[10], m[11]);
    sw_g(state, 2, 7, 8, 13, m[12], m[13]);
    sw_g(state, 3, 4, 9, 14, m[14], m[15]);
}

static inline void sw_permute(uint32_t m[16]) {
    uint32_t permuted[16];
    for (size_t i = 0; i < 16; i++) {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    memcpy(m, permuted, sizeof(permuted));
}

static inline void sw_compress(const uint32_t chaining_value[8], const uint32_t block_words[16], 
                               uint64_t counter, uint32_t block_len, uint32_t flags, uint32_t out[16]) {
    uint32_t state[16] = {
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        SW_IV[0], SW_IV[1], SW_IV[2], SW_IV[3],
        (uint32_t)counter, (uint32_t)(counter >> 32),
        block_len, flags
    };
    
    uint32_t block[16];
    memcpy(block, block_words, sizeof(block));

    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block); sw_permute(block);
    sw_round_function(state, block);

    for (int i = 0; i < 8; i++) {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    memcpy(out, state, sizeof(state));
}

static inline void sw_parent_cv(const uint32_t left[8], const uint32_t right[8], uint32_t flags, uint32_t out_cv[8]) {
    uint32_t block_words[16];
    for (int i = 0; i < 8; i++) block_words[i] = left[i];
    for (int i = 0; i < 8; i++) block_words[i + 8] = right[i];

    uint32_t out16[16];
    sw_compress(SW_IV, block_words, 0, 64, PARENT | flags, out16);

    for (int i = 0; i < 8; i++) out_cv[i] = out16[i];
}

void sw_blake3_full_tree(const std::vector<std::vector<uint32_t>>& input_data, uint32_t final_out[8], uint32_t num_chunks) {
    uint32_t cv_stack[16][8]; // 256청크면 depth 8까지만 쓰이므로 16이면 아주 넉넉함. cv_stack[현재 스택에 있는 CV][그 CV의 one word(4B)]
    int cv_stack_len = 0;

    for (uint32_t c = 0; c < num_chunks; c++) {
        uint32_t current_cv[8];
        for (int i = 0; i < 8; i++) current_cv[i] = SW_IV[i];

        for (int b = 0; b < 16; b++) {
            uint32_t flags = 0;
            if (b == 0)  flags |= CHUNK_START;
            if (b == 15) flags |= CHUNK_END;

            uint32_t out16[16];
            sw_compress(current_cv, input_data[c * 16 + b].data(), c, 64, flags, out16);
            // flatten indexing은 유지하고, .data로 넘겨야할듯?
            for (int i = 0; i < 8; i++) current_cv[i] = out16[i];
        }
        
        uint32_t new_cv[8];
        for (int i = 0; i < 8; i++) new_cv[i] = current_cv[i];

        uint64_t total_chunks = c + 1; 

        while ((total_chunks & 1) == 0) {
            cv_stack_len--; 
            uint32_t left_child[8];
            for (int i = 0; i < 8; i++) left_child[i] = cv_stack[cv_stack_len][i];

            // 트리의 진짜 마지막 병합 순간에만 ROOT 부여. 마지막 청크고, cv_stack가 비어있을때.
            uint32_t flags = 0;
            if ((c == num_chunks - 1) && (cv_stack_len == 0)) {
                flags = ROOT;
            }
            
            sw_parent_cv(left_child, new_cv, flags, new_cv);
            
            total_chunks >>= 1;
        }
        
        for (int i = 0; i < 8; i++) cv_stack[cv_stack_len][i] = new_cv[i];
        cv_stack_len++;
    }

    // Finalize 과정 (공식 C 구현체의 blake3_hasher_finalize 역할)
    // 청크 개수가 128처럼 2의 거듭제곱이면 while문 스킵 (이미 스택 길이가 1이므로).
    // 청크 개수가 384처럼 2의 거듭제곱이 아니면, 스택에 남은 서브트리들을 병합
    while (cv_stack_len > 1) {
        cv_stack_len--;
        uint32_t right_child[8];
        for (int i = 0; i < 8; i++) right_child[i] = cv_stack[cv_stack_len][i];

        cv_stack_len--;
        uint32_t left_child[8];
        for (int i = 0; i < 8; i++) left_child[i] = cv_stack[cv_stack_len][i];

        // 마지막 두 서브트리 병합 시 ROOT 부여
        uint32_t flags = (cv_stack_len == 0) ? ROOT : 0;
        
        uint32_t merged_cv[8];
        sw_parent_cv(left_child, right_child, flags, merged_cv);
        
        for (int i = 0; i < 8; i++) cv_stack[cv_stack_len][i] = merged_cv[i];
        cv_stack_len++;
    }

    for (int i = 0; i < 8; i++) final_out[i] = cv_stack[0][i];
}

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "1");
    parser.addSwitch("--num_chunks", "-n", "number of chunks to process (multiple of 128)", std::to_string(32 * NUM_ENGINES));
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));
    uint32_t num_chunks = stoi(parser.value("num_chunks"));

    if (num_chunks % 128 != 0) {
        std::cerr << "Error: num_chunks must be a multiple of 128." << std::endl;
        return EXIT_FAILURE;
    }

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    uint32_t num_passes = num_chunks / 128;
    uint32_t blocks_per_pe = num_passes * 512; // 한 PE당 한 패스에 512개 블락 처리(32개 청크)
    uint32_t total_blocks = num_chunks * 16;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "blake3_accelerator");

    std::cout << "Allocate Buffer in Global Memory\n";
    size_t in_size_bytes = sizeof(uint32_t) * blocks_per_pe * 16; // 4B * 16 = one block size
    size_t out_size_bytes = sizeof(uint32_t) * 8; // 4B * 8

    auto in_bo_0 = xrt::bo(device, in_size_bytes, krnl.group_id(0));
    auto in_bo_1 = xrt::bo(device, in_size_bytes, krnl.group_id(1));
    auto in_bo_2 = xrt::bo(device, in_size_bytes, krnl.group_id(2));
    auto in_bo_3 = xrt::bo(device, in_size_bytes, krnl.group_id(3));
    auto out_bo  = xrt::bo(device, out_size_bytes, krnl.group_id(4));

    // Map the contents of the buffer object into host memory
    uint32_t* in_map_0 = in_bo_0.map<uint32_t*>();
    uint32_t* in_map_1 = in_bo_1.map<uint32_t*>();
    uint32_t* in_map_2 = in_bo_2.map<uint32_t*>();
    uint32_t* in_map_3 = in_bo_3.map<uint32_t*>();
    uint32_t* out_map  = out_bo.map<uint32_t*>();

    std::vector<std::vector<uint32_t>> sw_in(total_blocks, std::vector<uint32_t>(16));
    uint32_t sw_out[8];

    printf("[Host] Generating Data with HW/SW interleaving mapping...\n");

    for (uint32_t p = 0; p < num_passes; p++) {
        for (int e = 0; e < 4; e++) {
            for (int c = 0; c < 32; c++) {
                uint32_t global_chunk_idx = p * 128 + e * 32 + c; 
                
                for (int b = 0; b < 16; b++) {
                    uint32_t local_idx = p * 512 + c * 16 + b;
                    
                    for (int w = 0; w < 16; w++) {
                        uint32_t val = (global_chunk_idx * 16 + b + w) * 0x11223344;
                        sw_in[global_chunk_idx * 16 + b][w] = val; 

                        if (e == 0) in_map_0[local_idx * 16 + w] = val;
                        if (e == 1) in_map_1[local_idx * 16 + w] = val;
                        if (e == 2) in_map_2[local_idx * 16 + w] = val;
                        if (e == 3) in_map_3[local_idx * 16 + w] = val;
                    }
                }
            }
        }
    }

    // Synchronize buffer content with device side
    std::cout << "\n[Host -> Device] Transferring data to HBM...\n";
    auto h2d_start = std::chrono::steady_clock::now();
    in_bo_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto h2d_end = std::chrono::steady_clock::now();

    std::cout << "[Kernel] Running BLAKE3 Accelerator...\n";
    auto krnl_start = std::chrono::steady_clock::now();
    auto run = krnl(in_bo_0, in_bo_1, in_bo_2, in_bo_3, out_bo, num_chunks);
    run.wait();
    auto krnl_end = std::chrono::steady_clock::now();

    std::cout << "[Device -> Host] Retrieving Hash Output...\n\n";
    auto d2h_start = std::chrono::steady_clock::now();
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto d2h_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> h2d_time = h2d_end - h2d_start;
    std::chrono::duration<double> krnl_time = krnl_end - krnl_start;
    std::chrono::duration<double> d2h_time = d2h_end - d2h_start;
    std::cout << "H2D Transfer Time: " << h2d_time.count() << " s"<<std::endl;
    std::cout << "Kernel Exec: " << krnl_time.count() << " s" <<std::endl;
    std::cout << "D2H Transfer Time: " << d2h_time.count() << " s" <<std::endl;

    std::cout<<std::endl;

    std::cout << "[Host] Running Software Golden Reference...\n";
    
    sw_blake3_full_tree(sw_in, sw_out, num_chunks);

    int errors = 0;
    std::cout << "\n[Result Comparison (Root Hash), num_chunks: " << num_chunks << "]\n";
    for (int i = 0; i < 8; i++) {
        if (out_map[i] != sw_out[i]) {
            printf("MISMATCH [%d] HW: %08X, SW: %08X\n", i, out_map[i], sw_out[i]);
            errors++;
        } else {
            printf("MATCH    [%d] HW: %08X, SW: %08X\n", i, out_map[i], sw_out[i]);
        }
    }

    std::cout<<std::endl;

    if (errors == 0) {
        std::cout << ">>> TEST PASSED! HW Output perfectly matches SW Golden Reference.\n";
    } else {
        std::cout << ">>> TEST FAILED! Found " << errors << " mismatches.\n";
    }

    double port_data_gb = (double)in_size_bytes / 1e9; // GB 단위 변환 (10^9)
    double total_data_gb = port_data_gb * 4; // 4개의 포트 사용

    double effective_bw_per_port = port_data_gb / krnl_time.count(); // GB/s
    double total_effective_bw = total_data_gb / krnl_time.count();   // GB/s

    std::cout << "\n================ [ Performance Report ] ================\n";
    std::cout << "Data Size per Port : " << port_data_gb * 1000 << " MB\n";
    std::cout << "Total Data Size    : " << total_data_gb * 1000 << " MB\n";
    std::cout << "Effective BW (Port): " << effective_bw_per_port << " GB/s\n";
    std::cout << "Effective BW (Total): " << total_effective_bw << " GB/s\n";
    std::cout << "========================================================\n\n";

    return errors;
}
