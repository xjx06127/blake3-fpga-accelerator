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

void sw_blake3_full_tree(uint32_t input_data[512][16], uint32_t final_out[8]) {
    uint32_t cv_stack[5][8]; // depth 5 is enough for the 32 chunks
    int cv_stack_len = 0;

    for (int c = 0; c < 32; c++) {
        uint32_t current_cv[8];
        for (int i = 0; i < 8; i++) current_cv[i] = SW_IV[i];

        for (int b = 0; b < 16; b++) {
            uint32_t flags = 0;
            if (b == 0)  flags |= CHUNK_START;
            if (b == 15) flags |= CHUNK_END;

            uint32_t out16[16];
            sw_compress(current_cv, input_data[c * 16 + b], c, 64, flags, out16);
            for (int i = 0; i < 8; i++) current_cv[i] = out16[i];
        }
        
        uint32_t new_cv[8];
        for (int i = 0; i < 8; i++) new_cv[i] = current_cv[i];

        uint64_t total_chunks = c + 1; 

        while ((total_chunks & 1) == 0) {
            cv_stack_len--; 
            uint32_t left_child[8];
            for (int i = 0; i < 8; i++) left_child[i] = cv_stack[cv_stack_len][i];

            uint32_t flags = (total_chunks == 2 && c == 31) ? ROOT : 0;
            
            sw_parent_cv(left_child, new_cv, flags, new_cv);
            
            total_chunks >>= 1;
        }
        
        for (int i = 0; i < 8; i++) cv_stack[cv_stack_len][i] = new_cv[i];
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
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "blake3");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto in_bo = xrt::bo(device, sizeof(uint32_t) * 512 * 16, krnl.group_id(0));
    auto out_bo = xrt::bo(device, sizeof(uint32_t) * 8, krnl.group_id(1));

    // Map the contents of the buffer object into host memory
    uint32_t* in_map = in_bo.map<uint32_t*>();
    uint32_t* out_map = out_bo.map<uint32_t*>();

    uint32_t test_data[512][16];
    for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 16; j++) {
            uint32_t val = (i * 16 + j) * 0x11223344;
            in_map[i * 16 + j] = val;
            test_data[i][j] = val;
        }
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto krnl_start = std::chrono::steady_clock::now();
    auto run = krnl(in_bo, out_bo);
    run.wait();
    auto krnl_end = std::chrono::steady_clock::now();

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::chrono::duration<double> krnl_time = krnl_end - krnl_start;
    std::cout << "Kernel Exec: " << krnl_time.count() << " s" <<std::endl;

    uint32_t sw_out[8];
    sw_blake3_full_tree(test_data, sw_out);

    int errors = 0;
    for (int i = 0; i < 8; i++) {
        if (out_map[i] != sw_out[i]) {
            printf("Mismatch [%d] HW: %08X, SW: %08X\n", i, out_map[i], sw_out[i]);
            errors++;
        }
    }

    if (errors == 0) std::cout << ">>> TEST PASSED!" << std::endl;
    else std::cout << ">>> TEST FAILED!" << std::endl;

    return errors;
}
