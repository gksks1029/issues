#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#if defined(__riscv_vector)
    #include <riscv_vector.h>
#endif

void printCpuInstructionSet() {
    printf("🧠 CPU:");
#if defined(__riscv)
    printf(" risc-v");
    #if defined(__riscv_vector)
        printf(" rvv");
    #else
        printf(" without SIMD");
    #endif
#endif
    printf("\n");
}

void test_mul_F32(float *y, const float *x, const float *m) {
    unsigned int i = 0;
#if defined(__riscv_vector)
    size_t vl = __riscv_vsetvl_e32m1(8);
    for(; i < 4096; i += 8){
        vfloat32m1_t out_vec = __riscv_vle32_v_f32m1(&x[i], vl);
        vfloat32m1_t x_vec = __riscv_vle32_v_f32m1(&m[i], vl);
        vfloat32m1_t res_vec = __riscv_vfmul_vv_f32m1(out_vec, x_vec, vl);
        __riscv_vse32_v_f32m1(&y[i], res_vec, vl); 
    }
#endif
    for (; i < 4096; i++)
        y[i] = x[i] * m[i];

}

int main() {
    int nums = 4096;
    struct timespec start, end;
    
    float *x = (float *)malloc(nums * sizeof(float));
    float *y = (float *)malloc(nums * sizeof(float));
    float *m = (float *)malloc(nums * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < nums; i++) {
        float rand_x = (float)rand() / (float)RAND_MAX;
        float rand_m = (float)rand() / (float)RAND_MAX;
        x[i] = rand_x * 5.0f - 2.5f;
        m[i] = rand_m * 5.0f - 2.5f;
    }

    size_t vl = __riscv_vsetvl_e32m1(8);
    for(unsigned int i = 0; i < 4096; i += 8){
        vfloat32m1_t out_vec = __riscv_vle32_v_f32m1(&x[i], vl);
        vfloat32m1_t x_vec = __riscv_vle32_v_f32m1(&m[i], vl);
        vfloat32m1_t res_vec = __riscv_vfmul_vv_f32m1(out_vec, x_vec, vl);
    if(i==0){
        clock_gettime(CLOCK_MONOTONIC, &start);
        __riscv_vse32_v_f32m1(&y[i], res_vec, vl); 
        clock_gettime(CLOCK_MONOTONIC, &end);
    }
    }

    //printCpuInstructionSet();
    //clock_gettime(CLOCK_MONOTONIC, &start);
    //======= test function =======
    //test_mul_F32(y, x, m);
    //=============================
    //clock_gettime(CLOCK_MONOTONIC, &end);

    long sec_diff  = end.tv_sec - start.tv_sec;
    long nsec_diff = end.tv_nsec - start.tv_nsec;
    long total_nsec = sec_diff * 1000000000L + nsec_diff;
    printf("Elapsed time: %.3f us\n", (float)total_nsec/1000);
    free(x);
    free(y);
    free(m);
    return 0;
}