__kernel void matvec_mult_float_loop(__global float* matrix,
                           __global float* vector,
                           __global float* result) {
    int id = get_global_id(0);
    float acc = 0.0;
  for (int i = 0; i < 4; i++) {
           acc += matrix[id * 4 + i] * vector[i];
           }
    result[id] = acc;
}

__kernel void matvec_mult_float_loop(__global float8* matrix,
                           __global float8* vector,
                           __global float* result) {
    int id = get_global_id(0);
    result[id] = dot(matrix[id].s0123, vector[0]) + dot(matrix[id].s4567, vector[0]);

}
