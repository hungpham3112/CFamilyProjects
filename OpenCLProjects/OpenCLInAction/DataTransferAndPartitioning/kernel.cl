__kernel void matvec_mult(__global float4* matrix,
			__global float4* vector,
			__global float* result) {
	int i = get_global_id(0);
	// Print information about vector[0]
    printf("vector[0].x: %f, vector[0].y: %f, vector[0].z: %f, vector[0].w: %f\n",
           vector[10].s1, vector[10].s1, vector[10].s2, vector[10].s3);
	result[i] = dot(matrix[i], vector[0]);
}
