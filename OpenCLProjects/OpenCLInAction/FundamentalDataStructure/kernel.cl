__kernel void vec_add(__global float4 *vec1,
		      __global float4 *vec2,
		      __global float4 *vec3) {
	int id = get_global_id(0);
 	vec3[id] = vec1[id] + vec2[id];
}
