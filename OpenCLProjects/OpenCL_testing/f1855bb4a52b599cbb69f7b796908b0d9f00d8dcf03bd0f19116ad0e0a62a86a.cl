kernel void A(global double* o, const unsigned int v, const unsigned int y, const double n) {
    printf("before o[%u], %lf\n", get_global_id(0) * y, o[get_global_id(0) * y]);
    unsigned int s = get_global_id(0);
    o[s * y] = o[s * y] + n;
    printf("after: o[%u], %lf\n", get_global_id(0) * y, o[get_global_id(0) * y]);
}
