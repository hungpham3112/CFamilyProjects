kernel void A(global double* c, const double k) {
    int e = get_global_id(0) + k;
    c[e] = 42.0 / (1.0 + exp(-c[e]));
}
