// src/kernels.ispc
// Optimized ISPC kernels for segment tree queries
// Each SIMD lane processes a separate query for better vectorization

export void build_leaves(uniform int tree[], uniform const int data[], uniform int n) {
    // Vectorized leaf copy with high throughput
    foreach (i = 0 ... n) {
        tree[n + i] = data[i];
    }
}

// Highly optimized batch query - processes multiple queries with SIMD
// Each SIMD lane (AVX: 8 lanes, SSE: 4 lanes) processes one query in parallel
export void batch_query(uniform int tree[], uniform int n, 
                        uniform const int q_L[], uniform const int q_R[], 
                        uniform int results[], uniform int count) {
    // foreach automatically vectorizes across multiple queries
    // with each SIMD lane handling an independent query
    foreach (i = 0 ... count) {
        int l = q_L[i] + n;
        int r = q_R[i] + n;
        int sum = 0;
        
        // Each lane executes independently - no data dependencies
        // between lanes enables true SIMD parallelism
        // Use int types consistently to avoid conversion overhead
        while (l < r) {
            if ((l & 1) != 0) sum += tree[l++];
            if ((r & 1) != 0) sum += tree[--r];
            l = l >> 1;  // Use bit shift instead of division for performance
            r = r >> 1;
        }
        results[i] = sum;
    }
}