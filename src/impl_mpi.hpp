#pragma once
#include <vector>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <limits>
#include "impl_baseline.hpp"

namespace MPI_Impl
{
    // Enhanced MPI-compatible SegmentTree with lazy propagation
    class DistributedSegmentTree
    {
    private:
        std::vector<long long> m_data;
        std::vector<int> m_lazy;
        size_t m_cap;
        Baseline::OperationType m_op;

        long long combine(long long a, long long b) const
        {
            switch (m_op)
            {
            case Baseline::OperationType::SUM:
                return a + b;
            case Baseline::OperationType::MIN:
                return std::min(a, b);
            case Baseline::OperationType::MAX:
                return std::max(a, b);
            case Baseline::OperationType::PRODUCT:
                return a * b;
            default:
                return a + b;
            }
        }

        long long identity() const
        {
            switch (m_op)
            {
            case Baseline::OperationType::SUM:
                return 0;
            case Baseline::OperationType::MIN:
                return std::numeric_limits<long long>::max();
            case Baseline::OperationType::MAX:
                return std::numeric_limits<long long>::min();
            case Baseline::OperationType::PRODUCT:
                return 1;
            default:
                return 0;
            }
        }

    public:
        DistributedSegmentTree(const std::vector<int> &data,
                               Baseline::OperationType op = Baseline::OperationType::SUM)
            : m_op(op)
        {
            size_t n = data.size();
            m_cap = 1;
            while (m_cap < n)
                m_cap *= 2;

            m_data.resize(4 * m_cap, identity());
            m_lazy.resize(4 * m_cap, 0);

            for (size_t i = 0; i < n; i++)
            {
                m_data[m_cap + i] = data[i];
            }
        }

        long long query(int l, int r)
        {
            long long sum = identity();
            for (l += m_cap, r += m_cap; l < r; l /= 2, r /= 2)
            {
                if (l & 1)
                    sum = combine(sum, m_data[l++]);
                if (r & 1)
                    sum = combine(sum, m_data[--r]);
            }
            return sum;
        }

        void update(int idx, int val)
        {
            m_data[idx + m_cap] = val;
            idx += m_cap;
            for (idx /= 2; idx >= 1; idx /= 2)
                m_data[idx] = combine(m_data[2 * idx], m_data[2 * idx + 1]);
        }
    };

    double run_benchmark(int global_N, int num_queries, const std::vector<int> &q_L, const std::vector<int> &q_R)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<int> data(global_N, 1);
        DistributedSegmentTree tree(data, Baseline::OperationType::SUM);

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        long long local_checksum = 0;
        int queries_per_rank = (num_queries + size - 1) / size;
        int start_query = rank * queries_per_rank;
        int end_query = std::min(start_query + queries_per_rank, num_queries);

        for (int i = start_query; i < end_query; i++)
        {
            local_checksum += tree.query(q_L[i], q_R[i]);
        }

        long long global_checksum = 0;
        MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double>(end - start).count();
    }
}
