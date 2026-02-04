#pragma once
#include <vector>
#include <omp.h>
#include <algorithm>
#include <limits>

namespace OpenMP
{
    // Enum for operation type - makes segment tree more complex
    enum class OperationType
    {
        SUM = 0,
        MIN = 1,
        MAX = 2,
        PRODUCT = 3
    };

    class SegmentTree
    {
    private:
        std::vector<long long> m_data;
        std::vector<int> m_lazy; // Lazy propagation for range updates
        size_t m_cap;
        OperationType m_op;

        size_t next_pow_2(size_t v)
        {
            if (v == 0)
                return 1;
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            return v + 1;
        }

        long long combine(long long a, long long b) const
        {
            switch (m_op)
            {
            case OperationType::SUM:
                return a + b;
            case OperationType::MIN:
                return std::min(a, b);
            case OperationType::MAX:
                return std::max(a, b);
            case OperationType::PRODUCT:
                return a * b;
            default:
                return a + b;
            }
        }

        long long identity() const
        {
            switch (m_op)
            {
            case OperationType::SUM:
                return 0;
            case OperationType::MIN:
                return std::numeric_limits<long long>::max();
            case OperationType::MAX:
                return std::numeric_limits<long long>::min();
            case OperationType::PRODUCT:
                return 1;
            default:
                return 0;
            }
        }

        void push_down(int node, int start, int end)
        {
            if (m_lazy[node] != 0)
            {
                if (m_op == OperationType::SUM)
                {
                    m_data[node] += (long long)m_lazy[node] * (end - start);
                }
                else if (m_op == OperationType::MIN || m_op == OperationType::MAX)
                {
                    m_data[node] += m_lazy[node];
                }
                else if (m_op == OperationType::PRODUCT)
                {
                    m_data[node] *= m_lazy[node];
                }

                if (start != end - 1)
                {
                    m_lazy[2 * node] += m_lazy[node];
                    m_lazy[2 * node + 1] += m_lazy[node];
                }
                m_lazy[node] = 0;
            }
        }

        void update_range_rec(int node, int start, int end, int l, int r, int val)
        {
            push_down(node, start, end);
            if (start >= r || end <= l)
                return;

            if (start >= l && end <= r)
            {
                m_lazy[node] += val;
                push_down(node, start, end);
                return;
            }

            int mid = (start + end) / 2;
            update_range_rec(2 * node, start, mid, l, r, val);
            update_range_rec(2 * node + 1, mid, end, l, r, val);
            push_down(2 * node, start, mid);
            push_down(2 * node + 1, mid, end);
            m_data[node] = combine(m_data[2 * node], m_data[2 * node + 1]);
        }

        long long query_rec(int node, int start, int end, int l, int r)
        {
            if (start >= r || end <= l)
                return identity();

            push_down(node, start, end);

            if (start >= l && end <= r)
                return m_data[node];

            int mid = (start + end) / 2;
            long long p1 = query_rec(2 * node, start, mid, l, r);
            long long p2 = query_rec(2 * node + 1, mid, end, l, r);
            return combine(p1, p2);
        }

    public:
        SegmentTree(const std::vector<int> &data, OperationType op = OperationType::SUM)
            : m_op(op)
        {
            size_t n = data.size();
            m_cap = 1;
            while (m_cap < n)
                m_cap *= 2;

            m_data.resize(4 * m_cap, identity());
            m_lazy.resize(4 * m_cap, 0);

            // Build tree
            for (size_t i = 0; i < n; i++)
            {
                update_point(i, data[i]);
            }
        }

        void update_point(int idx, int val)
        {
            update_range_rec(1, 0, m_cap, idx, idx + 1, val);
        }

        // Backward compatibility wrapper
        void update(int idx, int val)
        {
            update_point(idx, val);
        }

        void update_range(int l, int r, int val)
        {
            update_range_rec(1, 0, m_cap, l, r, val);
        }

        long long query(int l, int r)
        {
            return query_rec(1, 0, m_cap, l, r);
        }

        // Parallel batch query for OpenMP optimization
        std::vector<long long> batch_query(const std::vector<int> &l_vals, const std::vector<int> &r_vals)
        {
            std::vector<long long> results(l_vals.size());

#pragma omp parallel for schedule(dynamic) if (l_vals.size() > 100)
            for (int i = 0; i < l_vals.size(); i++)
            {
                results[i] = query(l_vals[i], r_vals[i]);
            }

            return results;
        }
    };
}
