#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <map>
#include <algorithm>

#include "impl_baseline.hpp"
#include "impl_openmp.hpp"
#include "impl_ispc.hpp"
#include "impl_mpi.hpp"

using namespace std;

struct BenchmarkResult
{
    string test_name;
    int data_size;
    int num_ops;
    int num_threads;
    int grain_size;
    string schedule_type;
    double exec_time;
    double speedup;
};

vector<BenchmarkResult> results;

void generate_queries(vector<int> &q_L, vector<int> &q_R, int N, int Q, int seed)
{
    srand(seed);
    for (int i = 0; i < Q; i++)
    {
        q_L[i] = rand() % (N / 2);
        q_R[i] = q_L[i] + (rand() % 1000);
    }
}

void generate_updates(vector<int> &u_idx, vector<int> &u_val, int N, int U, int seed)
{
    srand(seed);
    for (int i = 0; i < U; i++)
    {
        u_idx[i] = rand() % N;
        u_val[i] = rand() % 100 + 1;
    }
}

double test_baseline_queries(const vector<int> &data, const vector<int> &q_L, const vector<int> &q_R)
{
    auto start = chrono::high_resolution_clock::now();
    Baseline::SegmentTree tree(data);
    long long checksum = 0;
    for (int i = 0; i < q_L.size(); i++)
    {
        checksum += tree.query(q_L[i], q_R[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

double test_baseline_mixed(const vector<int> &data, const vector<int> &u_idx, const vector<int> &u_val,
                           const vector<int> &q_L, const vector<int> &q_R)
{
    auto start = chrono::high_resolution_clock::now();
    Baseline::SegmentTree tree(data);
    for (int i = 0; i < u_idx.size(); i++)
    {
        tree.update(u_idx[i], u_val[i]);
    }
    long long checksum = 0;
    for (int i = 0; i < q_L.size(); i++)
    {
        checksum += tree.query(q_L[i], q_R[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

double test_openmp_queries(const vector<int> &data, const vector<int> &q_L, const vector<int> &q_R,
                           int num_threads, int grain_size, const string &schedule_type)
{
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();
    OpenMP::SegmentTree tree(data);
    long long checksum = 0;

    if (schedule_type == "static")
    {
#pragma omp parallel for schedule(static, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }
    else if (schedule_type == "dynamic")
    {
#pragma omp parallel for schedule(dynamic, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }
    else if (schedule_type == "guided")
    {
#pragma omp parallel for schedule(guided, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

double test_openmp_mixed(const vector<int> &data, const vector<int> &u_idx, const vector<int> &u_val,
                         const vector<int> &q_L, const vector<int> &q_R,
                         int num_threads, int grain_size, const string &schedule_type)
{
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();
    OpenMP::SegmentTree tree(data);

    for (int i = 0; i < u_idx.size(); i++)
    {
        tree.update(u_idx[i], u_val[i]);
    }

    long long checksum = 0;
    if (schedule_type == "static")
    {
#pragma omp parallel for schedule(static, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }
    else if (schedule_type == "dynamic")
    {
#pragma omp parallel for schedule(dynamic, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }
    else if (schedule_type == "guided")
    {
#pragma omp parallel for schedule(guided, grain_size) reduction(+ : checksum)
        for (int i = 0; i < q_L.size(); i++)
        {
            checksum += tree.query(q_L[i], q_R[i]);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

double test_ispc_queries(const vector<int> &data, const vector<int> &q_L, const vector<int> &q_R)
{
    auto start = chrono::high_resolution_clock::now();
    ISPC_Impl::SegmentTree tree(data);
    vector<int> results(q_L.size());
    tree.query_batch(q_L, q_R, results);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

double test_ispc_mixed(const vector<int> &data, const vector<int> &u_idx, const vector<int> &u_val,
                       const vector<int> &q_L, const vector<int> &q_R)
{
    auto start = chrono::high_resolution_clock::now();
    ISPC_Impl::SegmentTree tree(data);
    for (int i = 0; i < u_idx.size(); i++)
    {
        tree.update(u_idx[i], u_val[i]);
    }
    vector<int> res(q_L.size());
    tree.query_batch(q_L, q_R, res);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

void run_comprehensive_query_benchmarks(int rank, int N, int Q)
{
    if (rank != 0)
        return;

    vector<int> data(N, 1);
    vector<int> q_L(Q), q_R(Q);
    generate_queries(q_L, q_R, N, Q, 42);

    cout << "\n=== QUERY-ONLY BENCHMARKS (N=" << N << ", Q=" << Q << ") ===" << endl;
    cout << "Test,Time(s),Speedup" << endl;
    cout.flush();

    double baseline_time = test_baseline_queries(data, q_L, q_R);
    cout << "Baseline," << fixed << setprecision(4) << baseline_time << ",1.0" << endl;
    cout.flush();

    cout << "\n--- OpenMP Configuration Sweep ---" << endl;
    cout.flush();
    double best_omp_time = baseline_time;
    int best_threads = 1;
    int best_grain = 1;
    string best_schedule = "static";

    vector<int> thread_counts = {1, 2, 4, 8, 16};
    vector<int> grain_sizes = {1, 4, 8, 16, 32, 64, 128};
    vector<string> schedules = {"static", "dynamic", "guided"};

    for (int threads : thread_counts)
    {
        for (int grain : grain_sizes)
        {
            for (const string &sched : schedules)
            {
                double t = test_openmp_queries(data, q_L, q_R, threads, grain, sched);
                double speedup = baseline_time / t;
                cout << "OMP(t=" << threads << ",g=" << grain << "," << sched << "),"
                     << fixed << setprecision(4) << t << "," << speedup << endl;
                cout.flush();

                if (t < best_omp_time)
                {
                    best_omp_time = t;
                    best_threads = threads;
                    best_grain = grain;
                    best_schedule = sched;
                }
            }
        }
    }

    cout << "\nBest OpenMP Config: threads=" << best_threads << ", grain=" << best_grain
         << ", schedule=" << best_schedule << ", Time=" << fixed << setprecision(4) << best_omp_time
         << ", Speedup=" << (baseline_time / best_omp_time) << endl;
    cout.flush();

    cout << "\n--- ISPC Batch Query ---" << endl;
    cout.flush();
    double ispc_time = test_ispc_queries(data, q_L, q_R);
    cout << "ISPC," << fixed << setprecision(4) << ispc_time << "," << (baseline_time / ispc_time) << endl;
    cout.flush();
}

void run_comprehensive_mixed_benchmarks(int rank, int N, int U, int Q)
{
    if (rank != 0)
        return;

    vector<int> data(N, 1);
    vector<int> u_idx(U), u_val(U);
    vector<int> q_L(Q), q_R(Q);

    generate_updates(u_idx, u_val, N, U, 42);
    generate_queries(q_L, q_R, N, Q, 43);

    cout << "\n=== MIXED WORKLOAD BENCHMARKS (N=" << N << ", U=" << U << ", Q=" << Q << ") ===" << endl;
    cout << "Test,Time(s),Speedup" << endl;

    double baseline_time = test_baseline_mixed(data, u_idx, u_val, q_L, q_R);
    cout << "Baseline," << fixed << setprecision(4) << baseline_time << ",1.0" << endl;

    cout << "\n--- OpenMP Configuration Sweep ---" << endl;
    double best_omp_time = baseline_time;
    int best_threads = 1;
    int best_grain = 1;
    string best_schedule = "static";

    vector<int> thread_counts = {1, 2, 4, 8, 16};
    vector<int> grain_sizes = {1, 4, 8, 16, 32, 64};
    vector<string> schedules = {"static", "dynamic"};

    for (int threads : thread_counts)
    {
        for (int grain : grain_sizes)
        {
            for (const string &sched : schedules)
            {
                double t = test_openmp_mixed(data, u_idx, u_val, q_L, q_R, threads, grain, sched);
                double speedup = baseline_time / t;
                cout << "OMP(t=" << threads << ",g=" << grain << "," << sched << "),"
                     << fixed << setprecision(4) << t << "," << speedup << endl;

                if (t < best_omp_time)
                {
                    best_omp_time = t;
                    best_threads = threads;
                    best_grain = grain;
                    best_schedule = sched;
                }
            }
        }
    }

    cout << "\nBest OpenMP Config: threads=" << best_threads << ", grain=" << best_grain
         << ", schedule=" << best_schedule << ", Time=" << fixed << setprecision(4) << best_omp_time
         << ", Speedup=" << (baseline_time / best_omp_time) << endl;

    cout << "\n--- ISPC Batch ---" << endl;
    double ispc_time = test_ispc_mixed(data, u_idx, u_val, q_L, q_R);
    cout << "ISPC," << fixed << setprecision(4) << ispc_time << "," << (baseline_time / ispc_time) << endl;
}

void run_update_heavy_benchmark(int rank, int N, int U)
{
    if (rank != 0)
        return;

    vector<int> data(N, 1);
    vector<int> u_idx(U), u_val(U);
    generate_updates(u_idx, u_val, N, U, 44);

    cout << "\n=== UPDATE-HEAVY BENCHMARK (N=" << N << ", U=" << U << ") ===" << endl;
    cout << "Test,Time(s),Speedup" << endl;

    auto start = chrono::high_resolution_clock::now();
    Baseline::SegmentTree tree(data);
    for (int i = 0; i < U; i++)
    {
        tree.update(u_idx[i], u_val[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    double baseline_time = chrono::duration<double>(end - start).count();
    cout << "Baseline," << fixed << setprecision(4) << baseline_time << ",1.0" << endl;
}

void run_scalability_test(int rank, int N, int Q)
{
    if (rank != 0)
        return;

    cout << "\n=== SCALABILITY TEST (Varying Data Size) ===" << endl;
    cout << "DataSize,BaselineTime,BestOMPTime,Speedup" << endl;
    cout.flush();

    vector<int> data_sizes = {1000000, 5000000, 10000000, 50000000};

    for (int size : data_sizes)
    {
        cout << "Running for N=" << size << "..." << endl;
        cout.flush();

        vector<int> test_data(size, 1);
        vector<int> test_q_L(Q), test_q_R(Q);

        srand(45);
        for (int i = 0; i < Q; i++)
        {
            test_q_L[i] = rand() % (size / 2);
            test_q_R[i] = test_q_L[i] + (rand() % 1000);
        }

        double base = test_baseline_queries(test_data, test_q_L, test_q_R);
        double omp = test_openmp_queries(test_data, test_q_L, test_q_R, 8, 16, "dynamic");

        cout << size << "," << fixed << setprecision(4) << base << "," << omp
             << "," << (base / omp) << endl;
        cout.flush();
    }
}

void run_contention_test(int rank, int N, int Q)
{
    if (rank != 0)
        return;

    cout << "\n=== CONTENTION ANALYSIS ===" << endl;
    cout << "Workload,NumThreads,Time(s),Efficiency(%)" << endl;

    vector<int> data(N, 1);
    vector<int> q_L(Q), q_R(Q);
    generate_queries(q_L, q_R, N, Q, 46);

    double baseline = test_baseline_queries(data, q_L, q_R);

    for (int t : {1, 2, 4, 8, 16})
    {
        double exec = test_openmp_queries(data, q_L, q_R, t, 16, "dynamic");
        double efficiency = (baseline / exec) / t * 100;
        cout << "QueryHeavy," << t << "," << fixed << setprecision(4) << exec
             << "," << efficiency << endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    int Q = (argc > 2) ? atoi(argv[2]) : 10000;
    int U = (argc > 3) ? atoi(argv[3]) : 1000;
    int test_type = (argc > 4) ? atoi(argv[4]) : 0;

    if (rank == 0)
    {
        cout << "===========================================\n";
        cout << "COMPREHENSIVE SEGMENT TREE BENCHMARKING\n";
        cout << "===========================================\n";
    }

    switch (test_type)
    {
    case 0:
        run_comprehensive_query_benchmarks(rank, N, Q);
        break;
    case 1:
        run_comprehensive_mixed_benchmarks(rank, N, U, Q);
        break;
    case 2:
        run_update_heavy_benchmark(rank, N, U);
        break;
    case 3:
        run_scalability_test(rank, N, Q);
        break;
    case 4:
        run_contention_test(rank, N, Q);
        break;
    case 5:
    {
        // MPI distributed benchmark: prepare queries on rank 0, broadcast,
        // run the distributed benchmark on all ranks and report a single line
        // that `benchmark.py` can parse: "MPI,<time>,<speedup>" from rank 0.
        vector<int> q_L(Q), q_R(Q);
        if (rank == 0)
        {
            generate_queries(q_L, q_R, N, Q, 42);
        }

        MPI_Bcast(q_L.data(), Q, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(q_R.data(), Q, MPI_INT, 0, MPI_COMM_WORLD);

        double baseline_time = 0.0;
        if (rank == 0)
        {
            vector<int> data(N, 1);
            baseline_time = test_baseline_queries(data, q_L, q_R);
        }

        double mpi_time = MPI_Impl::run_benchmark(N, Q, q_L, q_R);

        if (rank == 0)
        {
            double speedup = (mpi_time > 0.0) ? (baseline_time / mpi_time) : 0.0;
            cout << "MPI," << fixed << setprecision(4) << mpi_time << "," << speedup << endl;
            cout.flush();
        }

        break;
    }
    default:
        if (rank == 0)
        {
            cout << "\nUsage: ./main [N] [Q] [U] [TEST_TYPE]\n";
            cout << "TEST_TYPE:\n";
            cout << "  0 - Query-only benchmarks (default)\n";
            cout << "  1 - Mixed workload benchmarks\n";
            cout << "  2 - Update-heavy benchmarks\n";
            cout << "  3 - Scalability tests\n";
            cout << "  4 - Contention analysis\n";
        }
        run_comprehensive_query_benchmarks(rank, N, Q);
    }

    MPI_Finalize();
    return 0;
}
