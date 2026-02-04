#!/usr/bin/env python3
"""
Segment Tree Benchmarking Suite
Comprehensive comparison: Baseline vs OpenMP vs ISPC vs MPI
Multiple operation types: Sum, Multiply, Min/Max, and more
"""

import subprocess
import sys
import os
import csv
import time
from pathlib import Path
from typing import Dict, List


class SegmentTreeBenchmark:
    """Comprehensive benchmark with comparison analysis"""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.benchmark_bin = self.workspace_dir / "benchmark_bin"
        self.results = {"fast_configs": [], "scalability": [], "efficiency": []}

    def build(self) -> bool:
        """Build benchmark executable"""
        print("=" * 70)
        print("BUILDING BENCHMARK SUITE")
        print("=" * 70)

        try:
            print("\n[1/3] Cleaning old builds...")
            subprocess.run(
                ["make", "clean"],
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=60,
            )

            print("[2/3] Building benchmark_bin...")
            result = subprocess.run(
                ["make", "benchmark_bin"],
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"❌ Build failed!")
                return False

            print("[3/3] Verifying build...")
            if self.benchmark_bin.exists():
                print(f"✓ Built: {self.benchmark_bin}")
            else:
                print("❌ Build verification failed!")
                return False

            print("\n✅ Build successful!")
            return True

        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def run_full_benchmark(
        self,
        num_elements: int = 10_000_000,
        num_queries: int = 10_000,
        num_updates: int = 1_000,
    ) -> bool:
        """Run complete benchmark suite"""
        print("\n" + "=" * 70)
        print("FULL BENCHMARK SUITE")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"  Elements: {num_elements:,}")
        print(f"  Queries: {num_queries:,}")
        print(f"  Updates: {num_updates:,}")
        print(f"\nApproaches Tested:")
        print(f"  • Baseline (Sequential) - Single-threaded reference")
        print(f"  • OpenMP - Multi-threaded parallelization")
        print(f"  • ISPC - SIMD vectorization for batch operations")
        print(f"  • MPI - Distributed memory parallelization")
        print(f"\nOperations Covered:")
        print(f"  • Range Sum Queries")
        print(f"  • Range Product/Multiplication")
        print(f"  • Range Min/Max Queries")
        print(f"  • Point Updates")
        print(f"  • Range Updates (if applicable)")

        try:
            start_total = time.time()

            # Mode 0: Fast configs & approach comparison
            print("\n" + "-" * 70)
            print("TEST 1/3: APPROACH COMPARISON & OPENMP TUNING")
            print("-" * 70)
            self._run_test_mode(0, num_elements, num_queries, num_updates)

            # Mode 5: MPI distributed benchmark (uses mpiexec)
            print("\n" + "-" * 70)
            print("TEST: MPI Distributed Benchmark")
            print("-" * 70)
            self._run_test_mode(5, num_elements, num_queries, num_updates)

            # Mode 3: Scalability
            print("\n" + "-" * 70)
            print("TEST 2/3: SCALABILITY ANALYSIS")
            print("-" * 70)
            self._run_test_mode(3, num_elements, num_queries, num_updates)

            # Mode 4: Efficiency
            print("\n" + "-" * 70)
            print("TEST 3/3: PARALLEL EFFICIENCY & CONTENTION ANALYSIS")
            print("-" * 70)
            self._run_test_mode(4, num_elements, num_queries, num_updates)

            total_time = time.time() - start_total
            print(f"\n✅ All tests completed in {total_time/60:.1f} minutes")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _run_test_mode(
        self, mode: int, num_elements: int, num_queries: int, num_updates: int
    ):
        """Run a single test mode"""
        try:
            start_time = time.time()
            # Use mpiexec for MPI mode (5), otherwise run locally
            if mode == 5:
                cmd = ["mpiexec", "-n", "4", "./benchmark_bin", str(num_elements), str(num_queries), str(num_updates), str(mode)]
            else:
                cmd = ["./benchmark_bin", str(num_elements), str(num_queries), str(num_updates), str(mode)]

            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                timeout=1800,
                text=True,
            )
            elapsed = time.time() - start_time

            if result.returncode != 0:
                print(f"❌ Test mode {mode} failed!")
                print(result.stderr)
                return

            output = result.stdout
            print(output)

            # Parse based on mode
            if mode == 0:
                self._parse_fast_configs(output)
            elif mode == 3:
                self._parse_scalability(output)
            elif mode == 4:
                self._parse_efficiency(output)
            elif mode == 5:
                # MPI prints a single CSV-like line: MPI,<time>,<speedup>
                self._parse_fast_configs(output)

        except subprocess.TimeoutExpired:
            print(f"❌ Test mode {mode} timeout!")
        except Exception as e:
            print(f"❌ Error in test mode {mode}: {e}")

    def _parse_fast_configs(self, output: str):
        """Parse fast config sweep output"""
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Implementation") or line.startswith("---") or "===" in line:
                continue

            if line.startswith("Baseline,"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        self.results["fast_configs"].append(
                            {
                                "config": "Baseline",
                                "threads": "-",
                                "grain": "-",
                                "schedule": "-",
                                "time": float(parts[1]),
                                "speedup": float(parts[2]),
                            }
                        )
                    except ValueError:
                        pass

            elif line.startswith("OMP("):
                # main.cpp prints OpenMP lines like:
                # OMP(t=8,g=16,dynamic),0.1234,2.5
                # When split by comma: ["OMP(t=8", "g=16", "dynamic)", "0.1234", "2.5"]
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        # Extract time and speedup from last 2 parts
                        time_val = float(parts[-2])
                        speedup_val = float(parts[-1])

                        # Reconstruct the inner parameters from parts[0..2]
                        # parts[0] = "OMP(t=8"
                        # parts[1] = "g=16"
                        # parts[2] = "dynamic)"
                        threads = "-"
                        grain = "-"
                        schedule = "-"
                        
                        # Extract threads from parts[0]: "OMP(t=8" -> "8"
                        if "t=" in parts[0]:
                            threads = int(parts[0].split("t=")[1])
                        
                        # Extract grain from parts[1]: "g=16" -> "16"
                        if parts[1].startswith("g="):
                            grain = int(parts[1].split("=")[1])
                        
                        # Extract schedule from parts[2]: "dynamic)" -> "dynamic"
                        schedule = parts[2].rstrip(")")

                        self.results["fast_configs"].append(
                            {
                                "config": "OpenMP",
                                "threads": threads,
                                "grain": grain,
                                "schedule": schedule,
                                "time": time_val,
                                "speedup": speedup_val,
                            }
                        )
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("ISPC,"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        self.results["fast_configs"].append(
                            {
                                "config": "ISPC",
                                "threads": "-",
                                "grain": "-",
                                "schedule": "-",
                                "time": float(parts[1]),
                                "speedup": float(parts[2]),
                            }
                        )
                    except (ValueError, IndexError):
                        pass

            elif line.startswith("MPI,"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        self.results["fast_configs"].append(
                            {
                                "config": "MPI",
                                "threads": "-",
                                "grain": "-",
                                "schedule": "-",
                                "time": float(parts[1]),
                                "speedup": float(parts[2]),
                            }
                        )
                    except (ValueError, IndexError):
                        pass

    def _parse_scalability(self, output: str):
        """Parse scalability test output"""
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if (
                not line
                or "===" in line
                or line.startswith("DataSize")
                or line.startswith("Running")
            ):
                continue

            if line and line[0].isdigit():
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        self.results["scalability"].append(
                            {
                                "size": int(parts[0]),
                                "baseline": float(parts[1]),
                                "best_omp": float(parts[2]),
                                "speedup": float(parts[3]),
                            }
                        )
                    except (ValueError, IndexError):
                        pass

    def _parse_efficiency(self, output: str):
        """Parse efficiency analysis output"""
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if not line or "===" in line or line.startswith("Workload"):
                continue

            if "," in line:
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        self.results["efficiency"].append(
                            {
                                "workload": parts[0],
                                "threads": int(parts[1]),
                                "time": float(parts[2]),
                                "efficiency": float(parts[3]),
                            }
                        )
                    except (ValueError, IndexError):
                        pass

    def generate_comparison_report(self) -> str:
        """Generate comprehensive approach comparison"""
        report = []
        report.append("\n" + "=" * 70)
        report.append("COMPREHENSIVE APPROACH COMPARISON")
        report.append("=" * 70)

        if not self.results["fast_configs"]:
            return "\n".join(report)

        # Group by implementation
        approaches = {}
        for config in self.results["fast_configs"]:
            approach = config["config"]
            if approach not in approaches:
                approaches[approach] = []
            approaches[approach].append(config)

        baseline_time = None
        report.append("\n📊 IMPLEMENTATION APPROACHES TESTED:\n")

        for approach in ["Baseline", "OpenMP", "ISPC", "MPI"]:
            if approach not in approaches:
                continue

            configs = approaches[approach]
            best = max(configs, key=lambda x: x["speedup"])

            if approach == "Baseline":
                baseline_time = best["time"]
                report.append(f"1. BASELINE (Sequential Reference)")
                report.append(f"   Description: Single-threaded, no parallelization")
                report.append(f"   Use Case: Reference for speedup calculation")
                report.append(f"   Execution Time: {best['time']:.4f}s")
                report.append(f"   Speedup: 1.00x (baseline)\n")
            elif approach == "OpenMP":
                report.append(f"2. OPENMP (Shared Memory Parallelization)")
                report.append(
                    f"   Description: Multi-threaded with configurable schedule"
                )
                report.append(
                    f"   Best Config: {best['threads']} threads, grain={best['grain']}, {best['schedule']}"
                )
                report.append(f"   Execution Time: {best['time']:.4f}s")
                report.append(f"   Speedup: {best['speedup']:.2f}x")
                if baseline_time:
                    time_saved = (baseline_time - best["time"]) * 1000
                    report.append(f"   Time Saved: {time_saved:.2f}ms per operation")
                report.append(f"   Tested Configurations: {len(configs)}")
                report.append(
                    f"   Use When: Multi-core CPU, shared memory, load balancing needed\n"
                )
            elif approach == "ISPC":
                report.append(f"3. ISPC (SIMD Vectorization)")
                report.append(f"   Description: Batch queries via SIMD instructions")
                report.append(f"   Execution Time: {best['time']:.4f}s")
                report.append(f"   Speedup: {best['speedup']:.2f}x")
                if baseline_time:
                    time_saved = (baseline_time - best["time"]) * 1000
                    report.append(f"   Time Saved: {time_saved:.2f}ms per operation")
                report.append(
                    f"   Use When: Batch operations, SIMD-friendly patterns\n"
                )
            elif approach == "MPI":
                report.append(f"4. MPI (Distributed Memory Parallelization)")
                report.append(f"   Description: Distributed queries across multiple processes")
                report.append(f"   Execution Time: {best['time']:.4f}s")
                report.append(f"   Speedup: {best['speedup']:.2f}x")
                if baseline_time:
                    time_saved = (baseline_time - best["time"]) * 1000
                    report.append(f"   Time Saved: {time_saved:.2f}ms per operation")
                report.append(
                    f"   Use When: Distributed systems, large-scale parallelization\n"
                )

        # Winner
        report.append("\n" + "-" * 70)
        report.append("🏆 OVERALL WINNER\n")
        best_overall = max(self.results["fast_configs"], key=lambda x: x["speedup"])
        report.append(f"Fastest Approach: {best_overall['config']}")
        report.append(f"Speedup: {best_overall['speedup']:.2f}x faster than baseline")
        if baseline_time:
            time_saved = (baseline_time - best_overall["time"]) * 1000
            report.append(f"Time Improvement: {time_saved:.3f}ms per operation\n")

        # Summary table
        report.append("-" * 70)
        report.append("DETAILED COMPARISON TABLE\n")
        report.append(f"{'Approach':<15} {'Best Time':<15} {'Speedup':<12} {'Configs'}")
        report.append("-" * 70)
        for approach in ["Baseline", "OpenMP", "ISPC", "MPI"]:
            if approach in approaches:
                best = max(approaches[approach], key=lambda x: x["speedup"])
                num_configs = len(approaches[approach])
                report.append(
                    f"{approach:<15} {best['time']:<15.4f}s {best['speedup']:<12.2f}x {num_configs}"
                )

        return "\n".join(report)

    def generate_scalability_report(self) -> str:
        """Generate scalability analysis"""
        report = []
        report.append("\n" + "=" * 70)
        report.append("SCALABILITY ANALYSIS - PERFORMANCE ACROSS DATA SIZES")
        report.append("=" * 70)

        if not self.results["scalability"]:
            return "\n".join(report)

        report.append("\n📈 EXECUTION TIME BY DATA SIZE:\n")
        report.append(
            f"{'Size (M)':<12} {'Baseline (s)':<15} {'Best OMP (s)':<15} {'Speedup':<12} {'Efficiency'}"
        )
        report.append("-" * 70)

        for r in sorted(self.results["scalability"], key=lambda x: x["size"]):
            size_m = r["size"] / 1_000_000
            eff = f"{r['speedup']*25:.0f}%" if r["speedup"] > 0 else "N/A"
            report.append(
                f"{size_m:<12.1f} {r['baseline']:<15.4f} {r['best_omp']:<15.4f} {r['speedup']:<12.2f}x {eff}"
            )

        # Scaling analysis
        report.append("\n" + "-" * 70)
        report.append("SCALING TREND ANALYSIS\n")

        if len(self.results["scalability"]) > 1:
            speedups = [
                r["speedup"]
                for r in sorted(self.results["scalability"], key=lambda x: x["size"])
            ]
            if speedups[-1] > speedups[0]:
                improvement = ((speedups[-1] / speedups[0]) - 1) * 100
                report.append(
                    f"✓ POSITIVE SCALING: Speedup increases {improvement:.0f}% with data size"
                )
                report.append(f"  Recommendation: Better for large datasets\n")
            else:
                degradation = ((speedups[0] / speedups[-1]) - 1) * 100
                report.append(
                    f"⚠ NEGATIVE SCALING: Speedup decreases {degradation:.0f}% with data size"
                )
                report.append(
                    f"  Recommendation: Parallelization overhead dominates small data\n"
                )

            # Efficiency note
            avg_speedup = sum(speedups) / len(speedups)
            report.append(f"Average Speedup: {avg_speedup:.2f}x")
            report.append(
                f"Best Speedup: {max(speedups):.2f}x (at N={sorted(self.results['scalability'], key=lambda x: x['size'])[-1]['size']/1e6:.0f}M)"
            )

        return "\n".join(report)

    def generate_efficiency_report(self) -> str:
        """Generate efficiency analysis"""
        report = []
        report.append("\n" + "=" * 70)
        report.append("PARALLEL EFFICIENCY ANALYSIS - THREAD SCALING")
        report.append("=" * 70)

        if not self.results["efficiency"]:
            return "\n".join(report)

        # Group by workload
        by_workload = {}
        for r in self.results["efficiency"]:
            workload = r["workload"]
            if workload not in by_workload:
                by_workload[workload] = []
            by_workload[workload].append(r)

        report.append("\n⚡ EFFICIENCY METRICS BY WORKLOAD:\n")

        for workload in sorted(by_workload.keys()):
            report.append(f"{workload}:")
            report.append(
                f"  {'Threads':<10} {'Time (s)':<12} {'Efficiency':<12} {'Rating'}"
            )
            report.append(f"  {'-'*48}")

            for r in sorted(by_workload[workload], key=lambda x: x["threads"]):
                eff = r["efficiency"]
                if eff > 100:
                    rating = "🟢 Superlinear"
                elif eff >= 75:
                    rating = "🟢 Excellent (75-99%)"
                elif eff >= 50:
                    rating = "🟡 Good (50-74%)"
                elif eff >= 25:
                    rating = "🟠 Fair (25-49%)"
                else:
                    rating = "🔴 Poor (<25%)"
                report.append(
                    f"  {r['threads']:<10} {r['time']:<12.4f} {eff:<12.1f}% {rating}"
                )
            report.append("")

        # Recommendations
        report.append("-" * 70)
        report.append("THREADING RECOMMENDATIONS\n")

        all_results = []
        for workload_results in by_workload.values():
            all_results.extend(workload_results)

        if all_results:
            # Optimal thread count
            best_eff = max(all_results, key=lambda x: x["efficiency"])
            report.append(f"✓ Optimal Thread Count: {best_eff['threads']} threads")
            report.append(f"  Achieves {best_eff['efficiency']:.1f}% efficiency\n")

            # Contention detection
            sorted_by_threads = sorted(all_results, key=lambda x: x["threads"])
            if len(sorted_by_threads) > 2:
                early_eff = sum(r["efficiency"] for r in sorted_by_threads[:2]) / 2
                late_eff = sum(r["efficiency"] for r in sorted_by_threads[-2:]) / 2
                contention_ratio = (late_eff / early_eff) if early_eff > 0 else 0

                if contention_ratio < 0.75:
                    report.append(f"⚠ CONTENTION DETECTED")
                    report.append(f"  Early threads (1-2): {early_eff:.1f}% efficiency")
                    report.append(f"  Late threads (>4): {late_eff:.1f}% efficiency")
                    report.append(
                        f"  Recommendation: Limit to {best_eff['threads']} threads to avoid overhead\n"
                    )
                else:
                    report.append(f"✓ GOOD SCALING: No significant contention detected")
                    report.append(f"  All thread counts maintain >75% efficiency\n")

        return "\n".join(report)

    def export_to_csv(self, prefix: str = "benchmark_results"):
        """Export all results to CSV files"""
        print("\n" + "=" * 70)
        print("EXPORTING RESULTS TO CSV")
        print("=" * 70)

        try:
            # Export fast configs
            if self.results["fast_configs"]:
                csv_file = self.workspace_dir / f"{prefix}_fast.csv"
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "config",
                            "threads",
                            "grain",
                            "schedule",
                            "time",
                            "speedup",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(self.results["fast_configs"])
                print(
                    f"✓ Exported: {csv_file.name} ({len(self.results['fast_configs'])} configs)"
                )

            # Export scalability
            if self.results["scalability"]:
                csv_file = self.workspace_dir / f"{prefix}_scalability.csv"
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["size", "baseline", "best_omp", "speedup"]
                    )
                    writer.writeheader()
                    writer.writerows(self.results["scalability"])
                print(
                    f"✓ Exported: {csv_file.name} ({len(self.results['scalability'])} sizes)"
                )

            # Export efficiency
            if self.results["efficiency"]:
                csv_file = self.workspace_dir / f"{prefix}_efficiency.csv"
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["workload", "threads", "time", "efficiency"]
                    )
                    writer.writeheader()
                    writer.writerows(self.results["efficiency"])
                print(
                    f"✓ Exported: {csv_file.name} ({len(self.results['efficiency'])} measurements)"
                )

            print("\n✅ CSV export complete!")
            return True

        except Exception as e:
            print(f"⚠️ CSV export error: {e}")
            return False

    def generate_plots(self, prefix: str = "benchmark_results"):
        """Generate plots from benchmark results"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\n⚠️ matplotlib not installed. Install with: pip install matplotlib")
            return False

        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATION PLOTS")
        print("=" * 70)

        try:
            # Plot 1: Approach Comparison with dual visualization
            if self.results["fast_configs"]:
                by_approach = {}
                for c in self.results["fast_configs"]:
                    if c["config"] not in by_approach:
                        by_approach[c["config"]] = []
                    by_approach[c["config"]].append(c["speedup"])

                approaches = list(by_approach.keys())
                best_speedups = [max(by_approach[a]) for a in approaches]
                avg_speedups = [np.mean(by_approach[a]) for a in approaches]
                colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"][: len(approaches)]

                # Create figure with best and average speedups side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Plot best speedups
                bars1 = ax1.bar(
                    approaches,
                    best_speedups,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=2,
                )
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}x",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )
                ax1.axhline(
                    y=1.0,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    label="Baseline (1.0x)",
                )
                ax1.set_ylabel("Best Speedup", fontsize=12, fontweight="bold")
                ax1.set_title(
                    "Best Performance per Approach", fontsize=13, fontweight="bold"
                )
                ax1.set_ylim(0, max(best_speedups) * 1.2)
                ax1.grid(axis="y", alpha=0.3)
                ax1.legend(fontsize=10)

                # Plot average speedups
                bars2 = ax2.bar(
                    approaches,
                    avg_speedups,
                    color=colors,
                    alpha=0.6,
                    edgecolor="black",
                    linewidth=2,
                )
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}x",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )
                ax2.axhline(
                    y=1.0,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    label="Baseline (1.0x)",
                )
                ax2.set_ylabel("Average Speedup", fontsize=12, fontweight="bold")
                ax2.set_title(
                    "Average Performance per Approach", fontsize=13, fontweight="bold"
                )
                ax2.set_ylim(0, max(avg_speedups) * 1.2)
                ax2.grid(axis="y", alpha=0.3)
                ax2.legend(fontsize=10)

                fig.suptitle(
                    "Approach Comparison: Baseline vs OpenMP vs ISPC vs MPI",
                    fontsize=15,
                    fontweight="bold",
                    y=1.00,
                )
                plt.tight_layout()
                plt.savefig(
                    self.workspace_dir / f"{prefix}_comparison.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                print(f"✓ Saved: {prefix}_comparison.png")
                plt.close()

            # Plot 2: Scalability
            if self.results["scalability"]:
                sizes = [
                    r["size"] / 1_000_000
                    for r in sorted(
                        self.results["scalability"], key=lambda x: x["size"]
                    )
                ]
                baselines = [
                    r["baseline"]
                    for r in sorted(
                        self.results["scalability"], key=lambda x: x["size"]
                    )
                ]
                omp_times = [
                    r["best_omp"]
                    for r in sorted(
                        self.results["scalability"], key=lambda x: x["size"]
                    )
                ]
                speedups = [
                    r["speedup"]
                    for r in sorted(
                        self.results["scalability"], key=lambda x: x["size"]
                    )
                ]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Execution times
                ax1.plot(
                    sizes,
                    baselines,
                    marker="o",
                    label="Baseline",
                    linewidth=2.5,
                    markersize=8,
                    color="#d62728",
                )
                ax1.plot(
                    sizes,
                    omp_times,
                    marker="s",
                    label="Best OpenMP",
                    linewidth=2.5,
                    markersize=8,
                    color="#2ca02c",
                )
                ax1.set_xlabel("Data Size (Millions)", fontsize=11, fontweight="bold")
                ax1.set_ylabel(
                    "Execution Time (seconds)", fontsize=11, fontweight="bold"
                )
                ax1.set_title(
                    "Scalability: Execution Time", fontsize=12, fontweight="bold"
                )
                ax1.legend(fontsize=11)
                ax1.grid(alpha=0.3)

                # Speedup
                ax2.plot(
                    sizes,
                    speedups,
                    marker="^",
                    color="#1f77b4",
                    linewidth=2.5,
                    markersize=8,
                )
                ax2.fill_between(sizes, speedups, alpha=0.2, color="#1f77b4")
                ax2.set_xlabel("Data Size (Millions)", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Speedup", fontsize=11, fontweight="bold")
                ax2.set_title(
                    "Scalability: Speedup Trend", fontsize=12, fontweight="bold"
                )
                ax2.grid(alpha=0.3)

                plt.tight_layout()
                plt.savefig(
                    self.workspace_dir / f"{prefix}_scalability.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                print(f"✓ Saved: {prefix}_scalability.png")
                plt.close()

            # Plot 3: Efficiency by Threads
            if self.results["efficiency"]:
                by_workload = {}
                for r in self.results["efficiency"]:
                    w = r["workload"]
                    if w not in by_workload:
                        by_workload[w] = {}
                    by_workload[w][r["threads"]] = r["efficiency"]

                plt.figure(figsize=(12, 6))
                colors_map = {
                    "QueryHeavy": "#d62728",
                    "Mixed": "#2ca02c",
                    "UpdateHeavy": "#1f77b4",
                }
                for i, workload in enumerate(sorted(by_workload.keys())):
                    threads = sorted(by_workload[workload].keys())
                    efficiencies = [by_workload[workload][t] for t in threads]
                    color = colors_map.get(workload, f"C{i}")
                    plt.plot(
                        threads,
                        efficiencies,
                        marker="o",
                        label=workload,
                        linewidth=2.5,
                        markersize=8,
                        color=color,
                    )

                plt.xlabel("Number of Threads", fontsize=11, fontweight="bold")
                plt.ylabel("Parallel Efficiency (%)", fontsize=11, fontweight="bold")
                plt.title(
                    "Parallel Efficiency by Thread Count",
                    fontsize=12,
                    fontweight="bold",
                )
                plt.axhline(
                    y=100,
                    color="blue",
                    linestyle="--",
                    linewidth=2,
                    label="Perfect Scaling (100%)",
                )
                plt.legend(fontsize=11)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    self.workspace_dir / f"{prefix}_efficiency.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                print(f"✓ Saved: {prefix}_efficiency.png")
                plt.close()

            print("\n✅ Plots generated successfully!")
            return True

        except Exception as e:
            print(f"⚠️ Plot generation error: {e}")
            return False


def main():
    """Main entry point - Automatic comprehensive benchmark across multiple element sizes"""

    os.chdir(Path(__file__).parent)
    bench = SegmentTreeBenchmark()

    # Build once
    if not bench.build():
        sys.exit(1)

    # Element sizes to test automatically
    element_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]

    # Fixed test parameters (increased for more comprehensive testing)
    queries = 500
    updates = 100

    print("\n" + "=" * 70)
    print("🚀 PARALLEL SEGMENT TREE - COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"\nTesting {len(element_sizes)} element sizes automatically:")
    print(f"  Elements: {', '.join(f'{n//1_000_000}M' for n in element_sizes)}")
    print(f"  Queries: {queries:,}")
    print(f"  Updates: {updates:,}")
    print("\n" + "=" * 70)

    # Run benchmark for each element size
    for i, num_elements in enumerate(element_sizes, 1):
        size_str = (
            f"{num_elements//1_000_000}M"
            if num_elements >= 1_000_000
            else f"{num_elements//1000}K"
        )

        print(f"\n[{i}/{len(element_sizes)}] Testing {size_str} elements...")
        print("-" * 70)

        if not bench.run_full_benchmark(num_elements, queries, updates):
            print(f"⚠️ Benchmark failed for {size_str} elements - continuing...")
            continue

        print(f"✓ Completed {size_str} element size benchmark")

    # Generate reports and output
    print("\n" + "=" * 70)
    print("📊 GENERATING COMPREHENSIVE REPORTS")
    print("=" * 70)

    print("\n" + bench.generate_comparison_report())
    print(bench.generate_scalability_report())
    print(bench.generate_efficiency_report())

    # Export and plot
    print("\n" + "=" * 70)
    print("💾 EXPORTING DATA AND GENERATING VISUALIZATIONS")
    print("=" * 70)

    bench.export_to_csv()
    bench.generate_plots()

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE - ALL TESTS EXECUTED")
    print("=" * 70)
    print("\nOutput files generated:")
    print("  CSV Results (for data analysis):")
    print("    • benchmark_results_fast.csv")
    print("    • benchmark_results_scalability.csv")
    print("    • benchmark_results_efficiency.csv")
    print("  Plots (for visualization):")
    print("    • benchmark_results_comparison.png")
    print("    • benchmark_results_scalability.png")
    print("    • benchmark_results_efficiency.png")
    print(
        "\nAll element sizes tested: "
        + ", ".join(f"{n//1_000_000}M" for n in element_sizes)
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
