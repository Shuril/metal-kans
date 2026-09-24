"""
Render publication-quality benchmark charts for metal-KANs on Apple Silicon Metal GPU.
Saves high-resolution PNG assets into the metal-kans/assets/ directory.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["SF Pro Display", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#D0D5DD",
    "axes.linewidth": 1.2,
    "grid.color": "#E4E7EC",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.autolayout": True,
})


def render_metal_throughput_chart():
    batch_sizes = [64, 128, 256, 1024, 4096, 16384, 65536]
    labels = ["64", "128", "256", "1k", "4k", "16k", "65k"]

    cheby_thru = [229173, 426104, 871780, 2827655, 5047158, 9875593, 7285873]
    fast_thru = [239070, 512196, 960585, 3216459, 5766527, 6925390, 4423676]
    bspline_thru = [247694, 495284, 936478, 3022023, 5240281, 6982302, 4411250]
    relu_thru = [235894, 470282, 941487, 3275053, 5523783, 6936623, 4402722]

    cheby_lat = [0.279, 0.300, 0.294, 0.362, 0.812, 1.659, 8.995]
    fast_lat = [0.268, 0.250, 0.267, 0.318, 0.710, 2.366, 14.815]
    bspline_lat = [0.258, 0.258, 0.273, 0.339, 0.782, 2.347, 14.857]
    relu_lat = [0.271, 0.272, 0.272, 0.313, 0.742, 2.362, 14.885]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(batch_sizes))
    ax1.plot(x, [v / 1e6 for v in cheby_thru], marker="o", lw=2.4, color="#3B82F6", label="ChebyKAN (deg=4)")
    ax1.plot(x, [v / 1e6 for v in fast_thru], marker="s", lw=2.4, color="#10B981", label="FastKAN (centers=8)")
    ax1.plot(x, [v / 1e6 for v in bspline_thru], marker="^", lw=2.4, color="#F59E0B", label="BSplineKAN (grid=5)")
    ax1.plot(x, [v / 1e6 for v in relu_thru], marker="d", lw=2.4, color="#8B5CF6", label="ReLUKAN (grids=8)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="semibold")
    ax1.set_xlabel("Batch Size", fontweight="bold")
    ax1.set_ylabel("Throughput (Million samples / sec)", fontweight="bold")
    ax1.set_title("metal-KANs Forward Throughput on Metal GPU", fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9.5)

    ax1.annotate("Peak: 9.88M samples/sec",
                 xy=(5, 9.875), xytext=(3.6, 9.2),
                 arrowprops=dict(facecolor="#1E3A8A", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight="bold", color="#1E3A8A")

    ax2.plot(x, cheby_lat, marker="o", lw=2.4, color="#3B82F6", label="ChebyKAN")
    ax2.plot(x, fast_lat, marker="s", lw=2.4, color="#10B981", label="FastKAN")
    ax2.plot(x, bspline_lat, marker="^", lw=2.4, color="#F59E0B", label="BSplineKAN")
    ax2.plot(x, relu_lat, marker="d", lw=2.4, color="#8B5CF6", label="ReLUKAN")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="semibold")
    ax2.set_xlabel("Batch Size", fontweight="bold")
    ax2.set_ylabel("Latency (ms)", fontweight="bold")
    ax2.set_title("metal-KANs Forward Latency (Lower is better)", fontweight="bold", pad=12)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.6, which="both")
    ax2.legend(frameon=True, fontsize=9.5)

    out_path = "assets/metal_throughput_benchmark.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


def render_metal_training_chart():
    categories = [
        "BSplineKAN (B=128)", "BSplineKAN (B=1k)", "BSplineKAN (B=4k)",
        "FastKAN (B=128)", "FastKAN (B=1k)", "FastKAN (B=4k)",
        "ChebyKAN (B=128)", "ChebyKAN (B=1k)", "ChebyKAN (B=4k)",
    ]

    metal_lat = [0.336, 0.458, 1.280, 0.273, 0.441, 1.431, 0.329, 0.548, 0.952]
    mlx_lat   = [1.631, 0.982, 2.494, 0.305, 0.514, 1.323, 0.413, 0.670, 1.629]
    torch_lat = [1.116, 6.542, 28.329, 0.790, 1.421, 4.377, 0.486, 0.607, 3.620]

    y_pos = np.arange(len(categories))
    bar_height = 0.26

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

    rects1 = ax.barh(y_pos - bar_height, metal_lat, bar_height, label="metal-KANs (Fused GPU Step)", color="#10B981")
    rects2 = ax.barh(y_pos, mlx_lat, bar_height, label="Apple MLX", color="#3B82F6")
    rects3 = ax.barh(y_pos + bar_height, torch_lat, bar_height, label="PyTorch MPS", color="#94A3B8")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontweight="semibold")
    ax.invert_yaxis()
    ax.set_xlabel("Training Step Latency (ms) [Log Scale]", fontweight="bold")
    ax.set_xscale("log")
    ax.set_title("End-to-End Training Step: metal-KANs vs MLX vs PyTorch MPS\nForward + MSE Loss + Analytical Backward + Adam Optimizer", fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.6, which="both")
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    ax.text(1.280 * 1.15, 2 - bar_height, "22.1x vs MPS\n1.95x vs MLX", va="center", fontsize=8.5, fontweight="bold", color="#065F46")
    ax.text(0.952 * 1.15, 8 - bar_height, "3.80x vs MPS\n1.71x vs MLX", va="center", fontsize=8.5, fontweight="bold", color="#065F46")

    out_path = "assets/metal_training_benchmark.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered: {out_path}")


if __name__ == "__main__":
    print("Rendering high-quality assets for metal-KANs...")
    render_metal_throughput_chart()
    render_metal_training_chart()
    print("metal-KANs assets rendered successfully!")
