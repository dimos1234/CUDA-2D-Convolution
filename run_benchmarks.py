import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import custom_convolution

# Resolution test points (Sweeping up to 4K)
SIZES = [512, 1024, 2048, 4096]
NUM_TRIALS = 10  # Number of full benchmark passes to average out
ITERATIONS_PER_TRIAL = 50

# Hardcoded filter matching our C++ tests
filter_weights = torch.tensor([
     0.0, -1.0,  0.0,
    -1.0,  5.0, -1.0,
     0.0, -1.0,  0.0
], dtype=torch.float32, device='cuda')

pytorch_filter = filter_weights.view(1, 1, 3, 3)

results = {size: {'native': [], 'custom': []} for size in SIZES}

print(f"Running {NUM_TRIALS} profiling passes per resolution to calculate mean execution times...")
print(f"{'Resolution':<12} | {'PyTorch cuDNN (ms)':<20} | {'Custom Kernel (ms)':<23} | {'Correctness Match':<18}")
print("-" * 80)

for size in SIZES:
    img_np = np.random.randint(0, 256, (size, size)).astype(np.uint8)
    img_custom = torch.from_numpy(img_np).cuda()
    img_pytorch = torch.from_numpy(img_np).float().cuda().view(1, 1, size, size)

    # --- WARM-UP RUNS ---
    for _ in range(10):
        _ = custom_convolution.forward(img_custom, filter_weights)
        _ = F.conv2d(img_pytorch, pytorch_filter, padding=1)
    torch.cuda.synchronize()

    trial_custom_times = []
    trial_native_times = []

    # Setup timing events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # --- MULTI-PASS PROFILING LOOP ---
    for trial in range(NUM_TRIALS):
        
        # Benchmark Custom Kernel for this trial
        start_evt.record()
        for _ in range(ITERATIONS_PER_TRIAL):
            custom_out = custom_convolution.forward(img_custom, filter_weights)
        end_evt.record()
        torch.cuda.synchronize()
        trial_custom_times.append(start_evt.elapsed_time(end_evt) / ITERATIONS_PER_TRIAL)

        # Benchmark PyTorch Native (cuDNN) for this trial
        start_evt.record()
        for _ in range(ITERATIONS_PER_TRIAL):
            py_out = F.conv2d(img_pytorch, pytorch_filter, padding=1)
        end_evt.record()
        torch.cuda.synchronize()
        trial_native_times.append(start_evt.elapsed_time(end_evt) / ITERATIONS_PER_TRIAL)

    # Calculate the statistical mean across all 10 independent trials
    mean_custom_time = np.mean(trial_custom_times)
    mean_native_time = np.mean(trial_native_times)

    # --- CORRECTNESS VERIFICATION ---
    py_out_flat = py_out.squeeze().clamp(0, 255).to(torch.uint8)
    mismatches = torch.sum(custom_out != py_out_flat).item()
    correct = "PASSED" if mismatches == 0 else f"FAILED ({mismatches}px)"

    results[size]['native'].append(mean_native_time)
    results[size]['custom'].append(mean_custom_time)

    print(f"{f'{size}x{size}':<12} | {mean_native_time:<20.4f} | {mean_custom_time:<23.4f} | {correct:<18}")

# --- GENERATE PERFORMANCE CHART ---
plt.figure(figsize=(10, 6))
plt.plot(SIZES, [results[s]['native'][0] for s in SIZES], marker='o', label='PyTorch Native (cuDNN) [Mean]', linewidth=2)
plt.plot(SIZES, [results[s]['custom'][0] for s in SIZES], marker='s', label='Custom Unrolled Kernel [Mean]', linewidth=2)
plt.xlabel('Image Resolution (Square Matrix Size)', fontsize=12)
plt.ylabel('Mean Execution Time (ms)', fontsize=12)
plt.title('GPU Convolution Processing Speed Scales (Averaged Over 10 Trials)', fontsize=14, fontweight='bold')
plt.xticks(SIZES)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.savefig('performance_curves.png', dpi=300, bbox_inches='tight')
print("\n[SUITE COMPLETE] Multi-pass performance scaling chart saved as 'performance_curves.png'")
