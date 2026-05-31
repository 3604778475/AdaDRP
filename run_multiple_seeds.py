"""
运行多次实验，使用不同随机种子，评估稳定性
"""
import subprocess
import sys

seeds = [2023, 2024, 2025, 2026, 2027]

for seed in seeds:
    print(f"\n{'='*70}")
    print(f"Running experiment with seed = {seed}")
    print('='*70)
    
    # 修改 train_mmpa_program.py 中的 seed
    cmd = f"sed -i 's/seed = 2025/seed = {seed}/' train_mmpa_program.py"
    subprocess.run(cmd, shell=True)
    
    # 运行实验
    result = subprocess.run([
        'python', 'train_mmpa_program.py',
        '--measurement', 'Z_SCORE',
        '--thres_gdsc', '0.0',
        '--thres_label', '0.1',
        '--n', '5',
        '--alph', '0.3',
        '--beta', '0.3',
        '--train',
        '--num_programs', '8',
        '--program_dim', '64',
        '--lambda_div', '0.01',
        '--lambda_sparse', '0.001',
        '--use_ot_alignment',
        '--lambda_ot', '0.005',
        '--ot_reg', '0.1',
        '--ot_cost_type', 'cosine',
        '--ot_marginal_type', 'uniform',
        '--backbone_type', 'convnext1d',
        '--convnext_depth', '1',
        '--convnext_kernel_size', '7',
        '--K', '3'
    ], capture_output=True, text=True)
    
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    # 恢复原始 seed
    subprocess.run("sed -i 's/seed = [0-9]*/seed = 2025/' train_mmpa_program.py", shell=True)

print("\n" + "="*70)
print("所有实验完成！请查看各次结果的一致性。")
print("="*70)
