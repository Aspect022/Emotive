import json
r = json.load(open("results/model_results.json"))
print(f"{'Model':<15} {'Params':>10} {'Val F1':>8} {'Test Acc':>10} {'Test F1':>10}")
print("-"*58)
for m, v in r.items():
    print(f"{m:<15} {v['n_params']:>10,} {v['best_val_f1']:>8.4f} {v['test_accuracy']:>10.4f} {v['test_macro_f1']:>10.4f}")
print(f"{'Chance (20%)':<15} {'-':>10} {'-':>8} {'0.2000':>10} {'0.2000':>10}")
print()
for m, v in r.items():
    f1s = v.get("test_f1_per_class", [])
    tasks = ["Arith","Patt","Mem","Comp","Attn"]
    print(f"  {m} per-class F1: " + " | ".join(f"{t}={f:.3f}" for t,f in zip(tasks, f1s)))
