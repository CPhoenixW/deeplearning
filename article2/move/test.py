# -*- coding: utf-8 -*-
"""
批量运行 main.py：
- attack_type ∈ ["label_flipping", "gaussian_noise", "sign_flipping"]
- use_svdd ∈ [False, True]
共 3 × 2 = 6 组实验。
"""

from main import main


def run_all():
    configs = [
        (False, "label_flipping"),
        (True, "label_flipping"),
        (False, "gaussian_noise"),
        (True, "gaussian_noise"),
        (False, "sign_flipping"),
        (True, "sign_flipping"),
    ]

    for idx, (use_svdd, attack_type) in enumerate(configs, start=1):
        tag = f"run{idx}_{'svdd' if use_svdd else 'fedavg'}_{attack_type}"
        print(f"\n=== Run {idx}/6: use_svdd={use_svdd}, attack_type={attack_type} ===")
        main(use_svdd=use_svdd, attack_type=attack_type, tag=tag)


if __name__ == "__main__":
    run_all()

