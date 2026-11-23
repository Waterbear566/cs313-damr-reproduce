# Chạy lại DeepSVDD (2 biến thể) và ALOCC theo pipeline chung, seed 0/1/2, PSM -> MSL -> SMD.
# Chạy từ gốc repo:  powershell -ExecutionPolicy Bypass -File scripts/run_e3a_rerun.ps1 [-Python .venv\Scripts\python.exe]
# Lượt đã có trong CSV sẽ được bỏ qua, nên chạy lại script sau khi bị ngắt là an toàn.
param([string]$Python = "python")
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::UserWarning"
New-Item -ItemType Directory -Force logs | Out-Null
foreach ($ds in "PSM", "MSL", "SMD") {
    foreach ($seed in 0, 1, 2) {
        foreach ($variant in "full", "bn_train_each_epoch") {
            & $Python baselines/DeepSVDD/run.py --dataset $ds --seed $seed --variant $variant 2>&1 |
                Tee-Object -Append -FilePath "logs/E3_DeepSVDD_${variant}_${ds}_seed${seed}.log"
            if ($LASTEXITCODE -ne 0) { throw "DeepSVDD $variant $ds seed $seed lỗi" }
        }
        & $Python baselines/ALOCC/run.py --dataset $ds --seed $seed 2>&1 |
            Tee-Object -Append -FilePath "logs/E3_ALOCC_${ds}_seed${seed}.log"
        if ($LASTEXITCODE -ne 0) { throw "ALOCC $ds seed $seed lỗi" }
    }
}
