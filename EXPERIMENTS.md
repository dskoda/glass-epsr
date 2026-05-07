# Simplified Experiment Workflow

This guide demonstrates the new simplified experiment structure for the `glass` package.

## Overview

The new workflow uses a unified **experiment folder** that contains everything needed for training and generation:

```
my_experiment/
├── config.yaml          # All parameters
├── data/                # Training data (*.xyz files)
├── checkpoints/         # Model checkpoints
│   ├── best.ckpt
│   ├── last.ckpt
│   └── epoch_*.ckpt
├── inits/               # Initial structures for generation
├── outputs/             # Generated structures
└── logs/                # TensorBoard logs
    └── version_0/
```

## Training a Model

### 1. Create experiment and train from scratch

```bash
# Create new experiment and train score model
glass train ./my_experiment --model-type score --num-species 1

# Or train spectral surrogate (EXAFS)
glass train ./my_experiment --model-type spec --spec-type exafs --num-species 1
```

This will:
- Create the experiment directory structure
- Save configuration to `config.yaml`
- Train the model
- Save checkpoints to `checkpoints/` (best.ckpt, last.ckpt, epoch_*.ckpt)
- Save TensorBoard logs to `logs/version_0/`

### 2. Resume training

```bash
# Resume from last checkpoint
glass train ./my_experiment --resume
```

### 3. Override specific parameters

```bash
# Override training parameters
glass train ./my_experiment --max-epochs 5000 --lr 0.0005 --dim 256
```

All overrides are saved to `config.yaml`.

## Generating Structures

### 1. Prepare initial structures

Copy your initial .xyz files to the `inits/` folder:

```bash
cp /path/to/init_structures/*.xyz ./my_experiment/inits/
```

### 2. Run generation

```bash
# Generate using best checkpoint (default)
glass generate ./my_experiment --inits ./my_experiment/inits/

# Use last checkpoint
glass generate ./my_experiment --inits ./inits/ --checkpoint last.ckpt

# Use specific checkpoint
glass generate ./my_experiment --inits ./inits/ --checkpoint epoch_0100.ckpt
```

### 3. Conditional generation with guidance

```bash
# PDF guidance
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type pdf --ref-path ./reference/

# XRD guidance
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type xrd --ref-path ./reference/ \
    --element-names Si --rho 5

# EXAFS guidance (requires spectral model)
glass generate ./my_experiment --inits ./inits/ \
    --guidance-type exafs --spec-model-path ./spec_experiment/checkpoints/best.ckpt \
    --ref-path ./reference/ --rho 1e8
```

## Configuration File

The `config.yaml` contains all parameters:

```yaml
name: my_experiment
model_type: score  # or "spec"
num_species: 1
num_convs: 5
dim: 200
ema_decay: 0.9999
max_epochs: 12000
batch_size: 1
learning_rate: 0.001
cutoff: 5.0
k: 0.8
data_dir: data
save_top_k: 3
checkpoint: best
n_runs: 10
# ... (see glass/experiment.py for all options)
```

You can edit this file directly or use CLI overrides.

## Data Organization

### Training Data

Place your training .xyz files in the `data/` folder. The system searches in order:

1. `data/*.xyz` (flat - preferred)
2. `data/structures/*.xyz` (old structure)
3. `data/structures/train/*.xyz` (old structure)

### Initial Structures

Place initial structures for generation in `inits/*.xyz`.

### Outputs

Generated structures are saved to `outputs/` by default.

## Migration from Old Structure

If you have existing experiments with the old structure:

### Old structure:
```
./data/Si_1.5_2.5_3.5/structures/train/*.xyz
./models/Si_1.5_2.5_3.5/version_0/checkpoints/*.ckpt
```

### New structure:
```
./experiments/Si_1.5_2.5_3.5/
├── data/*.xyz                    # Copy training structures here
├── checkpoints/
│   └── best.ckpt                # Copy checkpoint here
└── config.yaml                  # Create this file
```

Migration script:

```bash
# Create new experiment
glass train ./experiments/Si_1.5_2.5_3.5 \
    --model-type score \
    --num-species 1 \
    --max-epochs 0  # Just create structure

# Copy data
cp ./data/Si_1.5_2.5_3.5/structures/train/*.xyz \
   ./experiments/Si_1.5_2.5_3.5/data/

# Copy best checkpoint
cp ./models/Si_1.5_2.5_3.5/version_0/checkpoints/*.ckpt \
   ./experiments/Si_1.5_2.5_3.5/checkpoints/best.ckpt
```

## Quick Reference

| Old Command | New Command |
|------------|-------------|
| `glass train_score TAG --data-root ./data` | `glass train ./experiment --model-type score` |
| `glass train_spec TAG --spec-type exafs` | `glass train ./experiment --model-type spec --spec-type exafs` |
| `glass uncond_denoise --score-data-path ... --score-model-path ...` | `glass generate ./experiment --inits ./inits/` |
| `glass cond_denoise --score-data-path ... --score-model-path ... --ref-path ...` | `glass generate ./experiment --inits ./inits/ --guidance-type pdf --ref-path ./ref/` |

## Key Changes

1. **Single experiment folder** - Everything in one place
2. **config.yaml** - All parameters in one file
3. **Flat checkpoint structure** - No more `version_X` in checkpoints/
4. **Simplified CLI** - Just point to the experiment folder
5. **Automatic checkpoint discovery** - No need to specify full paths
6. **Hybrid config** - Use CLI for quick overrides, config.yaml for persistence
