#!/bin/bash

# Script to run complete baseline vs LRM comparison
# This script:
# 1. Runs baseline emotion training (saves to workspaces/emotion_regression)
# 2. Runs LRM emotion training (saves to workspaces/emotion_feedback) 
# 3. Generates comparison plots
#
# Usage:
#   bash run_baseline_vs_lrm_comparison.sh
#   bash run_baseline_vs_lrm_comparison.sh --skip-baseline    # Only run LRM (if baseline already done)
#   bash run_baseline_vs_lrm_comparison.sh --skip-lrm        # Only run baseline (if LRM already done)
#   bash run_baseline_vs_lrm_comparison.sh --plots-only      # Only generate plots (if both trainings done)

# =============================================================================
# Parse Arguments
# =============================================================================

SKIP_BASELINE=false
SKIP_LRM=false
PLOTS_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-lrm)
            SKIP_LRM=true
            shift
            ;;
        --plots-only)
            PLOTS_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-baseline    Skip baseline training (use existing results)"
            echo "  --skip-lrm         Skip LRM training (use existing results)"
            echo "  --plots-only       Only generate comparison plots"
            echo "  --help             Show this help message"
            echo ""
            echo "The script saves results in separate directories:"
            echo "  Baseline: workspaces/emotion_regression/"
            echo "  LRM:      workspaces/emotion_feedback/"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configuration
# =============================================================================

BASELINE_WORKSPACE="workspaces/emotion_regression"
LRM_WORKSPACE="workspaces/emotion_feedback"
PLOTS_DIR="training_comparison_plots"

echo "🚀 Baseline vs LRM Emotion Model Comparison"
echo "============================================"
echo "Baseline workspace: $BASELINE_WORKSPACE"
echo "LRM workspace:      $LRM_WORKSPACE"  
echo "Plots directory:    $PLOTS_DIR"
echo ""

# =============================================================================
# Training Phase
# =============================================================================

if [ "$PLOTS_ONLY" = false ]; then
    
    # Run baseline training
    if [ "$SKIP_BASELINE" = false ]; then
        echo "📊 Step 1: Training Baseline Model"
        echo "=================================="
        echo "Model: FeatureEmotionRegression_Cnn6_NewAffective"
        echo "Workspace: $BASELINE_WORKSPACE"
        echo ""
        
        # Check if baseline already exists
        if [ -d "$BASELINE_WORKSPACE/checkpoints" ] && [ "$(ls -A $BASELINE_WORKSPACE/checkpoints 2>/dev/null)" ]; then
            echo "⚠️  Baseline checkpoints found in $BASELINE_WORKSPACE/checkpoints"
            echo "   Continuing will overwrite existing baseline results."
            read -p "   Continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "   Skipping baseline training. Use --skip-baseline to skip this check."
                SKIP_BASELINE=true
            fi
        fi
        
        if [ "$SKIP_BASELINE" = false ]; then
            echo "🏃‍♂️ Starting baseline training..."
            ./run_emotion.sh --skip-extraction
            
            if [ $? -ne 0 ]; then
                echo "❌ Baseline training failed!"
                exit 1
            fi
            
            echo "✅ Baseline training completed!"
            echo "   Results saved in: $BASELINE_WORKSPACE"
        fi
    else
        echo "⏭️  Skipping baseline training (--skip-baseline flag)"
    fi
    
    echo ""
    
    # Run LRM training  
    if [ "$SKIP_LRM" = false ]; then
        echo "🧠 Step 2: Training LRM Model with Feedback"
        echo "==========================================="
        echo "Model: FeatureEmotionRegression_Cnn6_LRM"
        echo "Workspace: $LRM_WORKSPACE"
        echo ""
        
        # Check if LRM already exists
        if [ -d "$LRM_WORKSPACE/checkpoints" ] && [ "$(ls -A $LRM_WORKSPACE/checkpoints 2>/dev/null)" ]; then
            echo "⚠️  LRM checkpoints found in $LRM_WORKSPACE/checkpoints"
            echo "   Continuing will overwrite existing LRM results."
            read -p "   Continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "   Skipping LRM training. Use --skip-lrm to skip this check."
                SKIP_LRM=true
            fi
        fi
        
        if [ "$SKIP_LRM" = false ]; then
            echo "🏃‍♂️ Starting LRM training with feedback..."
            ./run_emotion_feedback.sh --skip-extraction
            
            if [ $? -ne 0 ]; then
                echo "❌ LRM training failed!"
                exit 1
            fi
            
            echo "✅ LRM training completed!"
            echo "   Results saved in: $LRM_WORKSPACE"
        fi
    else
        echo "⏭️  Skipping LRM training (--skip-lrm flag)"
    fi
    
else
    echo "⏭️  Skipping training phase (--plots-only flag)"
fi

echo ""

# =============================================================================
# Plotting Phase
# =============================================================================

echo "📈 Step 3: Generating Comparison Plots"
echo "======================================"

# Check if training results exist
BASELINE_LOGS_EXIST=false
LRM_LOGS_EXIST=false

if [ -d "$BASELINE_WORKSPACE/logs" ] && [ "$(ls -A $BASELINE_WORKSPACE/logs 2>/dev/null)" ]; then
    BASELINE_LOGS_EXIST=true
    echo "✅ Found baseline logs in: $BASELINE_WORKSPACE/logs"
fi

if [ -d "$LRM_WORKSPACE/logs" ] && [ "$(ls -A $LRM_WORKSPACE/logs 2>/dev/null)" ]; then
    LRM_LOGS_EXIST=true
    echo "✅ Found LRM logs in: $LRM_WORKSPACE/logs"
fi

if [ "$BASELINE_LOGS_EXIST" = false ] && [ "$LRM_LOGS_EXIST" = false ]; then
    echo "❌ No training logs found in either workspace!"
    echo "   Please run training first or check the workspace directories."
    exit 1
fi

# Create plots directory
mkdir -p "$PLOTS_DIR"

# Generate comparison plots
echo ""
echo "🎨 Creating training performance comparison plots..."

# Try to find specific log files
BASELINE_LOG=""
LRM_LOG=""

# Look for baseline logs
if [ "$BASELINE_LOGS_EXIST" = true ]; then
    # Try common log file names
    for logfile in "train.log" "training.log" "emotion.log"; do
        if [ -f "$BASELINE_WORKSPACE/logs/$logfile" ]; then
            BASELINE_LOG="$BASELINE_WORKSPACE/logs/$logfile"
            break
        fi
    done
    
    # If no specific log found, use the first .log file
    if [ -z "$BASELINE_LOG" ]; then
        BASELINE_LOG=$(find "$BASELINE_WORKSPACE/logs" -name "*.log" -type f | head -1)
    fi
fi

# Look for LRM logs
if [ "$LRM_LOGS_EXIST" = true ]; then
    # Try common log file names
    for logfile in "train.log" "training.log" "emotion.log"; do
        if [ -f "$LRM_WORKSPACE/logs/$logfile" ]; then
            LRM_LOG="$LRM_WORKSPACE/logs/$logfile"
            break
        fi
    done
    
    # If no specific log found, use the first .log file
    if [ -z "$LRM_LOG" ]; then
        LRM_LOG=$(find "$LRM_WORKSPACE/logs" -name "*.log" -type f | head -1)
    fi
fi

# Generate plots
if [ -n "$BASELINE_LOG" ] && [ -n "$LRM_LOG" ]; then
    echo "📊 Using logs:"
    echo "   Baseline: $BASELINE_LOG"
    echo "   LRM:      $LRM_LOG"
    
    python plot_training_comparison.py \
        --baseline-logs "$BASELINE_LOG" \
        --lrm-logs "$LRM_LOG" \
        --output-dir "$PLOTS_DIR"
        
elif [ -n "$BASELINE_LOG" ] || [ -n "$LRM_LOG" ]; then
    echo "📊 Using auto-detection (only one model available):"
    [ -n "$BASELINE_LOG" ] && echo "   Baseline: $BASELINE_LOG"
    [ -n "$LRM_LOG" ] && echo "   LRM:      $LRM_LOG"
    
    python plot_training_comparison.py \
        --auto-find \
        --search-dirs "$BASELINE_WORKSPACE" "$LRM_WORKSPACE" \
        --output-dir "$PLOTS_DIR"
else
    echo "⚠️  No specific log files found, trying auto-detection..."
    
    python plot_training_comparison.py \
        --auto-find \
        --search-dirs "$BASELINE_WORKSPACE" "$LRM_WORKSPACE" \
        --output-dir "$PLOTS_DIR"
fi

if [ $? -eq 0 ]; then
    echo "✅ Comparison plots generated successfully!"
    echo "   Plots saved in: $PLOTS_DIR"
    echo ""
    echo "📁 Generated files:"
    ls -la "$PLOTS_DIR"
else
    echo "❌ Plot generation failed!"
    echo "   You can try manual plotting with:"
    echo "   python plot_training_comparison.py --auto-find --search-dirs $BASELINE_WORKSPACE $LRM_WORKSPACE"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "🎉 Baseline vs LRM Comparison Complete!"
echo "======================================="
echo ""
echo "📊 Results Summary:"
echo "   Baseline workspace: $BASELINE_WORKSPACE"
echo "   LRM workspace:      $LRM_WORKSPACE"
echo "   Comparison plots:   $PLOTS_DIR"
echo ""
echo "🔍 Next steps:"
echo "   1. Open the plots in $PLOTS_DIR to see the comparison"
echo "   2. Check training_comparison.png for the main overview"
echo "   3. Review individual metric plots for detailed analysis"
echo ""
echo "📈 Expected LRM advantages:"
echo "   - Lower validation loss (better generalization)"
echo "   - Lower validation MAE (better emotion prediction)"  
echo "   - Higher Pearson correlation (better linear relationship)"
echo "   - Potentially faster convergence"
echo ""

# Show plot files if they exist
if [ -d "$PLOTS_DIR" ] && [ "$(ls -A $PLOTS_DIR 2>/dev/null)" ]; then
    echo "📸 Available plots:"
    for plot in "$PLOTS_DIR"/*.png; do
        if [ -f "$plot" ]; then
            echo "   - $(basename "$plot")"
        fi
    done
fi

echo ""
echo "🚀 Comparison pipeline completed successfully!" 