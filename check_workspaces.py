#!/usr/bin/env python3
"""
Check workspace structure and existing training results.

This script helps verify the separate workspace setup for baseline vs LRM comparison.
"""

import os
from pathlib import Path

def check_workspace(workspace_path, model_name):
    """Check a workspace directory for training results."""
    print(f"\n📁 {model_name} Workspace: {workspace_path}")
    print("-" * 60)
    
    if not os.path.exists(workspace_path):
        print("❌ Workspace does not exist")
        return False
    
    # Check subdirectories
    subdirs = ['checkpoints', 'logs', 'statistics', 'features', 'predictions']
    has_results = False
    
    for subdir in subdirs:
        subdir_path = os.path.join(workspace_path, subdir)
        if os.path.exists(subdir_path):
            files = os.listdir(subdir_path)
            if files:
                print(f"✅ {subdir}/: {len(files)} files")
                # Show some example files
                for i, file in enumerate(files[:3]):
                    print(f"   - {file}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
                has_results = True
            else:
                print(f"📂 {subdir}/: empty")
        else:
            print(f"❌ {subdir}/: missing")
    
    return has_results

def check_logs_for_plotting(workspace_path):
    """Check if workspace has logs suitable for plotting."""
    logs_dir = os.path.join(workspace_path, 'logs')
    if not os.path.exists(logs_dir):
        return False, []
    
    log_files = []
    for file in os.listdir(logs_dir):
        if file.endswith('.log') or 'log' in file.lower():
            log_path = os.path.join(logs_dir, file)
            # Check if file has content
            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                    if len(content) > 100:  # Has substantial content
                        log_files.append(file)
            except:
                pass
    
    return len(log_files) > 0, log_files

def main():
    print("🔍 Workspace Structure Check")
    print("=" * 50)
    print("Checking separate workspaces for baseline vs LRM comparison...")
    
    # Define workspaces
    baseline_workspace = "workspaces/emotion_regression"
    lrm_workspace = "workspaces/emotion_feedback"
    
    # Check baseline workspace
    baseline_has_results = check_workspace(baseline_workspace, "Baseline Model")
    baseline_has_logs, baseline_log_files = check_logs_for_plotting(baseline_workspace)
    
    # Check LRM workspace  
    lrm_has_results = check_workspace(lrm_workspace, "LRM Model")
    lrm_has_logs, lrm_log_files = check_logs_for_plotting(lrm_workspace)
    
    # Summary
    print(f"\n📊 Summary")
    print("-" * 30)
    print(f"Baseline results: {'✅ Available' if baseline_has_results else '❌ Missing'}")
    print(f"Baseline logs:    {'✅ Available' if baseline_has_logs else '❌ Missing'}")
    if baseline_has_logs:
        print(f"   Log files: {', '.join(baseline_log_files)}")
    
    print(f"LRM results:      {'✅ Available' if lrm_has_results else '❌ Missing'}")
    print(f"LRM logs:         {'✅ Available' if lrm_has_logs else '❌ Missing'}")
    if lrm_has_logs:
        print(f"   Log files: {', '.join(lrm_log_files)}")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    print("-" * 30)
    
    if not baseline_has_results and not lrm_has_results:
        print("🚀 Run complete comparison:")
        print("   bash run_baseline_vs_lrm_comparison.sh")
    elif not baseline_has_results:
        print("📊 Run baseline training only:")
        print("   bash run_baseline_vs_lrm_comparison.sh --skip-lrm")
    elif not lrm_has_results:
        print("🧠 Run LRM training only:")
        print("   bash run_baseline_vs_lrm_comparison.sh --skip-baseline")
    elif baseline_has_logs and lrm_has_logs:
        print("📈 Generate comparison plots:")
        print("   bash run_baseline_vs_lrm_comparison.sh --plots-only")
        print("   # OR directly:")
        print("   python plot_training_comparison.py --auto-find")
    else:
        print("⚠️  Results exist but logs may be incomplete.")
        print("   Consider re-running training or check log files manually.")
    
    # Check if plots already exist
    plots_dir = "training_comparison_plots"
    if os.path.exists(plots_dir) and os.listdir(plots_dir):
        print(f"\n📸 Existing plots found in: {plots_dir}")
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        for plot in plot_files:
            print(f"   - {plot}")
    
    print(f"\n✅ Workspace check complete!")

if __name__ == '__main__':
    main() 