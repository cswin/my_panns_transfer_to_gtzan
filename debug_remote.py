#!/usr/bin/env python3

import os
import h5py

def debug_remote_issue():
    """Debug why the enhanced loading isn't working on remote server."""
    
    print("=== Remote Server Debugging ===")
    
    # Check file paths
    features_file = "/home/pengliu/Private/panns_transfer_to_gtzan/features/features.h5"
    
    print(f"1. Checking features file: {features_file}")
    if os.path.exists(features_file):
        print("   ✅ Features file exists")
        
        # Check the file content
        try:
            with h5py.File(features_file, 'r') as hf:
                audio_names = [name.decode() for name in hf['audio_name'][:]]
                unique_files = list(set(audio_names))
                print(f"   📊 Contains: {len(unique_files)} unique files")
                
                if len(unique_files) == 325:
                    print("   ❌ PROBLEM: Still has old 325 files!")
                    print("   💡 SOLUTION: Delete this file and re-run extraction")
                elif len(unique_files) > 900:
                    print("   ✅ SUCCESS: Enhanced extraction worked!")
                else:
                    print(f"   ⚠️  PARTIAL: Has {len(unique_files)} files")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print("   ❌ Features file doesn't exist")
    
    # Check if enhanced code is present
    features_script = "/home/pengliu/Private/my_panns_transfer_to_gtzan/utils/features.py"
    print(f"\n2. Checking enhanced code in: {features_script}")
    
    if os.path.exists(features_script):
        with open(features_script, 'r') as f:
            content = f.read()
            
        if "Method 1: librosa with target sample rate" in content:
            print("   ✅ Enhanced loading code is present")
        else:
            print("   ❌ Enhanced loading code is MISSING!")
            print("   💡 SOLUTION: Re-sync the files")
        
        if "Method 2: librosa with original sample rate" in content:
            print("   ✅ Method 2 code is present")
        else:
            print("   ❌ Method 2 code is MISSING!")
        
        if "Method 3: soundfile" in content:
            print("   ✅ Method 3 code is present")
        else:
            print("   ❌ Method 3 code is MISSING!")
    else:
        print("   ❌ Features script doesn't exist!")
    
    # Check workspace paths
    workspace = "/home/pengliu/Private/panns_transfer_to_gtzan/"
    print(f"\n3. Checking workspace: {workspace}")
    if os.path.exists(workspace):
        print("   ✅ Workspace exists")
        features_dir = os.path.join(workspace, "features")
        if os.path.exists(features_dir):
            print("   ✅ Features directory exists")
        else:
            print("   ❌ Features directory missing")
    else:
        print("   ❌ Workspace doesn't exist!")
    
    # Provide clear next steps
    print(f"\n=== Next Steps ===")
    print("1. Delete old features file:")
    print(f"   rm -f {features_file}")
    print("\n2. Run enhanced extraction:")
    print("   python3 utils/features.py pack_audio_files_to_hdf5 \\")
    print("     --dataset_dir=/DATA/pliu/EmotionData/GTZAN/genres_original/ \\")
    print("     --workspace=/home/pengliu/Private/panns_transfer_to_gtzan/")
    print("\n3. Verify results:")
    print("   python3 analyze_dataset.py")

if __name__ == '__main__':
    debug_remote_issue() 