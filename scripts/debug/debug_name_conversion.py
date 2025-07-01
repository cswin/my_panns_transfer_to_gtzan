#!/usr/bin/env python3

"""Debug script to test name conversion logic in forward hook."""

def test_name_conversion():
    # Test data from debug output
    mod_inputs_keys = [
        'from_affective_valence_128d_to_visual_system_base_conv_block4',
        'from_affective_arousal_128d_to_visual_system_base_conv_block4',
        'from_affective_valence_128d_to_visual_system_base_conv_block3',
        'from_affective_arousal_128d_to_visual_system_base_conv_block3'
    ]
    
    active_connections = [
        ('affective.valence.128d', 'visual_system.base.conv_block4'),
        ('affective.arousal.128d', 'visual_system.base.conv_block4'),
        ('affective.valence.128d', 'visual_system.base.conv_block3'),
        ('affective.arousal.128d', 'visual_system.base.conv_block3')
    ]
    
    print("=== Testing CORRECTED Name Conversion Logic ===")
    print(f"mod_inputs keys: {mod_inputs_keys}")
    print(f"active_connections: {active_connections}")
    print()
    
    for mod_name in mod_inputs_keys:
        print(f"Processing mod_name: {mod_name}")
        
        # Current logic in forward hook
        if not mod_name.startswith("from_") or "_to_" not in mod_name:
            print("  - Skipped: doesn't match pattern")
            continue
            
        # Extract source and target layer names
        parts = mod_name.split('_to_')
        if len(parts) != 2:
            print("  - Skipped: invalid split")
            continue
            
        source_part = parts[0].replace('from_', '')  # Remove 'from_' prefix
        target_part = parts[1]
        
        # CORRECTED logic: Convert underscores back to dots for layer names - but be selective
        # For source: affective_valence_128d -> affective.valence.128d
        source_layer = source_part.replace('_', '.')
        
        # For target: visual_system_base_conv_block4 -> visual_system.base.conv_block4
        # Only convert specific underscores to dots based on known patterns
        if target_part.startswith('visual_system_base_'):
            # visual_system_base_conv_block4 -> visual_system.base.conv_block4
            target_layer = target_part.replace('visual_system_base_', 'visual_system.base.')
        else:
            # Fallback: convert all underscores to dots
            target_layer = target_part.replace('_', '.')
        
        print(f"  - source_part: '{source_part}'")
        print(f"  - target_part: '{target_part}'")
        print(f"  - source_layer: '{source_layer}'")
        print(f"  - target_layer: '{target_layer}'")
        
        # Check if this matches any active connection
        matches = []
        for conn in active_connections:
            if (source_layer, target_layer) == conn:
                matches.append(conn)
        
        print(f"  - Matches: {matches}")
        print()

if __name__ == "__main__":
    test_name_conversion() 