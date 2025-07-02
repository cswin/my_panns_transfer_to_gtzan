# LRM Architecture Refactoring

## Overview

This document describes the refactoring of the LongRangeModulation (LRM) architecture to eliminate code duplication and fix hook management issues.

## Problems in Original Architecture

### 1. Code Duplication
Both `LongRangeModulation` and `LongRangeModulationSingle` had duplicate methods:

```python
# Duplicated in both classes:
- hook_fn()
- remove_hooks() 
- clear_stored_activations()
- enable()/disable()
```

### 2. Hook Management Issues
```python
# LongRangeModulation.remove_hooks()
def remove_hooks(self):
    # Remove its own hooks
    for hook_item in self.targ_hooks:
        hook_item.remove()
    for hook in self.mod_hooks:
        hook.remove()
    
    # Then call children's remove_hooks() - POTENTIAL DOUBLE REMOVAL!
    for block in self.children():
        block.remove_hooks()  # Calls LongRangeModulationSingle.remove_hooks()
```

### 3. Inconsistent Internal Feedback
- Custom source names `'affective_valence_128d'` and `'affective_arousal_128d'` don't exist in model layers
- `source_module` becomes `None`, so no hooks are registered
- Internal feedback works inconsistently (sometimes works, sometimes doesn't)

## Refactored Architecture

### 1. LRMBase Class
Common functionality extracted to base class:

```python
class LRMBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.targ_hooks = []
        self.mod_hooks = []
        self.mod_inputs = {}
        self.disable_modulation_during_inference = False
    
    def hook_fn(self, module, input, output, name):
        """Store activation in mod_inputs dictionary."""
        self.mod_inputs[name] = output
    
    def remove_hooks(self):
        """Remove all hooks safely with error handling."""
        for hook_item in self.targ_hooks:
            try:
                if isinstance(hook_item, tuple):
                    hook_item[0].remove()
                else:
                    hook_item.remove()
            except Exception as e:
                print(f"Warning: Error removing target hook: {e}")
        
        for hook in self.mod_hooks:
            try:
                hook.remove()
            except Exception as e:
                print(f"Warning: Error removing mod hook: {e}")
        
        self.targ_hooks.clear()
        self.mod_hooks.clear()
    
    def clear_stored_activations(self):
        """Clear stored activations."""
        self.mod_inputs.clear()
    
    def enable(self):
        """Enable modulation."""
        self.disable_modulation_during_inference = False
    
    def disable(self):
        """Disable modulation."""
        self.disable_modulation_during_inference = True
```

### 2. RefactoredLongRangeModulationSingle
Individual modulation unit with clear responsibilities:

```python
class RefactoredLongRangeModulationSingle(LRMBase):
    def __init__(self, model, mod_target, mod_sources, img_size=224, ...):
        super().__init__()
        # Initialize individual modulation unit
        # Register hooks for actual model layers
        # Create ModBlocks for each source
        
    def forward_hook_target(self, module, input, output):
        """Modulate target output."""
        # Apply modulation logic
        # No duplication with parent class
```

### 3. RefactoredLongRangeModulation
Container class that manages multiple instances:

```python
class RefactoredLongRangeModulation(LRMBase):
    def __init__(self, model, mod_connections, img_size=224):
        super().__init__()
        # Create RefactoredLongRangeModulationSingle for each destination
        # No duplicate hook management
        
    def remove_hooks(self):
        """Remove all hooks from children (no duplication)."""
        for block in self.children():
            block.remove_hooks()
    
    def clear_stored_activations(self):
        """Clear stored activations from children."""
        for block in self.children():
            block.clear_stored_activations()
```

### 4. RefactoredEmotionModel
Model with proper internal feedback:

```python
class RefactoredEmotionModel(FeatureEmotionRegression_Cnn6_LRM):
    def __init__(self, ...):
        super().__init__(...)
        # Use refactored LRM
        self.lrm = RefactoredLongRangeModulation(self, mod_connections, img_size=224)
        # Register hooks directly on affective pathways
        self._register_affective_hooks()
    
    def _register_affective_hooks(self):
        """Register hooks to capture affective pathway activations."""
        valence_128d_layer = self.affective_valence[2:4]
        valence_128d_layer.register_forward_hook(self._valence_128d_hook)
        
        arousal_128d_layer = self.affective_arousal[2:4]
        arousal_128d_layer.register_forward_hook(self._arousal_128d_hook)
```

## Key Improvements

### 1. ✅ Eliminated Code Duplication
- **Before**: Duplicate methods in both classes
- **After**: Common functionality in `LRMBase`

### 2. ✅ Clear Separation of Responsibilities
- **LRMBase**: Common hook management
- **RefactoredLongRangeModulationSingle**: Individual modulation
- **RefactoredLongRangeModulation**: Container management

### 3. ✅ Proper Hook Management
- **Before**: Potential double hook removal
- **After**: Safe hook removal with error handling
- **Before**: Unclear hook ownership
- **After**: Clear hook ownership

### 4. ✅ Consistent Internal Feedback
- **Before**: Depends on custom source names that don't exist
- **After**: Direct hook registration on actual layers
- **Before**: Inconsistent behavior
- **After**: Predictable behavior

## Benefits

1. **Eliminates Potential Bugs**
   - No more double hook removal
   - No more inconsistent internal feedback
   - Safe error handling

2. **Improves Maintainability**
   - Single source of truth for common functionality
   - Clear class responsibilities
   - Easier to debug and extend

3. **Ensures Consistent Behavior**
   - Internal feedback works reliably
   - Predictable hook management
   - No memory leaks

4. **Makes Debugging Easier**
   - Clear separation of concerns
   - Proper error handling
   - Consistent behavior

## Migration Guide

### For Immediate Use (External Steering)
```python
# Use external steering signals (already working)
steering_signals = [
    {
        'source': 'affective_valence_128d',
        'activation': valence_128d,
        'strength': 1.0,
        'alpha': 1.0
    }
]
output = model(feature, steering_signals=steering_signals)
```

### For Refactored Internal Feedback
```python
# Use refactored model with proper internal feedback
model = RefactoredEmotionModel(...)
output = model(feature, forward_passes=2)  # Internal feedback works
```

## Testing

The refactored architecture can be tested using:

```bash
python scripts/debug/refactored_lrm_architecture.py --test-only
```

This will show the improvements and benefits of the refactored architecture.

## Conclusion

The refactored LRM architecture eliminates code duplication, fixes hook management issues, and ensures consistent internal feedback. This makes the code more maintainable, debuggable, and reliable.

For immediate results, use external steering signals. For a proper fix, implement the refactored architecture. 