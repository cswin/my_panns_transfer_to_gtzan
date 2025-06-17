# 🧠 LRM Emotion Regression Model Architecture

## Overview
The `FeatureEmotionRegression_Cnn6_LRM` model implements **Long Range Modulation (LRM)** with psychologically-motivated feedback connections for emotion regression on audio data.

## 🎯 Psychological Motivation

### **Valence (Positive/Negative Emotion)**
- **Function**: Affects how we **interpret semantic content**
- **Neural Target**: Higher-level convolutional features that capture semantic information
- **Rationale**: Positive/negative emotional state influences content interpretation

### **Arousal (Energy/Activation Level)**  
- **Function**: Affects **attention and alertness** to fine-grained details
- **Neural Target**: Lower-level convolutional features that capture acoustic details
- **Rationale**: High arousal increases sensitivity to acoustic nuances

---

## 🏗️ Model Architecture

### **Base Model: Cnn6**
```
Input (mel-spectrogram) → conv_block1 → conv_block2 → conv_block3 → conv_block4 → fc1 → {valence, arousal}
                             ↑           ↑           ↑           ↑
                        (64 channels) (128 ch)   (256 ch)   (512 ch)
```

### **LRM Feedback Connections**

#### **Valence Feedback (Semantic Modulation)**
```python
# Valence affects semantic interpretation
{'source': 'fc_valence', 'target': 'base.conv_block4'},  # Highest semantic level (512 ch)
{'source': 'fc_valence', 'target': 'base.conv_block3'},  # Mid semantic level (256 ch)
```

#### **Arousal Feedback (Attention Modulation)**
```python
# Arousal affects attention to acoustic details  
{'source': 'fc_arousal', 'target': 'base.conv_block2'},  # Low-level acoustics (128 ch)
{'source': 'fc_arousal', 'target': 'base.conv_block1'},  # Lowest-level acoustics (64 ch)
```

---

## ⚡ Forward Pass Flow

### **Pass 1: Initial Processing (No Modulation)**
```
Input → conv1 → conv2 → conv3 → conv4 → fc1 → {valence₁, arousal₁}
```

### **Pass 2+: Feedback Modulation**
```
Input → conv1⊗arousal₁ → conv2⊗arousal₁ → conv3⊗valence₁ → conv4⊗valence₁ → fc1 → {valence₂, arousal₂}
         ↑                ↑                ↑                ↑
    attention mod    attention mod    semantic mod     semantic mod
```

**Where `⊗` represents**: `output = features * (1.0 + modulation_signal)`

---

## 🔧 ModBlock Architecture

Each feedback connection uses a **ModBlock** to transform emotion predictions:

```python
ModBlock(
    source_dim=1,              # Single emotion value
    target_channels=X,         # Target conv layer channels
    
    # Internal transformation:
    Linear(1 → 64),           # Expand emotion prediction  
    ReLU(),
    Linear(64 → target_channels), # Match target layer
    Tanh()                    # Modulation range [-1, 1]
)
```

### **Specific ModBlocks Created**

| **ModBlock** | **Source** | **Target** | **Channels** | **Purpose** |
|--------------|------------|------------|--------------|-------------|
| `from_fc_valence_to_base_conv_block4` | Valence | Conv4 | 512 | Semantic modulation (highest) |
| `from_fc_valence_to_base_conv_block3` | Valence | Conv3 | 256 | Semantic modulation (mid) |
| `from_fc_arousal_to_base_conv_block2` | Arousal | Conv2 | 128 | Attention modulation (low) |
| `from_fc_arousal_to_base_conv_block1` | Arousal | Conv1 | 64  | Attention modulation (lowest) |

---

## 🎛️ Key Features

### **Hierarchical Modulation**
- **Valence**: Top-down semantic influence (conv4 → conv3)
- **Arousal**: Bottom-up attention influence (conv1 → conv2)

### **Iterative Refinement**
- Multiple forward passes allow progressive improvement
- Each pass benefits from previous emotion predictions

### **Tunable Modulation Strength**
- **Learnable scaling**: Separate `neg_scale` and `pos_scale` parameters
- **Runtime control**: Adjust strength during inference
- **Asymmetric modulation**: Different strengths for inhibition vs facilitation

### **Psychological Grounding**
- Based on emotion psychology research
- Valence affects interpretation, arousal affects attention

---

## 📊 Usage Example

```python
# Initialize model
model = FeatureEmotionRegression_Cnn6_LRM(
    sample_rate=32000,
    window_size=1024, 
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    forward_passes=2  # Number of feedback iterations
)

# Forward pass with feedback
output = model(mel_features, forward_passes=2)
# Returns: {'valence': tensor, 'arousal': tensor}

# Control feedback
model.enable_feedback()   # Enable LRM modulation
model.disable_feedback()  # Standard feedforward mode

# Control modulation strength
model.set_modulation_strength(0.5)      # Reduce modulation by half
model.set_modulation_strength(2.0)      # Double modulation strength
model.set_modulation_strength((0.5, 1.5)) # Asymmetric: reduce inhibition, increase facilitation

# Runtime strength control
output = model(mel_features, modulation_strength=1.5)  # Stronger feedback for this forward pass
```

---

## 🧪 Training Script

```bash
# Train with LRM feedback
bash run_emotion_feedback.sh

# Key parameters:
# - MODEL_TYPE="FeatureEmotionRegression_Cnn6_LRM"
# - FORWARD_PASSES=2
# - Separate workspace: workspaces/emotion_feedback
```

---

## 🔬 Research Basis

This implementation is inspired by:
1. **Long Range Modulation (LRM)** research on visual feedback
2. **Emotion psychology** on valence/arousal dimensions  
3. **Neuroscience** on top-down emotional modulation
4. **Audio processing** hierarchical feature representations

The key insight is that **emotional context should influence how we process sensory information**, implemented through learned modulation of intermediate neural representations. 