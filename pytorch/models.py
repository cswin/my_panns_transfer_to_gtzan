from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import do_mixup, interpolate, pad_framewise_output
 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


class FeatureTransfer_Cnn14(Transfer_Cnn14):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module,
        but takes pre-computed features as input instead of waveform.
        """
        super(FeatureTransfer_Cnn14, self).__init__(sample_rate, window_size, 
            hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, time_steps, mel_bins)
        """
        # Skip spectrogram extraction since input is already features
        x = input.unsqueeze(1)  # Add channel dimension: (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.base.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
 
        return output_dict


class Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn6, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # Use mel_bins for batch normalization size
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Transfer_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn6 as a sub module.
        """
        super(Transfer_Cnn6, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        # Convert old model state dict to new model structure
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            # Skip loading the batch norm parameters since we've replaced it
            if not k.startswith('bn0.'):
                new_state_dict[k] = v
            
        self.base.load_state_dict(new_state_dict, strict=False)
        print("Pretrained model loaded with some missing keys (this is expected for transfer learning)")

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


class FeatureTransfer_Cnn6(Transfer_Cnn6):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn6 as a sub module,
        but takes pre-computed features as input instead of waveform.
        """
        super(FeatureTransfer_Cnn6, self).__init__(sample_rate, window_size, 
            hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)
            
        # Store mel_bins for later use
        self.mel_bins = mel_bins
        
        # Replace the batch norm layer with correct dimensions
        self._replace_batch_norm()
        
    def _replace_batch_norm(self):
        """Replace the batch normalization layer with correct dimensions."""
        # Create a new batch norm layer with the correct dimensions
        self.base.bn0 = nn.BatchNorm2d(self.mel_bins)
        
        # Initialize the batch norm layer
        init_bn(self.base.bn0)
        
        # Reset running statistics
        self.base.bn0.running_mean = torch.zeros(self.mel_bins)
        self.base.bn0.running_var = torch.ones(self.mel_bins)
        self.base.bn0.num_batches_tracked = torch.tensor(0)
        
    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        # Convert old model state dict to new model structure
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            # Skip loading ANY batch norm related parameters
            if not k.startswith('bn0.'):
                new_state_dict[k] = v
                
        self.base.load_state_dict(new_state_dict, strict=False)
        print("Pretrained model loaded with some missing keys (this is expected for transfer learning)")
        
        # Re-replace the batch norm layer after loading pretrained weights
        self._replace_batch_norm()
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, time_steps, mel_bins)
        """
        # Skip spectrogram extraction since input is already features
        x = input.unsqueeze(1)  # Add channel dimension: (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.base.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
 
        return output_dict


class AffectiveCnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_visual_system=True):
        """
        AffectiveCnn6: A model that uses Cnn6's convolutional layers as a frozen visual system
        and adds 3 fully connected layers as a trainable affective (emotion) system.
        
        Args:
            freeze_visual_system: If True, freezes the convolutional layers (visual system)
        """
        super(AffectiveCnn6, self).__init__()
        
        # Create the base Cnn6 model (will be used for visual system)
        audioset_classes_num = 527  # Original AudioSet classes
        self.visual_system = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
                                 fmax, audioset_classes_num)
        
        # Remove the original classification layers from visual system
        # We'll keep everything up to the embedding layer
        
        # Define the 3-layer affective system
        # Input dimension is 512 (from conv layers after pooling)
        self.affective_fc1 = nn.Linear(512, 256, bias=True)
        self.affective_fc2 = nn.Linear(256, 128, bias=True)
        self.affective_fc3 = nn.Linear(128, classes_num, bias=True)
        
        # Dropout layers for the affective system
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.5)
        
        if freeze_visual_system:
            # Freeze all parameters in the visual system (convolutional layers)
            for param in self.visual_system.parameters():
                param.requires_grad = False
            print("Visual system (convolutional layers) frozen - only affective system will be trained")
        
        self.init_affective_weights()
    
    def init_affective_weights(self):
        """Initialize weights for the affective system layers"""
        init_layer(self.affective_fc1)
        init_layer(self.affective_fc2)
        init_layer(self.affective_fc3)
    
    def load_visual_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights for the visual system (Cnn6 layers)"""
        checkpoint = torch.load(pretrained_checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            new_state_dict[k] = v
            
        self.visual_system.load_state_dict(new_state_dict, strict=False)
        print("Pretrained visual system loaded")
    
    def forward(self, input, mixup_lambda=None):
        """
        Forward pass through visual system (frozen) and affective system (trainable)
        Input: (batch_size, data_length) for raw audio or (batch_size, time_steps, mel_bins) for features
        """
        # Extract features using the visual system (convolutional layers)
        # We'll manually run through the visual system layers to get the embedding
        
        # Spectrogram and mel extraction
        x = self.visual_system.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.visual_system.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.visual_system.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.visual_system.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Pass through convolutional blocks (visual system - frozen)
        x = self.visual_system.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # Now pass through the affective system (3 FC layers - trainable)
        x = self.dropout1(x)
        x = F.relu(self.affective_fc1(x))
        
        x = self.dropout2(x)
        x = F.relu(self.affective_fc2(x))
        
        x = self.dropout3(x)
        affective_embedding = x.clone()  # Save embedding before final layer
        
        clipwise_output = torch.log_softmax(self.affective_fc3(x), dim=-1)
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': affective_embedding,
            'visual_features': x1 + x2  # Features from visual system
        }

        return output_dict


class FeatureAffectiveCnn6(AffectiveCnn6):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_visual_system=True):
        """
        FeatureAffectiveCnn6: Similar to AffectiveCnn6 but takes pre-computed features as input
        instead of raw waveform.
        """
        super(FeatureAffectiveCnn6, self).__init__(sample_rate, window_size, 
            hop_size, mel_bins, fmin, fmax, classes_num, freeze_visual_system)
            
        # Store mel_bins for later use
        self.mel_bins = mel_bins
        
        # Replace the batch norm layer with correct dimensions
        self._replace_batch_norm()
        
    def _replace_batch_norm(self):
        """Replace the batch normalization layer with correct dimensions."""
        # Create a new batch norm layer with the correct dimensions
        self.visual_system.bn0 = nn.BatchNorm2d(self.mel_bins)
        
        # Initialize the batch norm layer
        init_bn(self.visual_system.bn0)
        
        # Reset running statistics
        self.visual_system.bn0.running_mean = torch.zeros(self.mel_bins)
        self.visual_system.bn0.running_var = torch.ones(self.mel_bins)
        self.visual_system.bn0.num_batches_tracked = torch.tensor(0)
        
    def load_visual_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights for the visual system, handling batch norm replacement"""
        checkpoint = torch.load(pretrained_checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            # Skip loading ANY batch norm related parameters
            if not k.startswith('bn0.'):
                new_state_dict[k] = v
                
        self.visual_system.load_state_dict(new_state_dict, strict=False)
        print("Pretrained visual system loaded with batch norm replacement")
        
        # Re-replace the batch norm layer after loading pretrained weights
        self._replace_batch_norm()
    
    def forward(self, input, mixup_lambda=None):
        """
        Forward pass for pre-computed features
        Input: (batch_size, time_steps, mel_bins)
        """
        # Skip spectrogram extraction since input is already features
        x = input.unsqueeze(1)  # Add channel dimension: (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.visual_system.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.visual_system.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Pass through convolutional blocks (visual system - frozen)
        x = self.visual_system.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # Now pass through the affective system (3 FC layers - trainable)
        x = self.dropout1(x)
        x = F.relu(self.affective_fc1(x))
        
        x = self.dropout2(x)
        x = F.relu(self.affective_fc2(x))
        
        x = self.dropout3(x)
        affective_embedding = x.clone()  # Save embedding before final layer
        
        clipwise_output = torch.log_softmax(self.affective_fc3(x), dim=-1)
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': affective_embedding,
            'visual_features': x1 + x2  # Features from visual system
        }

        return output_dict


class EmotionRegression_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base=True):
        """Regression model for valence and arousal prediction using pretrained Cnn14.
        
        Args:
            sample_rate: int, sample rate of audio
            window_size: int, window size for STFT
            hop_size: int, hop size for STFT
            mel_bins: int, number of mel bins
            fmin: int, minimum frequency
            fmax: int, maximum frequency
            freeze_base: bool, whether to freeze the pretrained base model
        """
        super(EmotionRegression_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        # Use pretrained Cnn14 as base
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Regression heads for valence and arousal
        self.fc_valence = nn.Linear(2048, 1, bias=True)
        self.fc_arousal = nn.Linear(2048, 1, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_valence)
        init_layer(self.fc_arousal)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights from AudioSet checkpoint."""
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Forward pass.
        
        Args:
            input: (batch_size, data_length), raw audio waveform
            mixup_lambda: float, mixup parameter
            
        Returns:
            output_dict: dict containing:
                'valence': (batch_size, 1), predicted valence values
                'arousal': (batch_size, 1), predicted arousal values
                'embedding': (batch_size, 2048), feature embeddings
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        # Predict valence and arousal (no activation function for regression)
        valence = self.fc_valence(embedding)
        arousal = self.fc_arousal(embedding)
        
        output_dict = {
            'valence': valence,
            'arousal': arousal,
            'embedding': embedding
        }
 
        return output_dict


class FeatureEmotionRegression_Cnn14(EmotionRegression_Cnn14):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base=True):
        """Feature-based regression model for valence and arousal prediction.
        
        Takes pre-computed mel-spectrogram features as input instead of raw waveform.
        """
        super(FeatureEmotionRegression_Cnn14, self).__init__(sample_rate, window_size, 
            hop_size, mel_bins, fmin, fmax, freeze_base)
        
        # Replace batch normalization for feature input
        self._replace_batch_norm()

    def _replace_batch_norm(self):
        """Replace batch normalization for feature input."""
        # Create a new batch norm layer with the correct number of channels
        self.base.bn0 = nn.BatchNorm2d(self.base.bn0.num_features)
        init_bn(self.base.bn0)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights, excluding spectrogram/logmel extractors."""
        checkpoint = torch.load(pretrained_checkpoint_path)
        pretrained_dict = checkpoint['model']
        
        # Remove spectrogram and logmel extractor weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if not k.startswith('spectrogram_extractor') 
                          and not k.startswith('logmel_extractor')}
        
        # Load remaining weights
        model_dict = self.base.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.base.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Forward pass with pre-computed features.
        
        Args:
            input: (batch_size, time_steps, mel_bins), pre-computed mel-spectrogram
            mixup_lambda: float, mixup parameter
            
        Returns:
            output_dict: dict containing valence, arousal, and embedding
        """
        # Input is already mel-spectrogram features: (batch_size, time_steps, mel_bins)
        # Add channel dimension to match CNN input format
        x = input.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
        
        # Apply batch norm exactly like the original CNN
        x = x.transpose(1, 3)   # (batch_size, mel_bins, time_steps, 1)
        x = self.base.bn0(x)
        x = x.transpose(1, 3)   # (batch_size, 1, time_steps, mel_bins)
        
        if self.training:
            x = self.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Forward through convolutional blocks
        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        # Global pooling
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        # Predict valence and arousal
        valence = self.fc_valence(embedding)
        arousal = self.fc_arousal(embedding)
        
        output_dict = {
            'valence': valence,
            'arousal': arousal,
            'embedding': embedding
        }
        
        return output_dict


class EmotionRegression_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base=True):
        """Regression model for valence and arousal prediction using pretrained Cnn6.
        
        Args:
            sample_rate: int, sample rate of audio
            window_size: int, window size for STFT
            hop_size: int, hop size for STFT
            mel_bins: int, number of mel bins
            fmin: int, minimum frequency
            fmax: int, maximum frequency
            freeze_base: bool, whether to freeze the pretrained base model
        """
        super(EmotionRegression_Cnn6, self).__init__()
        audioset_classes_num = 527
        
        # Use pretrained Cnn6 as base
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Regression heads for valence and arousal
        self.fc_valence = nn.Linear(512, 1, bias=True)
        self.fc_arousal = nn.Linear(512, 1, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_valence)
        init_layer(self.fc_arousal)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights from AudioSet checkpoint."""
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Forward pass.
        
        Args:
            input: (batch_size, data_length), raw audio waveform
            mixup_lambda: float, mixup parameter
            
        Returns:
            output_dict: dict containing:
                'valence': (batch_size, 1), predicted valence values
                'arousal': (batch_size, 1), predicted arousal values
                'embedding': (batch_size, 512), feature embeddings
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        # Predict valence and arousal (no activation function for regression)
        valence = self.fc_valence(embedding)
        arousal = self.fc_arousal(embedding)
        
        output_dict = {
            'valence': valence,
            'arousal': arousal,
            'embedding': embedding
        }
 
        return output_dict


class FeatureEmotionRegression_Cnn6(EmotionRegression_Cnn6):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base=True):
        """Feature-based regression model for valence and arousal prediction using Cnn6.
        
        Takes pre-computed mel-spectrogram features as input instead of raw waveform.
        """
        super(FeatureEmotionRegression_Cnn6, self).__init__(sample_rate, window_size, 
            hop_size, mel_bins, fmin, fmax, freeze_base)
        
        # Replace batch normalization for feature input
        self._replace_batch_norm()

    def _replace_batch_norm(self):
        """Replace batch normalization for feature input."""
        # Create a new batch norm layer with the correct number of channels
        self.base.bn0 = nn.BatchNorm2d(self.base.bn0.num_features)
        init_bn(self.base.bn0)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights, excluding spectrogram/logmel extractors."""
        checkpoint = torch.load(pretrained_checkpoint_path)
        pretrained_dict = checkpoint['model']
        
        # Remove spectrogram and logmel extractor weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if not k.startswith('spectrogram_extractor') 
                          and not k.startswith('logmel_extractor')}
        
        # Load remaining weights
        model_dict = self.base.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.base.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Forward pass with pre-computed features.
        
        Args:
            input: (batch_size, time_steps, mel_bins), pre-computed mel-spectrogram
            mixup_lambda: float, mixup parameter
            
        Returns:
            output_dict: dict containing valence, arousal, and embedding
        """
        # Handle DataParallel dimension issue and extra dimensions
        if input.dim() == 5:
            # DataParallel case: (num_gpus, batch_size, time_steps, mel_bins)
            # Reshape to (batch_size * num_gpus, time_steps, mel_bins)
            original_shape = input.shape
            input = input.view(-1, original_shape[-2], original_shape[-1])
        elif input.dim() == 4 and input.shape[1] == 1:
            # Extra dimension case: (batch_size, 1, time_steps, mel_bins)
            # Squeeze out the extra dimension
            input = input.squeeze(1)  # (batch_size, time_steps, mel_bins)
        elif input.dim() != 3:
            raise ValueError(f"Expected 3D, 4D (with dim 1 = 1), or 5D input, got {input.dim()}D input with shape {input.shape}")
        
        # Input is already mel-spectrogram features: (batch_size, time_steps, mel_bins)
        # Add channel dimension to match CNN input format
        x = input.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
        
        # Apply batch norm exactly like the original CNN
        x = x.transpose(1, 3)   # (batch_size, mel_bins, time_steps, 1)
        x = self.base.bn0(x)
        x = x.transpose(1, 3)   # (batch_size, 1, time_steps, mel_bins)
        
        if self.training:
            x = self.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Forward through convolutional blocks (Cnn6 has 6 conv blocks)
        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        # Global pooling
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        # Predict valence and arousal
        valence = self.fc_valence(embedding)
        arousal = self.fc_arousal(embedding)
        
        output_dict = {
            'valence': valence,
            'arousal': arousal,
            'embedding': embedding
        }
        
        return output_dict


class FeatureEmotionRegression_Cnn6_NewAffective(nn.Module):
    """
    Feature-based emotion regression with new affective system architecture.
    
    This model has the SAME structure as the LRM model but WITHOUT feedback connections.
    It serves as a fair comparison baseline with:
    - Visual System: Frozen CNN6 backbone (same as LRM)
    - Affective System: New separate valence/arousal pathways (same as LRM)
    - No Feedback: Pure feedforward processing (different from LRM)
    
    This allows fair comparison between:
    - Old affective system (FeatureEmotionRegression_Cnn6) 
    - New affective system without feedback (this model)
    - New affective system with feedback (FeatureEmotionRegression_Cnn6_LRM)
    """
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, freeze_base=True):
        """
        Initialize new affective system emotion regression model.
        
        Args:
            sample_rate: int, sample rate of audio
            window_size: int, window size for STFT
            hop_size: int, hop size for STFT  
            mel_bins: int, number of mel bins
            fmin: int, minimum frequency
            fmax: int, maximum frequency
            freeze_base: bool, whether to freeze the pretrained base model
        """
        super(FeatureEmotionRegression_Cnn6_NewAffective, self).__init__()
        
        # Visual system - same as LRM model
        self.base_model = FeatureEmotionRegression_Cnn6(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, freeze_base
        )
        
        # New affective system - same pathways as LRM model but no feedback
        # Separate pathways for valence and arousal with rich representations
        self.affective_valence = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.affective_arousal = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize the new affective pathways
        for pathway in [self.affective_valence, self.affective_arousal]:
            for layer in pathway:
                if isinstance(layer, nn.Linear):
                    init_layer(layer)
    
    def forward(self, input, mixup_lambda=None):
        """
        Forward pass - pure feedforward through visual + new affective systems.
        
        Args:
            input: (batch_size, time_steps, mel_bins) pre-computed features
            mixup_lambda: float, mixup parameter
            
        Returns:
            output_dict: dict containing valence, arousal, and embedding
        """
        # Handle DataParallel dimension issue and extra dimensions
        if input.dim() == 5:
            # DataParallel case: (num_gpus, batch_size, time_steps, mel_bins)
            # Reshape to (batch_size * num_gpus, time_steps, mel_bins)
            original_shape = input.shape
            input = input.view(-1, original_shape[-2], original_shape[-1])
        elif input.dim() == 4 and input.shape[1] == 1:
            # Extra dimension case: (batch_size, 1, time_steps, mel_bins)
            # Squeeze out the extra dimension
            input = input.squeeze(1)  # (batch_size, time_steps, mel_bins)
        elif input.dim() != 3:
            raise ValueError(f"Expected 3D, 4D (with dim 1 = 1), or 5D input, got {input.dim()}D input with shape {input.shape}")
        
        # Forward through visual system (frozen CNN6 backbone)
        # This extracts the 512D embedding from conv features
        visual_output = self.base_model(input, mixup_lambda)
        embedding = visual_output['embedding']  # 512D visual embedding
        
        # Forward through new affective system (separate pathways)
        valence = self.affective_valence(embedding)
        arousal = self.affective_arousal(embedding)
        
        output_dict = {
            'valence': valence,
            'arousal': arousal,
            'embedding': embedding  # Return visual embedding for compatibility
        }
        
        return output_dict
    
    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights for visual system."""
        self.base_model.load_from_pretrain(pretrained_checkpoint_path)


