"""
Melodious GNN Assembler - Graph Neural Network for Music Symbol Assembly

This module implements a Graph Attention Network (GAT) for assembling detected
musical symbols into a structurally correct musical score.

Architecture:
- 3-layer GAT with 8 attention heads per layer
- Node features: class embedding + spatial features + confidence
- Edge types: proximity, staff membership, vertical overlap
- Edge classification: stem→notehead, beam→note_group, slur→phrase, tie→sustained_note, no_relation

Based on: Hajic & Pecina, ICDAR 2017; Tuggener et al., ICPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Symbol classes (15-class prototype)
SYMBOL_CLASSES = [
    'notehead-full', 'notehead-half', 'notehead-whole',
    'clefG', 'clefF', 'clefC',
    'rest-8th', 'rest-quarter', 'rest-half', 'rest-whole',
    'accidentalSharp', 'accidentalFlat', 'accidentalNatural',
    'beam', 'stem'
]

# Relationship types for edge classification
RELATIONSHIP_TYPES = [
    'no_relation',      # 0: No relationship
    'stem_notehead',    # 1: Stem owns notehead (determines pitch/duration)
    'beam_notegroup',   # 2: Beam groups notes rhythmically
    'slur_phrase',      # 3: Slur spans phrase
    'tie_sustained'     # 4: Tie connects sustained notes
]


@dataclass
class Detection:
    """Single symbol detection from YOLO."""
    class_id: int
    class_name: str
    x: float  # normalized center x
    y: float  # normalized center y
    w: float  # normalized width
    h: float  # normalized height
    confidence: float
    staff_index: int = 0  # which staff this symbol belongs to


@dataclass
class Relationship:
    """Predicted relationship between two symbols."""
    source_idx: int
    target_idx: int
    relationship_type: str
    confidence: float


class SymbolEmbedding(nn.Module):
    """Learnable embedding for symbol classes."""
    
    def __init__(self, num_classes: int = 15, embed_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(class_ids)


class NodeFeatureEncoder(nn.Module):
    """Encodes node features from detection outputs.
    
    Node features (10-dimensional):
    - class_embedding (4d): learned embedding for symbol class
    - x_norm, y_norm (2d): normalized bounding box center
    - w_norm, h_norm (2d): normalized bounding box size
    - detection_confidence (1d): YOLO confidence score
    - staff_row_index (1d): which staff line this symbol belongs to
    """
    
    def __init__(self, num_classes: int = 15, embed_dim: int = 4):
        super().__init__()
        self.symbol_embedding = SymbolEmbedding(num_classes, embed_dim)
        self.feature_dim = embed_dim + 6  # embedding + 6 spatial/confidence features
        
    def forward(self, detections: List[Detection]) -> torch.Tensor:
        """
        Args:
            detections: List of Detection objects
            
        Returns:
            Tensor of shape (num_nodes, feature_dim)
        """
        if len(detections) == 0:
            return torch.zeros((0, self.feature_dim))
            
        features = []
        for det in detections:
            # Class embedding
            class_embed = self.symbol_embedding(
                torch.tensor([det.class_id], dtype=torch.long)
            ).squeeze(0)
            
            # Spatial features
            spatial = torch.tensor([
                det.x, det.y, det.w, det.h,
                det.confidence,
                float(det.staff_index)
            ], dtype=torch.float32)
            
            features.append(torch.cat([class_embed, spatial]))
            
        return torch.stack(features)


class GraphConstructor:
    """Constructs graph from YOLO detections.
    
    Edge types:
    - Proximity (e_prox): k-nearest neighbors within each staff
    - Staff membership (e_staff): symbols on same staff line
    - Vertical overlap (e_vert): bounding boxes overlap vertically
    """
    
    def __init__(self, k_neighbors: int = 5, staff_threshold: float = 0.1):
        self.k_neighbors = k_neighbors
        self.staff_threshold = staff_threshold
        
    def construct(self, detections: List[Detection], image_width: float = 1.0, image_height: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs graph edges from detections.
        
        Args:
            detections: List of Detection objects
            image_width: Image width for normalization
            image_height: Image height for normalization
            
        Returns:
            edge_index: Tensor of shape (2, num_edges) with source/target indices
            edge_type: Tensor of shape (num_edges,) with edge type indices
        """
        num_nodes = len(detections)
        if num_nodes < 2:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
        
        edges = []
        edge_types = []
        
        # Extract positions
        positions = np.array([[d.x, d.y] for d in detections])
        staff_indices = np.array([d.staff_index for d in detections])
        
        # 1. Proximity edges (k-nearest neighbors within same staff)
        for i in range(num_nodes):
            # Find nodes on same staff
            same_staff = np.where(staff_indices == staff_indices[i])[0]
            same_staff = same_staff[same_staff != i]  # exclude self
            
            if len(same_staff) > 0:
                # Compute distances
                dists = np.sqrt(
                    (positions[same_staff, 0] - positions[i, 0])**2 +
                    (positions[same_staff, 1] - positions[i, 1])**2
                )
                
                # Get k nearest
                k = min(self.k_neighbors, len(same_staff))
                nearest = same_staff[np.argsort(dists)[:k]]
                
                for j in nearest:
                    edges.append([i, j])
                    edge_types.append(0)  # proximity edge type
        
        # 2. Staff membership edges (all symbols on same staff connected)
        unique_staffs = np.unique(staff_indices)
        for staff_idx in unique_staffs:
            staff_nodes = np.where(staff_indices == staff_idx)[0]
            if len(staff_nodes) > 1:
                # Connect all pairs (this could be dense, so we limit it)
                for i in range(len(staff_nodes)):
                    for j in range(i+1, len(staff_nodes)):
                        edges.append([staff_nodes[i], staff_nodes[j]])
                        edges.append([staff_nodes[j], staff_nodes[i]])
                        edge_types.append(1)  # staff edge type
                        edge_types.append(1)
        
        # 3. Vertical overlap edges (for stem-notehead candidates)
        for i in range(num_nodes):
            det_i = detections[i]
            for j in range(num_nodes):
                if i == j:
                    continue
                    
                det_j = detections[j]
                
                # Check vertical overlap
                y_min_i = det_i.y - det_i.h / 2
                y_max_i = det_i.y + det_i.h / 2
                y_min_j = det_j.y - det_j.h / 2
                y_max_j = det_j.y + det_j.h / 2
                
                overlap = min(y_max_i, y_max_j) - max(y_min_i, y_min_j)
                min_height = min(det_i.h, det_j.h)
                
                if overlap > min_height * 0.3:  # 30% overlap threshold
                    edges.append([i, j])
                    edge_types.append(2)  # vertical overlap edge type
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_types = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_types


class GNNAssembler(nn.Module):
    """Graph Attention Network for music symbol assembly.
    
    Architecture:
    - 3-layer GAT with 8 attention heads per layer
    - Edge classification head for relationship prediction
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        node_feature_dim: int = 10,
        hidden_dim: int = 64,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        num_relationships: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_relationships = num_relationships
        
        # Node feature encoder
        self.node_encoder = NodeFeatureEncoder(num_classes, embed_dim=4)
        
        # Input projection
        self.input_proj = nn.Linear(self.node_encoder.feature_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            # First layer: hidden_dim -> hidden_dim * num_heads
            # Subsequent layers: hidden_dim * num_heads -> hidden_dim * num_heads
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            out_channels = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=True  # concatenate heads
                )
            )
        
        # Final projection to get node embeddings
        self.node_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Edge classification head
        # Input: concat(source_embedding, target_embedding, edge_type_embedding, relative_position)
        edge_type_embed_dim = 8
        self.edge_type_embedding = nn.Embedding(3, edge_type_embed_dim)  # 3 edge types
        
        relative_pos_dim = 4  # dx, dy, dw, dh
        edge_input_dim = hidden_dim * 2 + edge_type_embed_dim + relative_pos_dim
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relationships)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for edge classification.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: node features (num_nodes, feature_dim)
                - edge_index: edge indices (2, num_edges)
                - edge_type: edge type indices (num_edges,)
                - detections: list of Detection objects (for relative position)
                
        Returns:
            edge_logits: Tensor of shape (num_edges, num_relationships)
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        
        # Project input features
        x = self.input_proj(x)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)
        
        # Get node embeddings
        node_embeddings = self.node_proj(x)
        
        # Edge classification
        # Get embeddings for source and target nodes of each edge
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        
        src_embeds = node_embeddings[src_nodes]
        tgt_embeds = node_embeddings[tgt_nodes]
        
        # Edge type embeddings
        edge_type_embeds = self.edge_type_embedding(edge_type)
        
        # Relative position features
        if hasattr(data, 'detections') and data.detections is not None:
            rel_pos = self._compute_relative_positions(data.detections, src_nodes, tgt_nodes)
        else:
            rel_pos = torch.zeros((edge_index.shape[1], 4), device=x.device)
        
        # Concatenate all edge features
        edge_features = torch.cat([src_embeds, tgt_embeds, edge_type_embeds, rel_pos], dim=-1)
        
        # Classify edges
        edge_logits = self.edge_classifier(edge_features)
        
        return edge_logits
    
    def _compute_relative_positions(
        self,
        detections: List[Detection],
        src_nodes: torch.Tensor,
        tgt_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative position features between node pairs."""
        rel_positions = []
        
        for src_idx, tgt_idx in zip(src_nodes.cpu().numpy(), tgt_nodes.cpu().numpy()):
            src_det = detections[src_idx]
            tgt_det = detections[tgt_idx]
            
            rel_pos = [
                tgt_det.x - src_det.x,  # dx
                tgt_det.y - src_det.y,  # dy
                tgt_det.w - src_det.w,  # dw
                tgt_det.h - src_det.h   # dh
            ]
            rel_positions.append(rel_pos)
        
        return torch.tensor(rel_positions, dtype=torch.float32, device=src_nodes.device)
    
    def predict_relationships(
        self,
        detections: List[Detection],
        image_width: float = 1.0,
        image_height: float = 1.0,
        conf_threshold: float = 0.5
    ) -> List[Relationship]:
        """
        Predict relationships between detected symbols.
        
        Args:
            detections: List of Detection objects from YOLO
            image_width: Image width for normalization
            image_height: Image height for normalization
            conf_threshold: Minimum confidence for relationship prediction
            
        Returns:
            List of Relationship objects
        """
        if len(detections) < 2:
            return []
        
        # Construct graph
        graph_constructor = GraphConstructor()
        edge_index, edge_type = graph_constructor.construct(
            detections, image_width, image_height
        )
        
        if edge_index.shape[1] == 0:
            return []
        
        # Encode node features
        node_features = self.node_encoder(detections)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            detections=detections
        )
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            edge_logits = self.forward(data)
            edge_probs = F.softmax(edge_logits, dim=-1)
        
        # Extract predictions
        relationships = []
        for i, (src_idx, tgt_idx) in enumerate(edge_index.t().cpu().numpy()):
            probs = edge_probs[i].cpu().numpy()
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            if pred_class > 0 and confidence >= conf_threshold:  # Skip 'no_relation'
                relationships.append(Relationship(
                    source_idx=int(src_idx),
                    target_idx=int(tgt_idx),
                    relationship_type=RELATIONSHIP_TYPES[pred_class],
                    confidence=float(confidence)
                ))
        
        return relationships


class AssembledScore:
    """Represents an assembled musical score from detected symbols and relationships."""
    
    def __init__(self, detections: List[Detection], relationships: List[Relationship]):
        self.detections = detections
        self.relationships = relationships
        self.notes = []  # List of assembled notes
        self.staffs = {}  # Staff index -> list of symbols
        self._assemble()
    
    def _assemble(self):
        """Assemble notes from stem-notehead relationships."""
        # Group symbols by staff
        for i, det in enumerate(self.detections):
            if det.staff_index not in self.staffs:
                self.staffs[det.staff_index] = []
            self.staffs[det.staff_index].append(i)
        
        # Build adjacency list for relationships
        adjacency = {}  # source_idx -> [(target_idx, relationship_type)]
        for rel in self.relationships:
            if rel.source_idx not in adjacency:
                adjacency[rel.source_idx] = []
            adjacency[rel.source_idx].append((rel.target_idx, rel.relationship_type))
        
        # Find stem-notehead pairs
        stem_noteheads = {}  # stem_idx -> list of notehead indices
        for rel in self.relationships:
            if rel.relationship_type == 'stem_notehead':
                if rel.source_idx not in stem_noteheads:
                    stem_noteheads[rel.source_idx] = []
                stem_noteheads[rel.source_idx].append(rel.target_idx)
        
        # Assemble notes
        for stem_idx, notehead_indices in stem_noteheads.items():
            stem_det = self.detections[stem_idx]
            for notehead_idx in notehead_indices:
                notehead_det = self.detections[notehead_idx]
                
                # Determine note duration from notehead type and stem presence
                if notehead_det.class_name == 'notehead-whole':
                    duration = 'whole'
                elif notehead_det.class_name == 'notehead-half':
                    duration = 'half'
                else:  # notehead-full
                    # Check for beams
                    has_beam = any(
                        r.relationship_type == 'beam_notegroup'
                        for r in self.relationships
                        if r.source_idx == stem_idx or r.target_idx == stem_idx
                    )
                    if has_beam:
                        duration = 'eighth'  # Simplified - could be 16th, etc.
                    else:
                        duration = 'quarter'
                
                self.notes.append({
                    'notehead_idx': notehead_idx,
                    'stem_idx': stem_idx,
                    'pitch_position': notehead_det.y,  # Relative to staff
                    'duration': duration,
                    'staff_index': notehead_det.staff_index
                })
        
        # Handle noteheads without stems (whole notes, half notes)
        for i, det in enumerate(self.detections):
            if det.class_name in ['notehead-whole', 'notehead-half']:
                # Check if already part of a note
                if not any(n['notehead_idx'] == i for n in self.notes):
                    duration = 'whole' if det.class_name == 'notehead-whole' else 'half'
                    self.notes.append({
                        'notehead_idx': i,
                        'stem_idx': None,
                        'pitch_position': det.y,
                        'duration': duration,
                        'staff_index': det.staff_index
                    })


def train_gnn(
    model: GNNAssembler,
    train_data: List[Data],
    val_data: List[Data],
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = 'cuda'
) -> Dict:
    """
    Train the GNN assembler.
    
    Args:
        model: GNNAssembler model
        train_data: List of PyG Data objects for training
        val_data: List of PyG Data objects for validation
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Class weights for imbalanced edge types (most edges are 'no_relation')
    class_weights = torch.tensor([0.1, 2.0, 2.0, 1.5, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            
            edge_logits = model(data)
            loss = criterion(edge_logits, data.edge_label)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            preds = edge_logits.argmax(dim=-1)
            train_correct += (preds == data.edge_label).sum().item()
            train_total += data.edge_label.shape[0]
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                edge_logits = model(data)
                loss = criterion(edge_logits, data.edge_label)
                
                val_loss += loss.item()
                
                preds = edge_logits.argmax(dim=-1)
                val_correct += (preds == data.edge_label).sum().item()
                val_total += data.edge_label.shape[0]
        
        history['train_loss'].append(train_loss / len(train_data))
        history['val_loss'].append(val_loss / len(val_data))
        history['train_acc'].append(train_correct / train_total if train_total > 0 else 0)
        history['val_acc'].append(val_correct / val_total if val_total > 0 else 0)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")
    
    return history


if __name__ == "__main__":
    # Test GNN model
    print("Testing GNN Assembler...")
    
    # Create dummy detections
    detections = [
        Detection(0, 'notehead-full', 0.3, 0.5, 0.02, 0.03, 0.85, 0),
        Detection(14, 'stem', 0.31, 0.45, 0.01, 0.15, 0.92, 0),
        Detection(0, 'notehead-full', 0.4, 0.5, 0.02, 0.03, 0.88, 0),
        Detection(14, 'stem', 0.41, 0.45, 0.01, 0.15, 0.90, 0),
        Detection(13, 'beam', 0.35, 0.35, 0.15, 0.02, 0.78, 0),
        Detection(3, 'clefG', 0.1, 0.5, 0.05, 0.15, 0.95, 0),
    ]
    
    # Create model
    model = GNNAssembler(num_classes=15, hidden_dim=64)
    print(f"Created GNN Assembler with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test prediction
    relationships = model.predict_relationships(detections)
    print(f"\nPredicted {len(relationships)} relationships:")
    for rel in relationships:
        src_class = detections[rel.source_idx].class_name
        tgt_class = detections[rel.target_idx].class_name
        print(f"  {src_class} -> {tgt_class}: {rel.relationship_type} (conf: {rel.confidence:.2f})")
    
    # Assemble score
    score = AssembledScore(detections, relationships)
    print(f"\nAssembled {len(score.notes)} notes")
    for i, note in enumerate(score.notes):
        print(f"  Note {i+1}: duration={note['duration']}, staff={note['staff_index']}")