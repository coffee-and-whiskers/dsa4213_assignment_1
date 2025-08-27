"""
Enhanced Embedding Space Visualization Module
Advanced t-SNE, UMAP, and PCA visualizations with comprehensive analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnhancedEmbeddingVisualizer:
    """
    Advanced visualization of word embeddings with multiple dimensionality reduction techniques
    """
    
    def __init__(self, embeddings_dir: str = 'embeddings'):
        """
        Initialize enhanced visualizer with improved defaults
        """
        self.embeddings_dir = embeddings_dir
        self.embeddings = None
        self.word2idx = None
        self.idx2word = None
        self.normalized_embeddings = None
        
    def load_embeddings(self, model_name: str, normalize: bool = True) -> bool:
        """
        Load embeddings with optional normalization
        
        Args:
            model_name: Name of model ('skipgram', 'ppmi_svd', 'glove')
            normalize: Whether to L2-normalize embeddings for cosine similarity
        """
        try:
            if model_name == 'skipgram':
                from gensim.models import KeyedVectors
                model_path = os.path.join(self.embeddings_dir, 'skipgram.bin')
                wv = KeyedVectors.load_word2vec_format(model_path, binary=True)
                
                vocab = wv.index_to_key
                self.embeddings = np.array([wv[word] for word in vocab])
                self.word2idx = {word: idx for idx, word in enumerate(vocab)}
                self.idx2word = {idx: word for word, idx in self.word2idx.items()}
                
            elif model_name in ['ppmi_svd', 'glove']:
                model_path = os.path.join(self.embeddings_dir, f'{model_name}.pkl')
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.embeddings = model_data['embeddings']
                self.word2idx = model_data['word2idx']
                self.idx2word = model_data['idx2word']
            
            else:
                print(f"unknown model: {model_name}")
                return False
            
            # Normalize embeddings for cosine similarity
            if normalize:
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                self.normalized_embeddings = self.embeddings / (norms + 1e-10)
            else:
                self.normalized_embeddings = self.embeddings
            
            print(f"loaded {model_name} embeddings: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"error loading {model_name}: {e}")
            return False
    
    def advanced_visualize(self,
                          ai_platforms: Optional[Dict[str, List[str]]] = None,
                          perception_terms: Optional[Dict[str, List[str]]] = None,
                          custom_terms: Optional[Dict[str, List[str]]] = None,
                          methods: List[str] = ['tsne', 'umap', 'pca'],
                          figsize: Tuple[int, int] = None,
                          save_path: Optional[str] = None,
                          show_connections: bool = False,
                          perplexity_values: List[int] = [30],
                          n_neighbors_values: List[int] = [15],
                          min_dist: float = 0.1):
        """
        Create advanced embedding visualizations with all three methods
        
        Args:
            ai_platforms: Dictionary of AI platform terms
            perception_terms: Dictionary of perception/quality terms  
            custom_terms: Additional custom term groups
            methods: List of methods to use
            figsize: Figure size
            save_path: Path to save visualization
            show_connections: Whether to show connections between related terms
            perplexity_values: List of perplexity values for t-SNE
            n_neighbors_values: List of n_neighbors values for UMAP
            min_dist: Minimum distance for UMAP
        """
        if self.embeddings is None:
            print("no embeddings loaded. use load_embeddings() first.")
            return
        
        # Default term sets if not provided
        if ai_platforms is None:
            ai_platforms = {
                'OpenAI': ['openai', 'gpt', 'chatgpt', 'gpt4', 'gpt5', 'o1', 'o3'],
                'Anthropic': ['anthropic', 'claude', 'opus', 'sonnet', 'haiku'],
                'Google': ['google', 'gemini', 'bard', 'palm'],
                'Meta': ['meta', 'llama', 'llama2', 'llama3', 'llama4'],
                'Mistral': ['mistral', 'mixtral', 'mistral7b'],
                'DeepSeek': ['deepseek', 'deepseekr1', 'deepseekv2', 'deepseekv3'],
                'Microsoft': ['microsoft', 'copilot', 'bing']
            }
        
        if perception_terms is None:
            perception_terms = {
                'Quality+': ['good', 'best', 'powerful', 'smart', 'advanced'],
                'Quality-': ['bad', 'worst', 'weak', 'dumb', 'basic'],
                'Cost': ['expensive', 'cheap', 'free', 'paid'],
                'Restrictions': ['censored', 'uncensored', 'filtered', 'restricted'],
                'Changes': ['nerfed', 'improved', 'degraded', 'updated']
            }
        
        # Combine all term groups
        all_term_groups = {}
        all_term_groups.update(ai_platforms)
        all_term_groups.update(perception_terms)
        if custom_terms:
            all_term_groups.update(custom_terms)
        
        # Collect terms and prepare data
        indices = []
        labels = []
        groups = []
        group_types = []  # To distinguish platforms from perceptions
        
        for group, terms in ai_platforms.items():
            for term in terms:
                if term in self.word2idx:
                    indices.append(self.word2idx[term])
                    labels.append(term)
                    groups.append(group)
                    group_types.append('platform')
        
        for group, terms in perception_terms.items():
            for term in terms:
                if term in self.word2idx:
                    indices.append(self.word2idx[term])
                    labels.append(term)
                    groups.append(group)
                    group_types.append('perception')
        
        if custom_terms:
            for group, terms in custom_terms.items():
                for term in terms:
                    if term in self.word2idx:
                        indices.append(self.word2idx[term])
                        labels.append(term)
                        groups.append(group)
                        group_types.append('custom')
        
        if not indices:
            print("No terms found in vocabulary")
            return
        
        print(f"visualizing {len(indices)} terms across {len(set(groups))} groups")
        selected_embeddings = self.normalized_embeddings[indices]
        
        # Determine figure size
        n_methods = len(methods)
        if figsize is None:
            figsize = (8 * n_methods, 7)
        
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        # Color and marker setup
        unique_groups = list(set(groups))
        n_colors = len(unique_groups)
        
        # Use a better color palette
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
        
        color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
        
        # Marker styles for different types
        marker_map = {
            'platform': 'o',  # Circle for platforms
            'perception': '^',  # Triangle for perceptions
            'custom': 's'  # Square for custom terms
        }
        
        # Store reduced coordinates for potential further analysis
        reduction_results = {}
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            
            if method.lower() == 'pca':
                print(f"running pca...")
                # Standardize before PCA
                scaler = StandardScaler()
                scaled_embeddings = scaler.fit_transform(selected_embeddings)
                
                reducer = PCA(n_components=2, random_state=42)
                reduced = reducer.fit_transform(scaled_embeddings)
                
                # Calculate explained variance
                explained_var = reducer.explained_variance_ratio_.sum()
                title = f'PCA\n(Explained Variance: {explained_var:.1%})'
                
                # Add loading vectors for top contributing dimensions
                if len(indices) < 50:  # Only show loadings for smaller visualizations
                    loadings = reducer.components_.T * np.sqrt(reducer.explained_variance_)
                    
            elif method.lower() == 'tsne':
                # Use best perplexity value
                perplexity = min(perplexity_values[0], len(indices) - 1)
                print(f"running t-sne (perplexity={perplexity})...")
                
                reducer = TSNE(n_components=2, 
                             perplexity=perplexity,
                             early_exaggeration=12,
                             learning_rate='auto',
                             max_iter=1500,
                             init='pca',
                             metric='cosine',
                             random_state=42,
                             n_jobs=-1)
                reduced = reducer.fit_transform(selected_embeddings)
                
                # Calculate KL divergence if available
                kl_div = reducer.kl_divergence_ if hasattr(reducer, 'kl_divergence_') else 'N/A'
                title = f't-SNE\n(Perplexity: {perplexity}, KL: {kl_div:.2f})'
                
            elif method.lower() == 'umap':
                try:
                    import umap
                    n_neighbors = min(n_neighbors_values[0], len(indices) - 1)
                    print(f"running umap (n_neighbors={n_neighbors})...")
                    
                    reducer = umap.UMAP(n_components=2,
                                       n_neighbors=n_neighbors,
                                       min_dist=min_dist,
                                       metric='cosine',
                                       random_state=42)
                    reduced = reducer.fit_transform(selected_embeddings)
                    title = f'UMAP\n(n_neighbors: {n_neighbors})'
                    
                except ImportError:
                    print("umap not installed. install with: pip install umap-learn")
                    ax.text(0.5, 0.5, 'umap not installed\npip install umap-learn',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title('umap')
                    ax.axis('off')
                    continue
            else:
                continue
            
            reduction_results[method] = reduced
            
            # Plot points with different markers and colors
            for group in unique_groups:
                group_mask = [g == group for g in groups]
                group_points = reduced[group_mask]
                group_types_subset = [gt for gt, m in zip(group_types, group_mask) if m]
                
                # Determine marker style
                marker = marker_map.get(group_types_subset[0], 'o')
                
                # Adjust size based on type
                size = 120 if group_types_subset[0] == 'platform' else 80
                
                ax.scatter(group_points[:, 0], group_points[:, 1],
                          c=[color_map[group]], 
                          label=group,
                          s=size, 
                          alpha=0.7,
                          edgecolors='white',
                          linewidth=1.5,
                          marker=marker)
            
            # Show connections between related terms if requested
            if show_connections and method.lower() != 'pca':
                self._add_connections(ax, reduced, labels, groups)
            
            # Add annotations with smart positioning
            texts = []
            for point, label in zip(reduced, labels):
                txt = ax.annotate(label, 
                                 (point[0], point[1]),
                                 fontsize=8,
                                 alpha=0.8)
                texts.append(txt)
            
            # Avoid overlapping labels (requires adjustText)
            try:
                from adjustText import adjust_text
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))
            except ImportError:
                pass  # adjustText not installed
            
            # Styling
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlabel('Component 1', fontsize=9)
            ax.set_ylabel('Component 2', fontsize=9)
            
            # Add legend with better positioning
            if idx == n_methods - 1:  # Only show legend on last plot
                ax.legend(bbox_to_anchor=(1.05, 1), 
                         loc='upper left',
                         fontsize=8,
                         framealpha=0.9,
                         title='Groups',
                         title_fontsize=9)
            else:
                ax.legend().set_visible(False)
            
            # Add subtle background coloring for regions
            if method.lower() in ['tsne', 'umap']:
                self._add_density_contours(ax, reduced, groups)
        
        # Main title
        plt.suptitle('Advanced Word Embedding Space Visualization', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        # Return reduction results for further analysis
        return reduction_results
    
    def _add_connections(self, ax, reduced, labels, groups):
        """Add connections between semantically related terms"""
        # Define connections (you can customize these)
        connections = [
            ('gpt', 'gpt4'), ('gpt4', 'gpt5'),
            ('claude', 'opus'), ('opus', 'sonnet'),
            ('llama', 'llama2'), ('llama2', 'llama3'),
            ('good', 'best'), ('bad', 'worst'),
            ('cheap', 'free'), ('expensive', 'paid')
        ]
        
        for term1, term2 in connections:
            if term1 in labels and term2 in labels:
                idx1 = labels.index(term1)
                idx2 = labels.index(term2)
                ax.plot([reduced[idx1, 0], reduced[idx2, 0]],
                       [reduced[idx1, 1], reduced[idx2, 1]],
                       'gray', alpha=0.2, linewidth=0.5, linestyle='--')
    
    def _add_density_contours(self, ax, reduced, groups):
        """Add density contours to show clustering regions"""
        try:
            from scipy.stats import gaussian_kde
            
            # Create density estimation for major groups
            major_groups = ['OpenAI', 'Anthropic', 'Google', 'Meta']
            
            for group in major_groups:
                if group in groups:
                    group_mask = [g == group for g in groups]
                    if sum(group_mask) > 2:  # Need at least 3 points
                        group_points = reduced[group_mask]
                        
                        # Estimate density
                        kde = gaussian_kde(group_points.T)
                        
                        # Create grid
                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                           np.linspace(y_min, y_max, 50))
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        
                        # Calculate density
                        density = kde(positions).reshape(xx.shape)
                        
                        # Add contour
                        ax.contour(xx, yy, density, levels=1, 
                                  colors='gray', alpha=0.2, linewidths=1)
        except:
            pass  # Skip if scipy not available
    
    def comparative_analysis(self,
                            models: List[str],
                            test_terms: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = None):
        """
        Compare multiple embedding models using all three methods
        
        Args:
            models: List of model names to compare
            test_terms: Specific terms to focus on
            figsize: Figure size for the comparison plot
        """
        if test_terms is None:
            test_terms = ['gpt', 'claude', 'gemini', 'llama', 'good', 'bad', 
                         'expensive', 'cheap', 'censored', 'uncensored']
        
        n_models = len(models)
        n_methods = 3  # PCA, t-SNE, UMAP
        
        if figsize is None:
            figsize = (6 * n_methods, 5 * n_models)
        
        fig, axes = plt.subplots(n_models, n_methods, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, model_name in enumerate(models):
            print(f"\nProcessing model: {model_name}")
            
            if not self.load_embeddings(model_name):
                for method_idx in range(n_methods):
                    axes[model_idx, method_idx].text(0.5, 0.5, 
                                                    f'{model_name}\nnot found',
                                                    ha='center', va='center',
                                                    transform=axes[model_idx, method_idx].transAxes)
                continue
            
            # Get indices for test terms
            indices = []
            labels = []
            for term in test_terms:
                if term in self.word2idx:
                    indices.append(self.word2idx[term])
                    labels.append(term)
            
            if not indices:
                continue
            
            selected_embeddings = self.normalized_embeddings[indices]
            
            # Apply each reduction method
            methods = ['PCA', 't-SNE', 'UMAP']
            for method_idx, method in enumerate(methods):
                ax = axes[model_idx, method_idx]
                
                if method == 'PCA':
                    reducer = PCA(n_components=2, random_state=42)
                    reduced = reducer.fit_transform(selected_embeddings)
                    title = f'{model_name.upper()} - PCA ({reducer.explained_variance_ratio_.sum():.1%})'
                    
                elif method == 't-SNE':
                    perplexity = min(10, len(indices) - 1)
                    reducer = TSNE(n_components=2, perplexity=perplexity,
                                 random_state=42, n_iter=1000)
                    reduced = reducer.fit_transform(selected_embeddings)
                    title = f'{model_name.upper()} - t-SNE'
                    
                else:  # UMAP
                    try:
                        import umap
                        n_neighbors = min(5, len(indices) - 1)
                        reducer = umap.UMAP(n_components=2, 
                                          n_neighbors=n_neighbors,
                                          random_state=42)
                        reduced = reducer.fit_transform(selected_embeddings)
                        title = f'{model_name.upper()} - UMAP'
                    except ImportError:
                        ax.text(0.5, 0.5, 'UMAP not installed',
                               ha='center', va='center', transform=ax.transAxes)
                        continue
                
                # Color by term type
                colors = []
                for label in labels:
                    if label in ['gpt', 'claude', 'gemini', 'llama']:
                        colors.append('blue')
                    elif label in ['good', 'bad', 'expensive', 'cheap']:
                        colors.append('green')
                    else:
                        colors.append('red')
                
                ax.scatter(reduced[:, 0], reduced[:, 1], 
                          c=colors, s=100, alpha=0.7)
                
                for point, label in zip(reduced, labels):
                    ax.annotate(label, (point[0], point[1]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, alpha=0.9)
                
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Model Embedding Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def similarity_heatmap(self,
                          ai_platforms: Dict[str, List[str]],
                          perception_terms: Dict[str, List[str]],
                          method: str = 'average',
                          save_path: Optional[str] = None):
        """
        Create heatmap showing similarities between platforms and perception terms
        
        Args:
            ai_platforms: Dictionary of AI platform terms
            perception_terms: Dictionary of perception terms
            method: How to aggregate multiple terms ('average', 'max', 'min')
            save_path: Path to save heatmap
        """
        import pandas as pd
        
        # Calculate platform representations
        platform_vectors = {}
        for platform, terms in ai_platforms.items():
            vectors = []
            for term in terms:
                if term in self.word2idx:
                    vectors.append(self.normalized_embeddings[self.word2idx[term]])
            
            if vectors:
                if method == 'average':
                    platform_vectors[platform] = np.mean(vectors, axis=0)
                elif method == 'max':
                    platform_vectors[platform] = np.max(vectors, axis=0)
                elif method == 'min':
                    platform_vectors[platform] = np.min(vectors, axis=0)
        
        # Calculate perception term representations
        perception_vectors = {}
        for category, terms in perception_terms.items():
            vectors = []
            for term in terms:
                if term in self.word2idx:
                    vectors.append(self.normalized_embeddings[self.word2idx[term]])
            
            if vectors:
                if method == 'average':
                    perception_vectors[category] = np.mean(vectors, axis=0)
                elif method == 'max':
                    perception_vectors[category] = np.max(vectors, axis=0)
                elif method == 'min':
                    perception_vectors[category] = np.min(vectors, axis=0)
        
        # Calculate similarity matrix
        similarity_matrix = []
        platforms_list = list(platform_vectors.keys())
        perceptions_list = list(perception_vectors.keys())
        
        for platform in platforms_list:
            row = []
            for perception in perceptions_list:
                sim = cosine_similarity(
                    platform_vectors[platform].reshape(1, -1),
                    perception_vectors[perception].reshape(1, -1)
                )[0, 0]
                row.append(sim)
            similarity_matrix.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(similarity_matrix, 
                         index=platforms_list,
                         columns=perceptions_list)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1,
                   cbar_kws={'label': 'Cosine Similarity'},
                   linewidths=0.5, linecolor='gray')
        
        plt.title('AI Platform - Perception Similarity Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Perception Categories', fontsize=12)
        plt.ylabel('AI Platforms', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
        
        return df


def main():
    """Enhanced example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced embedding visualization')
    parser.add_argument('--model', type=str, required=True,
                       choices=['skipgram', 'ppmi_svd', 'glove'],
                       help='Model to visualize')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings',
                       help='Directory containing embeddings')
    parser.add_argument('--methods', nargs='+', 
                       default=['tsne', 'umap', 'pca'],
                       choices=['tsne', 'umap', 'pca'],
                       help='Visualization methods')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--show-connections', action='store_true',
                       help='Show connections between related terms')
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate similarity heatmap')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = EnhancedEmbeddingVisualizer(args.embeddings_dir)
    
    # Define term groups
    ai_platforms = {
        'OpenAI': ['openai', 'gpt', 'chatgpt', 'gpt4', 'gpt5', 'o1', 'o3'],
        'Anthropic': ['anthropic', 'claude', 'opus', 'sonnet', 'haiku'],
        'Google': ['google', 'gemini', 'bard'],
        'Meta': ['meta', 'llama', 'llama2', 'llama3', 'llama4'],
        'Mistral': ['mistral', 'mixtral', 'mistral7b', 'mixtral8x7b'],
        'DeepSeek': ['deepseek', 'deepseekr1', 'deepseekv2', 'deepseekv3']
    }
    
    perception_terms = {
        'Cost': ['expensive', 'cheap', 'free', 'paid', 'subscription'],
        'Restrictions': ['censored', 'uncensored', 'filtered', 'restricted', 'jailbreak'],
        'Quality': ['good', 'bad', 'better', 'worse', 'best', 'worst'],
        'Capability': ['smart', 'dumb', 'powerful', 'weak', 'advanced'],
        'Changes': ['nerfed', 'buffed', 'improved', 'degraded', 'updated']
    }
    
    # Load model and visualize
    if viz.load_embeddings(args.model):
        # Main visualization
        viz.advanced_visualize(
            ai_platforms=ai_platforms,
            perception_terms=perception_terms,
            methods=args.methods,
            save_path=args.save,
            show_connections=args.show_connections
        )
        
        # Generate heatmap if requested
        if args.heatmap:
            viz.similarity_heatmap(
                ai_platforms=ai_platforms,
                perception_terms=perception_terms,
                save_path=args.save.replace('.png', '_heatmap.png') if args.save else None
            )
        
        # Compare models if requested
        if args.compare:
            viz.comparative_analysis(
                models=['skipgram', 'ppmi_svd', 'glove']
            )


if __name__ == "__main__":
    main()