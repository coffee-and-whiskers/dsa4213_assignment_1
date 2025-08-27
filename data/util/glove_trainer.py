"""
GloVe training module with efficient implementation
"""

import numpy as np
from scipy import sparse
import time
import pickle
import os
from typing import Dict, List, Tuple, Optional


class GloVeTrainer:
    """
    Memory-efficient GloVe trainer with convergence checking
    """
    
    def __init__(self, output_dir: str = 'embeddings'):
        """
        Initialize GloVe trainer
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        
    def train(self, training_data: Dict, embedding_dim: int = 100,
             window: int = 5, learning_rate: float = 0.05,
             max_iter: int = 50, x_max: int = 10, alpha: float = 0.75,
             tolerance: float = 1e-4, patience: int = 5,
             use_adagrad: bool = True, verbose: bool = True) -> Dict:
        """
        Train GloVe model with specified parameters
        
        Args:
            training_data: Dictionary with preprocessed data
            embedding_dim: Dimension of embeddings
            window: Context window size
            learning_rate: Initial learning rate
            max_iter: Maximum training iterations
            x_max: Cutoff for weighting function
            alpha: Exponent for weighting function
            tolerance: Convergence tolerance
            patience: Early stopping patience
            use_adagrad: Whether to use AdaGrad optimizer
            verbose: Whether to print progress
            
        Returns:
            Dictionary with model components
        """
        if verbose:
            print("\n" + "=" * 60)
            print("GLOVE MODEL TRAINING")
            print("=" * 60)
            print(f"Parameters:")
            print(f"  Embedding dimensions: {embedding_dim}")
            print(f"  Window size: {window}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Max iterations: {max_iter}")
            print(f"  X_max: {x_max}, Alpha: {alpha}")
            print(f"  Using AdaGrad: {use_adagrad}")
            print(f"  Convergence: tolerance={tolerance}, patience={patience}")
        
        start_time = time.time()
        
        # Extract data
        documents = training_data['documents']
        word2idx = training_data['word2idx']
        idx2word = training_data['idx2word']
        vocab_size = training_data['vocab_size']
        
        # Build co-occurrence matrix
        if verbose:
            print(f"\nBuilding co-occurrence matrix...")
        cooc_matrix = self._build_cooccurrence_matrix(
            documents, word2idx, vocab_size, window, verbose
        )
        
        # Train GloVe
        if verbose:
            print(f"\nTraining GloVe embeddings...")
        embeddings, losses, converged = self._train_glove(
            cooc_matrix, vocab_size, embedding_dim,
            learning_rate, max_iter, x_max, alpha,
            tolerance, patience, use_adagrad, verbose
        )
        
        training_time = time.time() - start_time
        
        # Store model
        self.model = {
            'embeddings': embeddings,
            'word2idx': word2idx,
            'idx2word': idx2word,
            'losses': losses,
            'converged': converged,
            'parameters': {
                'embedding_dim': embedding_dim,
                'window': window,
                'learning_rate': learning_rate,
                'x_max': x_max,
                'alpha': alpha
            }
        }
        
        # Save model
        model_path = os.path.join(self.output_dir, 'glove.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if verbose:
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Model saved to: {model_path}")
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Converged: {converged}")
            
            # Quality check
            self._quality_check(embeddings, idx2word, verbose)
        
        return self.model
    
    def _build_cooccurrence_matrix(self, documents: list, word2idx: Dict,
                                   vocab_size: int, window: int,
                                   verbose: bool) -> sparse.csr_matrix:
        """
        Build co-occurrence matrix from documents
        """
        rows, cols, data = [], [], []
        
        total_docs = len(documents)
        for doc_idx, doc in enumerate(documents):
            if verbose and (doc_idx + 1) % 5000 == 0:
                print(f"  Processing document {doc_idx + 1}/{total_docs}")
            
            for i, word in enumerate(doc):
                if word not in word2idx:
                    continue
                
                word_idx = word2idx[word]
                
                # Context window
                context_start = max(0, i - window)
                context_end = min(len(doc), i + window + 1)
                
                for j in range(context_start, context_end):
                    if i == j:
                        continue
                    
                    context_word = doc[j]
                    if context_word in word2idx:
                        context_idx = word2idx[context_word]
                        
                        # Weight by distance
                        weight = 1.0 / abs(i - j)
                        
                        rows.append(word_idx)
                        cols.append(context_idx)
                        data.append(weight)
        
        # Create sparse matrix
        cooc_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(vocab_size, vocab_size)
        ).tocsr()
        
        if verbose:
            nnz = cooc_matrix.nnz
            density = 100 * nnz / (vocab_size ** 2)
            print(f"  Co-occurrence matrix: {vocab_size}x{vocab_size}")
            print(f"  Non-zero entries: {nnz:,}")
            print(f"  Density: {density:.4f}%")
        
        return cooc_matrix
    
    def _weight_func(self, x: np.ndarray, x_max: float, alpha: float) -> np.ndarray:
        """
        GloVe weighting function
        """
        return np.minimum(1.0, (x / x_max) ** alpha)
    
    def _train_glove(self, cooc_matrix: sparse.csr_matrix, vocab_size: int,
                    embedding_dim: int, learning_rate: float, max_iter: int,
                    x_max: float, alpha: float, tolerance: float,
                    patience: int, use_adagrad: bool, verbose: bool) -> Tuple:
        """
        Train GloVe embeddings using iterative optimization
        """
        # Initialize embeddings
        np.random.seed(42)
        W = np.random.randn(vocab_size, embedding_dim) * 0.01
        W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.01
        b = np.zeros(vocab_size)
        b_tilde = np.zeros(vocab_size)
        
        # AdaGrad gradient history
        if use_adagrad:
            grad_sq_W = np.ones((vocab_size, embedding_dim)) * 1e-8
            grad_sq_W_tilde = np.ones((vocab_size, embedding_dim)) * 1e-8
            grad_sq_b = np.ones(vocab_size) * 1e-8
            grad_sq_b_tilde = np.ones(vocab_size) * 1e-8
        
        # Convert to COO for efficient iteration
        if not sparse.isspmatrix_coo(cooc_matrix):
            cooc_matrix = cooc_matrix.tocoo()
        
        rows = cooc_matrix.row
        cols = cooc_matrix.col
        cooc_data = cooc_matrix.data
        
        # Pre-compute weights and log co-occurrences
        weights = self._weight_func(cooc_data, x_max, alpha)
        log_cooc = np.log(cooc_data + 1e-10)
        
        n_pairs = len(cooc_data)
        
        # Training loop
        losses = []
        best_loss = float('inf')
        no_improvement = 0
        converged = False
        
        for epoch in range(max_iter):
            epoch_start = time.time()
            
            # Shuffle for SGD
            shuffle_idx = np.random.permutation(n_pairs)
            
            epoch_loss = 0.0
            
            # Process in mini-batches
            batch_size = min(2000, n_pairs)
            n_batches = (n_pairs + batch_size - 1) // batch_size
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_pairs)
                
                batch_indices = shuffle_idx[start_idx:end_idx]
                batch_rows = rows[batch_indices]
                batch_cols = cols[batch_indices]
                batch_weights = weights[batch_indices]
                batch_log_cooc = log_cooc[batch_indices]
                
                # Forward pass
                predictions = (
                    np.sum(W[batch_rows] * W_tilde[batch_cols], axis=1) +
                    b[batch_rows] + b_tilde[batch_cols]
                )
                
                # Compute errors
                errors = predictions - batch_log_cooc
                weighted_errors = batch_weights * errors
                epoch_loss += np.sum(batch_weights * errors ** 2)
                
                # Compute gradients
                grad_w = weighted_errors[:, np.newaxis] * W_tilde[batch_cols]
                grad_w_tilde = weighted_errors[:, np.newaxis] * W[batch_rows]
                grad_b = weighted_errors
                grad_b_tilde = weighted_errors
                
                # Gradient clipping
                grad_w = np.clip(grad_w, -5, 5)
                grad_w_tilde = np.clip(grad_w_tilde, -5, 5)
                grad_b = np.clip(grad_b, -5, 5)
                grad_b_tilde = np.clip(grad_b_tilde, -5, 5)
                
                # Update parameters
                unique_rows, unique_row_inverse = np.unique(batch_rows, return_inverse=True)
                unique_cols, unique_col_inverse = np.unique(batch_cols, return_inverse=True)
                
                # Aggregate and apply updates for W and b
                for i, row_idx in enumerate(unique_rows):
                    mask = (unique_row_inverse == i)
                    agg_grad_w = grad_w[mask].sum(axis=0)
                    agg_grad_b = grad_b[mask].sum()
                    
                    if use_adagrad:
                        grad_sq_W[row_idx] += agg_grad_w ** 2
                        grad_sq_b[row_idx] += agg_grad_b ** 2
                        
                        lr_w = learning_rate / np.sqrt(grad_sq_W[row_idx])
                        lr_b = learning_rate / np.sqrt(grad_sq_b[row_idx])
                    else:
                        lr_w = learning_rate
                        lr_b = learning_rate
                    
                    W[row_idx] -= lr_w * agg_grad_w
                    b[row_idx] -= lr_b * agg_grad_b
                
                # Aggregate and apply updates for W_tilde and b_tilde
                for i, col_idx in enumerate(unique_cols):
                    mask = (unique_col_inverse == i)
                    agg_grad_w_tilde = grad_w_tilde[mask].sum(axis=0)
                    agg_grad_b_tilde = grad_b_tilde[mask].sum()
                    
                    if use_adagrad:
                        grad_sq_W_tilde[col_idx] += agg_grad_w_tilde ** 2
                        grad_sq_b_tilde[col_idx] += agg_grad_b_tilde ** 2
                        
                        lr_w_tilde = learning_rate / np.sqrt(grad_sq_W_tilde[col_idx])
                        lr_b_tilde = learning_rate / np.sqrt(grad_sq_b_tilde[col_idx])
                    else:
                        lr_w_tilde = learning_rate
                        lr_b_tilde = learning_rate
                    
                    W_tilde[col_idx] -= lr_w_tilde * agg_grad_w_tilde
                    b_tilde[col_idx] -= lr_b_tilde * agg_grad_b_tilde
            
            # Average loss
            epoch_loss /= n_pairs
            losses.append(epoch_loss)
            
            # Check convergence
            if epoch_loss < best_loss - tolerance:
                best_loss = epoch_loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Print progress
            if verbose and ((epoch + 1) % 5 == 0 or no_improvement >= patience):
                epoch_time = time.time() - epoch_start
                print(f"  epoch {epoch + 1}/{max_iter}: loss = {epoch_loss:.6f}, "
                     f"time = {epoch_time:.1f}s")
            
            # Early stopping
            if no_improvement >= patience:
                if verbose:
                    print(f"  converged. no improvement for {patience} epochs.")
                converged = True
                break
        
        # Combine embeddings
        embeddings = (W + W_tilde) / 2
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        
        return embeddings, losses, converged
    
    def _quality_check(self, embeddings: np.ndarray, idx2word: Dict,
                      verbose: bool = True):
        """
        Perform quality check on embeddings similar to Skip-gram and PPMI-SVD
        """
        if not verbose:
            return
        
        # Get word2idx mapping
        word2idx = {word: idx for idx, word in idx2word.items()}
        
        # First, analyze AI platform associations with quality/perception terms
        print("\nai platform perception analysis (glove)")
        
        # Define AI platforms
        ai_platforms = {
            'openai': ['openai', 'gpt', 'chatgpt', 'gpt4', 'gpt5', 'o1', 'o3'],
            'anthropic': ['anthropic', 'claude', 'opus', 'sonnet', 'haiku'],
            'google': ['google', 'gemini'],
            'meta': ['meta', 'llama', 'llama2', 'llama3', 'llama4'],
            'mistral': ['mistral', 'mixtral', 'mistral7b', 'mixtral8x7b'],
            'deepseek': ['deepseek', 'deepseekr1', 'deepseekv2', 'deepseekv3']
        }
        
        # Define perception/quality terms
        perception_terms = {
            'cost': ['expensive', 'cheap', 'free', 'paid', 'subscription', 
                    'affordable', 'overpriced', 'worth'],
            
            'restrictions': ['censored', 'uncensored', 'filtered', 'restricted', 
                            'jailbreak', 'refuses', 'woke', 'lobotomized', 
                            'alignment', 'safety', 'guardrails'],
            
            'quality': ['good', 'bad', 'better', 'worse', 'best', 'worst',
                        'trash', 'goat', 'mid', 'peak'],
            
            'capability': ['smart', 'dumb', 'powerful', 'weak', 'advanced',
                        'basic', 'capable', 'limited'],
            
            'changes': ['nerfed', 'buffed', 'improved', 'degraded', 
                        'updated', 'broken', 'fixed'],
            
            'preference': ['prefer', 'hate', 'love', 'switched', 'abandoned',
                        'recommend', 'avoid']
        }
        
        # Analyze each platform
        for platform_name, platform_terms in ai_platforms.items():
            # Get platform vector (average of its terms)
            platform_vectors = []
            found_terms = []
            
            for term in platform_terms:
                if term in word2idx:
                    platform_vectors.append(embeddings[word2idx[term]])
                    found_terms.append(term)
            
            if not platform_vectors:
                continue
            
            # Average vector for the platform
            platform_vector = np.mean(platform_vectors, axis=0)
            
            print(f"\n{platform_name} associations (terms: {', '.join(found_terms)}):")
            
            # Check similarity with each perception category
            for category, terms in perception_terms.items():
                similarities = []
                for term in terms:
                    if term in word2idx:
                        term_vec = embeddings[word2idx[term]]
                        sim = np.dot(platform_vector, term_vec) / (
                            np.linalg.norm(platform_vector) * np.linalg.norm(term_vec)
                        )
                        similarities.append((term, sim))
                
                if similarities:
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_term = similarities[0]
                    avg_sim = np.mean([s[1] for s in similarities])
                    print(f"  {category:12s}: {top_term[0]}({top_term[1]:.3f}) avg={avg_sim:.3f}")
        
        print("\nsample nearest neighbors")
        
        # Test words covering different domains
        test_words = [
            # SANITY CHECK 
            'good', 'bad', 'work', 'works', 'use', 'using',
            
            # AI PLATFORM/COMPANY
            'openai', 'anthropic', 'google', 'meta', 'mistral',
            
            # SPECIFIC MODEL
            'gpt', 'gpt4', 'gpt5', 'o1', 'o1-mini',
            'claude', 'opus', 'sonnet', 'haiku',
            'gemini', 'flash', 'pro', 'turbo',
            'llama', 'llama3', 'mistral', 'mixtral', 'deepseek', 'qwen',
            
            # ACCESS/INTERFACE
            'chatgpt', 'copilot', 'cursor', 'perplexity',
            'api', 'playground', 'token', 'context', 'prompt',
            'huggingface', 'replicate',
            
            # CAPABILITY/FEATURE
            'answer', 'coding', 'debug', 'refactor', 'generate',
            'vision', 'image', 'multimodal', 'reading', 'summarizing',
            'reasoning', 'benchmark', 'score', 'performance',
            
            # COMMUNITY PERCEPTION
            'censored', 'uncensored', 'jailbreak', 'refuses',
            'hallucinate', 'accurate', 'reliable', 'biased',
            'expensive', 'cheap', 'free', 'paid',
            'fast', 'slow', 'better'
        ]
        
        for word in test_words:
            if word in word2idx:
                word_idx = word2idx[word]
                word_emb = embeddings[word_idx]
                
                # Compute similarities
                similarities = embeddings @ word_emb
                
                # Get top neighbors (excluding self)
                top_indices = np.argsort(similarities)[::-1][1:9]  # Get 8 neighbors
                
                neighbors = []
                for idx in top_indices:
                    neighbor_word = idx2word[idx]
                    sim = similarities[idx]
                    neighbors.append(f"{neighbor_word}({sim:.3f})")
                
                print(f"  {word:12s} -> {', '.join(neighbors)}")
            else:
                print(f"  {word:12s} -> [not in vocabulary]")
    
    def get_embeddings_matrix(self) -> np.ndarray:
        """
        Get embeddings matrix
        
        Returns:
            Embeddings matrix
        """
        if not self.model:
            raise ValueError("Model not trained")
        
        return self.model['embeddings']
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot training convergence
        
        Args:
            save_path: Optional path to save plot
        """
        if not self.model or 'losses' not in self.model:
            print("No loss history available")
            return
        
        import matplotlib.pyplot as plt
        
        losses = self.model['losses']
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GloVe Training Convergence')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.output_dir, 'glove_convergence.png'))
        
        plt.show()