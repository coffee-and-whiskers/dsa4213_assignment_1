# ppmi-svd trainer

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import time
import pickle
import os
from typing import Dict, Tuple, Optional


class PPMISVDTrainer:
    # trains ppmi-svd model
    
    def __init__(self, output_dir: str = 'embeddings'):
        # setup output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        
    def train(self, training_data: Dict, embedding_dim: int = 300,
             window: int = 5, ppmi_shift: int = 1,
             use_context_weights: bool = True,
             verbose: bool = True) -> Dict:
        # train with given params
        if verbose:
            print("\ntraining ppmi-svd model")
            print(f"parameters:")
            print(f"  embedding dim: {embedding_dim}")
            print(f"  window: {window}")
            print(f"  ppmi shift: {ppmi_shift}")
            print(f"  context weighting: {use_context_weights}")
        
        start_time = time.time()
        
        # get data
        documents = training_data['documents']
        word2idx = training_data['word2idx']
        idx2word = training_data['idx2word']
        vocab_size = training_data['vocab_size']
        
        # build cooc matrix
        if verbose:
            print(f"\nbuilding co-occurrence matrix...")
        cooc_matrix = self._build_cooccurrence_matrix(
            documents, word2idx, vocab_size, window, use_context_weights, verbose
        )
        
        # calc ppmi
        if verbose:
            print(f"\ncalculating ppmi matrix...")
        ppmi_matrix, total_variance = self._calculate_ppmi(
            cooc_matrix, ppmi_shift, verbose
        )
        
        # do svd
        if verbose:
            print(f"\nperforming svd to {embedding_dim} dimensions...")
        embeddings, singular_values, variance_explained = self._perform_svd(
            ppmi_matrix, embedding_dim, total_variance, verbose
        )
        
        training_time = time.time() - start_time
        
        # store
        self.model = {
            'embeddings': embeddings,
            'word2idx': word2idx,
            'idx2word': idx2word,
            'singular_values': singular_values,
            'total_variance': total_variance,
            'variance_explained': variance_explained,
            'parameters': {
                'embedding_dim': embedding_dim,
                'window': window,
                'ppmi_shift': ppmi_shift
            }
        }
        
        # save
        model_path = os.path.join(self.output_dir, 'ppmi_svd.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if verbose:
            print(f"\ntraining completed in {training_time:.1f} seconds")
            print(f"model saved to: {model_path}")
            print(f"variance explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
            
            # check quality
            self._quality_check(embeddings, idx2word, verbose)
        
        return self.model
    
    def _build_cooccurrence_matrix(self, documents: list, word2idx: Dict,
                                   vocab_size: int, window: int,
                                   use_weights: bool, verbose: bool) -> sparse.csr_matrix:
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
                        
                        # Weight by distance (optional)
                        if use_weights:
                            weight = 1.0 / abs(i - j)
                        else:
                            weight = 1.0
                        
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
    
    def _calculate_ppmi(self, cooc_matrix: sparse.csr_matrix,
                       ppmi_shift: int, verbose: bool) -> Tuple[sparse.csr_matrix, float]:
        """
        Calculate Positive Pointwise Mutual Information matrix
        """
        # Calculate probabilities
        epsilon = 1e-10
        
        # Word probabilities (row sums)
        word_counts = np.array(cooc_matrix.sum(axis=1)).flatten()
        total_count = word_counts.sum()
        p_word = word_counts / total_count
        
        # Context probabilities (column sums)
        context_counts = np.array(cooc_matrix.sum(axis=0)).flatten()
        p_context = context_counts / total_count
        
        # Calculate total variance (Frobenius norm squared) before PPMI
        total_variance = sparse.linalg.norm(cooc_matrix, 'fro') ** 2
        
        if verbose:
            print(f"  Total co-occurrence variance: {total_variance:.2f}")
        
        # calc ppmi for non-zero entries
        rows, cols = cooc_matrix.nonzero()
        ppmi_data = []
        
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            
            # PMI calculation with numerical stability
            joint_prob = cooc_matrix[i, j] / total_count
            independent_prob = p_word[i] * p_context[j]
            
            # PMI with shift
            pmi = np.log2((joint_prob + epsilon) / (independent_prob + epsilon))
            ppmi = max(0, pmi - ppmi_shift)
            
            if ppmi > 0:
                ppmi_data.append((i, j, ppmi))
        
        # Create PPMI matrix
        ppmi_matrix = sparse.csr_matrix(
            ([d[2] for d in ppmi_data],
             ([d[0] for d in ppmi_data], [d[1] for d in ppmi_data])),
            shape=cooc_matrix.shape
        )
        
        if verbose:
            nnz_ppmi = ppmi_matrix.nnz
            reduction = 100 * (1 - nnz_ppmi / cooc_matrix.nnz)
            print(f"  PPMI non-zero entries: {nnz_ppmi:,}")
            print(f"  Sparsity reduction: {reduction:.1f}%")
        
        return ppmi_matrix, total_variance
    
    def _perform_svd(self, ppmi_matrix: sparse.csr_matrix,
                    embedding_dim: int, total_variance: float,
                    verbose: bool) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform SVD on PPMI matrix
        """
        # Determine actual dimension
        svd_dim = min(embedding_dim, min(ppmi_matrix.shape) - 1)
        
        if verbose and svd_dim < embedding_dim:
            print(f"  Note: Using {svd_dim} dimensions (matrix constraint)")
        
        # Perform truncated SVD
        U, s, Vt = svds(ppmi_matrix.astype(np.float64), k=svd_dim)
        
        # Sort by singular values (descending)
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]
        
        # Create embeddings (weighted by singular values)
        embeddings = U @ np.diag(np.sqrt(s))
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        
        # Calculate variance explained
        variance_captured = np.sum(s ** 2)
        variance_explained = variance_captured / total_variance
        
        if verbose:
            print(f"  Singular values range: [{s.min():.3f}, {s.max():.3f}]")
            print(f"  Variance captured: {variance_captured:.2f}")
        
        return embeddings, s, variance_explained
    
    def _quality_check(self, embeddings: np.ndarray, idx2word: Dict,
                      verbose: bool = True):
        """
        Perform quality check on embeddings similar to Skip-gram
        """
        if not verbose:
            return
        
        # Get word2idx mapping
        word2idx = {word: idx for idx, word in idx2word.items()}
        
        # First, analyze AI platform associations with quality/perception terms
        print("\n" + "=" * 50)
        print("AI PLATFORM PERCEPTION ANALYSIS (PPMI-SVD)")
        print("=" * 50)
        
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
            
            print(f"\n{platform_name.upper()} associations (terms: {', '.join(found_terms)}):")
            
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
        
        print("\n" + "=" * 50)
        print("SAMPLE NEAREST NEIGHBORS")
        print("=" * 50)
        
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
    
    def analyze_singular_values(self) -> Dict:
        """
        Analyze singular value distribution
        
        Returns:
            Analysis results
        """
        if not self.model:
            return {'error': 'Model not trained'}
        
        s = self.model['singular_values']
        
        # Calculate cumulative variance
        s_squared = s ** 2
        cumsum = np.cumsum(s_squared)
        cumsum_ratio = cumsum / cumsum[-1]
        
        # Find dimensions needed for different variance thresholds
        dims_for_variance = {}
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            dims_needed = np.argmax(cumsum_ratio >= threshold) + 1
            dims_for_variance[f'{int(threshold*100)}%'] = dims_needed
        
        return {
            'n_components': len(s),
            'singular_value_range': [float(s.min()), float(s.max())],
            'variance_explained': float(self.model['variance_explained']),
            'dims_for_variance': dims_for_variance,
            'singular_values': s.tolist()[:20]  # First 20 values
        }