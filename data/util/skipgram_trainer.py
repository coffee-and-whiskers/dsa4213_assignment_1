# skipgram word2vec trainer

from gensim.models import Word2Vec
import numpy as np
import time
import os
from typing import Dict, Optional, List


class SkipGramTrainer:
    # trains skipgram model
    
    def __init__(self, output_dir: str = 'embeddings'):
        # setup output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        
    def train(self, training_data: Dict, embedding_dim: int = 100, 
             window: int = 5, epochs: int = 30, negative: int = 10,
             min_count: int = 5, learning_rate: float = 0.025,
             min_learning_rate: float = 0.0001, sample: float = 1e-4,
             verbose: bool = True) -> Word2Vec:
        # train model with given params
        if verbose:
            print("\ntraining skip-gram model")
            print(f"parameters:")
            print(f"  embedding dim: {embedding_dim}")
            print(f"  window: {window}")
            print(f"  epochs: {epochs}")
            print(f"  negative samples: {negative}")
            print(f"  learning rate: {learning_rate} -> {min_learning_rate}")
            print(f"  downsampling: {sample}")
        
        start_time = time.time()
        
        # get documents
        documents = training_data['documents']
        
        if verbose:
            print(f"\ntraining on {len(documents):,} documents...")
        
        # train
        self.model = Word2Vec(
            sentences=documents,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            sg=1,  # Skip-gram
            hs=0,  # No hierarchical softmax
            negative=negative,
            epochs=epochs,
            alpha=learning_rate,
            min_alpha=min_learning_rate,
            sample=sample,
            seed=42,
            workers=4
        )
        
        training_time = time.time() - start_time
        
        # save
        model_path = os.path.join(self.output_dir, 'skipgram.bin')
        self.model.wv.save_word2vec_format(model_path, binary=True)
        
        if verbose:
            print(f"\ntraining completed in {training_time:.1f} seconds")
            print(f"model saved to: {model_path}")
            print(f"vocabulary size: {len(self.model.wv):,}")
            
            # check quality
            self._quality_check(verbose)
        
        return self.model
    
    def _quality_check(self, verbose: bool = True):
        """
        Perform quality check on trained model
        """
        if not self.model or not verbose:
            return
        
        # First, analyze AI platform associations with quality/perception terms
        print("\n" + "=" * 50)
        print("AI PLATFORM PERCEPTION ANALYSIS")
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
                if term in self.model.wv:
                    platform_vectors.append(self.model.wv[term])
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
                    if term in self.model.wv:
                        sim = np.dot(platform_vector, self.model.wv[term]) / (
                            np.linalg.norm(platform_vector) * np.linalg.norm(self.model.wv[term])
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
    
    # AI PLATFORM/COMPANY BIAS (30%)
    'openai', 'anthropic', 'google', 'meta', 'mistral', 
    
    # SPECIFIC MODEL BIAS (25%)
    'gpt', 'gpt4', 'gpt5', 'o1', 'o1-mini', 'o4-mini' , 'o3-mini',       
    'claude', 'opus', 'sonnet', 'haiku',                    
    'gemini', 'flash', 'pro', 'turbo',             
    'llama', 'llama3', 'mistral', 'mixtral', 'distilled', 'deepseek', 'qwen',            
    
    # ACCESS/INTERFACE BIAS (15%)
    'chatgpt', 'copilot', 'cursor', 'perplexity',  'cc', # products/interfaces
    'api', 'playground', 'token', 'context', 'prompt', 'rate', 'limit', 
    'huggingface', 'replicate',
    
    # CAPABILITY/FEATURE BIAS (10%)
    'answer', 'coding', 'debug', 'refactor', 'generate',       
    'vision', 'image', 'multimodal', 'reading', 'summarizing',     
    'reasoning', 'benchmark', 'score', 'performance',         
    
    # COMMUNITY PERCEPTION BIAS (10%)
    'censored', 'uncensored', 'jailbreak', 'refuses',       
    'hallucinate', 'accurate', 'reliable', 'biased',        
    'expensive', 'cheap', 'free', 'paid' ,
    'fast', 'slow', 'better'          
    ]
        
        for word in test_words:
            if word in self.model.wv:
                try:
                    similar = self.model.wv.most_similar(word, topn=8)
                    neighbors = ', '.join([f"{w}({s:.3f})" for w, s in similar])
                    print(f"  {word:12s} -> {neighbors}")
                except:
                    print(f"  {word:12s} -> [error computing neighbors]")
            else:
                print(f"  {word:12s} -> [not in vocabulary]")
    
    
    def get_embeddings_matrix(self, word2idx: Dict) -> np.ndarray:
        """
        Get embeddings matrix for vocabulary
        
        Args:
            word2idx: Word to index mapping
            
        Returns:
            Embeddings matrix
        """
        if not self.model:
            raise ValueError("Model not trained")
        
        vocab_size = len(word2idx)
        embedding_dim = self.model.wv.vector_size
        embeddings = np.zeros((vocab_size, embedding_dim))
        
        for word, idx in word2idx.items():
            if word in self.model.wv:
                embeddings[idx] = self.model.wv[word]
            else:
                # Random initialization for OOV words
                embeddings[idx] = np.random.randn(embedding_dim) * 0.01
        
        return embeddings