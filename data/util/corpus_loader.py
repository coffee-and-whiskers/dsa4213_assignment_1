# loads corpus and prepares for training

import numpy as np
from collections import Counter, defaultdict
from typing import List, Set, Dict, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords

# get nltk stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("downloading nltk stopwords...")
    nltk.download('stopwords')


class CorpusLoader:
    # loads corpus for training
    
    def __init__(self, min_freq: int = 10, max_vocab_size: int = 10000, 
                 remove_stopwords: bool = True, remove_artifacts: bool = True,
                 keep_domain_words: bool = True):
        # setup params for loading
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.remove_stopwords = remove_stopwords
        self.remove_artifacts = remove_artifacts
        self.keep_domain_words = keep_domain_words
        
        # english stopwords
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
        
        # keep ai domain words
        self.domain_words = {
            # AI/ML terms
            'ai', 'ml', 'model', 'models', 'data', 'training', 'neural', 'network',
            'deep', 'learning', 'algorithm', 'artificial', 'intelligence', 'machine',
            
            # LLM specific
            'llm', 'gpt', 'chatgpt', 'claude', 'bard', 'gemini', 'llama',
            'prompt', 'prompts', 'prompting', 'token', 'tokens', 'embedding',
            
            # Technical terms
            'code', 'python', 'api', 'github', 'software', 'programming',
            'function', 'variable', 'class', 'method', 'parameter',
            
            # Platform terms
            'reddit', 'post', 'comment', 'thread', 'user', 'community',
            
            # General tech
            'computer', 'technology', 'digital', 'online', 'internet',
            'system', 'application', 'platform', 'tool', 'service'
        } if keep_domain_words else set()
        
        # Data attributes
        self.documents = []
        self.vocabulary = set()
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.doc_freq = Counter()
        self.total_docs = 0
        
    def load_corpus(self, corpus_path: str, verbose: bool = True) -> Dict:
        """
        Load pre-processed corpus and prepare for model training
        
        Args:
            corpus_path: Path to processed corpus file (from prepare_corpus.py)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training-ready data
        """
        if verbose:
            print("=" * 60)
            print("LOADING CORPUS FOR MODEL TRAINING")
            print("=" * 60)
            print(f"Corpus file: {corpus_path}")
            print(f"Configuration:")
            print(f"  Min frequency: {self.min_freq}")
            print(f"  Max vocab size: {self.max_vocab_size}")
            print(f"  Remove stopwords: {self.remove_stopwords}")
            print(f"  Keep domain words: {self.keep_domain_words}")
            print()
        
        # Load pre-processed corpus
        documents = self._load_corpus_file(corpus_path, verbose)
        
        # Build initial vocabulary
        vocab, word_freq, doc_freq = self._build_vocabulary(documents, verbose)
        
        # Apply filters
        filtered_vocab = self._apply_filters(vocab, word_freq, verbose)
        
        # Filter documents with final vocabulary
        filtered_docs = self._filter_documents(documents, filtered_vocab, verbose)
        
        # Store results
        self.documents = filtered_docs
        self.vocabulary = filtered_vocab
        self.word2idx = {word: idx for idx, word in enumerate(sorted(filtered_vocab))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_freq = word_freq
        self.doc_freq = doc_freq
        self.total_docs = len(filtered_docs)
        
        if verbose:
            self._print_statistics()
        
        return {
            'documents': self.documents,
            'vocabulary': self.vocabulary,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'doc_freq': self.doc_freq,
            'vocab_size': len(self.vocabulary),
            'total_docs': self.total_docs
        }
    
    def _load_corpus_file(self, corpus_path: str, verbose: bool) -> List[List[str]]:
        """Load pre-processed corpus from file"""
        if verbose:
            print("Loading corpus file...")
        
        documents = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    # Corpus is already cleaned and lowercased
                    tokens = line.split()
                    
                    # Additional artifact removal if requested
                    if self.remove_artifacts:
                        tokens = self._clean_tokens(tokens)
                    
                    if len(tokens) > 2:  # Keep documents with at least 3 tokens
                        documents.append(tokens)
                
                if verbose and (line_num + 1) % 10000 == 0:
                    print(f"  Loaded {line_num + 1} lines...")
        
        if verbose:
            print(f"  Loaded {len(documents)} documents")
        
        return documents
    
    def _clean_tokens(self, tokens: List[str]) -> List[str]:
        """Clean tokens by removing punctuation artifacts"""
        cleaned = []
        for token in tokens:
            # Remove trailing punctuation
            token = re.sub(r'[.,;:!?\'"]+$', '', token)
            # Remove leading punctuation
            token = re.sub(r'^[.,;:!?\'"]+', '', token)
            # Skip pure punctuation or empty tokens
            if token and not re.match(r'^[.,;:!?\'"]+$', token):
                cleaned.append(token)
        return cleaned
    
    def _build_vocabulary(self, documents: List[List[str]], verbose: bool) -> Tuple:
        """Build vocabulary with frequency counts"""
        if verbose:
            print("\nBuilding vocabulary...")
        
        word_freq = Counter()
        doc_freq = Counter()
        
        for doc in documents:
            word_freq.update(doc)
            doc_freq.update(set(doc))  # Count unique words per document
        
        vocab = set(word_freq.keys())
        
        if verbose:
            print(f"  Initial vocabulary size: {len(vocab)}")
            print(f"  Total tokens: {sum(word_freq.values())}")
        
        return vocab, word_freq, doc_freq
    
    def _apply_filters(self, vocab: Set[str], word_freq: Counter, verbose: bool) -> Set[str]:
        """Apply frequency and stopword filters to vocabulary"""
        if verbose:
            print("\nApplying filters:")
            print(f"  Initial vocabulary size: {len(vocab)}")
        
        # 1. Minimum frequency filter
        vocab = {word for word in vocab if word_freq[word] >= self.min_freq}
        if verbose:
            print(f"  After min frequency ({self.min_freq}): {len(vocab)} words")
        
        # 2. Stopword filtering (with domain word preservation)
        if self.remove_stopwords:
            # Remove stopwords but keep domain-specific words
            words_to_remove = set()
            for word in vocab:
                if word in self.stopwords and word not in self.domain_words:
                    words_to_remove.add(word)
            
            if verbose and words_to_remove:
                print(f"  Removing {len(words_to_remove)} stopwords")
                print(f"    Examples: {', '.join(list(words_to_remove)[:15])}")
            
            vocab = vocab - words_to_remove
            
            if verbose:
                print(f"  After stopword removal: {len(vocab)} words")
                
                # Show preserved domain words that would have been removed
                preserved = [w for w in self.domain_words if w in vocab and w in self.stopwords]
                if preserved:
                    print(f"  Preserved domain words: {', '.join(preserved[:10])}")
        
        # 3. Vocabulary size limit
        if len(vocab) > self.max_vocab_size:
            # Sort by frequency but prioritize domain words
            def sort_key(word):
                # Domain words get a boost in ranking
                freq = word_freq[word]
                if word in self.domain_words:
                    freq *= 1.5  # Boost domain words
                return freq
            
            sorted_words = sorted(vocab, key=sort_key, reverse=True)
            vocab = set(sorted_words[:self.max_vocab_size])
            if verbose:
                print(f"  Limited to top {self.max_vocab_size} words")
        
        if verbose:
            print(f"  Final vocabulary size: {len(vocab)}")
            
            # Show statistics about the final vocabulary
            domain_in_vocab = len([w for w in vocab if w in self.domain_words])
            stopwords_in_vocab = len([w for w in vocab if w in self.stopwords])
            print(f"  Domain words retained: {domain_in_vocab}")
            print(f"  Stopwords retained: {stopwords_in_vocab}")
        
        return vocab
    
    def _filter_documents(self, documents: List[List[str]], 
                         vocabulary: Set[str], verbose: bool) -> List[List[str]]:
        """Filter documents to only include vocabulary words"""
        if verbose:
            print("\nFiltering documents...")
        
        filtered_docs = []
        for doc in documents:
            filtered_tokens = [token for token in doc if token in vocabulary]
            if len(filtered_tokens) > 2:  # Keep documents with at least 3 vocab words
                filtered_docs.append(filtered_tokens)
        
        if verbose:
            print(f"  Kept {len(filtered_docs)} documents")
        
        return filtered_docs
    
    def _print_statistics(self):
        """Print corpus statistics"""
        print("\n" + "=" * 60)
        print("CORPUS STATISTICS")
        print("=" * 60)
        print(f"Documents: {self.total_docs:,}")
        print(f"Vocabulary: {len(self.vocabulary):,}")
        print(f"Total tokens: {sum(len(doc) for doc in self.documents):,}")
        print(f"Avg doc length: {np.mean([len(doc) for doc in self.documents]):.1f}")
        
        # Top words
        top_words = sorted(self.vocabulary, key=lambda w: self.word_freq[w], reverse=True)[:10]
        print(f"\nTop 10 words by frequency:")
        for word in top_words:
            print(f"  {word}: {self.word_freq[word]:,} (in {self.doc_freq[word]} docs)")
        
        # Check for artifacts
        artifacts = []
        for word in list(self.vocabulary)[:1000]:  # Check sample
            if re.match(r'.*[.,;:!?]$', word) or re.match(r'^[.,;:!?].*', word):
                artifacts.append(word)
        
        if artifacts:
            print(f"\nWarning: Found {len(artifacts)} potential artifacts")
            print(f"  Examples: {', '.join(artifacts[:5])}")
        else:
            print(f"\nNo punctuation artifacts detected")
    
    def get_training_data(self) -> Dict:
        """Get preprocessed data for training"""
        return {
            'documents': self.documents,
            'vocabulary': self.vocabulary,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'doc_freq': self.doc_freq,
            'vocab_size': len(self.vocabulary),
            'total_docs': self.total_docs
        }


def detect_platform_bias(vocabulary: Set[str], word_freq: Counter, 
                         verbose: bool = True) -> Dict:
    """
    Detect potential platform-specific biases in vocabulary
    
    Args:
        vocabulary: Set of vocabulary words
        word_freq: Word frequency counter
        verbose: Whether to print results
        
    Returns:
        Dictionary with bias analysis results
    """
    # Platform-specific terms to check
    platform_terms = {
        'reddit': ['reddit', 'subreddit', 'upvote', 'downvote', 'karma', 'redditor'],
        'twitter': ['tweet', 'retweet', 'hashtag', 'twitter', 'follower'],
        'facebook': ['facebook', 'like', 'share', 'timeline', 'status'],
        'news': ['article', 'reporter', 'journalist', 'newspaper', 'headline'],
        'academic': ['paper', 'research', 'study', 'professor', 'university'],
        'tech': ['github', 'stackoverflow', 'repository', 'commit', 'pull'],
        'ai_platforms': ['openai', 'chatgpt', 'gpt', 'claude', 'anthropic', 'bard', 'gemini']
    }
    
    # AI/ML specific terms
    ai_terms = {
        'models': ['model', 'neural', 'network', 'deep', 'learning', 'training'],
        'frameworks': ['tensorflow', 'pytorch', 'keras', 'sklearn', 'scikit'],
        'techniques': ['embedding', 'vector', 'transformer', 'attention', 'bert'],
        'data': ['dataset', 'corpus', 'tokenize', 'preprocess', 'feature']
    }
    
    results = {
        'platform_presence': {},
        'ai_presence': {},
        'total_platform_terms': 0,
        'total_ai_terms': 0
    }
    
    # Check platform terms
    for platform, terms in platform_terms.items():
        found_terms = []
        total_freq = 0
        for term in terms:
            if term in vocabulary:
                found_terms.append(term)
                total_freq += word_freq[term]
        
        if found_terms:
            results['platform_presence'][platform] = {
                'terms': found_terms,
                'count': len(found_terms),
                'total_frequency': total_freq
            }
            results['total_platform_terms'] += len(found_terms)
    
    # Check AI terms
    for category, terms in ai_terms.items():
        found_terms = []
        total_freq = 0
        for term in terms:
            if term in vocabulary:
                found_terms.append(term)
                total_freq += word_freq[term]
        
        if found_terms:
            results['ai_presence'][category] = {
                'terms': found_terms,
                'count': len(found_terms),
                'total_frequency': total_freq
            }
            results['total_ai_terms'] += len(found_terms)
    
    if verbose:
        print("\n" + "=" * 60)
        print("BIAS DETECTION ANALYSIS")
        print("=" * 60)
        
        if results['platform_presence']:
            print("\nPlatform-specific terms detected:")
            for platform, data in results['platform_presence'].items():
                print(f"  {platform.upper()}:")
                print(f"    Terms: {', '.join(data['terms'])}")
                print(f"    Total frequency: {data['total_frequency']:,}")
        else:
            print("\nNo significant platform-specific bias detected")
        
        if results['ai_presence']:
            print("\nAI/ML terms detected:")
            for category, data in results['ai_presence'].items():
                print(f"  {category.upper()}:")
                print(f"    Terms: {', '.join(data['terms'][:5])}")  # Show first 5
                print(f"    Total frequency: {data['total_frequency']:,}")
        
        # Overall assessment
        total_vocab = len(vocabulary)
        platform_ratio = results['total_platform_terms'] / total_vocab * 100
        ai_ratio = results['total_ai_terms'] / total_vocab * 100
        
        print(f"\nOverall bias metrics:")
        print(f"  Platform terms: {results['total_platform_terms']} ({platform_ratio:.2f}% of vocab)")
        print(f"  AI/ML terms: {results['total_ai_terms']} ({ai_ratio:.2f}% of vocab)")
        
        if platform_ratio > 1.0:
            print("  ⚠ Warning: High platform-specific bias detected")
        if ai_ratio > 2.0:
            print("  ⚠ Warning: High AI/ML bias detected")
    
    return results