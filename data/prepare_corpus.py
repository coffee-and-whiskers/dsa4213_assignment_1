# prepare corpus from raw json files for training

import json
import re
import os
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from collections import Counter
import argparse


class CorpusProcessor:
    # processes json data into clean corpus
    
    def __init__(self, raw_data_dir: str = 'raw', output_dir: str = 'embedding_ready_corpus'):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # track stats
        self.stats = {
            'files_processed': [],
            'total_raw_texts': 0,
            'total_cleaned_texts': 0,
            'vocabulary_size': 0,
            'total_words': 0
        }
    
    def clean_text_initial(self, text: str, preserve_sentences: bool = True) -> str:
        # clean urls, markdown, reddit stuff
        if not text:
            return ""
        
        # escape sequences
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        text = text.replace('\\"', '"').replace("\\'", "'")
        text = text.replace('\\`', '').replace('\x00', '')
        
        # code blocks
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`\n]*`', ' ', text)
        
        # urls and emails
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)
        
        # reddit stuff
        text = re.sub(r'/?u/[\w-]+', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'/?r/[\w-]+', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\[deleted\]|\[removed\]', ' ', text)
        
        # edit markers
        text = re.sub(r'(EDIT|Edit|edit):\s*.*?(?=\n|$)', ' ', text, flags=re.MULTILINE)
        
        # markdown
        text = re.sub(r'^#{1,6}\s+', ' ', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'\*{1,3}([^\*\n]+)\*{1,3}', r'\1', text)  # Bold/italic
        text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', ' ', text)  # Images
        text = re.sub(r'^>\s*', ' ', text, flags=re.MULTILINE)  # Quotes
        
        # metadata in parens
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', ' ', text)  # Years
        text = re.sub(r'\([^)]*\b\d+[kKmM]\b[^)]*\)', ' ', text)  # Numbers with k/M
        
        # empty brackets
        text = re.sub(r'\(\s*\)', ' ', text)
        text = re.sub(r'\[\s*\]', ' ', text)
        text = re.sub(r'\{\s*\}', ' ', text)
        
        # balance brackets
        text = self._balance_brackets(text)
        
        # technical stuff
        text = re.sub(r'[./\\]+[\w/\\.-]+\.\w+', ' ', text)  # File paths
        text = re.sub(r'\b\w*_\w+\b', ' ', text)  # Variable names
        text = re.sub(r'0x[0-9a-fA-F]+', ' ', text)  # Hex codes
        text = re.sub(r'\bv?\d+\.\d+(\.\d+)*\b', ' ', text)  # Version numbers
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle punctuation
        if preserve_sentences:
            # Keep sentence-ending punctuation
            text = re.sub(r'[^a-z0-9\s.,!?\'"-]', ' ', text)
        else:
            # Remove all punctuation
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        if preserve_sentences:
            text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Remove very short sentences
        if preserve_sentences:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.split()) > 2]
            text = '. '.join(sentences)
            if sentences and not text.endswith('.'):
                text += '.'
        
        # Final cleanup
        text = text.strip()
        text = re.sub(r'([.,!?]){2,}', r'\1', text)  # Remove multiple punctuation
        
        return text
    
    def _balance_brackets(self, text: str) -> str:
        """Remove unmatched brackets from text"""
        pairs = {'(': ')', '[': ']', '{': '}'}
        reverse_pairs = {v: k for k, v in pairs.items()}
        
        result = []
        stack = []
        
        for char in text:
            if char in pairs:
                stack.append((char, len(result)))
                result.append(char)
            elif char in reverse_pairs:
                if stack and stack[-1][0] == reverse_pairs[char]:
                    stack.pop()
                    result.append(char)
            else:
                result.append(char)
        
        # Remove unmatched opening brackets
        if stack:
            positions_to_remove = {pos for _, pos in stack}
            result = [char for i, char in enumerate(result) if i not in positions_to_remove]
        
        return ''.join(result)
    
    def clean_token(self, token: str) -> str:
        """Clean individual token - remove punctuation artifacts"""
        # Remove trailing punctuation
        token = re.sub(r'[.,;:!?\'"]+$', '', token)
        # Remove leading punctuation
        token = re.sub(r'^[.,;:!?\'"]+', '', token)
        # Remove brackets
        token = re.sub(r'^\[|\]$', '', token)
        token = re.sub(r'^\(|\)$', '', token)
        return token.lower()
    
    def filter_artifacts(self, tokens: List[str]) -> List[str]:
        """Filter out known artifacts and noise"""
        artifacts = {
            'redacted', 'redact', 'anonymized', 'deleted', 'removed',
            'edit:', 'update:', 'tldr', 'tl;dr', '[deleted]', '[removed]',
            'edit', 'update', 'edited', 'updated'
        }
        return [t for t in tokens if t.lower() not in artifacts and len(t) > 1]
    
    def process_json_files(self, json_files: Optional[List[str]] = None) -> List[str]:
        """
        Process raw JSON files into initially cleaned texts
        
        Args:
            json_files: List of JSON files to process (if None, process all in raw_data_dir)
            
        Returns:
            List of cleaned texts
        """
        if json_files is None:
            # Get all JSON files from raw data directory
            json_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.json')]
            json_files = [os.path.join(self.raw_data_dir, f) for f in json_files]
        
        all_cleaned_texts = []
        
        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"Warning: {json_file} not found, skipping...")
                continue
            
            print(f"\nProcessing {os.path.basename(json_file)}...")
            
            # Load data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each entry
            cleaned_texts = []
            for item in data:
                # Try multiple possible text fields
                text = item.get('comment_text') or item.get('text') or item.get('body', '')
                cleaned = self.clean_text_initial(text, preserve_sentences=True)
                
                if cleaned and len(cleaned.split()) >= 3:
                    cleaned_texts.append(cleaned)
            
            all_cleaned_texts.extend(cleaned_texts)
            
            # Update statistics
            self.stats['files_processed'].append({
                'file': os.path.basename(json_file),
                'original_count': len(data),
                'cleaned_count': len(cleaned_texts)
            })
            
            print(f"  Extracted {len(cleaned_texts)} texts from {len(data)} entries")
        
        self.stats['total_raw_texts'] = len(all_cleaned_texts)
        return all_cleaned_texts
    
    def apply_vocabulary_filters(self, texts: List[str], min_freq: int = 10, 
                                max_vocab: int = 10000, 
                                max_doc_freq_ratio: float = 0.5) -> Tuple[List[List[str]], Set[str]]:
        """
        Apply vocabulary filters and tokenize texts
        
        Args:
            texts: List of cleaned text strings
            min_freq: Minimum word frequency
            max_vocab: Maximum vocabulary size
            max_doc_freq_ratio: Maximum document frequency ratio (remove too common words)
            
        Returns:
            Tuple of (tokenized documents, vocabulary)
        """
        print("\nTokenizing and building vocabulary...")
        
        # Tokenize all texts
        documents = []
        word_freq = Counter()
        doc_freq = Counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(texts)} texts...")
            
            # Tokenize
            tokens = text.split()
            
            # Clean tokens
            tokens = [self.clean_token(t) for t in tokens]
            
            # Filter artifacts
            tokens = self.filter_artifacts(tokens)
            
            if len(tokens) > 2:
                documents.append(tokens)
                word_freq.update(tokens)
                doc_freq.update(set(tokens))
        
        print(f"Initial vocabulary size: {len(word_freq)}")
        
        # Apply frequency filters
        print("\nApplying vocabulary filters...")
        
        # 1. Minimum frequency filter
        vocab = {word for word, freq in word_freq.items() if freq >= min_freq}
        print(f"  After min frequency ({min_freq}): {len(vocab)} words")
        
        # 2. Maximum document frequency filter (remove too common words)
        total_docs = len(documents)
        max_doc_freq = max_doc_freq_ratio * total_docs
        high_freq_words = {word for word in vocab if doc_freq[word] > max_doc_freq}
        
        if high_freq_words:
            print(f"  Removing {len(high_freq_words)} words appearing in >{max_doc_freq:.0f} docs")
            print(f"    Examples: {', '.join(list(high_freq_words)[:10])}")
            vocab = vocab - high_freq_words
        
        # 3. Vocabulary size limit
        if len(vocab) > max_vocab:
            # Keep most frequent (but not TOO frequent) words
            sorted_words = sorted(vocab, key=lambda w: word_freq[w], reverse=True)
            vocab = set(sorted_words[:max_vocab])
            print(f"  Limited to top {max_vocab} words")
        
        print(f"Final vocabulary size: {len(vocab)}")
        
        # Filter documents with final vocabulary
        print("\nFiltering documents with final vocabulary...")
        filtered_docs = []
        for doc in documents:
            filtered_tokens = [t for t in doc if t in vocab]
            if len(filtered_tokens) > 2:
                filtered_docs.append(filtered_tokens)
        
        print(f"Final document count: {len(filtered_docs)}")
        
        # Update statistics
        self.stats['total_cleaned_texts'] = len(filtered_docs)
        self.stats['vocabulary_size'] = len(vocab)
        self.stats['total_words'] = sum(len(doc) for doc in filtered_docs)
        
        return filtered_docs, vocab
    
    def validate_corpus(self, documents: List[List[str]], vocab: Set[str]) -> Dict[str, Any]:
        """
        Validate the quality of the final corpus
        
        Args:
            documents: List of tokenized documents
            vocab: Final vocabulary
            
        Returns:
            Validation report
        """
        print("\nValidating corpus quality...")
        
        # Check for remaining artifacts
        punct_patterns = [
            r'.*[.,;:!?]$', r'^[.,;:!?].*',
            r'^\[.*\]$', r'^\(.*\)$'
        ]
        
        remaining_artifacts = []
        for word in list(vocab)[:1000]:  # Check sample
            for pattern in punct_patterns:
                if re.match(pattern, word):
                    remaining_artifacts.append(word)
                    break
        
        # Calculate statistics
        word_freq = Counter()
        for doc in documents:
            word_freq.update(doc)
        
        # Get most common content words
        common_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'
        }
        
        content_words = [(w, c) for w, c in word_freq.most_common(100) 
                        if w not in common_stopwords]
        
        validation = {
            'total_documents': len(documents),
            'vocabulary_size': len(vocab),
            'total_tokens': sum(len(doc) for doc in documents),
            'avg_doc_length': sum(len(doc) for doc in documents) / len(documents),
            'remaining_artifacts': remaining_artifacts[:20],
            'artifact_count': len(remaining_artifacts),
            'top_content_words': content_words[:20],
            'hapax_legomena': sum(1 for count in word_freq.values() if count == 1),
            'type_token_ratio': len(vocab) / sum(word_freq.values())
        }
        
        return validation
    
    def save_corpus(self, documents: List[List[str]], vocab: Set[str], 
                   validation: Dict[str, Any]):
        """
        Save the processed corpus and related files
        
        Args:
            documents: List of tokenized documents
            vocab: Final vocabulary
            validation: Validation report
        """
        print("\nSaving processed corpus...")
        
        # Save main corpus file
        corpus_path = os.path.join(self.output_dir, 'corpus.txt')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(' '.join(doc) + '\n')
        print(f"  Corpus saved to: {corpus_path}")
        
        # Save vocabulary
        vocab_path = os.path.join(self.output_dir, 'vocabulary.txt')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in sorted(vocab):
                f.write(word + '\n')
        print(f"  Vocabulary saved to: {vocab_path}")
        
        # Save processing report
        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'raw_data_dir': self.raw_data_dir,
                'output_dir': self.output_dir
            },
            'statistics': self.stats,
            'validation': validation,
            'processing_steps': [
                '1. Load raw JSON files',
                '2. Initial text cleaning (URLs, code, markdown)',
                '3. Tokenization and token cleaning',
                '4. Vocabulary filtering (frequency, document frequency)',
                '5. Document filtering with final vocabulary',
                '6. Quality validation'
            ]
        }
        
        report_path = os.path.join(self.output_dir, 'processing_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {report_path}")
    
    def run_full_pipeline(self, json_files: Optional[List[str]] = None,
                         min_freq: int = 10, max_vocab: int = 10000,
                         max_doc_freq_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Run the complete corpus preparation pipeline
        
        Args:
            json_files: Optional list of JSON files to process
            min_freq: Minimum word frequency
            max_vocab: Maximum vocabulary size
            max_doc_freq_ratio: Maximum document frequency ratio
            
        Returns:
            Processing statistics and validation report
        """
        print("=" * 70)
        print("CORPUS PREPARATION PIPELINE")
        print("=" * 70)
        
        # Step 1: Process JSON files
        print("\nSTEP 1: Processing JSON files...")
        cleaned_texts = self.process_json_files(json_files)
        
        # Step 2: Apply vocabulary filters
        print("\nSTEP 2: Building and filtering vocabulary...")
        documents, vocab = self.apply_vocabulary_filters(
            cleaned_texts, min_freq, max_vocab, max_doc_freq_ratio
        )
        
        # Step 3: Validate corpus
        print("\nSTEP 3: Validating corpus quality...")
        validation = self.validate_corpus(documents, vocab)
        
        # Step 4: Save results
        print("\nSTEP 4: Saving processed corpus...")
        self.save_corpus(documents, vocab, validation)
        
        # Print summary
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Files processed: {len(self.stats['files_processed'])}")
        print(f"Documents: {validation['total_documents']:,}")
        print(f"Vocabulary: {validation['vocabulary_size']:,}")
        print(f"Total tokens: {validation['total_tokens']:,}")
        print(f"Avg doc length: {validation['avg_doc_length']:.1f}")
        
        if validation['artifact_count'] > 0:
            print(f"\nWarning: {validation['artifact_count']} potential artifacts remaining")
            print(f"Examples: {', '.join(validation['remaining_artifacts'][:5])}")
        else:
            print("\nNo artifacts detected in vocabulary!")
        
        print(f"\nTop content words:")
        for word, count in validation['top_content_words'][:10]:
            print(f"  {word}: {count:,}")
        
        print(f"\nOutput saved to: {self.output_dir}/")
        print(f"  - corpus.txt: Ready for embedding training")
        print(f"  - vocabulary.txt: Final vocabulary")
        print(f"  - processing_report.json: Detailed statistics")
        
        return {
            'statistics': self.stats,
            'validation': validation
        }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Prepare corpus for word embedding training')
    
    parser.add_argument('--raw-dir', type=str, default='raw',
                       help='Directory containing raw JSON files')
    parser.add_argument('--output-dir', type=str, default='corpus',
                       help='Output directory for processed corpus')
    parser.add_argument('--min-freq', type=int, default=10,
                       help='Minimum word frequency (default: 10)')
    parser.add_argument('--max-vocab', type=int, default=10000,
                       help='Maximum vocabulary size (default: 10000)')
    parser.add_argument('--max-doc-freq', type=float, default=0.5,
                       help='Maximum document frequency ratio (default: 0.5)')
    parser.add_argument('--files', nargs='+', default=None,
                       help='Specific JSON files to process')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CorpusProcessor(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    results = processor.run_full_pipeline(
        json_files=args.files,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
        max_doc_freq_ratio=args.max_doc_freq
    )
    
    return results


if __name__ == "__main__":
    # If running directly without arguments, use defaults
    import sys
    
    if len(sys.argv) == 1:
        print("Running with default configuration...")
        print("Raw data directory: raw/")
        print("Output directory: embedding_ready_corpus/")
        print()
    
    results = main()