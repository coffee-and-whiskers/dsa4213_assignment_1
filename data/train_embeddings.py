"""
Trains Skip-gram, PPMI-SVD, and GloVe models on prepared corpus
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional
from util.corpus_loader import CorpusLoader
from util.skipgram_trainer import SkipGramTrainer
from util.ppmi_svd_trainer import PPMISVDTrainer
from util.glove_trainer import GloVeTrainer


class EmbeddingPipeline:
    def __init__(self, corpus_path: str = 'corpus/cleaned_corpus_v2.txt', 
                 output_dir: str = 'embeddings',
                 config_path: Optional[str] = None):
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.config = self._load_config(config_path)
        self.corpus_loader = None
        self.training_data = None
        self.models = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        default_config = {
            'corpus_loading': {
                'min_freq': 10,
                'max_vocab_size': 10000,
                'remove_stopwords': True,
                'remove_artifacts': True,
                'keep_domain_words': True
            },
            'skipgram': {
                'embedding_dim': 100,
                'window': 5,
                'epochs': 30,
                'negative': 10,
                'learning_rate': 0.025,
                'min_learning_rate': 0.0001
            },
            'ppmi_svd': {
                'embedding_dim': 300,
                'window': 5,
                'ppmi_shift': 1,
                'use_context_weights': True
            },
            'glove': {
                'embedding_dim': 100,
                'window': 5,
                'learning_rate': 0.05,
                'max_iter': 50,
                'x_max': 10,
                'alpha': 0.75,
                'tolerance': 1e-4,
                'patience': 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
        
        return default_config
    
    def load_corpus(self, verbose: bool = True) -> Dict:

        print("\n" + "=" * 70)
        print("LOADING PREPARED CORPUS")
        print("=" * 70)
        
        self.corpus_loader = CorpusLoader(
            min_freq=self.config['corpus_loading']['min_freq'],
            max_vocab_size=self.config['corpus_loading']['max_vocab_size'],
            remove_stopwords=self.config['corpus_loading']['remove_stopwords'],
            remove_artifacts=self.config['corpus_loading']['remove_artifacts'],
            keep_domain_words=self.config['corpus_loading']['keep_domain_words']
        )

        self.training_data = self.corpus_loader.load_corpus(
            self.corpus_path, verbose=verbose
        )
    
        import pickle
        data_info_path = os.path.join(self.output_dir, 'training_data_info.pkl')
        with open(data_info_path, 'wb') as f:
            pickle.dump({
                'vocab_size': len(self.training_data['vocabulary']),
                'total_docs': self.training_data['total_docs'],
                'word2idx': self.training_data['word2idx'],
                'idx2word': self.training_data['idx2word']
            }, f)
        
        if verbose:
            print(f"\nTraining data info saved to: {data_info_path}")
        
        return self.training_data
    
    
    def train_skipgram(self, verbose: bool = True) -> Dict:
        """Train Skip-gram model"""
        if not self.training_data:
            raise ValueError("Corpus must be loaded first")
        
        print("\n" + "=" * 70)
        print("TRAINING SKIP-GRAM MODEL")
        print("=" * 70)
        
        trainer = SkipGramTrainer(self.output_dir)
        
        model = trainer.train(
            training_data=self.training_data,
            **self.config['skipgram'],
            verbose=verbose
        )
        
        self.models['skipgram'] = {
            'model': model,
            'trainer': trainer
        }
        
        return self.models['skipgram']
    
    def train_ppmi_svd(self, verbose: bool = True) -> Dict:
        """Train PPMI-SVD model"""
        if not self.training_data:
            raise ValueError("Corpus must be loaded first")
        
        print("\n" + "=" * 70)
        print("TRAINING PPMI-SVD MODEL")
        print("=" * 70)
        
        trainer = PPMISVDTrainer(self.output_dir)
        
        model = trainer.train(
            training_data=self.training_data,
            **self.config['ppmi_svd'],
            verbose=verbose
        )
        
        self.models['ppmi_svd'] = {
            'model': model,
            'trainer': trainer
        }
        
        if verbose:
            print("\nAnalyzing singular value distribution...")
            svd_analysis = trainer.analyze_singular_values()
            if svd_analysis and 'dims_for_variance' in svd_analysis:
                print(f"  Dimensions for 90% variance: {svd_analysis['dims_for_variance']['90%']}")
                print(f"  Dimensions for 95% variance: {svd_analysis['dims_for_variance']['95%']}")
        
        return self.models['ppmi_svd']
    
    def train_glove(self, verbose: bool = True) -> Dict:
        """Train GloVe model"""
        if not self.training_data:
            raise ValueError("Corpus must be loaded first")
        
        print("\n" + "=" * 70)
        print("TRAINING GLOVE MODEL")
        print("=" * 70)
        
        trainer = GloVeTrainer(self.output_dir)
        
        model = trainer.train(
            training_data=self.training_data,
            **self.config['glove'],
            verbose=verbose
        )
        
        self.models['glove'] = {
            'model': model,
            'trainer': trainer
        }
        
        if verbose and self.config['bias_analysis']['visualize']:
            print("\nPlotting GloVe convergence...")
            trainer.plot_convergence()
        
        return self.models['glove']
    
    def train_all_models(self, models: Optional[List[str]] = None, 
                        verbose: bool = True) -> Dict:
     
        if models is None:
            models = ['skipgram', 'ppmi_svd', 'glove']
        
        print("\n" + "=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70)
        print(f"Models to train: {', '.join(models)}")
        
        overall_start = time.time()
    
        if 'skipgram' in models:
            self.train_skipgram(verbose)
        
        if 'ppmi_svd' in models:
            self.train_ppmi_svd(verbose)
        
        if 'glove' in models:
            self.train_glove(verbose)
        
        overall_time = time.time() - overall_start
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total training time: {overall_time:.1f} seconds")
        print(f"Models trained: {', '.join(self.models.keys())}")
        
        return self.models
    
    
    def generate_final_report(self) -> str:
        """
        Generate a comprehensive final report
        
        Returns:
            Path to the report file
        """
        print("\n" + "=" * 70)
        print("GENERATING FINAL REPORT")
        print("=" * 70)
        
        report_path = os.path.join(self.output_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("WORD EMBEDDING TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATA FLOW:\n")
            f.write("-" * 40 + "\n")
            f.write("1. reddit_scraper.py to raw/*.json\n")
            f.write("2. prepare_corpus.py to corpus/*.txt\n")
            f.write("3. train_embeddings.py to embeddings/*\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Corpus: {self.corpus_path}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Training statistics
            if self.training_data:
                f.write("TRAINING DATA STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Documents: {self.training_data['total_docs']:,}\n")
                f.write(f"Vocabulary size: {len(self.training_data['vocabulary']):,}\n")
                f.write(f"Total tokens: {sum(len(doc) for doc in self.training_data['documents']):,}\n\n")
            
            # Models trained
            f.write("MODELS TRAINED\n")
            f.write("-" * 40 + "\n")
            for model_name in self.models:
                f.write(f"Done: {model_name.upper()}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Training pipeline complete!\n")
        
        print(f"Report saved to: {report_path}")
        
        return report_path
    


def main():
    parser = argparse.ArgumentParser(
        description='train word embeddings on prepared corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DATA FLOW:
  1. reddit_scraper.py to raw/*.json (Reddit data collection)
  2. prepare_corpus.py to corpus/*.txt (Text preprocessing)
  3. train_embeddings.py to embeddings/* (Model training)

EXAMPLES:
  # Train all models on default corpus
  python train_embeddings.py
  
  # Train specific models
  python train_embeddings.py --models skipgram glove
  
  # Use custom corpus and output directory
  python train_embeddings.py --corpus corpus/combined_corpus.txt --output my_embeddings
  
  # Skip bias analysis for faster training
  python train_embeddings.py --skip-bias-analysis
        """
    )
    
    parser.add_argument('--corpus', type=str, default='corpus/cleaned_corpus_v2.txt',
                       help='Path to the prepared corpus file (default: corpus/cleaned_corpus_v2.txt)')
    parser.add_argument('--output', type=str, default='embeddings',
                       help='Output directory for models and results (default: embeddings)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--models', nargs='+', 
                       default=['skipgram', 'ppmi_svd', 'glove'],
                       choices=['skipgram', 'ppmi_svd', 'glove'],
                       help='Models to train (default: all)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found: {args.corpus}")
        return
    pipeline = EmbeddingPipeline(
        corpus_path=args.corpus,
        output_dir=args.output,
        config_path=args.config
    )
    
    verbose = not args.quiet
    try:
      
        pipeline.load_corpus(verbose=verbose)
        pipeline.train_all_models(models=args.models, verbose=verbose)
        report_path = pipeline.generate_final_report()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"All results saved to: {args.output}/")
        print(f"Final report: {report_path}")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()