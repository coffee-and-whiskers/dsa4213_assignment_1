"""run this after training models with train_embeddings.py to visualize tsne,umap,pca
"""

from util.visualize import EnhancedEmbeddingVisualizer
import argparse
import json
import os


def load_custom_terms(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='visualization of word embeddings using t-SNE, UMAP, and PCA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python visualize_embeddings.py --model skipgram --methods tsne umap pca
  python visualize_embeddings.py --model glove --analysis-mode full --save results/glove_viz.png
  python visualize_embeddings.py --model skipgram --analysis-mode platforms
  python visualize_embeddings.py --compare skipgram ppmi_svd glove
        """
    )
    

    parser.add_argument('--model', type=str, 
                       choices=['skipgram', 'ppmi_svd', 'glove'],
                       help='Model to visualize')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings',
                       help='Directory containing trained embeddings')
    parser.add_argument('--methods', nargs='+', default=['tsne', 'umap', 'pca'],
                       choices=['tsne', 'umap', 'pca'],
                       help='Visualization methods (default: all three)')
    parser.add_argument('--analysis-mode', type=str, default='platforms',
                       choices=['platforms', 'perceptions', 'full', 'custom'],
                       help='What to analyze: platforms only, perceptions only, or full analysis')
    parser.add_argument('--custom-terms', type=str, default=None,
                       help='Path to JSON file with custom term groups')
    parser.add_argument('--focus-platforms', nargs='+', 
                       choices=['openai', 'anthropic', 'google', 'meta', 'mistral', 'deepseek', 'microsoft'],
                       help='Focus on specific platforms')
    parser.add_argument('--show-connections', action='store_true',
                       help='Show connections between related terms')
    parser.add_argument('--perplexity', type=int, default=None,
                       help='Perplexity parameter for t-SNE (auto-selected based on model if not specified)')
    parser.add_argument('--n-neighbors', type=int, default=None,
                       help='n_neighbors parameter for UMAP (auto-selected based on model if not specified)')
    parser.add_argument('--min-dist', type=float, default=None,
                       help='min_dist parameter for UMAP (default: auto-selected)')
    parser.add_argument('--use-presets', action='store_true', default=True,
                       help='Use optimal presets for each model type (default: True)')
    parser.add_argument('--figsize', nargs=2, type=int, default=None,
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (e.g., --figsize 20 10)')
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate similarity heatmap between platforms and perceptions')
    parser.add_argument('--compare', nargs='+',
                       choices=['skipgram', 'ppmi_svd', 'glove'],
                       help='Compare multiple models side by side')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save main visualization')
    parser.add_argument('--save-heatmap', type=str, default=None,
                       help='Path to save heatmap (if --heatmap is used)')
    parser.add_argument('--save-all', type=str, default=None,
                       help='Directory to save all outputs')
    
    args = parser.parse_args()
    viz = EnhancedEmbeddingVisualizer(args.embeddings_dir)
    ai_platforms = {
        'OpenAI': ['openai', 'gpt', 'chatgpt', 'gpt3', 'gpt4', 'gpt5', 'o1', 'o3'],
        'Anthropic': ['anthropic', 'claude', 'claude2', 'claude3', 'opus', 'sonnet', 'haiku'],
        'Google': ['google', 'gemini', 'bard', 'palm', 'lamda'],
        'Meta': ['meta', 'llama', 'llama2', 'llama3', 'llama4'],
        'Mistral': ['mistral', 'mixtral', 'mistral7b', 'mixtral8x7b'],
        'DeepSeek': ['deepseek', 'deepseekr1', 'deepseekv2', 'deepseekv3'],
        'Microsoft': ['microsoft', 'copilot', 'bing', 'sydney']
    }
    perception_terms = {
        'Cost': ['expensive', 'cheap', 'free', 'paid', 'subscription', 
                'affordable', 'overpriced', 'worth', 'costly'],
        
        'Restrictions': ['censored', 'uncensored', 'filtered', 'restricted', 
                        'jailbreak', 'refuses', 'woke', 'lobotomized', 
                        'alignment', 'safety', 'guardrails', 'limited'],
        
        'Quality+': ['good', 'better', 'best', 'excellent', 'amazing',
                    'powerful', 'smart', 'advanced', 'capable', 'goat', 'peak'],
        
        'Quality-': ['bad', 'worse', 'worst', 'terrible', 'trash',
                    'weak', 'dumb', 'basic', 'limited', 'mid', 'useless'],
        
        'Changes': ['nerfed', 'buffed', 'improved', 'degraded', 
                   'updated', 'broken', 'fixed', 'enhanced', 'ruined'],
        
        'Preference': ['prefer', 'hate', 'love', 'switched', 'abandoned',
                      'recommend', 'avoid', 'favorite', 'dislike']
    }
    
    
    if args.focus_platforms:
        filtered_platforms = {}
        for platform in args.focus_platforms:
            platform_key = platform.capitalize()
            if platform_key in ai_platforms:
                filtered_platforms[platform_key] = ai_platforms[platform_key]
        ai_platforms = filtered_platforms
    custom_terms = None
    if args.custom_terms:
        custom_terms = load_custom_terms(args.custom_terms)
        if custom_terms:
            print(f"Loaded custom terms from {args.custom_terms}")

    vis_platforms = None
    vis_perceptions = None
    
    if args.analysis_mode == 'platforms':
        vis_platforms = ai_platforms
        vis_perceptions = None
    elif args.analysis_mode == 'perceptions':
        vis_platforms = None
        vis_perceptions = perception_terms
    elif args.analysis_mode == 'full':
        vis_platforms = ai_platforms
        vis_perceptions = perception_terms
    elif args.analysis_mode == 'custom' and custom_terms:
        vis_platforms = custom_terms.get('platforms', None)
        vis_perceptions = custom_terms.get('perceptions', None)
    
    save_path = args.save
    heatmap_path = args.save_heatmap
    if args.save_all:
        os.makedirs(args.save_all, exist_ok=True)
        if not save_path:
            save_path = os.path.join(args.save_all, f'{args.model}_visualization.png')
        if not heatmap_path and args.heatmap:
            heatmap_path = os.path.join(args.save_all, f'{args.model}_heatmap.png')
    
    if args.compare:
        print(f"\ncomparing models: {', '.join(args.compare)}\n")
        test_terms = ['gpt', 'claude', 'gemini', 'llama', 'mistral',
                     'good', 'bad', 'expensive', 'cheap', 
                     'censored', 'uncensored', 'smart', 'dumb']
        
        viz.comparative_analysis(
            models=args.compare,
            test_terms=test_terms,
            figsize=tuple(args.figsize) if args.figsize else None
        )
    
    elif args.model:
        print(f"\nloading model: {args.model}")
        print(f"embeddings directory: {args.embeddings_dir}/\n")
        
        # model-specific optimal parameters
        model_presets = {
            'skipgram': {
                'perplexity': 15,      # Lower for tighter clusters
                'n_neighbors': 10,     # Tighter UMAP groups
                'min_dist': 0.1        # More compact layout
            },
            'ppmi_svd': {
                'perplexity': 5,       # Very low - sparse embeddings need local focus
                'n_neighbors': 5,      # Small neighborhoods
                'min_dist': 0.05       # Very tight packing to see any structure
            },
            'glove': {
                'perplexity': 20,      # Balanced
                'n_neighbors': 15,     # Standard
                'min_dist': 0.15       # Slightly spread out
            }
        }
        
        if args.use_presets and args.model in model_presets:
            preset = model_presets[args.model]
            perplexity = args.perplexity if args.perplexity is not None else preset['perplexity']
            n_neighbors = args.n_neighbors if args.n_neighbors is not None else preset['n_neighbors']
            min_dist = args.min_dist if args.min_dist is not None else preset['min_dist']
            print(f"using optimized parameters for {args.model}:")
            print(f"  t-sne perplexity: {perplexity}")
            print(f"  umap n_neighbors: {n_neighbors}")
            print(f"  umap min_dist: {min_dist}")
        else:
            # Use defaults or user-specified values
            perplexity = args.perplexity if args.perplexity is not None else 30
            n_neighbors = args.n_neighbors if args.n_neighbors is not None else 15
            min_dist = args.min_dist if args.min_dist is not None else 0.1
        
        if viz.load_embeddings(args.model, normalize=True):
            print(f"loaded {args.model} embeddings")
            print(f"visualization methods: {', '.join(args.methods)}")
            
            if args.analysis_mode != 'custom' or (args.analysis_mode == 'custom' and custom_terms):
                print(f"analysis mode: {args.analysis_mode}")
                print(f"\ngenerating visualization...")
                
                results = viz.advanced_visualize(
                    ai_platforms=vis_platforms,
                    perception_terms=vis_perceptions,
                    custom_terms=custom_terms if args.analysis_mode == 'custom' else None,
                    methods=args.methods,
                    figsize=tuple(args.figsize) if args.figsize else None,
                    save_path=save_path,
                    show_connections=args.show_connections,
                    perplexity_values=[perplexity],
                    n_neighbors_values=[n_neighbors],
                    min_dist=min_dist
                )
                
                if save_path:
                    print(f"visualization saved to: {save_path}")
                
                if args.heatmap and vis_platforms and vis_perceptions:
                    print(f"\ngenerating similarity heatmap...")
                    
                    df = viz.similarity_heatmap(
                        ai_platforms=vis_platforms,
                        perception_terms=vis_perceptions,
                        method='average',
                        save_path=heatmap_path
                    )
                    
                    if heatmap_path:
                        print(f"heatmap saved to: {heatmap_path}")
                    
                    # Print top similarities
                    print("\ntop platform-perception similarities:")
                    for platform in df.index[:5]:  # Top 5 platforms
                        max_perception = df.loc[platform].idxmax()
                        max_value = df.loc[platform].max()
                        min_perception = df.loc[platform].idxmin()
                        min_value = df.loc[platform].min()
                        print(f"{platform:12} | Closest: {max_perception:12} ({max_value:.3f}) | "
                              f"Furthest: {min_perception:12} ({min_value:.3f})")
                
                print(f"\nvisualization complete")
            
            else:
                print("✗ No terms to visualize. Check your custom terms file.")
        
        else:
            print(f"failed to load {args.model} embeddings")
            print(f"make sure you've trained the model first with:")
            print(f"  python train_embeddings.py --models {args.model}")
    
    else:
        # Default: try to visualize the first available model
        print("\nno model specified. searching for available models...")
        for model in ['skipgram', 'ppmi_svd', 'glove']:
            print(f"trying to load {model}...")
            if viz.load_embeddings(model):
                print(f"found and loaded {model}")
                print(f"visualizing with methods: {', '.join(args.methods)}")
                
                viz.advanced_visualize(
                    ai_platforms=ai_platforms if args.analysis_mode in ['platforms', 'full'] else None,
                    perception_terms=perception_terms if args.analysis_mode in ['perceptions', 'full'] else None,
                    methods=args.methods,
                    save_path=save_path,
                    show_connections=args.show_connections
                )
                break
        else:
            print("\nno trained models found in embeddings/")
            print("train models first with: python train_embeddings.py")
            print("\nexample commands:")
            print("  python train_embeddings.py --models skipgram")
            print("  python train_embeddings.py --models ppmi_svd")
            print("  python train_embeddings.py --models glove")


if __name__ == "__main__":
    main()