import asyncpraw
import asyncio
import json
import time
import os
import sys
from datetime import datetime
import re
import string

# fix windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class FastRedditScraper:
    def __init__(self):
        self.reddit = asyncpraw.Reddit(
            client_id="weFEJCQ0LcZfnna7WHpKUQ",
            client_secret="aERbsDRisUXQPe9pL-I3zbWtykquDg",
            user_agent="subsum:v1.0.0 by /u/survivalife",
            ratelimit_seconds=600  # Wait up to 10 minutes for rate limits
        )
        
        # track rate limits
        self.requests_made = 0
        self.last_request_time = time.time()
        self.rate_limit_errors = 0
        self.consecutive_429s = 0
        
        self.subreddits = ['artificial'] # add more subreddits if needed
        self.use_hard_target = False  # collect top content without hard limit
        self.target_words = None  # no word target
        self.save_interval_words = 50000  # save every 50k words
        self.min_comment_words = 10  # min words per comment
        self.top_posts_percentile = 0.7  # top 70% posts by score
        self.top_comments_percentile = 0.7  # top 70% comments by score
        self.max_posts_to_process = 200  # safety limit
        self.bot_patterns = [
            r'bot', r'auto', r'moderator'
        ]
        
        # track duplicates
        self.seen_comments = {}  # hash -> first occurrence
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.reddit.close()
    
    def clean_text(self, text):
        """Basic cleaning for Reddit text - removing only the obvious noise"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove Reddit user/subreddit mentions
        text = re.sub(r'/?u/[\w-]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'/?r/[\w-]+', '', text, flags=re.IGNORECASE)
        
        # Remove deleted/removed markers
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        
        # Remove edit markers and everything after them
        text = re.sub(r'(EDIT|Edit|edit):\s*.*?(?=\n|$)', '', text, flags=re.MULTILINE)
        
        # Remove quotes (lines starting with >)
        lines = text.split('\n')
        lines = [line for line in lines if not line.strip().startswith('>')]
        text = '\n'.join(lines)
        
        # Basic markdown removal - just the most common
        text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)  # Bold/italic
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n+', '\n', text)  # Multiple newlines to single
        
        # Final strip
        text = text.strip()
        
        return text
    
    def get_comment_hash(self, comment_text):
        """Generate a hash for deduplication - using first 100 chars of cleaned text"""
        cleaned = self.clean_text(comment_text)
        # Use first 100 characters for hash to catch near-duplicates
        text_sample = cleaned[:100].lower().strip()
        return hash(text_sample)
        
    def is_valid_comment(self, comment):
        if not hasattr(comment, 'body') or comment.body in ['[deleted]', '[removed]', None]:
            return False
        
        author = str(comment.author).lower() if comment.author else ''
        if any(pattern in author for pattern in self.bot_patterns):
            return False
        
        # Skip comments that are primarily about images/videos
        body_lower = comment.body.lower()
        image_indicators = [
            'imgur.com', 'i.redd.it', 'v.redd.it', '.jpg', '.jpeg', '.png']
        if any(indicator in body_lower for indicator in image_indicators):
            return False
        
        # Clean text before counting words
        cleaned_text = self.clean_text(comment.body)
        word_count = len(cleaned_text.split())
        
        if word_count < self.min_comment_words:
            return False
        
        # Check for duplicate/near-duplicate comments
        comment_hash = self.get_comment_hash(comment.body)
        if comment_hash in self.seen_comments:
            return False
        
        # Mark as seen
        self.seen_comments[comment_hash] = True
            
        return word_count  # Return word count for tracking
    
    def is_valid_post(self, post):
        """Check if post is text-based and not image/video"""
        # Skip image/video posts
        if hasattr(post, 'is_video') and post.is_video:
            return False
        
        if hasattr(post, 'url'):
            url_lower = post.url.lower()
            media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webm']
            media_domains = ['imgur.com', 'i.redd.it', 'v.redd.it', 'youtube.com', 'youtu.be']
            
            if any(ext in url_lower for ext in media_extensions):
                return False
            if any(domain in url_lower for domain in media_domains):
                return False
        
        # Check if it's a self/text post
        if hasattr(post, 'is_self') and not post.is_self:
            # External link, check if it's to an image
            if hasattr(post, 'post_hint') and post.post_hint in ['image', 'hosted:video', 'rich:video']:
                return False
        
        return True
    
    def save_checkpoint(self, data, subreddit_name, checkpoint_num):
        filename = f"{subreddit_name}_comments_{checkpoint_num}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {filename} ({len(data)} comments)", flush=True)
    
    async def get_comment_forest(self, submission):
        """Get top-scoring comment threads to preserve conversation flow"""
        valid_comments = []
        
        # Get top-level comments
        submission.comment_limit = 200  # Get more comments to filter from
        submission.comment_sort = 'best'
        
        await submission.load()  # Load the submission
        
        # First, collect and score top-level comments
        top_level_comments = []
        for comment in submission.comments:
            # Skip MoreComments objects
            if isinstance(comment, asyncpraw.models.MoreComments):
                continue
            
            word_count = self.is_valid_comment(comment)
            if word_count:
                top_level_comments.append((comment, comment.score))
        
        # Sort top-level comments by score and take top 70%
        if top_level_comments:
            top_level_comments.sort(key=lambda x: x[1], reverse=True)
            cutoff_index = int(len(top_level_comments) * self.top_comments_percentile)
            selected_threads = top_level_comments[:cutoff_index]
            
            # Now collect entire conversation threads for selected top-level comments
            for comment, _ in selected_threads:
                # Add the top-level comment
                if self.is_valid_comment(comment):
                    valid_comments.append(comment)
                
                # Recursively add all replies in the thread
                reply_queue = list(comment.replies) if hasattr(comment, 'replies') else []
                
                while reply_queue and len(valid_comments) < 1000:  # Safety limit
                    reply = reply_queue.pop(0)
                    
                    if isinstance(reply, asyncpraw.models.MoreComments):
                        continue
                    
                    # Add reply if valid (preserving conversation even if individual reply is low score)
                    if self.is_valid_comment(reply):
                        valid_comments.append(reply)
                    
                    # Add sub-replies to queue to get full thread
                    if hasattr(reply, 'replies'):
                        reply_queue.extend(reply.replies)
        
        return valid_comments
    
    async def handle_rate_limit(self):
        """Handle rate limiting with exponential backoff"""
        self.consecutive_429s += 1
        wait_time = min(2 ** self.consecutive_429s * 5, 300)  # 5, 10, 20, 40, 80, 160, max 300 seconds
        print(f"Rate limit hit ({self.consecutive_429s}x). Waiting {wait_time} seconds...", flush=True)
        await asyncio.sleep(wait_time)
    
    async def process_post(self, post, subreddit_name):
        """Process a single post and return comment data"""
        try:
            comments = await self.get_comment_forest(post)
            self.consecutive_429s = 0  # Reset on success
            
            comment_data = []
            for comment in comments:
                # Clean the text before storing
                cleaned_text = self.clean_text(comment.body)
                
                comment_obj = {
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'post_title': self.clean_text(post.title),  # Clean post title too
                    'comment_id': comment.id,
                    'comment_text': cleaned_text,  # Store cleaned text
                    'original_text': comment.body,  # Keep original for reference
                    'comment_score': comment.score,
                    'created_utc': comment.created_utc,
                    'author': str(comment.author) if comment.author else 'deleted'
                }
                comment_data.append(comment_obj)
            
            return comment_data
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                await self.handle_rate_limit()
                return []  # Return empty and let the batch processor retry
            else:
                print(f"Error processing post: {e}", flush=True)
                return []
    
    async def scrape_subreddit(self, subreddit_name):
        print(f"\n{'='*60}", flush=True)
        print(f"Starting scrape for r/{subreddit_name}", flush=True)
        print(f"Collecting top {int(self.top_posts_percentile*100)}% quality content", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Reset deduplication for each subreddit
        self.seen_comments = {}
        
        subreddit = await self.reddit.subreddit(subreddit_name)
        comments_data = []
        comment_ids_seen = set()
        checkpoint_num = 1
        total_words = 0
        last_checkpoint_words = 0
        duplicates_skipped = 0
        
        try:
            # Fetch more posts to ensure we get enough comments
            print("\nFetching posts...", flush=True)
            
            # Get posts from different time periods
            hot_posts = []
            new_posts = []
            top_week = []
            top_month = []
            top_year = []
            
            async for post in subreddit.hot(limit=300):
                hot_posts.append(post)
            print(f"  Hot posts: {len(hot_posts)}", flush=True)
            
            async for post in subreddit.new(limit=300):
                new_posts.append(post)
            print(f"  New posts: {len(new_posts)}", flush=True)
            
            async for post in subreddit.top(time_filter='week', limit=300):
                top_week.append(post)
            print(f"  Top week: {len(top_week)}", flush=True)
            
            async for post in subreddit.top(time_filter='month', limit=300):
                top_month.append(post)
            print(f"  Top month: {len(top_month)}", flush=True)
            
            async for post in subreddit.top(time_filter='year', limit=300):
                top_year.append(post)
            print(f"  Top year: {len(top_year)}", flush=True)
            
            # Combine and deduplicate, filtering out image/video posts
            all_posts = []
            seen_ids = set()
            skipped_media = 0
            for post in hot_posts + new_posts + top_week + top_month + top_year:
                if post.id not in seen_ids:
                    if self.is_valid_post(post):
                        all_posts.append(post)
                        seen_ids.add(post.id)
                    else:
                        skipped_media += 1
            
            print(f"  Skipped {skipped_media} image/video posts", flush=True)
            
            # Sort by score and take top 70%
            all_posts.sort(key=lambda x: x.score, reverse=True)
            cutoff_index = int(len(all_posts) * self.top_posts_percentile)
            all_posts = all_posts[:cutoff_index]
            
            # Then sort by comment count within top 70% to prioritize discussion
            all_posts.sort(key=lambda x: x.num_comments, reverse=True)
            
            print(f"\nUsing top {int(self.top_posts_percentile*100)}% of posts: {len(all_posts)} posts", flush=True)
            print(f"Will collect top {int(self.top_comments_percentile*100)}% of comments from each post", flush=True)
            print("Processing posts (prioritizing by comment count)...\n", flush=True)
            
            # Process posts in batches of 10
            batch_size = 10  # Increased from 5 to 10
            post_count = 0
            
            # Process all selected posts (top 70%)
            for i in range(0, min(len(all_posts), self.max_posts_to_process), batch_size):
                batch = all_posts[i:i+batch_size]
                
                # Add small delay between batches to respect rate limits
                if i > 0 and self.consecutive_429s == 0:
                    await asyncio.sleep(1)  # 1 second between batches if no rate limit issues
                
                # Process batch concurrently
                tasks = [self.process_post(post, subreddit_name) for post in batch]
                batch_results = await asyncio.gather(*tasks)
                
                # Collect results
                for post_comments in batch_results:
                    for comment_obj in post_comments:
                        if comment_obj['comment_id'] not in comment_ids_seen:
                            comment_ids_seen.add(comment_obj['comment_id'])
                            
                            # Count words in cleaned text
                            comment_words = len(comment_obj['comment_text'].split())
                            total_words += comment_words
                            comment_obj['word_count'] = comment_words
                            
                            comments_data.append(comment_obj)
                            
                            # Progress updates based on words
                            if total_words // 25000 > (total_words - comment_words) // 25000:
                                print(f"Progress: {total_words:,} words ({len(comments_data)} comments, {len(self.seen_comments)} unique)", flush=True)
                            
                            # Save checkpoint based on words
                            if total_words - last_checkpoint_words >= self.save_interval_words:
                                self.save_checkpoint(
                                    comments_data[-(len(comments_data) - (checkpoint_num-1)*1000):],
                                    subreddit_name, checkpoint_num
                                )
                                checkpoint_num += 1
                                last_checkpoint_words = total_words
                
                post_count += len(batch)
                if post_count % 20 == 0:  # Update every 20 posts with batch size 10
                    print(f"Processed {post_count} posts: {total_words:,} words, {len(comments_data)} comments", flush=True)
            
        except Exception as e:
            print(f"Error: {e}", flush=True)
        
        # Save final file
        final_filename = f"{subreddit_name}_final_{total_words}words_{len(comments_data)}comments.json"
        with open(final_filename, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}", flush=True)
        print(f"Completed r/{subreddit_name}", flush=True)
        print(f"Total words collected: {total_words:,}", flush=True)
        print(f"Total comments collected: {len(comments_data)}", flush=True)
        print(f"Unique comments (deduped): {len(self.seen_comments)}", flush=True)
        print(f"Average words per comment: {total_words//len(comments_data) if comments_data else 0}", flush=True)
        print(f"Final file: {final_filename}", flush=True)
        print(f"{'='*60}", flush=True)
        
        return {'words': total_words, 'comments': len(comments_data), 'unique': len(self.seen_comments)}
    
    async def run(self):
        print("Fast Reddit Scraper Starting...", flush=True)
        print(f"Mode: Collecting top {int(self.top_posts_percentile*100)}% quality content (no hard limit)", flush=True)
        print(f"Subreddits: {', '.join(self.subreddits)}", flush=True)
        print(f"Batch size: 10 posts", flush=True)
        print(f"Text cleaning: Enabled", flush=True)
        print(f"Deduplication: Enabled", flush=True)
        
        # Test connection
        try:
            print("\nTesting connection...", flush=True)
            subreddit = await self.reddit.subreddit("test")
            async for submission in subreddit.hot(limit=1):
                print(f"[SUCCESS] Connected!", flush=True)
                break
        except Exception as e:
            print(f"Connection failed: {e}", flush=True)
            return
        
        all_results = {}
        
        for subreddit_name in self.subreddits:
            try:
                results = await self.scrape_subreddit(subreddit_name)
                all_results[subreddit_name] = results
                
                # Small delay between subreddits
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Fatal error scraping r/{subreddit_name}: {e}", flush=True)
                all_results[subreddit_name] = 0
        
        print("\n" + "="*60)
        print("SCRAPING COMPLETE - FINAL SUMMARY")
        print("="*60)
        
        total_words_all = 0
        total_comments_all = 0
        total_unique_all = 0
        
        for sub, results in all_results.items():
            if isinstance(results, dict):
                print(f"r/{sub}: {results['words']:,} words, {results['comments']} comments ({results['unique']} unique)")
                total_words_all += results['words']
                total_comments_all += results['comments']
                total_unique_all += results['unique']
            else:
                print(f"r/{sub}: No data")
        
        print(f"\nTotal corpus: {total_words_all:,} words, {total_comments_all} comments ({total_unique_all} unique)")
        print(f"Average words per comment: {total_words_all//total_comments_all if total_comments_all else 0}")
        
        summary_filename = f"scraping_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
                'total_comments': total_comments_all,
                'total_unique': total_unique_all,
                'total_words': total_words_all
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_filename}", flush=True)

async def main():
    async with FastRedditScraper() as scraper:
        await scraper.run()

if __name__ == "__main__":
    # Windows event loop policy fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())