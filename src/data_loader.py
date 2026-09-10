"""
Data loading and preprocessing module for customer support on Twitter dataset.
"""
import pandas as pd
from typing import Dict, List, Any

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Loads the raw CSV data into a pandas DataFrame."""
    print(f"Loading raw data from {csv_path}...")
    dtypes = {
        'tweet_id': 'str',
        'author_id': 'str',
        'inbound': 'bool',
        'created_at': 'str',
        'text': 'str',
        'response_tweet_id': 'str',
        'in_response_to_tweet_id': 'str'
    }
    df = pd.read_csv(csv_path, dtype=dtypes)
    print(f"Loaded {len(df)} rows.")
    return df

def build_threads(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Reconstructs conversation threads by following in_response_to_tweet_id chains."""
    print("Building conversation threads...")
    records = df.to_dict(orient='records')
    tweet_map = {str(r['tweet_id']): r for r in records}
    
    threads = {}
    children_map = {}
    roots = []
    
    for tweet in records:
        parent_id = str(tweet.get('in_response_to_tweet_id'))
        if parent_id != 'nan' and parent_id and parent_id in tweet_map:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(tweet)
        else:
            roots.append(tweet)
            
    for root in roots:
        tid = str(root['tweet_id'])
        thread = []
        stack = [root]
        while stack:
            curr = stack.pop(0)
            thread.append(curr)
            curr_id = str(curr['tweet_id'])
            if curr_id in children_map:
                stack.extend(children_map[curr_id])
        threads[tid] = thread
        
    print(f"Built {len(threads)} threads.")
    return threads

def get_brand_conversations(df: pd.DataFrame, brand: str = 'SpotifyCares') -> pd.DataFrame:
    """Filters for brand conversations and builds a clean DataFrame."""
    print(f"Filtering conversations for brand: {brand}...")
    brand_tweets = df[df['author_id'] == brand]
    tweet_dict = df.set_index('tweet_id').to_dict('index')
    
    results = []
    count = 0
    total_brand = len(brand_tweets)
    
    for _, row in brand_tweets.iterrows():
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count}/{total_brand} brand tweets...")
            
        brand_reply_text = row['text']
        parent_id = row['in_response_to_tweet_id']
        
        if pd.notna(parent_id) and str(parent_id) in tweet_dict:
            parent_tweet = tweet_dict[str(parent_id)]
            user_tweet_text = parent_tweet['text']
            user_author = parent_tweet['author_id']
            
            curr_parent = str(parent_id)
            thread_id = curr_parent
            while curr_parent in tweet_dict and pd.notna(tweet_dict[curr_parent]['in_response_to_tweet_id']):
                curr_parent = str(tweet_dict[curr_parent]['in_response_to_tweet_id'])
                thread_id = curr_parent
                
            results.append({
                'thread_id': thread_id,
                'user_tweet': user_tweet_text,
                'brand_reply': brand_reply_text,
                'user_author': user_author
            })
            
    result_df = pd.DataFrame(results)
    print(f"Found {len(result_df)} brand conversation pairs.")
    return result_df

def get_first_contact_pairs(brand_convos_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """For each thread, gets the FIRST user message and FIRST brand response."""
    print("Extracting first contact pairs...")
    first_contacts = brand_convos_df.drop_duplicates(subset=['thread_id'], keep='first')
    records = first_contacts.to_dict('records')
    print(f"Extracted {len(records)} first contact pairs.")
    return records
