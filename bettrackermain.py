import discord
import os
import json
import datetime
import requests
import asyncio
import traceback
import time
import statistics
import threading
import subprocess
import sys
from discord.ext import commands, tasks
from discord import app_commands
import google.generativeai as genai
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# Auto-install FastAPI dependencies if missing
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", package])

# FastAPI imports for web dashboard
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("📦 Installing dashboard dependencies...")
    try:
        install_package("fastapi")
        install_package("uvicorn")
        install_package("pydantic")
        # Try importing again
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        FASTAPI_AVAILABLE = True
        print("✅ Dashboard dependencies installed!")
    except Exception as e:
        FASTAPI_AVAILABLE = False
        print(f"⚠️ Could not install FastAPI: {e}")

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyBjPdp_MybNk0ddDVUpzAU1ggz_fzE7cAs"
genai.configure(api_key=GEMINI_API_KEY)

# === MACHINE LEARNING ADAPTIVE FILTER SYSTEM ===
# This system learns from ALL bets to find profitable patterns
# and auto-adjusts filters when profitable strategies are discovered

ADMIN_USER_ID = 886065668137566239  # User to notify about algorithm changes/discoveries

# ML Configuration
ML_CONFIG = {
    "learning_mode": True,  # Allow all bets through for learning
    "min_sample_size": 100,  # Minimum bets before making recommendations (100 for statistical significance)
    "min_sample_size_combo": 50,  # Minimum for combination patterns (bet_type + odds, etc.)
    "confidence_threshold": 0.58,  # 58% win rate to consider profitable (above random chance)
    "min_profit_units": 50.0,  # Minimum profit in units to trigger filter change (50 units per 100 bets)
    "analysis_interval_hours": 2,  # How often to run analysis
    "auto_adjust_enabled": True,  # Auto-apply discovered profitable filters
    "stable_period_days": 3,  # Days a pattern must be profitable to apply
    "max_observed_bets": 50000,  # Maximum bets to keep in memory
    "max_bet_age_days": 90,  # Remove bets older than this
}

# Notification cooldown tracking to prevent spam
_last_notification_time = {}
NOTIFICATION_COOLDOWNS = {
    "reconnect": 300,  # 5 minutes between reconnect notifications
    "startup": 3600,  # 1 hour between startup notifications (if bot restarts rapidly)
    "ml_status": 21600,  # 6 hours between ML status updates
    "pattern_discovery": 0,  # Always send pattern discoveries immediately
}

def can_send_notification(notification_type: str) -> bool:
    """Check if enough time has passed to send this notification type again"""
    import time
    now = time.time()
    cooldown = NOTIFICATION_COOLDOWNS.get(notification_type, 0)
    last_sent = _last_notification_time.get(notification_type, 0)
    
    if now - last_sent >= cooldown:
        _last_notification_time[notification_type] = now
        return True
    return False

# ML Data Storage
ML_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_learning_data.json")

ml_learning_data = {
    "all_observed_bets": [],  # ALL bets seen (including filtered ones)
    "profitable_patterns": {},  # Discovered profitable patterns
    "filter_changes": [],  # History of filter changes made
    "last_analysis": None,  # Last time analysis was run
    "discoveries": [],  # Notable discoveries to report
}

def load_ml_data():
    """Load ML learning data from disk"""
    global ml_learning_data, ML_CONFIG
    try:
        if os.path.exists(ML_DATA_FILE):
            with open(ML_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for bet in data.get("all_observed_bets", []):
                    if isinstance(bet.get("timestamp"), str):
                        bet["timestamp"] = datetime.datetime.fromisoformat(bet["timestamp"])
                    if isinstance(bet.get("settled_at"), str) and bet.get("settled_at"):
                        bet["settled_at"] = datetime.datetime.fromisoformat(bet["settled_at"])
                ml_learning_data = data
                
                # Load saved ML_CONFIG if it exists
                if "ml_config" in data:
                    saved_config = data["ml_config"]
                    # Update ML_CONFIG with saved values (preserve defaults for any missing keys)
                    for key, value in saved_config.items():
                        if key in ML_CONFIG:
                            ML_CONFIG[key] = value
                    print(f"🧠 ML Config: Loaded saved settings (learning_mode={ML_CONFIG.get('learning_mode')})")
                # Backwards-compatibility: older code/readers used 'observed_bets'
                # Ensure 'observed_bets' points to the same list as 'all_observed_bets'
                if "observed_bets" not in ml_learning_data:
                    ml_learning_data["observed_bets"] = ml_learning_data.get("all_observed_bets", [])
                
                print(f"🧠 ML System: Loaded {len(ml_learning_data.get('all_observed_bets', []))} learning samples")
                print(f"🧠 ML System: Loaded {len(ml_learning_data.get('profitable_patterns', {}))} profitable patterns")
    except json.JSONDecodeError as e:
        print(f"⚠️ ML data file is corrupted: {e}")
        print(f"🔄 Creating backup and starting fresh...")
        # Backup the corrupted file
        try:
            backup_file = ML_DATA_FILE + ".corrupted_backup"
            if os.path.exists(ML_DATA_FILE):
                os.rename(ML_DATA_FILE, backup_file)
                print(f"📁 Corrupted file backed up to: {backup_file}")
        except Exception as backup_err:
            print(f"⚠️ Could not backup corrupted file: {backup_err}")
        # Reset to empty state
        ml_learning_data = {
            "all_observed_bets": [],
            "profitable_patterns": {},
            "filter_changes": [],
            "last_analysis": None,
            "discoveries": [],
        }
        print(f"🧠 ML System: Starting with fresh data")
    except Exception as e:
        print(f"⚠️ Failed to load ML data: {e}")

def cleanup_ml_data():
    """Clean up old ML data to prevent memory issues"""
    global ml_learning_data
    try:
        bets = ml_learning_data.get("all_observed_bets", [])
        if not bets:
            return
        
        now = datetime.datetime.now(datetime.timezone.utc)
        max_age = datetime.timedelta(days=ML_CONFIG.get("max_bet_age_days", 90))
        max_bets = ML_CONFIG.get("max_observed_bets", 50000)
        
        original_count = len(bets)
        
        # Remove bets older than max_age
        cutoff = now - max_age
        bets = [
            bet for bet in bets
            if isinstance(bet.get("timestamp"), datetime.datetime) and bet["timestamp"] > cutoff
        ]
        
        # If still too many, keep the most recent ones
        if len(bets) > max_bets:
            bets.sort(key=lambda x: x.get("timestamp", datetime.datetime.min), reverse=True)
            bets = bets[:max_bets]
        
        ml_learning_data["all_observed_bets"] = bets
        
        removed = original_count - len(bets)
        if removed > 0:
            print(f"🧹 ML Cleanup: Removed {removed} old bets, {len(bets)} remaining")
            save_ml_data()  # Persist the cleanup
            
    except Exception as e:
        print(f"⚠️ Error cleaning ML data: {e}")

def import_existing_bet_history():
    """Import existing bet history from tracked_bets into ML learning data"""
    global ml_learning_data
    try:
        # This will be called after tracked_bets is loaded
        from_bet_history = globals().get("tracked_bets", [])
        if not from_bet_history:
            return 0
        
        existing_ids = set()
        for bet in ml_learning_data.get("all_observed_bets", []):
            # Create a unique ID based on runner name + timestamp
            bet_id = f"{bet.get('runner_name', '')}_{bet.get('event_name', '')}_{str(bet.get('timestamp', ''))[:10]}"
            existing_ids.add(bet_id)
        
        imported = 0
        for bet in from_bet_history:
            # Create ID to check for duplicates
            bet_id = f"{bet.get('runner_name', '')}_{bet.get('event_name', '')}_{str(bet.get('timestamp', ''))[:10]}"
            if bet_id in existing_ids:
                continue
            
            # Extract data
            timestamp = bet.get("timestamp", datetime.datetime.now(datetime.timezone.utc))
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            odds = bet.get("odds")
            runner_name = bet.get("runner_name", "")
            event_name = bet.get("event_name", "")
            market_name = bet.get("market_name", "")
            volume = bet.get("volume", 0)
            
            # Convert tracked_bet format to ML format with ALL features
            ml_bet = {
                "timestamp": timestamp,
                "bet_type": bet.get("bet_type", "unknown"),
                "sport_key": bet.get("sport_key"),
                "odds": odds,
                "bet_amount": volume,
                "total_matched": volume,
                "matched_percentage": 100.0,
                "runner_name": runner_name,
                "event_name": event_name,
                "market_name": market_name,
                "recommended_units": bet.get("recommended_units"),
                "was_posted": True,
                "filter_reason": None,
                "status": bet.get("status", "pending"),
                "pnl": bet.get("pnl", 0.0),
                "settled_at": bet.get("settled_at"),
                "is_au_race": None,  # Unknown for historical
                
                # Time features
                "hour_of_day": timestamp.hour if isinstance(timestamp, datetime.datetime) else 12,
                "day_of_week": timestamp.weekday() if isinstance(timestamp, datetime.datetime) else 0,
                "is_weekend": timestamp.weekday() >= 5 if isinstance(timestamp, datetime.datetime) else False,
                "time_bucket": categorize_time(timestamp.hour) if isinstance(timestamp, datetime.datetime) else "afternoon",
                
                # Odds features
                "odds_bucket": categorize_odds(odds),
                "is_favorite": odds is not None and odds < 3.0,
                "is_longshot": odds is not None and odds > 10.0,
                
                # Amount features
                "amount_bucket": categorize_amount(volume),
                
                # Selection/Team features
                "selection_first_letter": runner_name[0].upper() if runner_name else "",
                "selection_word_count": len(runner_name.split()) if runner_name else 0,
                
                # Event features
                "venue": extract_venue(event_name),
                "is_group_match": "group" in event_name.lower() if event_name else False,
                "is_final": any(x in event_name.lower() for x in ["final", "championship", "cup"]) if event_name else False,
                "is_derby": "derby" in event_name.lower() or "derby" in market_name.lower() if event_name or market_name else False,
                
                # Market features
                "market_type": categorize_market(market_name),
            }
            ml_learning_data["all_observed_bets"].append(ml_bet)
            imported += 1
        
        if imported > 0:
            save_ml_data()
            print(f"🧠 ML: Imported {imported} bets from existing bet history!")
        
        return imported
    except Exception as e:
        print(f"⚠️ Failed to import bet history to ML: {e}")
        return 0

def save_ml_data():
    """Save ML learning data to disk"""
    try:
        serializable = {**ml_learning_data}
        serializable["all_observed_bets"] = []
        for bet in ml_learning_data.get("all_observed_bets", []):
            item = {**bet}
            if isinstance(bet.get("timestamp"), datetime.datetime):
                item["timestamp"] = bet["timestamp"].isoformat()
            if isinstance(bet.get("settled_at"), datetime.datetime):
                item["settled_at"] = bet["settled_at"].isoformat()
            serializable["all_observed_bets"].append(item)
        
        # Also save ML_CONFIG so settings persist across restarts
        serializable["ml_config"] = ML_CONFIG
        # Backwards-compatibility: include observed_bets alias for older API consumers
        serializable["observed_bets"] = serializable.get("all_observed_bets", [])
        
        with open(ML_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save ML data: {e}")

def record_observed_bet(bet_data: dict, was_posted: bool, filter_reason: str = None):
    """Record a bet for ML learning - INCLUDES bets that were filtered out"""
    now = datetime.datetime.now(datetime.timezone.utc)
    odds = bet_data.get("odds")
    runner_name = bet_data.get("runner_name", "")
    event_name = bet_data.get("event_name", "")
    market_name = bet_data.get("market_name", "")
    bet_amount = bet_data.get("bet_amount")
    matched_pct = bet_data.get("matched_percentage")
    
    observed = {
        "timestamp": now,
        "bet_type": bet_data.get("bet_type", "unknown"),
        "sport_key": bet_data.get("sport_key"),
        "odds": odds,
        "bet_amount": bet_amount,
        "total_matched": bet_data.get("total_matched"),
        "matched_percentage": matched_pct,
        "runner_name": runner_name,
        "event_name": event_name,
        "market_name": market_name,
        "recommended_units": bet_data.get("recommended_units"),
        "was_posted": was_posted,
        "filter_reason": filter_reason,
        "status": "pending",
        "pnl": 0.0,
        "settled_at": None,
        "is_au_race": bet_data.get("is_au_race"),
        
        # Stake update tracking
        "is_stake_update": bet_data.get("is_stake_update", False),
        "previous_total": bet_data.get("previous_total"),
        
        # === FEATURE EXTRACTION FOR ML ===
        # Time features
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": now.weekday() >= 5,
        "time_bucket": categorize_time(now.hour),
        
        # Odds features
        "odds_bucket": categorize_odds(odds),
        "is_favorite": odds is not None and odds < 3.0,
        "is_longshot": odds is not None and odds > 10.0,
        "odds_type": categorize_odds_movement(odds),
        
        # Amount features
        "amount_bucket": categorize_amount(bet_amount),
        
        # Selection/Team features
        "selection_first_letter": runner_name[0].upper() if runner_name else "",
        "selection_word_count": len(runner_name.split()) if runner_name else 0,
        
        # Event features
        "venue": extract_venue(event_name),
        "race_number": extract_race_number(event_name),
        "race_category": categorize_race_number(extract_race_number(event_name)),
        "is_group_match": "group" in event_name.lower() if event_name else False,
        "is_final": any(x in event_name.lower() for x in ["final", "championship", "cup"]) if event_name else False,
        "is_derby": "derby" in event_name.lower() or "derby" in market_name.lower() if event_name or market_name else False,
        
        # Market features
        "market_type": categorize_market(market_name),
        
        # === NEW ADVANCED FEATURES ===
        # Value/confidence indicators
        "value_indicator": calculate_value_indicator(odds, matched_pct),
        
        # Bet sizing relative to odds (kelly-like analysis)
        "stake_to_odds_ratio": (bet_amount / odds) if odds and bet_amount else None,
        "is_large_stake_longshot": bet_amount and odds and bet_amount >= 5000 and odds >= 5.0,
        "is_confident_bet": matched_pct and matched_pct >= 80,
        
        # Time until event (if available)
        "minutes_to_start": bet_data.get("minutes_to_start"),
        "is_late_bet": bet_data.get("minutes_to_start") is not None and bet_data.get("minutes_to_start") <= 5,
        "is_early_bet": bet_data.get("minutes_to_start") is not None and bet_data.get("minutes_to_start") >= 60,
        
        # Streak tracking (will be populated by analysis)
        "consecutive_wins_before": 0,
        "consecutive_losses_before": 0,
    }
    ml_learning_data["all_observed_bets"].append(observed)
    save_ml_data()
    return observed

def categorize_time(hour: int) -> str:
    """Categorize time of day into periods"""
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 21:
        return "evening"
    else:
        return "late_night"

def extract_venue(event_name: str) -> str:
    """Extract venue/location from event name"""
    if not event_name:
        return "unknown"
    # Common patterns: "Venue Name - Race X" or "Team vs Team"
    parts = event_name.split(" - ")
    if len(parts) > 1:
        return parts[0].strip().lower()
    return event_name.split()[0].lower() if event_name else "unknown"

def extract_race_number(event_name: str) -> int:
    """Extract race number from event name (e.g., 'Flemington - Race 5' -> 5)"""
    if not event_name:
        return 0
    import re
    match = re.search(r'race\s*(\d+)', event_name.lower())
    if match:
        return int(match.group(1))
    return 0

def categorize_race_number(race_num: int) -> str:
    """Categorize race number (early races vs feature races vs late races)"""
    if race_num == 0:
        return "unknown"
    elif race_num <= 3:
        return "early_races"
    elif race_num <= 6:
        return "mid_races"
    elif race_num <= 8:
        return "feature_races"  # Often the best races
    else:
        return "late_races"

def calculate_value_indicator(odds: float, matched_pct: float) -> str:
    """
    Calculate if bet shows value based on odds vs matched percentage.
    High matched % at longer odds = sharps backing it = potential value
    """
    if odds is None or matched_pct is None:
        return "unknown"
    
    # Value score: higher matched % at longer odds is better
    if odds >= 5.0 and matched_pct >= 80:
        return "high_value"
    elif odds >= 3.0 and matched_pct >= 70:
        return "good_value"
    elif odds < 2.0 and matched_pct >= 90:
        return "confident_favorite"
    elif matched_pct < 50:
        return "low_confidence"
    else:
        return "neutral"

def categorize_odds_movement(current_odds: float, implied_prob: float = None) -> str:
    """Categorize based on odds characteristics"""
    if current_odds is None:
        return "unknown"
    
    # Round number odds often indicate opening prices
    if current_odds in [2.0, 3.0, 4.0, 5.0, 10.0]:
        return "round_odds"
    # Precise odds suggest movement/adjustment
    elif len(str(current_odds).split('.')[-1]) >= 2:
        return "precise_odds"
    else:
        return "normal_odds"

def categorize_market(market_name: str) -> str:
    """Categorize market type"""
    if not market_name:
        return "unknown"
    market_lower = market_name.lower()
    if "win" in market_lower or "winner" in market_lower:
        return "win"
    elif "place" in market_lower:
        return "place"
    elif "over" in market_lower or "under" in market_lower:
        return "over_under"
    elif "handicap" in market_lower:
        return "handicap"
    elif "first" in market_lower:
        return "first_scorer"
    elif "correct" in market_lower:
        return "correct_score"
    elif "both" in market_lower and "score" in market_lower:
        return "btts"
    else:
        return "other"

def categorize_odds(odds: float) -> str:
    """Categorize odds into buckets for pattern analysis"""
    if odds is None:
        return "unknown"
    if odds < 1.5:
        return "very_short"
    elif odds < 2.0:
        return "short"
    elif odds < 3.0:
        return "medium_short"
    elif odds < 5.0:
        return "medium"
    elif odds < 10.0:
        return "medium_long"
    elif odds < 20.0:
        return "long"
    else:
        return "very_long"

def categorize_amount(amount: float) -> str:
    """Categorize bet amount into buckets for ML analysis"""
    if amount is None:
        return "unknown"
    if amount < 500:
        return "under_500"
    elif amount < 1000:
        return "500_to_1k"
    elif amount < 2000:
        return "1k_to_2k"
    elif amount < 3000:
        return "2k_to_3k"
    elif amount < 5000:
        return "3k_to_5k"
    elif amount < 7500:
        return "5k_to_7.5k"
    elif amount < 10000:
        return "7.5k_to_10k"
    elif amount < 15000:
        return "10k_to_15k"
    elif amount < 25000:
        return "15k_to_25k"
    elif amount < 50000:
        return "25k_to_50k"
    elif amount < 100000:
        return "50k_to_100k"
    else:
        return "100k_plus"

def get_amount_range_display(bucket: str) -> str:
    """Convert bucket name to readable display"""
    displays = {
        "under_500": "$0-$500",
        "500_to_1k": "$500-$1K",
        "1k_to_2k": "$1K-$2K",
        "2k_to_3k": "$2K-$3K",
        "3k_to_5k": "$3K-$5K",
        "5k_to_7.5k": "$5K-$7.5K",
        "7.5k_to_10k": "$7.5K-$10K",
        "10k_to_15k": "$10K-$15K",
        "15k_to_25k": "$15K-$25K",
        "25k_to_50k": "$25K-$50K",
        "50k_to_100k": "$50K-$100K",
        "100k_plus": "$100K+",
    }
    return displays.get(bucket, bucket)

async def notify_admin(message: str, embed: discord.Embed = None, notification_type: str = None):
    """Send a notification to the admin user about algorithm changes/discoveries
    
    Args:
        message: The message to send
        embed: Optional embed to include
        notification_type: Type of notification for cooldown checking (e.g., "reconnect", "startup", "ml_status")
                          If None, message is always sent immediately
    """
    try:
        # Check cooldown if notification type is specified
        if notification_type and not can_send_notification(notification_type):
            print(f"📧 Skipping {notification_type} notification (cooldown active)")
            return
        
        user = await bot.fetch_user(ADMIN_USER_ID)
        if user:
            if embed:
                await user.send(content=message, embed=embed)
            else:
                await user.send(message)
            print(f"📧 Notified admin: {message[:50]}...")
    except Exception as e:
        print(f"⚠️ Failed to notify admin: {e}")

def analyze_profitability_by_pattern(bets: List[dict], pattern_key: str, is_combo: bool = False) -> Dict[str, Any]:
    """Analyze profitability grouped by a specific pattern key
    
    Args:
        bets: List of bet dictionaries
        pattern_key: The key to group by
        is_combo: If True, uses lower sample size threshold for combination patterns
    """
    try:
        grouped = defaultdict(list)
        for bet in bets:
            if bet.get("status") in ["won", "lost"]:  # Only settled bets
                key_value = bet.get(pattern_key, "unknown")
                grouped[key_value].append(bet)
        
        # Use lower threshold for combo patterns
        min_samples = ML_CONFIG.get("min_sample_size_combo", 10) if is_combo else ML_CONFIG["min_sample_size"]
        
        results = {}
        for key_value, group_bets in grouped.items():
            if len(group_bets) < min_samples:
                continue
            
            won = [b for b in group_bets if b["status"] == "won"]
            lost = [b for b in group_bets if b["status"] == "lost"]
            total = len(won) + len(lost)
            win_rate = len(won) / total if total > 0 else 0
            
            total_pnl = sum(b.get("pnl", 0) for b in group_bets)
            
            # Safely calculate average odds
            odds_list = [b["odds"] for b in group_bets if b.get("odds") and isinstance(b.get("odds"), (int, float))]
            avg_odds = statistics.mean(odds_list) if odds_list else 2.0
            
            results[key_value] = {
                "sample_size": total,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_odds": avg_odds,
                "wins": len(won),
                "losses": len(lost),
                "profitable": win_rate >= ML_CONFIG["confidence_threshold"] and total_pnl >= ML_CONFIG["min_profit_units"]
            }
        
        return results
    except Exception as e:
        print(f"⚠️ Error in analyze_profitability_by_pattern: {e}")
        return {}

def find_profitable_combinations() -> List[Dict]:
    """Find profitable betting patterns across multiple dimensions"""
    try:
        bets = ml_learning_data.get("all_observed_bets", [])
        settled_bets = [b for b in bets if b.get("status") in ["won", "lost"]]
        
        print(f"🧠 Pattern search: {len(settled_bets)} settled bets available")
        
        if len(settled_bets) < ML_CONFIG["min_sample_size"]:
            print(f"🧠 Not enough settled bets for pattern search ({len(settled_bets)} < {ML_CONFIG['min_sample_size']})")
            return []
        
        discoveries = []
        
        # Analyze by bet type
        by_type = analyze_profitability_by_pattern(settled_bets, "bet_type")
        print(f"🧠 By type analysis: {len(by_type)} categories found")
        for bet_type, stats in by_type.items():
            print(f"   - {bet_type}: {stats.get('sample_size')} bets, {stats.get('win_rate', 0)*100:.1f}% win rate, profitable={stats.get('profitable')}")
            if stats.get("profitable"):
                discoveries.append({
                    "pattern": f"bet_type={bet_type}",
                    "description": f"{bet_type.title()} bets",
                    "category": "bet_type",
                    "recommended_action": f"FOCUS on {bet_type} bets",
                    **stats
                })
        
        # Analyze by odds bucket
        by_odds = analyze_profitability_by_pattern(settled_bets, "odds_bucket")
        print(f"🧠 By odds analysis: {len(by_odds)} categories found")
        for odds_bucket, stats in by_odds.items():
            if stats.get("profitable"):
                discoveries.append({
                    "pattern": f"odds_bucket={odds_bucket}",
                    "description": f"Odds range: {odds_bucket.replace('_', ' ')}",
                    "category": "odds",
                    "recommended_action": f"Target {odds_bucket.replace('_', ' ')} odds",
                    **stats
                })
        
        # Analyze by bet amount bucket - CRITICAL for finding optimal minimum
        by_amount = analyze_profitability_by_pattern(settled_bets, "amount_bucket")
        print(f"🧠 By amount analysis: {len(by_amount)} categories found")
        for amount_bucket, stats in by_amount.items():
            if stats.get("profitable"):
                amount_display = get_amount_range_display(amount_bucket)
                discoveries.append({
                    "pattern": f"amount_bucket={amount_bucket}",
                    "description": f"Bet size: {amount_display}",
                    "category": "amount",
                    "recommended_action": f"Set minimum to capture {amount_display} bets",
                    **stats
                })
        
        # Analyze by hour of day
        by_hour = analyze_profitability_by_pattern(settled_bets, "hour_of_day")
        for hour, stats in by_hour.items():
            if stats.get("profitable"):
                discoveries.append({
                    "pattern": f"hour={hour}",
                    "description": f"Bets at {hour}:00 UTC",
                    "category": "time",
                    "recommended_action": f"Most profitable at {hour}:00 UTC",
                    **stats
                })
        
        # Analyze by day of week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = analyze_profitability_by_pattern(settled_bets, "day_of_week")
        for day_num, stats in by_day.items():
            if stats.get("profitable"):
                day_name = days[int(day_num)] if isinstance(day_num, (int, str)) and str(day_num).isdigit() else str(day_num)
                discoveries.append({
                    "pattern": f"day={day_num}",
                    "description": f"Bets on {day_name}",
                    "category": "time",
                    "recommended_action": f"Focus on {day_name}",
                    **stats
                })
        
        # Analyze by sport (for sports bets)
        sports_bets = [b for b in settled_bets if b.get("bet_type") == "sports" and b.get("sport_key")]
        by_sport = analyze_profitability_by_pattern(sports_bets, "sport_key")
        for sport, stats in by_sport.items():
            if stats.get("profitable"):
                discoveries.append({
                    "pattern": f"sport={sport}",
                    "description": f"Sport: {sport.replace('_', ' ').title()}",
                    "category": "sport",
                    "recommended_action": f"Enable/focus on {sport.replace('_', ' ').title()}",
                    **stats
                })
        
        # Cross-dimensional analysis: bet_type + amount_bucket (find best minimums per type)
        for bet_type in ["horse", "greyhound", "sports"]:
            type_bets = [b for b in settled_bets if b.get("bet_type") == bet_type]
            by_amount_combo = analyze_profitability_by_pattern(type_bets, "amount_bucket", is_combo=True)
            for amount_bucket, stats in by_amount_combo.items():
                if stats.get("profitable"):
                    amount_display = get_amount_range_display(amount_bucket)
                    discoveries.append({
                        "pattern": f"bet_type={bet_type}&amount_bucket={amount_bucket}",
                        "description": f"{bet_type.title()} bets at {amount_display}",
                        "category": "combo",
                        "recommended_action": f"Set {bet_type} minimum to capture {amount_display}",
                        **stats
                    })
        
        # Cross-dimensional analysis: bet_type + odds_bucket
        for bet_type in ["horse", "greyhound", "sports"]:
            type_bets = [b for b in settled_bets if b.get("bet_type") == bet_type]
            by_odds_combo = analyze_profitability_by_pattern(type_bets, "odds_bucket", is_combo=True)
            for odds_bucket, stats in by_odds_combo.items():
                if stats.get("profitable"):
                    discoveries.append({
                        "pattern": f"bet_type={bet_type}&odds_bucket={odds_bucket}",
                        "description": f"{bet_type.title()} with {odds_bucket.replace('_', ' ')} odds",
                        "category": "combo",
                        "recommended_action": f"Target {bet_type} at {odds_bucket.replace('_', ' ')} odds",
                        **stats
                    })
        
        # Sport + amount analysis
        for sport in set(b.get("sport_key") for b in sports_bets if b.get("sport_key")):
            sport_specific = [b for b in sports_bets if b.get("sport_key") == sport]
            by_amount_sport = analyze_profitability_by_pattern(sport_specific, "amount_bucket", is_combo=True)
            for amount_bucket, stats in by_amount_sport.items():
                if stats.get("profitable"):
                    amount_display = get_amount_range_display(amount_bucket)
                    discoveries.append({
                        "pattern": f"sport={sport}&amount_bucket={amount_bucket}",
                        "description": f"{sport.replace('_', ' ').title()} at {amount_display}",
                        "category": "sport_combo",
                        "recommended_action": f"Set {sport} min to capture {amount_display}",
                        **stats
                    })
        
        # Analyze by time bucket (morning/afternoon/evening/night)
        by_time_bucket = analyze_profitability_by_pattern(settled_bets, "time_bucket")
        for time_bucket, stats in by_time_bucket.items():
            if stats.get("profitable"):
                discoveries.append({
                    "pattern": f"time_bucket={time_bucket}",
                    "description": f"Bets in the {time_bucket}",
                    "category": "time",
                    "recommended_action": f"Most profitable in the {time_bucket}",
                    **stats
                })
        
        # Analyze weekends vs weekdays
        by_weekend = analyze_profitability_by_pattern(settled_bets, "is_weekend")
        for is_weekend, stats in by_weekend.items():
            if stats.get("profitable"):
                label = "Weekend" if is_weekend else "Weekday"
                discoveries.append({
                    "pattern": f"is_weekend={is_weekend}",
                    "description": f"{label} bets",
                    "category": "time",
                    "recommended_action": f"Focus on {label}s",
                    **stats
                })
        
        # Analyze favorites vs longshots
        by_favorite = analyze_profitability_by_pattern(settled_bets, "is_favorite")
        for is_fav, stats in by_favorite.items():
            if stats.get("profitable"):
                label = "Favorites (odds < 3.0)" if is_fav else "Non-favorites (odds >= 3.0)"
                discoveries.append({
                    "pattern": f"is_favorite={is_fav}",
                    "description": label,
                    "category": "odds",
                    "recommended_action": f"Target {'favorites' if is_fav else 'value bets'}",
                    **stats
                })
        
        by_longshot = analyze_profitability_by_pattern(settled_bets, "is_longshot")
        for is_long, stats in by_longshot.items():
            if stats.get("profitable"):
                label = "Longshots (odds > 10.0)" if is_long else "Shorter odds (<=10.0)"
                discoveries.append({
                    "pattern": f"is_longshot={is_long}",
                    "description": label,
                    "category": "odds",
                    "recommended_action": f"{'Target longshots' if is_long else 'Avoid longshots'}",
                    **stats
                })
        
        # Analyze by venue (for horse racing)
        horse_bets = [b for b in settled_bets if b.get("bet_type") == "horse"]
        by_venue = analyze_profitability_by_pattern(horse_bets, "venue")
        for venue, stats in by_venue.items():
            if stats.get("profitable") and venue != "unknown":
                discoveries.append({
                    "pattern": f"venue={venue}",
                    "description": f"Venue: {venue.title()}",
                    "category": "venue",
                    "recommended_action": f"Strong results at {venue.title()}",
                    **stats
                })
        
        # Analyze AU vs UK/ROW horses
        by_au_race = analyze_profitability_by_pattern(horse_bets, "is_au_race")
        for is_au, stats in by_au_race.items():
            if stats.get("profitable"):
                label = "AU Horses" if is_au else "UK/International Horses"
                discoveries.append({
                    "pattern": f"is_au_race={is_au}",
                    "description": label,
                    "category": "region",
                    "recommended_action": f"Focus on {label}",
                    **stats
                })
        
        # Analyze by market type
        by_market = analyze_profitability_by_pattern(settled_bets, "market_type")
        for market_type, stats in by_market.items():
            if stats.get("profitable") and market_type != "unknown":
                discoveries.append({
                    "pattern": f"market_type={market_type}",
                    "description": f"Market: {market_type.replace('_', ' ').title()}",
                    "category": "market",
                    "recommended_action": f"Target {market_type.replace('_', ' ')} markets",
                    **stats
                })
        
        # Analyze finals/championships vs regular matches
        by_final = analyze_profitability_by_pattern(sports_bets, "is_final")
        for is_final, stats in by_final.items():
            if stats.get("profitable"):
                label = "Finals/Championships" if is_final else "Regular matches"
                discoveries.append({
                    "pattern": f"is_final={is_final}",
                    "description": label,
                    "category": "event_type",
                    "recommended_action": f"{'Target big events' if is_final else 'Focus on regular season'}",
                    **stats
                })
        
        # Analyze bets where stake was increased
        by_stake_update = analyze_profitability_by_pattern(settled_bets, "is_stake_update")
        for is_update, stats in by_stake_update.items():
            if stats.get("profitable") and is_update:
                discoveries.append({
                    "pattern": f"is_stake_update={is_update}",
                    "description": "Bets with stake increases",
                    "category": "stake_behavior",
                    "recommended_action": "Pay extra attention when stake increases on a bet",
                    **stats
                })
        
        # Stake updates by bet type
        for bet_type in ["horse", "greyhound", "sports"]:
            type_updates = [b for b in settled_bets if b.get("bet_type") == bet_type and b.get("is_stake_update")]
            if len(type_updates) >= 5:
                wins = len([b for b in type_updates if b["status"] == "won"])
                losses = len([b for b in type_updates if b["status"] == "lost"])
                total = wins + losses
                if total > 0:
                    win_rate = wins / total
                    pnl = sum(b.get("pnl", 0) for b in type_updates)
                    if win_rate >= ML_CONFIG["confidence_threshold"]:
                        discoveries.append({
                            "pattern": f"bet_type={bet_type}&is_stake_update=True",
                            "description": f"{bet_type.title()} bets with stake increases",
                            "category": "stake_combo",
                            "recommended_action": f"Stake increases on {bet_type} bets are profitable",
                            "sample_size": total,
                            "wins": wins,
                            "losses": losses,
                            "win_rate": win_rate,
                            "total_pnl": pnl,
                            "profitable": True
                        })
        
        # Stake updates by amount bucket
        stake_update_bets = [b for b in settled_bets if b.get("is_stake_update")]
        by_update_amount = analyze_profitability_by_pattern(stake_update_bets, "amount_bucket", is_combo=True)
        for amount_bucket, stats in by_update_amount.items():
            if stats.get("profitable"):
                amount_display = get_amount_range_display(amount_bucket)
                discoveries.append({
                    "pattern": f"is_stake_update=True&amount_bucket={amount_bucket}",
                    "description": f"Stake increases of {amount_display}",
                    "category": "stake_amount",
                    "recommended_action": f"Stake additions of {amount_display} are profitable",
                    **stats
                })
        
        # Triple combination analysis: bet_type + time_bucket + odds_bucket
        for bet_type in ["horse", "greyhound", "sports"]:
            type_bets = [b for b in settled_bets if b.get("bet_type") == bet_type]
            for time_bucket in ["morning", "afternoon", "evening", "night", "late_night"]:
                time_type_bets = [b for b in type_bets if b.get("time_bucket") == time_bucket]
                by_odds_triple = analyze_profitability_by_pattern(time_type_bets, "odds_bucket", is_combo=True)
                for odds_bucket, stats in by_odds_triple.items():
                    if stats.get("profitable") and stats.get("sample_size", 0) >= 8:
                        discoveries.append({
                            "pattern": f"bet_type={bet_type}&time={time_bucket}&odds={odds_bucket}",
                            "description": f"{bet_type.title()} in {time_bucket} at {odds_bucket.replace('_', ' ')} odds",
                            "category": "triple_combo",
                            "recommended_action": f"Sweet spot: {bet_type} {time_bucket} {odds_bucket.replace('_', ' ')}",
                            **stats
                        })
        
        # === NEW ADVANCED ANALYSES ===
        
        # Analyze by value indicator (high value bets from sharps)
        by_value = analyze_profitability_by_pattern(settled_bets, "value_indicator")
        for value_ind, stats in by_value.items():
            if stats.get("profitable") and value_ind != "unknown":
                discoveries.append({
                    "pattern": f"value_indicator={value_ind}",
                    "description": f"Value type: {value_ind.replace('_', ' ').title()}",
                    "category": "value",
                    "recommended_action": f"Target '{value_ind.replace('_', ' ')}' bets",
                    **stats
                })
        
        # Analyze by race number/category (feature races often more predictable)
        by_race_cat = analyze_profitability_by_pattern(horse_bets, "race_category")
        for race_cat, stats in by_race_cat.items():
            if stats.get("profitable") and race_cat != "unknown":
                discoveries.append({
                    "pattern": f"race_category={race_cat}",
                    "description": f"Race type: {race_cat.replace('_', ' ').title()}",
                    "category": "race_timing",
                    "recommended_action": f"Focus on {race_cat.replace('_', ' ')} for horses",
                    **stats
                })
        
        # Analyze confident bets (high matched %)
        by_confident = analyze_profitability_by_pattern(settled_bets, "is_confident_bet")
        for is_conf, stats in by_confident.items():
            if stats.get("profitable") and is_conf:
                discoveries.append({
                    "pattern": f"is_confident_bet=True",
                    "description": "High confidence bets (80%+ matched)",
                    "category": "confidence",
                    "recommended_action": "Prioritize bets with 80%+ matched percentage",
                    **stats
                })
        
        # Analyze large stake longshots (whale activity)
        by_whale = analyze_profitability_by_pattern(settled_bets, "is_large_stake_longshot")
        for is_whale, stats in by_whale.items():
            if stats.get("profitable") and is_whale:
                discoveries.append({
                    "pattern": f"is_large_stake_longshot=True",
                    "description": "Whale bets ($5K+ on odds 5.0+)",
                    "category": "whale_activity",
                    "recommended_action": "ALERT: Large stakes on longshots are profitable!",
                    **stats
                })
        
        # Analyze late bets (close to race start - often informed money)
        by_late = analyze_profitability_by_pattern(settled_bets, "is_late_bet")
        for is_late, stats in by_late.items():
            if stats.get("profitable") and is_late:
                discoveries.append({
                    "pattern": f"is_late_bet=True",
                    "description": "Late bets (within 5 mins of start)",
                    "category": "timing",
                    "recommended_action": "Late money often = informed money",
                    **stats
                })
        
        # Analyze early bets
        by_early = analyze_profitability_by_pattern(settled_bets, "is_early_bet")
        for is_early, stats in by_early.items():
            if stats.get("profitable") and is_early:
                discoveries.append({
                    "pattern": f"is_early_bet=True",
                    "description": "Early bets (60+ mins before start)",
                    "category": "timing",
                    "recommended_action": "Early money shows confidence",
                    **stats
                })
        
        # Analyze by odds type (round vs precise odds)
        by_odds_type = analyze_profitability_by_pattern(settled_bets, "odds_type")
        for odds_type, stats in by_odds_type.items():
            if stats.get("profitable") and odds_type != "unknown":
                discoveries.append({
                    "pattern": f"odds_type={odds_type}",
                    "description": f"Odds pattern: {odds_type.replace('_', ' ').title()}",
                    "category": "odds_analysis",
                    "recommended_action": f"Watch for {odds_type.replace('_', ' ')} odds",
                    **stats
                })
        
        # === CROSS-ANALYSIS: Value + Bet Type ===
        for bet_type in ["horse", "greyhound", "sports"]:
            type_bets = [b for b in settled_bets if b.get("bet_type") == bet_type]
            by_value_type = analyze_profitability_by_pattern(type_bets, "value_indicator", is_combo=True)
            for value_ind, stats in by_value_type.items():
                if stats.get("profitable") and value_ind in ["high_value", "good_value"]:
                    discoveries.append({
                        "pattern": f"bet_type={bet_type}&value_indicator={value_ind}",
                        "description": f"{bet_type.title()} with {value_ind.replace('_', ' ')}",
                        "category": "value_combo",
                        "recommended_action": f"Target {value_ind.replace('_', ' ')} {bet_type} bets",
                        **stats
                    })
        
        # === CROSS-ANALYSIS: Venue + Race Category (horse racing sweet spots) ===
        for venue in set(b.get("venue") for b in horse_bets if b.get("venue") and b.get("venue") != "unknown"):
            venue_bets = [b for b in horse_bets if b.get("venue") == venue]
            if len(venue_bets) >= 10:  # Lowered from 20 for faster pattern detection
                by_race_venue = analyze_profitability_by_pattern(venue_bets, "race_category", is_combo=True)
                for race_cat, stats in by_race_venue.items():
                    if stats.get("profitable") and race_cat != "unknown":
                        discoveries.append({
                            "pattern": f"venue={venue}&race_category={race_cat}",
                            "description": f"{venue.title()} - {race_cat.replace('_', ' ').title()}",
                            "category": "venue_race_combo",
                            "recommended_action": f"Sweet spot: {venue.title()} {race_cat.replace('_', ' ')}",
                            **stats
                        })
        
        # === LOSING PATTERN DETECTION (what to AVOID) ===
        # Find patterns with consistently bad results
        losing_patterns = []
        
        # Check for losing bet types
        for bet_type, stats in by_type.items():
            if stats.get("sample_size", 0) >= ML_CONFIG["min_sample_size"]:
                if stats.get("win_rate", 0.5) < 0.40 and stats.get("total_pnl", 0) < -5:
                    losing_patterns.append({
                        "pattern": f"AVOID: bet_type={bet_type}",
                        "description": f"⚠️ LOSING: {bet_type.title()} bets",
                        "category": "avoid",
                        "recommended_action": f"REDUCE exposure to {bet_type} bets",
                        "sample_size": stats.get("sample_size"),
                        "win_rate": stats.get("win_rate"),
                        "total_pnl": stats.get("total_pnl"),
                        "wins": stats.get("wins"),
                        "losses": stats.get("losses"),
                        "profitable": False
                    })
        
        # Check for losing odds ranges
        for odds_bucket, stats in by_odds.items():
            if stats.get("sample_size", 0) >= ML_CONFIG["min_sample_size"]:
                if stats.get("win_rate", 0.5) < 0.35 and stats.get("total_pnl", 0) < -10:
                    losing_patterns.append({
                        "pattern": f"AVOID: odds_bucket={odds_bucket}",
                        "description": f"⚠️ LOSING: {odds_bucket.replace('_', ' ')} odds",
                        "category": "avoid",
                        "recommended_action": f"AVOID {odds_bucket.replace('_', ' ')} odds range",
                        "sample_size": stats.get("sample_size"),
                        "win_rate": stats.get("win_rate"),
                        "total_pnl": stats.get("total_pnl"),
                        "wins": stats.get("wins"),
                        "losses": stats.get("losses"),
                        "profitable": False
                    })
        
        # Add losing patterns to discoveries (they're still important!)
        discoveries.extend(losing_patterns)
        
        # Sort by profitability (win rate * pnl)
        discoveries.sort(key=lambda x: (x.get("win_rate", 0) * x.get("total_pnl", 0), x.get("sample_size", 0)), reverse=True)
        
        return discoveries
    except Exception as e:
        print(f"⚠️ Error in find_profitable_combinations: {e}")
        traceback.print_exc()
        return []

async def run_ml_analysis():
    """Run the ML analysis and notify admin of any discoveries"""
    print("🧠 Running ML pattern analysis...")
    
    # Get data stats FIRST for debugging
    all_bets = ml_learning_data.get("all_observed_bets", [])
    total_observed = len(all_bets)
    settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
    settled_count = len(settled)
    pending_count = total_observed - settled_count
    
    print(f"🧠 ML Stats: {total_observed} total, {settled_count} settled, {pending_count} pending")
    
    if settled_count < ML_CONFIG["min_sample_size"]:
        print(f"🧠 Not enough settled bets ({settled_count}/{ML_CONFIG['min_sample_size']}) - skipping pattern analysis")
    
    discoveries = find_profitable_combinations()
    ml_learning_data["last_analysis"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    print(f"🧠 Found {len(discoveries)} patterns meeting criteria")
    
    wins = len([b for b in settled if b["status"] == "won"])
    losses = len([b for b in settled if b["status"] == "lost"])
    overall_win_rate = (wins / settled_count * 100) if settled_count > 0 else 0
    total_pnl = sum(b.get("pnl", 0) for b in settled)
    
    # Count by type
    horse_bets = [b for b in settled if b.get("bet_type") == "horse"]
    greyhound_bets = [b for b in settled if b.get("bet_type") == "greyhound"]
    sports_bets = [b for b in settled if b.get("bet_type") == "sports"]
    stake_updates = [b for b in settled if b.get("is_stake_update")]
    big_bets = [b for b in settled if b.get("amount_bucket") in ["50k_to_100k", "100k_plus"]]
    
    print(f"🧠 By type: 🐎{len(horse_bets)} 🐕{len(greyhound_bets)} ⚽{len(sports_bets)} | Win rate: {overall_win_rate:.1f}%")
    
    if not discoveries:
        print("🧠 No profitable patterns found yet (need more data or better win rate)")
        # Only send status update occasionally (with cooldown) to prevent spam
        if settled_count >= 5:  # Lowered from 10
            status_msg = (
                f"🧠 **ML STATUS UPDATE**\n\n"
                f"📊 **Data Collected:**\n"
                f"• Total Observed: {total_observed:,} bets\n"
                f"• Settled: {settled_count:,} | Pending: {pending_count:,}\n"
                f"• Wins: {wins} | Losses: {losses}\n"
                f"• Overall Win Rate: {overall_win_rate:.1f}%\n"
                f"• Total P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:.1f} units\n\n"
                f"📈 **By Type:**\n"
                f"• 🐎 Horse: {len(horse_bets)} bets\n"
                f"• 🐕 Greyhound: {len(greyhound_bets)} bets\n"
                f"• ⚽ Sports: {len(sports_bets)} bets\n"
                f"• 🔄 Stake Updates: {len(stake_updates)} bets\n"
                f"• 💰 Big Bets ($50k+): {len(big_bets)} bets\n\n"
                f"⏳ **Status:** Need {ML_CONFIG['min_sample_size']} settled bets per pattern with >{ML_CONFIG['confidence_threshold']*100:.0f}% win rate.\n"
                f"*Your server settings remain unchanged.*"
            )
            await notify_admin(status_msg, notification_type="ml_status")
        return
    
    # Check for NEW discoveries not previously reported
    existing_patterns = set(ml_learning_data.get("profitable_patterns", {}).keys())
    new_discoveries = [d for d in discoveries if d["pattern"] not in existing_patterns]
    
    # Update profitable patterns
    for d in discoveries:
        ml_learning_data["profitable_patterns"][d["pattern"]] = d
    
    save_ml_data()
    
    if new_discoveries:
        # Sort by most impactful (win rate * sample size)
        new_discoveries.sort(key=lambda x: x["win_rate"] * x["sample_size"], reverse=True)
        
        # Create DETAILED notification message
        header_msg = (
            f"🚨 **PROFITABLE PATTERN{'S' if len(new_discoveries) > 1 else ''} DISCOVERED!**\n\n"
            f"The ML algorithm has found {len(new_discoveries)} new profitable betting pattern{'s' if len(new_discoveries) > 1 else ''}!\n"
            f"These are based on analyzing {settled_count:,} settled bets.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await notify_admin(header_msg)
        
        # Send each discovery with full details
        for i, d in enumerate(new_discoveries[:10], 1):  # Top 10 discoveries
            category = d.get("category", "unknown")
            pattern = d.get("pattern", "")
            description = d.get("description", "Unknown pattern")
            win_rate = d.get("win_rate", 0) * 100
            sample_size = d.get("sample_size", 0)
            total_pnl_pattern = d.get("total_pnl", 0)
            avg_odds = d.get("avg_odds", 2.0)
            wins_d = d.get("wins", 0)
            losses_d = d.get("losses", 0)
            action = d.get("recommended_action", "Continue monitoring")
            
            # Determine category emoji and explanation
            category_info = {
                "bet_type": ("🎯", "Bet Type Analysis"),
                "odds": ("📊", "Odds Range Analysis"),
                "amount": ("💰", "Bet Size Analysis"),
                "time": ("🕐", "Time of Day Analysis"),
                "day": ("📅", "Day of Week Analysis"),
                "sport": ("⚽", "Sport Analysis"),
                "combo": ("🔗", "Combined Pattern"),
                "sport_combo": ("🎮", "Sport + Amount Combo"),
                "triple_combo": ("🎯🎯🎯", "Triple Combo Pattern"),
                "venue": ("📍", "Venue Analysis"),
                "region": ("🌏", "Region Analysis"),
                "market": ("📈", "Market Type Analysis"),
                "event_type": ("🏆", "Event Type Analysis"),
                "stake_behavior": ("🔄", "Stake Update Analysis"),
                "stake_combo": ("🔄+", "Stake Update by Type"),
                "stake_amount": ("💵", "Stake Addition Size"),
            }.get(category, ("❓", "Pattern Analysis"))
            
            emoji, category_label = category_info
            
            # Create detailed discovery message
            discovery_msg = (
                f"\n{emoji} **DISCOVERY #{i}: {description.upper()}**\n"
                f"*Category: {category_label}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 **PERFORMANCE STATS:**\n"
                f"```\n"
                f"Win Rate:     {win_rate:.1f}% ({'🔥 EXCELLENT' if win_rate >= 70 else '✅ GOOD' if win_rate >= 65 else '👍 PROFITABLE'})\n"
                f"Record:       {wins_d}W - {losses_d}L\n"
                f"Sample Size:  {sample_size} bets\n"
                f"Avg Odds:     {avg_odds:.2f}\n"
                f"Total P&L:    {'+' if total_pnl_pattern >= 0 else ''}{total_pnl_pattern:.1f} units\n"
                f"```\n\n"
                f"🔍 **WHAT THIS MEANS:**\n"
            )
            
            # Add specific explanation based on category
            if "bet_type" in category or category == "bet_type":
                bet_type_found = pattern.split("=")[1].split("&")[0] if "=" in pattern else "unknown"
                discovery_msg += f"When betting on **{bet_type_found}** events, you win {win_rate:.1f}% of the time.\n"
            elif "amount" in category:
                discovery_msg += f"Bets in this size range perform exceptionally well with a {win_rate:.1f}% win rate.\n"
            elif "odds" in category:
                discovery_msg += f"Bets at these odds have historically been profitable with consistent returns.\n"
            elif "time" in category or "hour" in pattern:
                discovery_msg += f"Betting during this time period yields better results than other times.\n"
            elif "stake" in category:
                discovery_msg += f"When someone adds more stake to an existing bet, it often signals inside confidence.\n"
            elif "venue" in category:
                discovery_msg += f"This venue/track consistently produces profitable outcomes.\n"
            elif "sport" in category:
                discovery_msg += f"This sport/league has shown consistent profitability patterns.\n"
            else:
                discovery_msg += f"This pattern shows reliable profitability above the {ML_CONFIG['confidence_threshold']*100:.0f}% threshold.\n"
            
            discovery_msg += (
                f"\n💡 **RECOMMENDED ACTION:**\n"
                f"*{action}*\n\n"
                f"⚙️ **TECHNICAL PATTERN:**\n"
                f"`{pattern}`\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            await notify_admin(discovery_msg)
        
        # Send summary and next steps
        summary_msg = (
            f"\n\n📋 **ANALYSIS SUMMARY**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 **Current Data:**\n"
            f"• Total Observed: {total_observed:,} bets\n"
            f"• Settled: {settled_count:,} bets\n"
            f"• Overall Win Rate: {overall_win_rate:.1f}%\n"
            f"• Overall P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:.1f} units\n\n"
            f"🔍 **Discoveries:**\n"
            f"• New Patterns Found: {len(new_discoveries)}\n"
            f"• Total Patterns Tracked: {len(discoveries)}\n\n"
            f"⚠️ **IMPORTANT:**\n"
            f"Your server settings have NOT been changed automatically.\n"
            f"The algorithm is in **learning mode** - observing all bets to find patterns.\n"
            f"When you're ready to apply these findings, use the panel or DM commands.\n\n"
            f"📱 **DM Commands:**\n"
            f"• `/patterns` - View all profitable patterns\n"
            f"• `/stats` - Detailed statistics\n"
            f"• `/analyze` - Run analysis now\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await notify_admin(summary_msg)
        
        ml_learning_data["discoveries"].extend([{
            **d,
            "discovered_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        } for d in new_discoveries])
        save_ml_data()
    
    print(f"🧠 ML Analysis complete: {len(discoveries)} profitable patterns, {len(new_discoveries)} new")

# ============================================
# ML STRATEGIES - DISPLAY & APPLY SYSTEM
# ============================================

def get_best_strategies() -> List[dict]:
    """Get the best ML strategies sorted by profitability"""
    patterns = ml_learning_data.get("profitable_patterns", {})
    if not patterns:
        return []
    
    strategies = []
    for pattern_key, data in patterns.items():
        # Calculate a score based on win rate, sample size, and P&L
        win_rate = data.get("win_rate", 0)
        sample_size = data.get("sample_size", 0)
        total_pnl = data.get("total_pnl", 0)
        avg_odds = data.get("avg_odds", 2.0)
        
        # Score formula: win_rate * log(sample_size) * pnl_factor
        import math
        sample_factor = math.log(max(sample_size, 1) + 1)
        pnl_factor = 1 + (total_pnl / 100) if total_pnl > 0 else 0.5
        score = win_rate * sample_factor * pnl_factor
        
        strategies.append({
            "pattern": pattern_key,
            "score": score,
            **data
        })
    
    # Sort by score
    strategies.sort(key=lambda x: x["score"], reverse=True)
    return strategies

def format_strategy_for_display(strategy: dict, rank: int) -> str:
    """Format a single strategy for detailed display"""
    pattern = strategy.get("pattern", "unknown")
    description = strategy.get("description", pattern)
    category = strategy.get("category", "unknown")
    win_rate = strategy.get("win_rate", 0) * 100
    sample_size = strategy.get("sample_size", 0)
    wins = strategy.get("wins", 0)
    losses = strategy.get("losses", 0)
    total_pnl = strategy.get("total_pnl", 0)
    avg_odds = strategy.get("avg_odds", 2.0)
    action = strategy.get("recommended_action", "Continue monitoring")
    
    # Determine rating
    if win_rate >= 90:
        rating = "🔥 ELITE"
    elif win_rate >= 80:
        rating = "💎 EXCELLENT"
    elif win_rate >= 70:
        rating = "⭐ GREAT"
    elif win_rate >= 65:
        rating = "✅ GOOD"
    else:
        rating = "📊 MODERATE"
    
    # Get category emoji
    category_emoji = {
        "bet_type": "🎯",
        "odds": "📊",
        "amount": "💰",
        "time": "🕐",
        "day": "📅",
        "sport": "⚽",
        "combo": "🔗",
        "sport_combo": "🎮",
        "triple_combo": "🎯",
        "venue": "📍",
        "region": "🌏",
        "market": "📈",
        "event_type": "🏆",
        "stake_behavior": "🔄",
        "stake_combo": "🔄",
        "stake_amount": "💵",
    }.get(category, "❓")
    
    # Build explanation based on pattern
    explanation = ""
    if "is_favorite=True" in pattern:
        explanation = f"Bets on favorites (odds below $3.00) win {win_rate:.0f}% of the time with avg odds of ${avg_odds:.2f}"
    elif "is_longshot=False" in pattern:
        explanation = f"Avoiding longshots (odds below $10) wins {win_rate:.0f}% with consistent ${avg_odds:.2f} avg odds"
    elif "odds_bucket=short" in pattern:
        explanation = f"Short odds bets ($1.50-$2.50) have {win_rate:.0f}% win rate at avg ${avg_odds:.2f}"
    elif "odds_bucket=medium" in pattern:
        explanation = f"Medium odds ($2.50-$5.00) hit {win_rate:.0f}% with avg ${avg_odds:.2f} returns"
    elif "bet_type=sports" in pattern:
        explanation = f"Sports betting wins {win_rate:.0f}% of the time - better than racing"
    elif "bet_type=horse" in pattern:
        explanation = f"Horse racing bets win {win_rate:.0f}% with ${avg_odds:.2f} avg odds"
    elif "bet_type=greyhound" in pattern:
        explanation = f"Greyhound bets hit {win_rate:.0f}% at ${avg_odds:.2f} average"
    elif "sport=" in pattern:
        sport_name = pattern.split("sport=")[1].split("&")[0]
        explanation = f"{sport_name.title()} betting wins {win_rate:.0f}% - focus on this sport"
    elif "is_weekend=False" in pattern:
        explanation = f"Weekday betting (Mon-Fri) wins {win_rate:.0f}% vs lower weekend rates"
    elif "day=" in pattern:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        try:
            day_num = int(pattern.split("day=")[1].split("&")[0])
            explanation = f"{days[day_num]} bets win {win_rate:.0f}% - best day of the week"
        except:
            explanation = f"This day of week wins {win_rate:.0f}% of bets"
    elif "amount_bucket=" in pattern:
        bucket = pattern.split("amount_bucket=")[1].split("&")[0]
        explanation = f"Bets in the {bucket.replace('_', '-')} range win {win_rate:.0f}%"
    elif "is_stake_update=True" in pattern:
        explanation = f"Bets where stake was increased win {win_rate:.0f}% - smart money adding"
    else:
        explanation = f"This pattern achieves {win_rate:.0f}% win rate with ${avg_odds:.2f} avg odds"
    
    return (
        f"**#{rank} {category_emoji} {description.upper()}** {rating}\n"
        f"┌─────────────────────────────\n"
        f"│ 📈 **Win Rate:** {win_rate:.1f}% ({wins}W - {losses}L)\n"
        f"│ 💵 **Avg Odds:** ${avg_odds:.2f}\n"
        f"│ 📊 **Sample Size:** {sample_size} bets\n"
        f"│ 💰 **Total P&L:** {'+' if total_pnl >= 0 else ''}{total_pnl:.1f} units\n"
        f"│\n"
        f"│ 📝 **What This Means:**\n"
        f"│ {explanation}\n"
        f"│\n"
        f"│ 💡 **Action:** {action}\n"
        f"└─────────────────────────────\n"
    )

def get_strategy_settings_changes(strategy: dict) -> dict:
    """Determine what settings changes a strategy would apply"""
    pattern = strategy.get("pattern", "")
    changes = {}
    
    # Parse pattern to determine settings
    if "bet_type=sports" in pattern:
        changes["focus_sports"] = True
        changes["description"] = "Enable/boost sports betting alerts"
    
    if "bet_type=horse" in pattern:
        changes["focus_horse"] = True
        changes["description"] = "Enable/boost horse racing alerts"
    
    if "bet_type=greyhound" in pattern:
        changes["focus_greyhound"] = True
        changes["description"] = "Enable/boost greyhound alerts"
    
    if "is_favorite=True" in pattern or "odds_bucket=short" in pattern:
        changes["prefer_favorites"] = True
        changes["max_odds"] = 3.0
        changes["description"] = "Focus on favorites (odds under $3.00)"
    
    if "is_longshot=False" in pattern:
        changes["avoid_longshots"] = True
        changes["max_odds"] = 10.0
        changes["description"] = "Avoid longshots (cap odds at $10.00)"
    
    if "sport=" in pattern:
        sport = pattern.split("sport=")[1].split("&")[0]
        changes["focus_sport"] = sport
        changes["description"] = f"Prioritize {sport.title()} betting"
    
    if "is_weekend=False" in pattern:
        changes["weekday_only"] = True
        changes["description"] = "Focus on weekday bets (Mon-Fri)"
    
    if "amount_bucket=" in pattern:
        bucket = pattern.split("amount_bucket=")[1].split("&")[0]
        # Set minimum based on bucket
        bucket_minimums = {
            "under_500": 100,
            "500_to_1k": 500,
            "1k_to_2k": 1000,
            "2k_to_3k": 2000,
            "3k_to_5k": 3000,
            "5k_to_7.5k": 5000,
            "7.5k_to_10k": 7500,
            "10k_to_15k": 10000,
            "15k_to_25k": 15000,
            "25k_to_50k": 25000,
            "50k_to_100k": 50000,
            "100k_plus": 100000,
        }
        if bucket in bucket_minimums:
            changes["min_bet"] = bucket_minimums[bucket]
            changes["description"] = f"Set minimum bet to ${bucket_minimums[bucket]:,}"
    
    return changes

# Active ML strategies that have been applied
ml_applied_strategies = {}

async def apply_strategy_to_all_servers(strategy: dict) -> Tuple[int, List[str]]:
    """Apply a strategy to all servers"""
    changes = get_strategy_settings_changes(strategy)
    if not changes:
        return 0, ["No applicable settings for this strategy"]
    
    applied_count = 0
    messages = []
    pattern = strategy.get("pattern", "unknown")
    
    for guild_id, settings in guild_settings.items():
        try:
            # Track what we're changing
            old_values = {}
            new_values = {}
            
            # Apply different changes based on strategy
            if changes.get("max_odds"):
                old_values["max_odds"] = settings.get("max_odds", "None")
                settings["max_odds"] = changes["max_odds"]
                new_values["max_odds"] = changes["max_odds"]
            
            if changes.get("min_bet"):
                # Apply to the appropriate bet type
                if "horse" in pattern:
                    old_values["au_horse_min_matched"] = settings.get("au_horse_min_matched", 1000)
                    old_values["uk_horse_min_matched"] = settings.get("uk_horse_min_matched", 1000)
                    settings["au_horse_min_matched"] = changes["min_bet"]
                    settings["uk_horse_min_matched"] = changes["min_bet"]
                    new_values["horse_min"] = changes["min_bet"]
                elif "greyhound" in pattern:
                    old_values["greyhound_min_matched"] = settings.get("greyhound_min_matched", 1000)
                    settings["greyhound_min_matched"] = changes["min_bet"]
                    new_values["greyhound_min"] = changes["min_bet"]
                elif "sports" in pattern:
                    old_values["min_matched"] = settings.get("min_matched", 1000)
                    settings["min_matched"] = changes["min_bet"]
                    new_values["sports_min"] = changes["min_bet"]
                else:
                    # Apply to all types
                    old_values["min_matched"] = settings.get("min_matched", 1000)
                    settings["min_matched"] = changes["min_bet"]
                    settings["au_horse_min_matched"] = changes["min_bet"]
                    settings["uk_horse_min_matched"] = changes["min_bet"]
                    settings["greyhound_min_matched"] = changes["min_bet"]
                    new_values["all_min"] = changes["min_bet"]
            
            if changes.get("focus_sport"):
                # Add to prioritized sports
                priority_sports = settings.get("priority_sports", [])
                if changes["focus_sport"] not in priority_sports:
                    priority_sports.append(changes["focus_sport"])
                    settings["priority_sports"] = priority_sports
                    new_values["priority_sport"] = changes["focus_sport"]
            
            if changes.get("weekday_only"):
                settings["weekday_only"] = True
                new_values["weekday_only"] = True
            
            if changes.get("prefer_favorites"):
                settings["prefer_favorites"] = True
                new_values["prefer_favorites"] = True
            
            if changes.get("avoid_longshots"):
                settings["avoid_longshots"] = True
                new_values["avoid_longshots"] = True
            
            applied_count += 1
            
        except Exception as e:
            messages.append(f"Error on guild {guild_id}: {e}")
    
    if applied_count > 0:
        # Save settings
        save_guild_settings()
        
        # Track applied strategy
        ml_applied_strategies[pattern] = {
            "applied_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "servers_affected": applied_count,
            "changes": changes
        }
        
        messages.append(f"✅ Applied to {applied_count} server(s): {changes.get('description', pattern)}")
    
    return applied_count, messages

def get_ml_strategies_embed() -> discord.Embed:
    """Create an embed showing all ML strategies with their details"""
    strategies = get_best_strategies()
    
    embed = discord.Embed(
        title="🧠 ML PROFITABLE STRATEGIES",
        description=(
            "These strategies have been discovered through analyzing thousands of bets.\n"
            "Click **Apply** to automatically configure your servers with a strategy.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ),
        color=discord.Color.gold(),
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    
    if not strategies:
        embed.add_field(
            name="📊 No Strategies Yet",
            value=(
                "The ML system is still collecting data.\n"
                f"Need {ML_CONFIG['min_sample_size']} settled bets per pattern.\n\n"
                "*Keep the bot running to gather more data!*"
            ),
            inline=False
        )
        return embed
    
    # Get overall stats
    all_bets = ml_learning_data.get("all_observed_bets", [])
    settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
    wins = len([b for b in settled if b["status"] == "won"])
    overall_wr = (wins / len(settled) * 100) if settled else 0
    
    embed.add_field(
        name="📊 Overall Stats",
        value=(
            f"**Total Observed:** {len(all_bets):,}\n"
            f"**Settled:** {len(settled):,}\n"
            f"**Overall Win Rate:** {overall_wr:.1f}%\n"
            f"**Strategies Found:** {len(strategies)}"
        ),
        inline=False
    )
    
    # Top strategies summary
    top_strategies_text = ""
    for i, strat in enumerate(strategies[:8], 1):
        win_rate = strat.get("win_rate", 0) * 100
        sample = strat.get("sample_size", 0)
        avg_odds = strat.get("avg_odds", 2.0)
        pnl = strat.get("total_pnl", 0)
        desc = strat.get("description", strat.get("pattern", ""))[:30]
        
        # Rating emoji
        if win_rate >= 90:
            rating = "🔥"
        elif win_rate >= 80:
            rating = "💎"
        elif win_rate >= 70:
            rating = "⭐"
        else:
            rating = "✅"
        
        top_strategies_text += (
            f"{rating} **{i}. {desc}**\n"
            f"   {win_rate:.0f}% WR | ${avg_odds:.2f} avg | {sample} bets | +{pnl:.0f}u\n"
        )
    
    embed.add_field(
        name="🏆 Top Strategies",
        value=top_strategies_text or "No strategies yet",
        inline=False
    )
    
    # Best odds recommendation
    best_odds_strategies = [s for s in strategies if "odds" in s.get("category", "") or "favorite" in s.get("pattern", "").lower()]
    if best_odds_strategies:
        best = best_odds_strategies[0]
        embed.add_field(
            name="💵 Best Odds Strategy",
            value=(
                f"**{best.get('description', 'Unknown')}**\n"
                f"Avg odds: ${best.get('avg_odds', 2.0):.2f}\n"
                f"Win rate: {best.get('win_rate', 0)*100:.0f}%\n"
                f"*{best.get('recommended_action', '')}*"
            ),
            inline=True
        )
    
    # Best sport
    sport_strategies = [s for s in strategies if s.get("category") == "sport"]
    if sport_strategies:
        best = sport_strategies[0]
        embed.add_field(
            name="⚽ Best Sport",
            value=(
                f"**{best.get('description', 'Unknown')}**\n"
                f"Win rate: {best.get('win_rate', 0)*100:.0f}%\n"
                f"+{best.get('total_pnl', 0):.0f} units profit"
            ),
            inline=True
        )
    
    # Best bet type
    type_strategies = [s for s in strategies if s.get("category") == "bet_type"]
    if type_strategies:
        best = type_strategies[0]
        embed.add_field(
            name="🎯 Best Bet Type",
            value=(
                f"**{best.get('description', 'Unknown')}**\n"
                f"Win rate: {best.get('win_rate', 0)*100:.0f}%\n"
                f"+{best.get('total_pnl', 0):.0f} units profit"
            ),
            inline=True
        )
    
    embed.set_footer(text="Use the buttons below to apply strategies • Your current settings shown for reference")
    
    return embed

def get_ml_recommendation(bet_data: dict) -> Tuple[bool, str, float]:
    """
    Get ML recommendation for whether to post a bet
    Returns: (should_post, reason, confidence_score)
    """
    if not ML_CONFIG["auto_adjust_enabled"]:
        return True, "ML auto-adjust disabled", 0.5
    
    patterns = ml_learning_data.get("profitable_patterns", {})
    if not patterns:
        return True, "No patterns learned yet - collecting data", 0.5
    
    # Check if bet matches any profitable patterns
    bet_type = bet_data.get("bet_type", "unknown")
    odds_bucket = categorize_odds(bet_data.get("odds"))
    amount_bucket = categorize_amount(bet_data.get("bet_amount"))
    sport_key = bet_data.get("sport_key")
    
    matching_patterns = []
    
    # Check simple patterns
    if f"bet_type={bet_type}" in patterns:
        matching_patterns.append(patterns[f"bet_type={bet_type}"])
    if f"odds_bucket={odds_bucket}" in patterns:
        matching_patterns.append(patterns[f"odds_bucket={odds_bucket}"])
    if f"amount_bucket={amount_bucket}" in patterns:
        matching_patterns.append(patterns[f"amount_bucket={amount_bucket}"])
    if sport_key and f"sport={sport_key}" in patterns:
        matching_patterns.append(patterns[f"sport={sport_key}"])
    
    # Check combo patterns
    combo_key = f"bet_type={bet_type}&odds_bucket={odds_bucket}"
    if combo_key in patterns:
        matching_patterns.append(patterns[combo_key])
    
    if matching_patterns:
        avg_win_rate = statistics.mean([p["win_rate"] for p in matching_patterns])
        avg_pnl = statistics.mean([p["total_pnl"] for p in matching_patterns])
        return True, f"Matches {len(matching_patterns)} profitable patterns", avg_win_rate
    
    # In learning mode, allow all bets through
    if ML_CONFIG["learning_mode"]:
        return True, "Learning mode - collecting data", 0.5
    
    return False, "No matching profitable patterns", 0.3

async def update_ml_bet_with_result(market_id: str, runner_name: str, won: bool):
    """Update ML learning data with actual result from a settled market"""
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        updated_count = 0
        
        for bet in ml_learning_data.get("all_observed_bets", []):
            # Match by runner name AND pending status
            # Also check if bet is recent (within last 7 days) to avoid false matches
            bet_timestamp = bet.get("timestamp")
            if isinstance(bet_timestamp, str):
                try:
                    bet_timestamp = datetime.datetime.fromisoformat(bet_timestamp)
                except:
                    bet_timestamp = None
            
            # Only match pending bets from the last 7 days
            is_recent = bet_timestamp and (now - bet_timestamp).days <= 7 if isinstance(bet_timestamp, datetime.datetime) else True
            
            if (bet.get("runner_name") == runner_name and 
                bet.get("status") == "pending" and 
                is_recent):
                bet["status"] = "won" if won else "lost"
                bet["settled_at"] = now
                
                odds = bet.get("odds", 2.0) or 2.0
                units = bet.get("recommended_units", 1.0) or 1.0
                
                if won:
                    bet["pnl"] = (odds - 1) * units
                else:
                    bet["pnl"] = -units
                
                updated_count += 1
                # Only update ONE matching bet per call (most recent pending one)
                break
        
        if updated_count > 0:
            save_ml_data()
            print(f"🧠 ML: Updated bet with result: {runner_name} - {'WON' if won else 'LOST'}")
            
            # Check if we have enough data to run analysis
            settled = len([b for b in ml_learning_data.get("all_observed_bets", []) if b.get("status") in ["won", "lost"]])
            if settled >= ML_CONFIG["min_sample_size"] and settled % 5 == 0:  # Every 5 settled bets
                print(f"🧠 Running auto-analysis after {settled} settled bets...")
                await run_ml_analysis()
                
    except Exception as e:
        print(f"⚠️ Error updating ML bet result: {e}")
        traceback.print_exc()

# Initialize ML data on load
load_ml_data()

# Loss tracking system
loss_tracking = {
    'losses': [],  # List of loss timestamps
    'cooldown_until': None  # When cooldown period ends
}

# Bet analysis configuration
BET_CONFIG = {
    'MIN_PERCENTAGE_THRESHOLD': 20.0,  # Lowered from 40% - minimum percentage for posting (bet amount / total matched)
    'AUSTRALIAN_LOCATIONS': [
        'australia', 'nsw', 'vic', 'qld', 'sa', 'wa', 'tas', 'nt', 'act',
        'albury', 'bendigo', 'caulfield', 'doomben', 'eagle farm', 'flemington',
        'gold coast', 'moonee valley', 'morphettville', 'randwick', 'rosehill'
    ]
}

TROT_KEYWORDS = ("trot", "pace", "harness", "standardbred")
AU_LOCATION_KEYWORDS = tuple(loc.lower() for loc in BET_CONFIG['AUSTRALIAN_LOCATIONS'])

# UK and AU track name sets for precise routing (lowercased)
UK_HORSE_TRACKS = {
    # England
    "aintree","ascot","bath","beverley","brighton","carlisle","cartmel","catterick bridge","chelmsford city","cheltenham","chester","doncaster","epsom downs","exeter","fakenham","fontwell park","goodwood","great yarmouth","haydock park","hereford","hexham","huntingdon","kempton park","leicester","lingfield park","ludlow","market rasen","newbury","newcastle","newmarket","newton abbot","nottingham","plumpton","pontefract","redcar","ripon","salisbury","sandown park","sedgefield","southwell","stratford-upon-avon","taunton","thirsk","uttoxeter","warwick","wetherby","wincanton","windsor","wolverhampton","worcester","york",
    # Scotland
    "ayr","hamilton park","kelso","musselburgh","perth",
    # Wales
    "bangor-on-dee","chepstow","ffos las",
    # Northern Ireland
    "down royal","downpatrick",
    # Republic of Ireland
    "ballinrobe","bellewstown","clonmel","cork","curragh","the curragh","dundalk","dundalk stadium","fairyhouse","galway","gowran park","kilbeggan","killarney","laytown","leopardstown","limerick","listowel","naas","navan","punchestown","roscommon","sligo","thurles","tipperary","tramore","wexford"
}

AU_HORSE_TRACKS = {
    # VIC
    "alexandra","ararat","avoca","bairnsdale","ballarat","balnarring","benalla","bendigo","camperdown","casterton","caulfield","colac","coleraine","cranbourne","donald","drouin","dunkeld","echuca","edenhope","flemington","geelong","hamilton","horsham","kilmore","kyneton","mansfield","mildura","moe","moonee valley","mornington","mortlake","murtoa","pakenham","sale","sandown","seymour","stawell","swan hill","tatura","terang","traralgon","wangaratta","warracknabeal","warrnambool","werribee","wodonga","yarra glen","yea",
    # NSW
    "albury","armidale","ballina","bathurst","binnaway","bombala","bourke","braidwood","broken hill","canterbury","casino","cessnock","cobar","coffs harbour","condobolin","cooma","coonamble","cootamundra","corowa","cowra","dubbo","forbes","glen innes","gosford","goulburn","grafton","griffith","gundagai","gunnedah","hawkesbury","inverell","kembla grange","kempsey","lismore","moree","mudgee","murwillumbah","muswellbrook","narrabri","narrandera","newcastle","nowra","orange","parkes","port macquarie","queanbeyan","randwick","rosehill","scone","tamworth","taree","wagga","warwick farm","wellington","wyong","young",
    # QLD
    "atherton","beaudesert","birdsville","bundaberg","cairns","charleville","charters towers","cloncurry","dalby","doomben","eagle farm","emerald","gatton","gladstone","gold coast","goondiwindi","gympie","ipswich","mackay","rockhampton","roma","sunshine coast","toowoomba","townsville","warwick",
    # SA
    "balaklava","clare","gawler","morphettville","mt gambier","murray bridge","port augusta","port lincoln","strathalbyn",
    # WA
    "albany","ascot","belmont park","broome","bunbury","geraldton","kalgoorlie","northam","pinjarra","york",
    # TAS
    "devonport","hobart","launceston",
    # NT
    "fannie bay","katherine","pioneer park",
    # ACT
    "canberra"
}

# Betfair sports configuration metadata
SPORT_CATEGORIES = [
    {"key": "soccer", "label": "Soccer", "betfair_name": "Soccer", "emoji": "⚽"},
    {"key": "tennis", "label": "Tennis", "betfair_name": "Tennis", "emoji": "🎾"},
    {"key": "cricket", "label": "Cricket", "betfair_name": "Cricket", "emoji": "🏏"},
    {"key": "basketball", "label": "Basketball (NBA)", "betfair_name": "Basketball", "emoji": "🏀"},
    {"key": "american_football", "label": "American Football (NFL)", "betfair_name": "American Football", "emoji": "🏈"},
    {"key": "baseball", "label": "Baseball (MLB)", "betfair_name": "Baseball", "emoji": "⚾"},
    {"key": "ice_hockey", "label": "Ice Hockey (NHL)", "betfair_name": "Ice Hockey", "emoji": "🏒"},
    {"key": "golf", "label": "Golf", "betfair_name": "Golf", "emoji": "🏌️"},
    {"key": "rugby_union", "label": "Rugby Union", "betfair_name": "Rugby Union", "emoji": "🏉"},
    {"key": "rugby_league", "label": "Rugby League", "betfair_name": "Rugby League", "emoji": "🏉"}
]

SPORT_CATEGORY_LOOKUP = {category["key"]: category for category in SPORT_CATEGORIES}
SPORT_NAME_TO_KEY = {category["betfair_name"].lower(): category["key"] for category in SPORT_CATEGORIES}
SPORT_EVENT_TYPE_CACHE = {}  # sport_key -> Betfair eventTypeId
SPORT_EVENT_TYPE_ID_TO_KEY = {}  # eventTypeId -> sport_key
SPORT_EVENT_TYPES_LAST_FETCH = 0.0
SPORT_EVENT_CACHE_TTL = 3600  # seconds

def calculate_bet_percentage(bet_amount, total_matched):
    """Calculate what percentage the bet amount is of total matched"""
    if total_matched == 0:
        return 0
    return (bet_amount / total_matched) * 100

def determine_post_channel(location, race_type):
    """
    Determine which channel to post in based on race location and type
    Returns: 'GLOBAL' or 'AU_HORSES'
    """
    # Convert to lowercase for comparison
    location = location.lower()
    race_type = race_type.lower() if race_type else ''
    
    # Check if it's a trot race
    is_trot = any(word in race_type for word in ['trot', 'pace', 'harness'])
    
    # Check if it's an Australian race
    is_australian = any(loc in location for loc in BET_CONFIG['AUSTRALIAN_LOCATIONS'])
    
    # Determine channel
    if is_trot:
        return 'AU_HORSES'  # Trots always go to AU Horses
    elif is_australian:
        return 'GLOBAL'  # Australian races go to Global
    else:
        return 'GLOBAL'  # All other horse races go to Global

def should_post_bet(bet_amount, total_matched):
    """Determine if bet should be posted based on percentage threshold"""
    percentage = calculate_bet_percentage(bet_amount, total_matched)
    return percentage >= BET_CONFIG['MIN_PERCENTAGE_THRESHOLD']

# Active big bet embeds for countdown tracking
active_big_bet_embeds = {}  # {market_id: {"message_id": id, "start_time": datetime, "embed_data": data, "last_display": str}}

def get_race_countdown(start_time_str):
    """Convert race start time to display string: hours/minutes normally, seconds inside last minute."""
    try:
        # Parse the time string (e.g., "02:21 UTC" or "21:45 UTC")
        time_part = start_time_str.replace(' UTC', '').strip()
        
        # Get current UTC time
        now = datetime.datetime.now(datetime.timezone.utc)
        print(f"🔍 DEBUG: Parsing race time '{start_time_str}' -> '{time_part}', current time: {now.strftime('%H:%M:%S UTC')}")
        
        # Parse the start time
        hour, minute = map(int, time_part.split(':'))
        
        # Create race start time for today first
        race_start_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        time_diff_today = race_start_today - now
        total_seconds_today = int(time_diff_today.total_seconds())
        
        print(f"🔍 DEBUG: Race today would be: {race_start_today.strftime('%H:%M:%S UTC')}, seconds from now: {total_seconds_today}")
        
        # If race is in the future today (even if just a few seconds), use today
        if total_seconds_today > -300:  # Allow up to 5 minutes grace period for races that just started
            race_start = race_start_today
            total_seconds = total_seconds_today
            print(f"🔍 DEBUG: Using today's race time")
        else:
            # Race time has passed today, so it must be tomorrow
            race_start = race_start_today + datetime.timedelta(days=1)
            time_diff = race_start - now
            total_seconds = int(time_diff.total_seconds())
            print(f"🔍 DEBUG: Using tomorrow's race time: {race_start.strftime('%H:%M:%S UTC')}, seconds from now: {total_seconds}")
        
        # If the race has already started (negative time or very small positive)
        if total_seconds <= 0:
            return "🏁 **『 RACE STARTED 』** 🏁", race_start
        
        # Format countdown with enhanced visual styling
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        # Minutes/hours normally; seconds inside last minute
        if total_seconds <= 60:
            countdown = f"⏱️ **in {total_seconds}s**"
        elif hours > 0:
            countdown = f"🕐 **in {hours}h {minutes}m**"
        else:
            countdown = f"🚀 **in {minutes}m**"
            
        print(f"🔍 DEBUG: Final countdown: {countdown}")
        return countdown, race_start
        
    except Exception as e:
        print(f"Error parsing race time '{start_time_str}': {e}")
        return f"⏰ **Time:** `{start_time_str}`", None

def get_race_countdown_from_datetime(start_dt: datetime.datetime):
    """Generate countdown from aware UTC datetime: hours/minutes normally; seconds inside last minute."""
    try:
        if start_dt.tzinfo is None:
            # Assume UTC if naive
            start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        time_diff = start_dt - now
        total_seconds = int(time_diff.total_seconds())
        print(f"🔎 COUNTDOWN DEBUG: start_dt(UTC)={start_dt.isoformat()} now(UTC)={now.isoformat()} total_seconds={total_seconds}")

        if total_seconds <= 0:
            # Race has started
            return "🏁 **RACE STARTED**", start_dt

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if total_seconds <= 60:
            countdown = f"⏱️ **in {total_seconds}s**"
        elif hours > 0:
            countdown = f"🕐 **in {hours}h {minutes}m**"
        else:
            countdown = f"🚀 **in {minutes}m**"

        return countdown, start_dt
    except Exception as e:
        print(f"Error in get_race_countdown_from_datetime: {e}")
        # Fallback to string version if something goes wrong
        return "⏰ **Time Unavailable**", start_dt

async def update_big_bet_countdowns():
    """Update countdown timers for active big bet embeds"""
    global active_big_bet_embeds
    
    for market_id, embed_info in list(active_big_bet_embeds.items()):
        try:
            start_time = embed_info["start_time"]
            if not start_time:
                continue
                
            # Calculate new countdown
            now = datetime.datetime.now(datetime.timezone.utc)
            time_diff = start_time - now
            total_seconds = int(time_diff.total_seconds())
            
            # If race finished more than 5 minutes ago, remove from tracking
            if total_seconds < -300:  # 5 minutes after start
                del active_big_bet_embeds[market_id]
                continue
            
            # Update countdown display: seconds inside last minute; otherwise minutes/hours
            if total_seconds <= 0:
                countdown_display = "🏁 **RACE STARTED**"
            elif total_seconds <= 60:
                countdown_display = f"⏱️ **in {total_seconds}s**"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                if hours > 0:
                    countdown_display = f"🕐 **in {hours}h {minutes}m**"
                else:
                    countdown_display = f"🚀 **in {minutes}m**"
            
            # Update the embed description
            embed_data = embed_info["embed_data"]
            original_description = embed_data["original_description"]
            
            # Replace the time line in the description
            lines = original_description.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("⏰"):
                    lines[i] = f"⏰ **Time:** {countdown_display}"
                    break
            
            updated_description = '\n'.join(lines)

            # Only edit when the countdown display actually changed
            if embed_info.get("last_display") == countdown_display:
                continue
            embed_info["last_display"] = countdown_display

            # Try to edit the message
            try:
                channel = bot.get_channel(embed_info["channel_id"])
                if channel:
                    # Use a partial message to avoid an extra GET request each update
                    message = channel.get_partial_message(embed_info["message_id"])
                    # Add a tiny deterministic jitter so multiple messages don't update simultaneously
                    try:
                        jitter = (abs(hash(str(embed_info["message_id"])) % 300)) / 1000.0  # up to 300ms
                        await asyncio.sleep(jitter)
                    except Exception:
                        pass
                    
                    embed = discord.Embed(
                        title=embed_data["title"],
                        description=updated_description,
                        color=embed_data["color"],
                        timestamp=embed_data["timestamp"]
                    )
                    
                    # Re-add any fields that were in the original embed
                    if "fields" in embed_data:
                        for field in embed_data["fields"]:
                            embed.add_field(
                                name=field["name"],
                                value=field["value"],
                                inline=field["inline"]
                            )
                    
                    await message.edit(embed=embed)
                    
            except Exception as edit_error:
                print(f"Error updating countdown for message {embed_info['message_id']}: {edit_error}")
                
        except Exception as e:
            print(f"Error processing countdown for {market_id}: {e}")

async def check_race_closure():
    """Check if tracked markets have been closed and update embeds accordingly."""
    if not active_big_bet_embeds:
        return
    try:
        session_token = get_session_token()
        for market_id, embed_info in list(active_big_bet_embeds.items()):
            start_time = embed_info.get("start_time")
            if not start_time:
                continue
            # Only check closure if we are past start time
            now = datetime.datetime.now(datetime.timezone.utc)
            if now < start_time:
                continue
            # Query market book status (lightweight)
            book_req = json.dumps({
                "jsonrpc": "2.0",
                "method": "SportsAPING/v1.0/listMarketBook",
                "params": {
                    "marketIds": [market_id],
                    "priceProjection": {"priceData": []},
                    "orderProjection": "EXECUTABLE",
                    "matchProjection": "NO_ROLLUP"
                },
                "id": 1
            })
            try:
                resp = call_betfair_api(book_req, session_token)
                if not resp or not resp.get("result"):
                    continue
                market_book = resp["result"][0]
                status = market_book.get("status") or market_book.get("marketDefinition", {}).get("status")
                if status in ("CLOSED", "SUSPENDED"):
                    # Update embed to show closed and stop tracking
                    try:
                        channel = bot.get_channel(embed_info["channel_id"])
                        if channel:
                            message = channel.get_partial_message(embed_info["message_id"])
                            embed_data = embed_info["embed_data"]
                            # Replace time line
                            lines = embed_data["original_description"].split('\n')
                            for i, line in enumerate(lines):
                                if line.strip().startswith("⏰"):
                                    lines[i] = f"⏰ **Time:** 🔒 **RACE CLOSED**"
                                    break
                            updated_description = '\n'.join(lines)
                            embed = discord.Embed(
                                title=embed_data["title"],
                                description=updated_description,
                                color=embed_data["color"],
                                timestamp=embed_data["timestamp"]
                            )
                            if "fields" in embed_data:
                                for field in embed_data["fields"]:
                                    embed.add_field(name=field["name"], value=field["value"], inline=field["inline"])
                            await message.edit(embed=embed)
                    except Exception as _:
                        pass
                    # Remove from tracking
                    del active_big_bet_embeds[market_id]
            except Exception as _:
                continue
    except Exception as e:
        print(f"Error in check_race_closure: {e}")

# Add delays to avoid rate limiting
async def safe_bot_login(bot, token):
    """Login to Discord with retry logic for rate limiting"""
    max_retries = 5
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Discord login attempt {attempt + 1}/{max_retries}")
            await bot.login(token)
            print("✅ Discord login successful!")
            return True
        except discord.HTTPException as e:
            if e.status == 429:  # Rate limited
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"⏱️ Rate limited! Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Discord login error: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error during login: {e}")
            await asyncio.sleep(retry_delay)
    
    print("❌ Failed to login after all retries")
    return False

def get_sport_emoji(event_name, market_name):
    name = ((event_name or "") + " " + (market_name or "")).lower()
    if "tennis" in name:
        return "🎾"
    if "cricket" in name:
        return "🏏"
    if "horse" in name or "racing" in name or "stakes" in name or "stakes" in market_name.lower():
        return "🏇"
    if "greyhound" in name or "dog" in name:
        return "🐕"
    if "golf" in name or "balls" in name:
        return "🏌️"
    if "soccer" in name or "football" in name or "premier" in name or "league" in name:
        return "⚽"
    if "basketball" in name or "nba" in name:
        return "🏀"
    if "rugby" in name:
        return "🏉"
    if "baseball" in name:
        return "⚾"
    if "hockey" in name:
        return "🏒"
    return "🌐"


def get_selection_label(bet_type, sport_key=None):
    """Get the appropriate label for a selection based on bet type and sport"""
    if bet_type == "horse":
        return "Horse"
    elif bet_type == "greyhound":
        return "Dog"
    elif bet_type == "trots":
        return "Horse"
    elif bet_type == "sports":
        # Map sports to appropriate selection labels
        if sport_key in ["tennis", "golf"]:
            return "Player"
        elif sport_key in ["soccer", "basketball", "american_football", "baseball", "ice_hockey", "rugby_union", "rugby_league", "cricket"]:
            return "Team"
        else:
            return "Selection"
    return "Selection"

def round_bet_amount(amount):
    """Round bet amounts to realistic looking figures"""
    if amount < 1000:
        # Round to nearest 50
        return round(amount / 50) * 50
    elif amount < 5000:
        # Round to nearest 100
        return round(amount / 100) * 100
    elif amount < 25000:
        # Round to nearest 500
        return round(amount / 500) * 500
    elif amount < 100000:
        # Round to nearest 1000
        return round(amount / 1000) * 1000
    else:
        # Round to nearest 5000
        return round(amount / 5000) * 5000

def calculate_unit_size(odds, bet_type, volume=0):
    """Calculate recommended unit size based on odds and bet type. Returns None if odds too short (NO BET)."""
    try:
        odds_float = float(odds)
    except:
        return 1.0  # Default to 1 unit if odds parsing fails
    
    if bet_type in ["horse", "greyhound", "trots"]:
        # Horse/Greyhound/Trots betting guidelines
        # NO BET if odds below 1.70 (too short, poor value)
        if odds_float < 1.70:
            return None  # NO BET
        elif 1.70 <= odds_float <= 2.20:
            return 2.0
        elif 2.20 < odds_float <= 3.00:
            return 1.5
        else:  # 3.00+
            return 1.0
    else:
        # Sports betting guidelines
        # NO BET if odds below 1.65 (too short, poor value)
        if odds_float < 1.65:
            return None  # NO BET
        elif 1.65 <= odds_float < 2.00:
            units = 2.0
        elif 2.00 <= odds_float < 3.00:
            units = 1.5
        else:  # 3.00+
            units = 1.0
        
        # Special exception: double units if volume exceeds $1M
        if volume >= 1000000:
            units *= 2
        
        return units

def track_new_bet(market_id, runner_name, odds, bet_type, amount, guild_id, channel_id, message_id, sport_key=None):
    """Track a new bet in the profit tracking system"""
    units = calculate_unit_size(odds, bet_type, amount)
    if units is None:
        units = 0.0
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    bet_data = {
        "id": f"{market_id}_{runner_name}_{guild_id}",
        "market_id": market_id,
        "runner_name": runner_name,
        "odds": float(odds),
        "bet_type": bet_type,
        "sport_key": sport_key,
        "volume": amount,
        "recommended_units": units,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "message_id": message_id,
        "timestamp": now_utc,
        "settled_at": None,
        "status": "pending",  # pending, won, lost
        "pnl": 0.0
    }
    
    tracked_bets.append(bet_data)
    save_bet_history()
    
    # Update daily stats
    today = now_utc.date()
    if daily_stats["date"] != today:
        # Reset daily stats for new day
        daily_stats.update({
            "date": today,
            "total_bets": 0,
            "pending_bets": 0,
            "won_bets": 0,
            "lost_bets": 0,
            "total_stakes": 0.0,
            "total_returns": 0.0,
            "pnl": 0.0
        })
    
    daily_stats["total_bets"] += 1
    daily_stats["pending_bets"] += 1
    daily_stats["total_stakes"] += units
    
    return bet_data

def update_bet_result(market_id, runner_name, guild_id, won=True):
    """Update the result of a tracked bet"""
    bet_id = f"{market_id}_{runner_name}_{guild_id}"
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    for bet in tracked_bets:
        if bet.get("id") == bet_id and bet.get("status") == "pending":
            bet["status"] = "won" if won else "lost"
            bet["settled_at"] = now_utc
            
            if won:
                # Calculate profit (odds - 1) * units
                profit = (bet["odds"] - 1) * bet["recommended_units"]
                bet["pnl"] = profit
                daily_stats["won_bets"] += 1
                daily_stats["total_returns"] += profit
                print(f"✅ Bet WON: {runner_name} @ ${bet['odds']:.2f} (+{profit:.1f} units)")
            else:
                # Loss is the stake
                bet["pnl"] = -bet["recommended_units"]
                daily_stats["lost_bets"] += 1
                print(f"❌ Bet LOST: {runner_name} @ ${bet['odds']:.2f} (-{bet['recommended_units']:.1f} units)")
            
            daily_stats["pending_bets"] = max(daily_stats["pending_bets"] - 1, 0)
            daily_stats["pnl"] = daily_stats["total_returns"] - daily_stats["total_stakes"]
            save_bet_history()
            return bet
    
    return None

def get_daily_summary():
    """Get current daily statistics"""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    today_bets = [bet for bet in tracked_bets if bet["timestamp"].date() == today]
    
    pending = len([bet for bet in today_bets if bet["status"] == "pending"])
    won = len([bet for bet in today_bets if bet["status"] == "won"])
    lost = len([bet for bet in today_bets if bet["status"] == "lost"])
    
    total_stakes = sum(bet["recommended_units"] for bet in today_bets)
    total_returns = sum(bet["pnl"] for bet in today_bets if bet["status"] == "won")
    total_losses = sum(bet["recommended_units"] for bet in today_bets if bet["status"] == "lost")
    net_pnl = total_returns - total_losses
    
    return {
        "date": today,
        "total_bets": len(today_bets),
        "pending_bets": pending,
        "won_bets": won,
        "lost_bets": lost,
        "total_stakes": total_stakes,
        "total_returns": total_returns,
        "total_losses": total_losses,
        "net_pnl": net_pnl,
        "win_rate": (won / len(today_bets) * 100) if today_bets else 0
    }


# === CONFIGURATION FROM ENVIRONMENT VARIABLES ===
# Set these in Railway dashboard: BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, DISCORD_BOT_TOKEN
USERNAME = os.environ.get("BETFAIR_USERNAME", "")
PASSWORD = os.environ.get("BETFAIR_PASSWORD", "")
APP_KEY = os.environ.get("BETFAIR_APP_KEY", "")
BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")

# Check if required env vars are set
if not BOT_TOKEN:
    print("❌ DISCORD_BOT_TOKEN not set! Add it to Railway environment variables.")
if not USERNAME or not PASSWORD or not APP_KEY:
    print("⚠️ Betfair credentials not fully set in environment variables.")

# === ACCESS RESTRICTIONS ===
AUTHORIZED_USER_IDS = {886065668137566239, 1239913614962331712}  # Authorized users who can use commands

# Always use local certificate files in the script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_FILE = os.path.join(BASE_DIR, 'client-2048.crt')
KEY_FILE = os.path.join(BASE_DIR, 'client-2048.key')

if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
    print("❌ Certificate files not available!")
    print("⚠️ Bot will start but Betfair features will be disabled")
    CERT_FILE = None
    KEY_FILE = None

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True

# Add connection settings to help with Railway rate limiting and optimize performance
bot = commands.Bot(
    command_prefix="!", 
    intents=intents,
    max_messages=500,  # Further reduce message cache
    chunk_guilds_at_startup=False,  # Don't chunk guilds immediately
    member_cache_flags=discord.MemberCacheFlags.none(),  # Disable member cache
    heartbeat_timeout=120.0,  # Increase heartbeat timeout for better stability
    guild_ready_timeout=15.0,  # Increase guild ready timeout
)
tree = bot.tree

# --- Bot Stability Tracking ---
bot_start_time = None  # Track when the bot started
last_gateway_refresh = None  # Track last gateway refresh
GATEWAY_REFRESH_HOURS = 48  # Refresh gateway connection every 48 hours (was 20, increased for stability)
consecutive_health_failures = 0  # Track health check failures

# --- Settings ---
guild_settings = {}  # {guild_id: {... per-alert configuration incl. sports_channels, min thresholds ...}}

# Persist guild settings so global scanning can auto-start across restarts
GUILD_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guild_settings.json")

def load_guild_settings():
    global guild_settings
    try:
        if os.path.exists(GUILD_SETTINGS_FILE):
            with open(GUILD_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Convert string keys back to integers (JSON only supports string keys)
                    guild_settings = {int(k): v for k, v in data.items()}
                    print(f"🗂️ Loaded guild settings for {len(guild_settings)} guild(s)")
                    # Log what was loaded for each guild
                    for gid, settings in guild_settings.items():
                        alerts_status = []
                        if settings.get("horse_alerts_running"):
                            alerts_status.append("🐎 Horse")
                        if settings.get("global_alerts_running"):
                            alerts_status.append("⚽ Sports")
                        if settings.get("greyhound_alerts_running"):
                            alerts_status.append("🐕 Greyhound")
                        if settings.get("trots_alerts_running"):
                            alerts_status.append("🚗 Trots")
                        if alerts_status:
                            print(f"   Guild {gid}: {', '.join(alerts_status)} alerts enabled")
                else:
                    print("⚠️ guild_settings.json is not a dict; ignoring")
        else:
            print("ℹ️ No guild_settings.json found; starting with empty settings")
    except Exception as e:
        print(f"⚠️ Failed to load guild settings: {e}")

def save_guild_settings():
    try:
        # Convert int keys to strings for JSON serialization
        serializable = {str(k): v for k, v in guild_settings.items()}
        # Write to temp file first, then rename to avoid corruption
        temp_file = GUILD_SETTINGS_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        # Atomic rename
        import shutil
        shutil.move(temp_file, GUILD_SETTINGS_FILE)
        print(f"💾 Saved settings for {len(guild_settings)} guild(s)")
    except Exception as e:
        print(f"⚠️ Failed to save guild settings: {e}")
        # Clean up temp file if it exists
        try:
            os.remove(temp_file)
        except:
            pass

async def save_guild_settings_async():
    """Async wrapper to save settings without blocking"""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_guild_settings)

# --- Event Deduplication ---
posted_events = {}  # {event_name: timestamp} - track posted events to prevent multiple bets per game

# --- Profit Tracking System ---
tracked_bets = []  # List of all tracked bets with full details
daily_stats = {
    "date": None,
    "total_bets": 0,
    "pending_bets": 0,
    "won_bets": 0,
    "lost_bets": 0,
    "total_stakes": 0.0,
    "total_returns": 0.0,
    "pnl": 0.0
}

# --- Persistent Bet History & Timezone Helpers ---
AWST_TZ = datetime.timezone(datetime.timedelta(hours=8))  # Perth time (UTC+8)
BET_HISTORY_FILE = os.path.join(BASE_DIR, "bet_history.json")

def _coerce_datetime(value):
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None

def load_bet_history():
    try:
        if not os.path.exists(BET_HISTORY_FILE):
            return []
        with open(BET_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            loaded = []
            for entry in data if isinstance(data, list) else []:
                entry["timestamp"] = _coerce_datetime(entry.get("timestamp")) or datetime.datetime.now(datetime.timezone.utc)
                entry["settled_at"] = _coerce_datetime(entry.get("settled_at"))
                loaded.append(entry)
            print(f"🗂️ Loaded {len(loaded)} bets from history")
            return loaded
    except Exception as e:
        print(f"⚠️ Failed to load bet history: {e}")
        return []

def save_bet_history():
    try:
        serializable = []
        for bet in tracked_bets:
            item = {**bet}
            # Convert datetimes to ISO strings for JSON
            ts = bet.get("timestamp")
            settled_ts = bet.get("settled_at")
            item["timestamp"] = ts.isoformat() if isinstance(ts, datetime.datetime) else ts
            item["settled_at"] = settled_ts.isoformat() if isinstance(settled_ts, datetime.datetime) else settled_ts
            serializable.append(item)
        with open(BET_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save bet history: {e}")

# Initialize tracked bets from history if present
try:
    tracked_bets = load_bet_history()
    # Import existing bet history into ML learning system
    import_existing_bet_history()
except Exception:
    tracked_bets = []

def record_loss():
    """Record a betting loss and check if cooldown should start"""
    current_time = datetime.datetime.now()
    loss_tracking['losses'].append(current_time)
    
    # Remove losses older than 1 hour
    one_hour_ago = current_time - datetime.timedelta(hours=1)
    loss_tracking['losses'] = [loss for loss in loss_tracking['losses'] if loss > one_hour_ago]
    
    # Check if we have 3 losses in the last hour
    if len(loss_tracking['losses']) >= 3:
        # Start 1-hour cooldown
        loss_tracking['cooldown_until'] = current_time + datetime.timedelta(hours=1)
        print(f"🛑 COOLDOWN ACTIVATED: 3 losses in 1 hour. Cooldown until {loss_tracking['cooldown_until'].strftime('%H:%M:%S')}")

def check_cooldown_status():
    """Check if we're in cooldown and return status message"""
    if not loss_tracking['cooldown_until']:
        return None, False
    
    current_time = datetime.datetime.now()
    if current_time >= loss_tracking['cooldown_until']:
        # Cooldown period ended
        loss_tracking['cooldown_until'] = None
        loss_tracking['losses'] = []  # Reset losses
        return None, False
    
    # Still in cooldown
    time_remaining = loss_tracking['cooldown_until'] - current_time
    minutes_remaining = int(time_remaining.total_seconds() / 60)
    
    cooldown_message = f"""
🛑 **BETTING SUSPENDED - COOLDOWN ACTIVE**
⏰ **Time Remaining:** {minutes_remaining} minutes
🚫 **Reason:** 3 losses detected in 1 hour
💤 **Status:** No bets until cooldown ends
"""
    return cooldown_message, True

def get_horse_betting_guidelines(odds, recommended_units):
    """Generate horse/greyhound betting guidelines with specific stake recommendation."""
    # Check cooldown status first
    cooldown_message, is_in_cooldown = check_cooldown_status()
    if is_in_cooldown:
        return cooldown_message

    caution = "" if odds > 1.70 else "⚠️ Odds on the shorter side; size your risk accordingly."
    lines = [f"🎯 **BET {recommended_units} UNITS ON THIS SELECTION**"]
    if caution:
        lines.append(caution)
    return "\n".join(lines)

async def get_gemini_bet_analysis(event_name, market_name, runner_name, odds, bet_amount, total_matched, sport_type):
    """Get AI analysis of the bet using Gemini 2.5 Flash with grounding"""
    try:
        # Create the model (Gemini 2.5 has built-in grounding capabilities)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        percentage = (bet_amount / total_matched * 100) if total_matched > 0 else 0
        
        prompt = f"""
You are a professional sports betting analyst. A massive ${bet_amount:,.0f} bet just came through.

**BET DETAILS:**
• Event: {event_name}
• Market: {market_name}
• Bet On: {runner_name}
• Odds: ${odds:.2f}
• Bet Amount: ${bet_amount:,.0f} ({percentage:.1f}% of total market)
• Sport: {sport_type}

**SEARCH AND ANALYZE:**
Look up current information about {event_name} including:
- Recent form and results for both sides
- Key injuries, suspensions, team news
- Head-to-head record
- Home/away factors
- Why this {runner_name} selection at ${odds:.2f} might attract a ${bet_amount:,.0f} bet

**PROVIDE:**
1. Quick summary of what's happening in this matchup
2. Key factors supporting or opposing this bet
3. Risk level (Low/Medium/High)
4. **CLEAR FINAL VERDICT:**
   - "✅ **BET ON IT** - [reason]" if it's a solid opportunity
   - "❌ **DON'T BET** - [reason]" if too risky or poor value

Be decisive and specific. Max 250 words.
"""
        
        # Generate response with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=25.0
        )
        
        if response and response.text:
            return response.text
        else:
            return "⚠️ AI analysis unavailable at this time."
            
    except asyncio.TimeoutError:
        return "⚠️ AI analysis timed out - bet analysis unavailable."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return f"⚠️ AI analysis error: {str(e)[:100]}"

def get_sports_betting_guidelines(odds, recommended_units, volume=0):
    """Generate sports betting guidelines with specific stake recommendation"""
    # Check cooldown status first
    cooldown_message, is_in_cooldown = check_cooldown_status()
    if is_in_cooldown:
        return cooldown_message
    
    volume_note = ""
    if volume >= 1000000:
        volume_note = "\n🚨 **Volume exceeds $1M - Units doubled!**"

    caution = "" if odds > 1.70 else "⚠️ Odds are short; monitor price movement."
    return (
        f"🎯 **BET {recommended_units} UNITS ON THIS SELECTION**{volume_note}\n"
        f"{caution}\n"
        "👯‍♂️ Multiple Bets on Same Game: Avoid stacking positions.\n"
        "✅ **FINAL VERDICT:** BET ON THIS (significant market stake detected)"
    )

BETFAIR_BETTING_API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
BETFAIR_ACCOUNT_API_URL = "https://api.betfair.com/exchange/account/json-rpc/v1"

# --- Betfair API ---
def log_login(success, details=""):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "FAIL"
    print(f"[LOGIN][{now}] {status}: {details}")
    with open("betfair_login.log", "a") as f:
        f.write(f"[{now}] {status}: {details}\n")

def log_event(event_type, details):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{event_type.upper()}][{now}] {details}")
    with open("betfair_events.log", "a") as f:
        f.write(f"[{now}] [{event_type.upper()}] {details}\n")

# --- Betfair Session Token Cache ---
session_token_cache = {
    "token": None,
    "expires": None  # datetime.datetime
}



# --- BETFAIR CREDENTIALS FROM ENVIRONMENT VARIABLES ---
BETFAIR_USERNAME = os.environ.get("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD = os.environ.get("BETFAIR_PASSWORD", "")
BETFAIR_APP_KEY = os.environ.get("BETFAIR_APP_KEY", "")

def get_session_token():
    """Get Betfair session token using hardcoded credentials and local cert files."""
    # Check if we have a cached valid token
    now = datetime.datetime.now(datetime.timezone.utc)
    if session_token_cache["token"] and session_token_cache["expires"]:
        if now < session_token_cache["expires"]:
            print("✅ Using cached Betfair session token")
            return session_token_cache["token"]
        else:
            print("🔄 Session token expired, getting new one...")
    
    # Check certificate files exist
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        print(f"❌ Certificate files not found:")
        print(f"   CERT: {CERT_FILE} (exists: {os.path.exists(CERT_FILE)})")
        print(f"   KEY:  {KEY_FILE} (exists: {os.path.exists(KEY_FILE)})")
        return None
    
    print(f"🔄 Attempting Betfair login with certificates...")
    print(f"   Username: {BETFAIR_USERNAME}")
    print(f"   App Key: {BETFAIR_APP_KEY}")
    print(f"   Cert File: {os.path.basename(CERT_FILE)}")
    
    # Prepare login request
    url = 'https://identitysso-cert.betfair.com/api/certlogin'
    headers = {
        'X-Application': BETFAIR_APP_KEY,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'username': BETFAIR_USERNAME,
        'password': BETFAIR_PASSWORD
    }
    
    try:
        # Make the request with certificate authentication
        response = requests.post(
            url, 
            data=data, 
            headers=headers, 
            cert=(CERT_FILE, KEY_FILE),
            timeout=30,
            verify=True  # Verify SSL certificates
        )
        
        print(f"🔍 HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        try:
            result = response.json()
        except Exception as json_error:
            print(f"❌ Failed to parse JSON response: {json_error}")
            print(f"Raw response: {response.text}")
            return None
        
        print(f"🔍 Betfair Response: {result}")
        
        if result.get('loginStatus') == 'SUCCESS':
            session_token = result['sessionToken']
            # Cache for 8 hours (Betfair tokens last 12 hours)
            session_token_cache["token"] = session_token
            session_token_cache["expires"] = now + datetime.timedelta(hours=8)
            
            print("✅ Betfair login successful!")
            log_login(True, "Login successful")
            return session_token
        else:
            error_code = result.get('error', 'Unknown error')
            login_status = result.get('loginStatus', 'Unknown status')
            
            print(f"❌ Betfair login failed:")
            print(f"   Status: {login_status}")
            print(f"   Error: {error_code}")
            
            # Provide helpful error messages
            if error_code == 'INVALID_USERNAME_OR_PASSWORD':
                print("💡 Check your username and password")
            elif error_code == 'CERT_AUTH_REQUIRED':
                print("💡 Certificate authentication failed - check your .crt and .key files")
                print("💡 Make sure the certificate files match your Betfair account")
            elif error_code == 'ACCOUNT_NOW_LOCKED':
                print("💡 Account is locked - wait or contact Betfair")
            
            log_login(False, f"Login failed: {login_status} - {error_code}")
            return None
            
    except requests.exceptions.SSLError as e:
        print(f"❌ SSL/Certificate Error: {e}")
        print("💡 This usually means:")
        print("   - Certificate files are corrupted or invalid")
        print("   - Certificate doesn't match your Betfair account") 
        print("   - Certificate files have wrong permissions")
        log_login(False, f"SSL Error: {e}")
        return None
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("💡 Network connectivity issue or Betfair server unreachable")
        log_login(False, f"Connection Error: {e}")
        return None
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout after 30 seconds")
        print("💡 Network is too slow or Betfair servers are slow")
        log_login(False, "Request timeout")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error during login: {e}")
        print(f"Exception type: {type(e).__name__}")
        log_login(False, f"Unexpected error: {e}")
        return None

def call_betfair_api(jsonrpc_req, session_token, api_type="betting", retries=3):
    """Call Betfair API with proper error handling and certificate authentication"""
    url = BETFAIR_BETTING_API_URL if api_type == "betting" else BETFAIR_ACCOUNT_API_URL
    headers = {
        'X-Application': BETFAIR_APP_KEY,
        'X-Authentication': session_token,
        'Content-Type': 'application/json'
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                url, 
                data=jsonrpc_req, 
                headers=headers, 
                cert=(CERT_FILE, KEY_FILE),
                timeout=30,
                verify=True
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Betfair API error (attempt {attempt + 1}): {response.status_code}")
                if attempt == retries - 1:  # Last attempt
                    print(f"Response: {response.text}")
                    
        except requests.exceptions.Timeout:
            print(f"⏱️ API timeout (attempt {attempt + 1})")
        except requests.exceptions.SSLError as e:
            print(f"❌ SSL error in API call: {e}")
            break  # Don't retry SSL errors
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error in API call (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"❌ Exception in call_betfair_api (attempt {attempt + 1}): {e}")
    
    return {}


def ensure_sport_event_type_cache(session_token, force=False):
    """Ensure we have up-to-date Betfair eventTypeIds for configured sports."""
    global SPORT_EVENT_TYPE_CACHE, SPORT_EVENT_TYPE_ID_TO_KEY, SPORT_EVENT_TYPES_LAST_FETCH

    now_ts = time.time()
    cache_fresh = SPORT_EVENT_TYPE_CACHE and not force and (now_ts - SPORT_EVENT_TYPES_LAST_FETCH) < SPORT_EVENT_CACHE_TTL
    if cache_fresh:
        return SPORT_EVENT_TYPE_CACHE

    try:
        req = json.dumps({
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listEventTypes",
            "params": {"filter": {}},
            "id": 1
        })
        resp = call_betfair_api(req, session_token)
        if not resp or not resp.get("result"):
            print("⚠️ Failed to refresh sport event type cache – using previous mapping")
            return SPORT_EVENT_TYPE_CACHE

        event_types = resp["result"]
        by_name = {
            (item.get("eventType", {}).get("name", "").lower()): item.get("eventType", {}).get("id")
            for item in event_types
            if item.get("eventType")
        }

        updated_cache = {}
        updated_reverse = {}
        missing = []

        for category in SPORT_CATEGORIES:
            name_key = category["betfair_name"].lower()
            event_type_id = by_name.get(name_key)
            if event_type_id:
                updated_cache[category["key"]] = event_type_id
                updated_reverse[event_type_id] = category["key"]
            else:
                missing.append(category["betfair_name"])

        if missing:
            print(f"⚠️ Betfair event types missing for: {', '.join(missing)}")

        if updated_cache:
            SPORT_EVENT_TYPE_CACHE = updated_cache
            SPORT_EVENT_TYPE_ID_TO_KEY = updated_reverse
            SPORT_EVENT_TYPES_LAST_FETCH = now_ts
            print(f"📘 Betfair sport mapping refreshed: {SPORT_EVENT_TYPE_CACHE}")

        return SPORT_EVENT_TYPE_CACHE
    except Exception as e:
        print(f"❌ Error refreshing sport event type cache: {e}")
        return SPORT_EVENT_TYPE_CACHE

def get_au_racing_event_type_id(session_token):
    """Get the Betfair event type ID for Australian horse racing"""
    return "7"  # Horse racing event type ID

def place_bet(market_id, runner_name, size=1.0, price=1.50):
    session_token = get_session_token()
    if not session_token:
        return "Login failed"
    # Find runner selectionId by name
    # For demo, just get runners from market catalogue
    event_type_id = get_au_racing_event_type_id(session_token)
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    market_catalogue_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "SportsAPING/v1.0/listMarketCatalogue",
        "params": {
            "filter": {
                "marketIds": [market_id]
            },
            "maxResults": "1",
            "marketProjection": ["RUNNER_METADATA"]
        },
        "id": 1
    })
    catalogue_resp = call_betfair_api(market_catalogue_req, session_token)
    selection_id = None
    for market in catalogue_resp.get("result", []):
        for runner in market["runners"]:
            if runner["runnerName"] == runner_name:
                selection_id = runner["selectionId"]
                break
    if not selection_id:
        return "Runner not found"
    place_order_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "SportsAPING/v1.0/placeOrders",
        "params": {
            "marketId": market_id,
            "instructions": [{
                "selectionId": selection_id,
                "handicap": "0",
                "side": "BACK",
                "orderType": "LIMIT",
                "limitOrder": {
                    "size": str(size),
                    "price": str(price),
                    "persistenceType": "LAPSE"
                }
            }],
            "customerRef": "discordbot"
        },
        "id": 1
    })
    resp = call_betfair_api(place_order_req, session_token)
    return resp

def get_balance():
    session_token = get_session_token()
    if not session_token:
        print("❌ No session token. Login likely failed.")
        return None

    req = json.dumps({
        "jsonrpc": "2.0",
        "method": "AccountAPING/v1.0/getAccountFunds",
        "params": {},
        "id": 1
    })

    resp = call_betfair_api(req, session_token, api_type="account")
    print("🔍 Raw response from getAccountFunds:")
    print(json.dumps(resp, indent=2))

    try:
        return float(resp["result"]["availableToBetBalance"])
    except Exception as e:
        print(f"❌ Error fetching balance: {e}")
        return None

class SetAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🔔 Set Alert Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        guild = interaction.guild
        if not guild:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return
        channels = [c for c in guild.text_channels][:25]
        options = [discord.SelectOption(label=c.name, value=str(c.id)) for c in channels]
        select = discord.ui.Select(placeholder="Select channel", options=options)
        async def select_callback(select_interaction):
            channel_id = int(select_interaction.data['values'][0])
            guild_settings.setdefault(guild.id, {})["alert_channel"] = channel_id
            await select_interaction.response.send_message(f"✅ Alert channel set to <#{channel_id}>", ephemeral=True)
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("Choose a channel for bet alerts:", view=view, ephemeral=True)

class SetMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="💵 Set Min $ Matched", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        guild = interaction.guild
        if not guild:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return
        options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
        select = discord.ui.Select(placeholder="Minimum $ matched for alerts", options=options)
        async def select_callback(select_interaction):
            min_amt = int(select_interaction.data['values'][0])
            guild_settings.setdefault(guild.id, {})["min_matched"] = min_amt
            await select_interaction.response.send_message(f"✅ Minimum matched set to ${min_amt}", ephemeral=True)
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("Choose minimum $ matched for alerts:", view=view, ephemeral=True)

class ChannelPaginationView(discord.ui.View):
    def __init__(self, channels, alert_type, guild_id, page=0):
        super().__init__(timeout=300)
        self.channels = channels
        self.alert_type = alert_type
        self.guild_id = guild_id
        self.page = page
        self.per_page = 20

        # Calculate pagination
        start_idx = page * self.per_page
        end_idx = min(start_idx + self.per_page, len(channels))
        current_channels = channels[start_idx:end_idx]

        # Create select menu for current page
        options = [discord.SelectOption(
            label=f"#{c.name}",
            value=str(c.id),
            description=f"Category: {c.category.name if c.category else 'No Category'}"
        ) for c in current_channels]

        select = discord.ui.Select(
            placeholder=f"Select channel (Page {page + 1}/{self.total_pages})",
            options=options
        )

        async def select_callback(interaction):
            try:
                channel_id = int(interaction.data['values'][0])
                settings = guild_settings.setdefault(self.guild_id, {})

                if isinstance(self.alert_type, str) and self.alert_type.startswith("sport:"):
                    sport_key = self.alert_type.split(":", 1)[1]
                    sport_meta = SPORT_CATEGORY_LOOKUP.get(sport_key, {})
                    sports_channels = settings.setdefault("sports_channels", {})
                    sports_channels[sport_key] = channel_id
                    save_guild_settings()
                    emoji = sport_meta.get("emoji", "⚽")
                    label = sport_meta.get("label", sport_key.title())
                    await interaction.response.send_message(
                        f"✅ {emoji} {label} alerts → <#{channel_id}>",
                        ephemeral=True
                    )
                    return

                settings_key = {
                    "sports": "alert_channel",
                    "horse": "horse_alert_channel",
                    "au_horse": "au_horse_alert_channel",
                    "uk_horse": "uk_horse_alert_channel",
                    "greyhound": "greyhound_alert_channel",
                    "trots": "trots_alert_channel",
                    "report": "report_channel"
                }.get(self.alert_type, "alert_channel")

                settings[settings_key] = channel_id
                save_guild_settings()
                emoji = {"sports": "⚽", "horse": "🐎", "au_horse": "🇦🇺", "uk_horse": "🇬🇧", "greyhound": "🐕", "trots": "🚗", "report": "📊"}.get(self.alert_type, "📢")
                await interaction.response.send_message(
                    f"✅ {emoji} {self.alert_type.title()} alert channel set to <#{channel_id}>",
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error setting {self.alert_type} channel: {e}")
                await interaction.response.send_message("❌ Error setting channel", ephemeral=True)
        select.callback = select_callback
        self.add_item(select)

        # Add navigation buttons
        if page > 0:
            self.add_item(PreviousPageButton(self))
        if end_idx < len(channels):
            self.add_item(NextPageButton(self))

    @property
    def total_pages(self):
        return (len(self.channels) + self.per_page - 1) // self.per_page

class PreviousPageButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="◀️ Previous", style=discord.ButtonStyle.secondary)
        self.parent_view = parent_view
    
    async def callback(self, interaction):
        new_view = ChannelPaginationView(
            self.parent_view.channels,
            self.parent_view.alert_type,
            self.parent_view.guild_id,
            self.parent_view.page - 1
        )
        await interaction.response.edit_message(view=new_view)

class NextPageButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="Next ▶️", style=discord.ButtonStyle.secondary)
        self.parent_view = parent_view
    
    async def callback(self, interaction):
        new_view = ChannelPaginationView(
            self.parent_view.channels,
            self.parent_view.alert_type,
            self.parent_view.guild_id,
            self.parent_view.page + 1
        )
        await interaction.response.edit_message(view=new_view)

class SetSportsAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="⚽ Set Sports Alert Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
            return

        options = [
            discord.SelectOption(
                label=category["label"],
                value=category["key"],
                emoji=category.get("emoji")
            )
            for category in SPORT_CATEGORIES
        ]

        sport_select = discord.ui.Select(placeholder="Select sport to configure", options=options)

        async def sport_select_callback(select_interaction: discord.Interaction):
            try:
                sport_key = select_interaction.data['values'][0]
                sport_meta = SPORT_CATEGORY_LOOKUP.get(sport_key, {})
                channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
                if not channels:
                    await select_interaction.response.send_message("❌ No accessible text channels found.", ephemeral=True)
                    return
                view = ChannelPaginationView(channels, f"sport:{sport_key}", guild.id)
                await select_interaction.response.send_message(
                    f"Choose a channel for {sport_meta.get('label', sport_key.title())} alerts:\n**Found {len(channels)} channels**",
                    view=view,
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error during sport channel selection: {e}")
                await select_interaction.response.send_message("❌ Error selecting sport.", ephemeral=True)

        sport_select.callback = sport_select_callback
        view = discord.ui.View()
        view.add_item(sport_select)
        await interaction.followup.send("Select which sport you want to configure:", view=view, ephemeral=True)

class SetAUHorseAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🇦🇺 Set AU Horse Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            guild = interaction.guild
            channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
            if not channels:
                await interaction.followup.send("❌ No accessible text channels found.", ephemeral=True)
                return
            view = ChannelPaginationView(channels, "au_horse", guild.id)
            await interaction.followup.send(
                f"Choose a channel for AU horse bet alerts:\n**Found {len(channels)} channels**",
                view=view,
                ephemeral=True
            )
        except Exception as e:
            print(f"❌ Error in SetAUHorseAlertChannelButton: {e}")
            await interaction.followup.send("❌ An error occurred. Please try again.", ephemeral=True)

class SetAUHorseMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🇦🇺 Set AU Horse Min $", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            guild = interaction.guild
            options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
            select = discord.ui.Select(placeholder="Minimum $ matched for AU horse alerts", options=options)
            async def select_callback(select_interaction):
                try:
                    min_amt = int(select_interaction.data['values'][0])
                    guild_settings.setdefault(guild.id, {})["au_horse_min_matched"] = min_amt
                    save_guild_settings()
                    await select_interaction.response.send_message(f"✅ AU Horse minimum matched set to ${min_amt:,}", ephemeral=True)
                except Exception as e:
                    print(f"❌ Error in AU horse min matched callback: {e}")
                    await select_interaction.response.send_message("❌ Error updating setting. Please try again.", ephemeral=True)
            select.callback = select_callback
            view = discord.ui.View(timeout=300)
            view.add_item(select)
            await interaction.followup.send("Choose minimum $ matched for AU horse alerts:", view=view, ephemeral=True)
        except Exception as e:
            print(f"❌ Error in SetAUHorseMinMatchedButton: {e}")
            await interaction.followup.send("❌ An error occurred. Please try again.", ephemeral=True)

class SetUKHorseAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🇬🇧 Set UK/ROW Horse Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            guild = interaction.guild
            channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
            if not channels:
                await interaction.followup.send("❌ No accessible text channels found.", ephemeral=True)
                return
            view = ChannelPaginationView(channels, "uk_horse", guild.id)
            await interaction.followup.send(
                f"Choose a channel for UK/Rest of World horse bet alerts:\n**Found {len(channels)} channels**",
                view=view,
                ephemeral=True
            )
        except Exception as e:
            print(f"❌ Error in SetUKHorseAlertChannelButton: {e}")
            await interaction.followup.send("❌ An error occurred. Please try again.", ephemeral=True)

class SetUKHorseMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🇬🇧 Set UK/ROW Horse Min $", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        try:
            await interaction.response.defer(ephemeral=True)
            guild = interaction.guild
            options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
            select = discord.ui.Select(placeholder="Minimum $ matched for UK/ROW horse alerts", options=options)
            async def select_callback(select_interaction):
                try:
                    min_amt = int(select_interaction.data['values'][0])
                    guild_settings.setdefault(guild.id, {})["uk_horse_min_matched"] = min_amt
                    save_guild_settings()
                    await select_interaction.response.send_message(f"✅ UK/ROW Horse minimum matched set to ${min_amt:,}", ephemeral=True)
                except Exception as e:
                    print(f"❌ Error in UK horse min matched callback: {e}")
                    await select_interaction.response.send_message("❌ Error updating setting. Please try again.", ephemeral=True)
            select.callback = select_callback
            view = discord.ui.View(timeout=300)
            view.add_item(select)
            await interaction.followup.send("Choose minimum $ matched for UK/ROW horse alerts:", view=view, ephemeral=True)
        except Exception as e:
            print(f"❌ Error in SetUKHorseMinMatchedButton: {e}")
            await interaction.followup.send("❌ An error occurred. Please try again.", ephemeral=True)



class RolePaginationView(discord.ui.View):
    def __init__(self, roles, guild_id, page=0):
        super().__init__(timeout=300)
        self.roles = roles
        self.guild_id = guild_id
        self.page = page
        self.per_page = 20
        
        # Calculate pagination
        start_idx = page * self.per_page
        end_idx = min(start_idx + self.per_page, len(roles))
        current_roles = roles[start_idx:end_idx]
        
        # Create select menu for current page
        options = [discord.SelectOption(
            label=r.name,
            value=str(r.id),
            description=f"Members: {len(r.members)} • Color: {str(r.color)}"
        ) for r in current_roles]
        
        if options:
            select = discord.ui.Select(
                placeholder=f"Select role to mention (Page {page + 1}/{self.total_pages})",
                options=options
            )
            select.callback = self.select_callback
            self.add_item(select)
        
        # Add navigation buttons
        if page > 0:
            self.add_item(PreviousRolePageButton(self))
        if end_idx < len(roles):
            self.add_item(NextRolePageButton(self))
    
    @property
    def total_pages(self):
        return (len(self.roles) + self.per_page - 1) // self.per_page
    
    async def select_callback(self, interaction):
        try:
            role_id = int(interaction.data['values'][0])
            guild_settings.setdefault(self.guild_id, {})["mention_role"] = role_id
            await interaction.response.send_message(f"✅ Mention role set to <@&{role_id}>", ephemeral=True)
        except Exception as e:
            print(f"❌ Error setting mention role: {e}")
            await interaction.response.send_message("❌ Error setting role", ephemeral=True)

class PreviousRolePageButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="◀️ Previous", style=discord.ButtonStyle.secondary)
        self.parent_view = parent_view
    
    async def callback(self, interaction):
        new_view = RolePaginationView(
            self.parent_view.roles,
            self.parent_view.guild_id,
            self.parent_view.page - 1
        )
        await interaction.response.edit_message(view=new_view)

class NextRolePageButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="Next ▶️", style=discord.ButtonStyle.secondary)
        self.parent_view = parent_view
    
    async def callback(self, interaction):
        new_view = RolePaginationView(
            self.parent_view.roles,
            self.parent_view.guild_id,
            self.parent_view.page + 1
        )
        await interaction.response.edit_message(view=new_view)

class SetMentionRoleButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="👤 Set Mention Role", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
            return
        
        roles = [r for r in guild.roles if not r.managed and r.name != "@everyone"]
        
        if not roles:
            await interaction.followup.send("❌ No mentionable roles found.", ephemeral=True)
            return
        
        view = RolePaginationView(roles, guild.id)
        await interaction.followup.send(
            f"Choose a role to mention for big bet alerts:\n**Found {len(roles)} roles**", 
            view=view, 
            ephemeral=True
        )

class ShowBalanceButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="💰 Show Betfair Balance", style=discord.ButtonStyle.success)

    async def callback(self, interaction: discord.Interaction):
        balance = get_balance()
        if balance is None:
            await interaction.response.send_message("💰 Error fetching balance.", ephemeral=True)
        else:
            await interaction.response.send_message(f"💰 Betfair Balance: **${balance:,.2f}**", ephemeral=True)

class ListMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="📊 List Matched", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        # Get races in next 12 hours
        session_token = get_session_token()
        if not session_token:
            await interaction.response.send_message("❌ Betfair login failed.", ephemeral=True)
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        later = now + datetime.timedelta(hours=12)
        event_type_id = get_au_racing_event_type_id(session_token)
        market_catalogue_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketCatalogue",
            "params": {
                "filter": {
                    "eventTypeIds": [event_type_id],
                    "marketTypeCodes": ["WIN"],
                    "marketStartTime": {
                        "from": now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "to": later.strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                },
                "sort": "FIRST_TO_START",
                "maxResults": "25",
                "marketProjection": ["RUNNER_METADATA", "EVENT"]
            },
            "id": 1
        })
        catalogue_resp = call_betfair_api(market_catalogue_req, session_token)
        markets = catalogue_resp.get("result", [])
        if not markets:
            await interaction.response.send_message("No races found in next 12 hours.", ephemeral=True)
            return
        options = []
        for market in markets:
            venue = market["event"].get("venue", "Unknown")
            market_name = market.get("marketName", "Unknown")
            start = market.get("marketStartTime", "")[:16].replace("T", " ")
            label = f"{venue} - {market_name} ({start})"
            options.append(discord.SelectOption(label=label, value=market["marketId"]))
        select = discord.ui.Select(placeholder="Select a race", options=options)
        async def select_callback(select_interaction):
            market_id = select_interaction.data['values'][0]
            view = discord.ui.View()
            view.add_item(ShowMatchedButton(market_id))
            await select_interaction.response.send_message(
                f"Selected race: `{market_id}`. Click below to show matched money.",
                view=view,
                ephemeral=True
            )
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("Select a race to view matched money:", view=view, ephemeral=True)

class ShowMatchedButton(discord.ui.Button):
    def __init__(self, market_id):
        super().__init__(label="Show Matched", style=discord.ButtonStyle.success)
        self.market_id = market_id

    async def callback(self, interaction: discord.Interaction):
        session_token = get_session_token()
        if not session_token:
            await interaction.response.send_message("❌ Betfair login failed.", ephemeral=True)
            return
        # Get market catalogue for runner names
        market_catalogue_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketCatalogue",
            "params": {
                "filter": {
                    "marketIds": [self.market_id]
                },
                "maxResults": "1",
                "marketProjection": ["RUNNER_METADATA"]
            },
            "id": 1
        })
        catalogue_resp = call_betfair_api(market_catalogue_req, session_token)
        runners_info = {}
        for market in catalogue_resp.get("result", []):
            for runner in market["runners"]:
                runners_info[runner["selectionId"]] = runner.get("runnerName", "Unknown")
        # Get market book for matched amounts
        market_book_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketBook",
            "params": {
                "marketIds": [self.market_id],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}
            },
            "id": 1
        })
        book_resp = call_betfair_api(market_book_req, session_token)
        if not book_resp or not book_resp.get("result"):
            await interaction.response.send_message("No market book data found.", ephemeral=True)
            return
        market_book = book_resp["result"][0]
        total_matched = market_book.get("totalMatched", 0)
        if total_matched is None:
            total_matched = 0
        msg = f"**Total Matched:** ${total_matched:,.2f}\n\n"
        for runner in market_book["runners"]:
            runner_id = runner["selectionId"]
            runner_name = runners_info.get(runner_id, str(runner_id))
            matched = runner.get("totalMatched", 0)
            if matched is None:
                matched = 0
            msg += f"🏇 {runner_name}: Matched ${matched:,.2f}\n"
        await interaction.response.send_message(msg, ephemeral=True)

class BetButton(discord.ui.Button):
    def __init__(self, market_id, runner_name):
        super().__init__(label=f"Auto Bet on {runner_name}", style=discord.ButtonStyle.danger)
        self.market_id = market_id
        self.runner_name = runner_name

    async def callback(self, interaction: discord.Interaction):
        result = place_bet(self.market_id, self.runner_name)
        log_event("BET", f"Bet attempted on {self.runner_name} in market {self.market_id} by {interaction.user}. Result: {result}")
        await interaction.response.send_message(f"Bet result: {result}", ephemeral=True)

# --- Track total staked ---
  # key: (market_id, runner_name), value: float

class SetGreyhoundAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🐕 Set Greyhound Alert Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
        
        if not channels:
            await interaction.followup.send("❌ No accessible text channels found.", ephemeral=True)
            return
        
        view = ChannelPaginationView(channels, "greyhound", guild.id)
        await interaction.followup.send(
            f"Choose a channel for greyhound bet alerts:\n**Found {len(channels)} channels**", 
            view=view, 
            ephemeral=True
        )

class SetGreyhoundMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🐕 Set Greyhound Min $ Matched", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        guild = interaction.guild
        options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
        select = discord.ui.Select(placeholder="Minimum $ matched for greyhound alerts", options=options)
        async def select_callback(select_interaction):
            min_amt = int(select_interaction.data['values'][0])
            guild_settings.setdefault(guild.id, {})["greyhound_min_matched"] = min_amt
            save_guild_settings()
            await select_interaction.response.send_message(f"✅ Greyhound minimum matched set to ${min_amt}", ephemeral=True)
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("Choose minimum $ matched for greyhound alerts:", view=view, ephemeral=True)

class SetTrotsAlertChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🚗 Set Trots Alert Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
        
        if not channels:
            await interaction.followup.send("❌ No accessible text channels found.", ephemeral=True)
            return
        
        view = ChannelPaginationView(channels, "trots", guild.id)
        await interaction.followup.send(
            f"Choose a channel for trots bet alerts:\n**Found {len(channels)} channels**", 
            view=view, 
            ephemeral=True
        )

class SetTrotsMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🚗 Set Trots Min $ Matched", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        guild = interaction.guild
        options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
        select = discord.ui.Select(placeholder="Minimum $ matched for trots alerts", options=options)
        async def select_callback(select_interaction):
            min_amt = int(select_interaction.data['values'][0])
            guild_settings.setdefault(guild.id, {})["trots_min_matched"] = min_amt
            save_guild_settings()
            await select_interaction.response.send_message(f"✅ Trots minimum matched set to ${min_amt}", ephemeral=True)
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("Choose minimum $ matched for trots alerts:", view=view, ephemeral=True)

class SetSportsMinMatchedButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="⚙️ Set Individual Sport Min", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
            return

        sport_options = [
            discord.SelectOption(
                label=category["label"],
                value=category["key"],
                emoji=category.get("emoji")
            )
            for category in SPORT_CATEGORIES
        ]

        sport_select = discord.ui.Select(placeholder="Select sport to configure", options=sport_options)

        async def sport_select_callback(select_interaction: discord.Interaction):
            try:
                sport_key = select_interaction.data['values'][0]
                sport_meta = SPORT_CATEGORY_LOOKUP.get(sport_key, {})
                amount_options = [discord.SelectOption(label=f"${amt:,}", value=str(amt)) for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]]
                amount_select = discord.ui.Select(placeholder=f"Minimum $ matched for {sport_meta.get('label', sport_key.title())}", options=amount_options)

                async def amount_callback(amount_interaction: discord.Interaction):
                    try:
                        min_amt = int(amount_interaction.data['values'][0])
                        settings = guild_settings.setdefault(guild.id, {})
                        sport_min = settings.setdefault("sports_min_matched", {})
                        sport_min[sport_key] = min_amt
                        save_guild_settings()
                        await amount_interaction.response.send_message(
                            f"✅ {sport_meta.get('label', sport_key.title())} minimum matched set to ${min_amt:,}",
                            ephemeral=True
                        )
                    except Exception as e:
                        print(f"❌ Error setting sport minimum: {e}")
                        await amount_interaction.response.send_message("❌ Error updating minimum.", ephemeral=True)

                amount_select.callback = amount_callback
                amount_view = discord.ui.View()
                amount_view.add_item(amount_select)
                await select_interaction.response.send_message(
                    f"Choose minimum $ matched for {sport_meta.get('label', sport_key.title())} alerts:",
                    view=amount_view,
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error during sport min selection: {e}")
                await select_interaction.response.send_message("❌ Error selecting sport.", ephemeral=True)

        sport_select.callback = sport_select_callback
        sport_view = discord.ui.View()
        sport_view.add_item(sport_select)
        await interaction.followup.send("Select which sport you want to configure a minimum for:", view=sport_view, ephemeral=True)


class SetAllSportsMinButton(discord.ui.Button):
    """Set the same minimum for ALL sports at once"""
    def __init__(self, guild):
        super().__init__(label="💰 Set ALL Sports Min", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
            return

        amount_options = [
            discord.SelectOption(label=f"${amt:,}", value=str(amt)) 
            for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
        ]
        amount_select = discord.ui.Select(placeholder="Set minimum for ALL sports", options=amount_options)

        async def amount_callback(amount_interaction: discord.Interaction):
            try:
                min_amt = int(amount_interaction.data['values'][0])
                settings = guild_settings.setdefault(guild.id, {})
                sport_min = settings.setdefault("sports_min_matched", {})
                
                # Set the same minimum for all sports
                for category in SPORT_CATEGORIES:
                    sport_min[category["key"]] = min_amt
                
                save_guild_settings()
                
                sports_list = ", ".join([cat["label"] for cat in SPORT_CATEGORIES[:5]]) + "..."
                await amount_interaction.response.send_message(
                    f"✅ **All sports** minimum set to **${min_amt:,}**\n"
                    f"Applied to: {sports_list}",
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error setting all sports minimum: {e}")
                await amount_interaction.response.send_message("❌ Error updating minimums.", ephemeral=True)

        amount_select.callback = amount_callback
        amount_view = discord.ui.View()
        amount_view.add_item(amount_select)
        await interaction.followup.send(
            "**Set minimum $ matched for ALL sports at once:**\n"
            "This will apply to: Soccer, Tennis, Cricket, Basketball, American Football, Baseball, Ice Hockey, Golf, Rugby Union, Rugby League",
            view=amount_view, 
            ephemeral=True
        )

# --- Start the tasks in on_ready ---
@bot.event
async def on_ready():
    global bot_start_time, last_gateway_refresh
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
    
    # Track bot start time for uptime monitoring
    bot_start_time = datetime.datetime.now(datetime.timezone.utc)
    last_gateway_refresh = bot_start_time
    
    # Load persisted guild settings once at startup
    try:
        load_guild_settings()
    except Exception as e:
        print(f"⚠️ Could not load guild settings on startup: {e}")
    
    # Test Betfair login
    session_token = get_session_token()
    if session_token:
        print("✅ Betfair login successful.")
    else:
        print("❌ Betfair login failed. Check credentials or certificate files.")
        print("⚠️ Bot will continue running but Betfair features may not work.")
    
    # Start daily summary report task
    if not daily_summary_report.is_running():
        daily_summary_report.start()
        print("📊 Daily summary report task started")
    
    # Check certificate files exist
    if not CERT_FILE or not os.path.exists(CERT_FILE):
        print(f"❌ Certificate file not available")
    if not KEY_FILE or not os.path.exists(KEY_FILE):
        print(f"❌ Key file not available")
    
    try:
        synced = await tree.sync()
        print(f"✅ Synced {len(synced)} commands.")
    except Exception as e:
        print(f"❌ Command sync failed: {e}")
    
    # Start tasks
    try:
        win_loss_check_task.start()
        print("✅ Started win/loss check task")
    except Exception as e:
        print(f"❌ Failed to start win/loss check task: {e}")
    
    try:
        countdown_update_task.start()
        print("✅ Started countdown update task")
    except Exception as e:
        print(f"❌ Failed to start countdown update task: {e}")
    
    try:
        cache_cleanup_task.start()
        print("✅ Started cache cleanup task")
    except Exception as e:
        print(f"❌ Failed to start cache cleanup task: {e}")
    
    try:
        ml_analysis_task.start()
        print("🧠 Started ML analysis task")
        
        # Notify admin about ML system status on startup (with cooldown to prevent spam on rapid restarts)
        total_bets = len(ml_learning_data.get("all_observed_bets", []))
        settled_bets = len([b for b in ml_learning_data.get("all_observed_bets", []) if b.get("status") in ["won", "lost"]])
        patterns = len(ml_learning_data.get("profitable_patterns", {}))
        
        await notify_admin(
            f"🤖 **Bot Started - ML System Active**\n\n"
            f"📊 **Learning Status:**\n"
            f"• Total observed bets: {total_bets}\n"
            f"• Settled bets: {settled_bets}\n"
            f"• Profitable patterns found: {patterns}\n"
            f"• Learning mode: {'ON' if ML_CONFIG.get('learning_mode') else 'OFF'}\n"
            f"• Auto-adjust: {'ON' if ML_CONFIG.get('auto_adjust_enabled') else 'OFF'}\n\n"
            f"*I'll message you when I discover profitable patterns!*",
            notification_type="startup"
        )
    except Exception as e:
        print(f"❌ Failed to start ML analysis task: {e}")
    
    # Start bot health monitoring task
    try:
        if not bot_health_check.is_running():
            bot_health_check.start()
            print("💓 Started bot health check task")
    except Exception as e:
        print(f"❌ Failed to start health check task: {e}")
    
    # Auto-start live stream if any guild has alerts enabled (after loading settings)
    try:
        await start_live_stream_if_needed()
        if live_stream_running:
            print("🔁 Global live stream already running based on persisted settings")
        else:
            print("🔄 Live stream will start when any alerts are enabled")
    except Exception as e:
        print(f"⚠️ Could not auto-start live stream: {e}")

@bot.event
async def on_error(event, *args, **kwargs):
    print(f"❌ Bot error in {event}: {args}")
    traceback.print_exc()


# Track disconnect count for monitoring
disconnect_count = 0
last_disconnect_time = None

@bot.event
async def on_disconnect():
    """Called when the bot disconnects from Discord"""
    global disconnect_count, last_disconnect_time
    disconnect_count += 1
    last_disconnect_time = datetime.datetime.now(datetime.timezone.utc)
    print(f"⚠️ Discord bot disconnected at {last_disconnect_time.isoformat()} (disconnect #{disconnect_count})")
    print("🔄 Discord.py will attempt automatic reconnection...")


@bot.event
async def on_resumed():
    """Called when the bot resumes a session after disconnect"""
    global last_gateway_refresh
    print(f"✅ Discord bot resumed session at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    last_gateway_refresh = datetime.datetime.now(datetime.timezone.utc)
    
    # Restart live stream if it was running but died
    try:
        await ensure_live_stream_healthy()
    except Exception as e:
        print(f"⚠️ Could not restart live stream on resume: {e}")
    
    # Notify admin about reconnection (with cooldown to prevent spam)
    try:
        await notify_admin("🔄 **Bot Reconnected**\nThe bot has successfully resumed its Discord session.", notification_type="reconnect")
    except:
        pass


@bot.event  
async def on_connect():
    """Called when the bot connects to Discord"""
    global last_gateway_refresh
    print(f"🔗 Discord bot connected at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    last_gateway_refresh = datetime.datetime.now(datetime.timezone.utc)


async def ensure_live_stream_healthy():
    """Check and restart the live stream monitor if it died"""
    global live_stream_task, live_stream_running
    
    # Check if any guild needs alerts
    alerts_needed = any(
        settings.get("horse_alerts_running") or 
        settings.get("global_alerts_running") or 
        settings.get("greyhound_alerts_running") or 
        settings.get("trots_alerts_running")
        for settings in guild_settings.values()
    )
    
    if alerts_needed:
        # Check if task is dead but should be running
        if live_stream_running and (live_stream_task is None or live_stream_task.done()):
            print("🔄 Live stream task died unexpectedly, restarting...")
            live_stream_task = asyncio.create_task(live_stream_monitor(), name="global_live_stream_monitor")
            print("✅ Live stream monitor restarted")
            return True
        elif not live_stream_running:
            # Start it if needed
            await start_live_stream_if_needed()
            return True
    return False


async def request_gateway_reconnect():
    """Request a clean gateway reconnection"""
    global last_gateway_refresh
    try:
        print("🔄 Requesting gateway reconnection for connection health...")
        # Close and reconnect - discord.py handles this gracefully
        await bot.close()
        # The main loop will restart the connection
        return True
    except Exception as e:
        print(f"❌ Gateway reconnect failed: {e}")
        return False


# Keep-alive task to monitor bot health
@tasks.loop(minutes=5)
async def bot_health_check():
    """Periodically check if the bot is still connected and healthy"""
    global consecutive_health_failures, last_gateway_refresh
    try:
        if bot.is_closed():
            print("❌ Bot connection is closed! This shouldn't happen.")
            consecutive_health_failures += 1
            return
        
        if not bot.is_ready():
            print("⚠️ Bot is connected but not ready yet...")
            consecutive_health_failures += 1
            return
        
        # Reset failure counter on successful check
        consecutive_health_failures = 0
        
        now = datetime.datetime.now(datetime.timezone.utc)
        latency = bot.latency * 1000  # Convert to ms
        
        # Check for high latency (indicates connection issues)
        if latency > 1000:  # Only warn for very high latency (>1 second)
            print(f"⚠️ High Discord latency detected: {latency:.0f}ms")
        
        # Gateway refresh is now much less aggressive - only refresh on serious latency issues
        # Removed automatic time-based gateway refresh as it causes unnecessary disconnects
        # Discord.py handles reconnection automatically and 48-hour connection is fine
        if latency > 2000:  # Only reconnect if latency is over 2 seconds
            print(f"⚠️ Very high latency ({latency:.0f}ms), connection may be degraded")
            # Don't force reconnect - let Discord.py handle it naturally
        
        # Check and restart live stream if needed
        try:
            restarted = await ensure_live_stream_healthy()
            if restarted:
                print("🔄 Live stream was restarted by health check")
        except Exception as e:
            print(f"⚠️ Could not check live stream health: {e}")
        
        # Periodic ML data cleanup (every 6 hours = every 72nd check)
        if bot_health_check.current_loop > 0 and bot_health_check.current_loop % 72 == 0:
            try:
                cleanup_ml_data()
            except Exception as e:
                print(f"⚠️ ML cleanup error: {e}")
        
        # Log heartbeat every 2 hours instead of every hour (reduce log spam)
        if bot_health_check.current_loop % 24 == 0:  # Every 24 checks = 2 hours
            guilds = len(bot.guilds)
            uptime_hours = (now - bot_start_time).total_seconds() / 3600 if bot_start_time else 0
            print(f"💓 Bot health: {latency:.0f}ms latency, {guilds} guilds, uptime: {uptime_hours:.1f}h, stream: {'🟢' if live_stream_running else '🔴'}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        consecutive_health_failures += 1


@bot_health_check.before_loop
async def before_health_check():
    """Wait for bot to be ready before starting health checks"""
    await bot.wait_until_ready()


# === ADMIN DM COMMAND SYSTEM ===
# Only responds to DMs from the admin user ID
@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author.id == bot.user.id:
        return
    
    # Only respond to DMs from admin
    if not isinstance(message.channel, discord.DMChannel):
        return
    
    if message.author.id != ADMIN_USER_ID:
        return
    
    content = message.content.lower().strip()
    
    # === HELP COMMAND ===
    if content in ["help", "/help", "commands", "?"]:
        help_text = """
🤖 **Bot Admin Commands** (DM Only)

**📊 ML/Analysis:**
• `status` - Bot status & ML overview
• `ml` - Detailed ML learning data
• `patterns` - All profitable patterns found
• `bets` - Recent observed bets
• `stats` - Win/loss statistics
• `analyze` - Run ML analysis now

**🔍 Live Data:**
• `watching` - What markets are being watched
• `alerts` - Active alerts status
• `cache` - Cache sizes & memory

**⚙️ Settings:**
• `settings` - View all guild settings
• `learning on/off` - Toggle learning mode

**📈 Specific Analysis:**
• `horses` - Horse racing analysis
• `dogs` - Greyhound analysis  
• `sports` - Sports betting analysis
• `amounts` - Analysis by bet amounts
• `odds` - Analysis by odds ranges
• `times` - Analysis by time of day
• `venues` - Analysis by venue

**🔧 Actions:**
• `reset ml` - Clear ML learning data
• `export` - Export data summary
"""
        await message.channel.send(help_text)
        return
    
    # === STATUS COMMAND ===
    if content in ["status", "stat", "s"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        patterns = ml_learning_data.get("profitable_patterns", {})
        
        # Count by status
        pending = len([b for b in all_bets if b.get("status") == "pending"])
        won = len([b for b in all_bets if b.get("status") == "won"])
        lost = len([b for b in all_bets if b.get("status") == "lost"])
        
        # Count by type
        by_type = {}
        for bet in all_bets:
            bt = bet.get("bet_type", "unknown")
            by_type[bt] = by_type.get(bt, 0) + 1
        
        learning_mode = ML_CONFIG.get('learning_mode', False)
        status_text = f"""
🤖 **Bot Status**

**🧠 ML Learning:**
• Mode: {'🟢 LEARNING (NOT POSTING)' if learning_mode else '🔴 POSTING ALERTS'}
• {'⚠️ Alerts are SILENCED while learning!' if learning_mode else '✅ Alerts are LIVE!'}
• Total Observed: {len(all_bets)} bets
• Pending: {pending} | Won: {won} | Lost: {lost}
• Profitable Patterns: {len(patterns)}

**📊 Bets by Type:**
{chr(10).join([f"• {k}: {v}" for k, v in by_type.items()])}

**🔄 Live Stream:** {'🟢 Running' if live_stream_running else '🔴 Stopped'}
**📡 Guilds:** {len(bot.guilds)}
**⏰ Last Analysis:** {ml_learning_data.get('last_analysis', 'Never')}
"""
        await message.channel.send(status_text)
        return
    
    # === ML DETAILED VIEW ===
    if content in ["ml", "learning", "ml data"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        
        # Amount breakdown
        amount_breakdown = {}
        for bet in all_bets:
            bucket = bet.get("amount_bucket", "unknown")
            amount_breakdown[bucket] = amount_breakdown.get(bucket, 0) + 1
        
        # Time breakdown
        time_breakdown = {}
        for bet in all_bets:
            tb = bet.get("time_bucket", "unknown")
            time_breakdown[tb] = time_breakdown.get(tb, 0) + 1
        
        # Odds breakdown
        odds_breakdown = {}
        for bet in all_bets:
            ob = bet.get("odds_bucket", "unknown")
            odds_breakdown[ob] = odds_breakdown.get(ob, 0) + 1
        
        ml_text = f"""
🧠 **ML Learning Data**

**💰 By Amount Range:**
{chr(10).join([f"• {get_amount_range_display(k)}: {v}" for k, v in sorted(amount_breakdown.items())])}

**⏰ By Time of Day:**
{chr(10).join([f"• {k}: {v}" for k, v in time_breakdown.items()])}

**📊 By Odds Range:**
{chr(10).join([f"• {k.replace('_', ' ')}: {v}" for k, v in odds_breakdown.items()])}

**📈 Total Samples:** {len(all_bets)}
"""
        await message.channel.send(ml_text)
        return
    
    # === PATTERNS COMMAND ===
    if content in ["patterns", "profitable", "discoveries"]:
        patterns = ml_learning_data.get("profitable_patterns", {})
        
        if not patterns:
            await message.channel.send("🧠 No profitable patterns discovered yet. Need more data!")
            return
        
        # Group by category
        by_cat = {}
        for key, data in patterns.items():
            cat = data.get("category", "other")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(data)
        
        patterns_text = "🎯 **Profitable Patterns Found**\n\n"
        for cat, items in by_cat.items():
            patterns_text += f"**{cat.upper()}:**\n"
            for item in items[:5]:  # Top 5 per category
                wr = item.get("win_rate", 0) * 100
                pnl = item.get("total_pnl", 0)
                patterns_text += f"• {item['description']}: {wr:.0f}% win, +{pnl:.1f}u\n"
            patterns_text += "\n"
        
        # Split if too long
        if len(patterns_text) > 1900:
            patterns_text = patterns_text[:1900] + "..."
        
        await message.channel.send(patterns_text)
        return
    
    # === RECENT BETS ===
    if content in ["bets", "recent", "observed"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        recent = all_bets[-20:]  # Last 20
        
        bets_text = "📋 **Recent Observed Bets**\n\n"
        for bet in reversed(recent):
            posted = "✅" if bet.get("was_posted") else "👁️"
            status = {"pending": "⏳", "won": "🏆", "lost": "❌"}.get(bet.get("status"), "❓")
            amt = bet.get("bet_amount", 0)
            odds = bet.get("odds", 0)
            runner = bet.get("runner_name", "Unknown")[:20]
            bet_type = bet.get("bet_type", "?")
            
            bets_text += f"{posted}{status} ${amt:,.0f} @ {odds:.2f} - {runner} ({bet_type})\n"
        
        await message.channel.send(bets_text)
        return
    
    # === STATS ===
    if content in ["stats", "statistics", "winrate"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        if not settled:
            await message.channel.send("📊 No settled bets yet to calculate stats.")
            return
        
        won = [b for b in settled if b["status"] == "won"]
        lost = [b for b in settled if b["status"] == "lost"]
        total_pnl = sum(b.get("pnl", 0) for b in settled)
        
        # By type
        stats_by_type = {}
        for bet in settled:
            bt = bet.get("bet_type", "unknown")
            if bt not in stats_by_type:
                stats_by_type[bt] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                stats_by_type[bt]["won"] += 1
            else:
                stats_by_type[bt]["lost"] += 1
            stats_by_type[bt]["pnl"] += bet.get("pnl", 0)
        
        stats_text = f"""
📊 **Win/Loss Statistics**

**Overall:**
• Settled: {len(settled)} bets
• Won: {len(won)} | Lost: {len(lost)}
• Win Rate: {len(won)/len(settled)*100:.1f}%
• Net P&L: {total_pnl:+.1f} units

**By Type:**
"""
        for bt, data in stats_by_type.items():
            total = data["won"] + data["lost"]
            wr = data["won"] / total * 100 if total > 0 else 0
            stats_text += f"• {bt}: {wr:.0f}% win ({data['won']}/{total}), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(stats_text)
        return
    
    # === RUN ANALYSIS ===
    if content in ["analyze", "analyse", "run analysis"]:
        await message.channel.send("🧠 Running ML analysis...")
        await run_ml_analysis()
        patterns = ml_learning_data.get("profitable_patterns", {})
        await message.channel.send(f"✅ Analysis complete! Found {len(patterns)} profitable patterns.")
        return
    
    # === WATCHING ===
    if content in ["watching", "markets", "live"]:
        await message.channel.send(f"""
🔍 **Live Monitoring**

• Stream Running: {'🟢 Yes' if live_stream_running else '🔴 No'}
• Markets in Cache: {len(market_data_cache)}
• Active Alerts: {len(active_runner_alerts)}
• Big Bet Embeds: {len(active_big_bet_embeds)}
• Posted Events: {len(posted_events)}
""")
        return
    
    # === CACHE INFO ===
    if content in ["cache", "memory"]:
        await message.channel.send(f"""
💾 **Cache Status**

• market_data_cache: {len(market_data_cache)} entries
• active_runner_alerts: {len(active_runner_alerts)} entries
• active_big_bet_embeds: {len(active_big_bet_embeds)} entries
• alert_cache: {len(alert_cache)} entries
• posted_events: {len(posted_events)} entries
• sent_alerts: {len(sent_alerts)} entries
• ML observed bets: {len(ml_learning_data.get('all_observed_bets', []))} entries
""")
        return
    
    # === AMOUNTS ANALYSIS ===
    if content in ["amounts", "amount", "money"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        by_amount = {}
        for bet in settled:
            bucket = bet.get("amount_bucket", "unknown")
            if bucket not in by_amount:
                by_amount[bucket] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_amount[bucket]["won"] += 1
            else:
                by_amount[bucket]["lost"] += 1
            by_amount[bucket]["pnl"] += bet.get("pnl", 0)
        
        text = "💰 **Analysis by Bet Amount**\n\n"
        for bucket, data in sorted(by_amount.items()):
            total = data["won"] + data["lost"]
            if total < 3:
                continue
            wr = data["won"] / total * 100 if total > 0 else 0
            display = get_amount_range_display(bucket)
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} **{display}**: {wr:.0f}% win ({total} bets), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text or "Not enough data yet.")
        return
    
    # === ODDS ANALYSIS ===
    if content in ["odds"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        by_odds = {}
        for bet in settled:
            bucket = bet.get("odds_bucket", "unknown")
            if bucket not in by_odds:
                by_odds[bucket] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_odds[bucket]["won"] += 1
            else:
                by_odds[bucket]["lost"] += 1
            by_odds[bucket]["pnl"] += bet.get("pnl", 0)
        
        text = "📊 **Analysis by Odds Range**\n\n"
        for bucket, data in by_odds.items():
            total = data["won"] + data["lost"]
            if total < 3:
                continue
            wr = data["won"] / total * 100 if total > 0 else 0
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} **{bucket.replace('_', ' ')}**: {wr:.0f}% win ({total} bets), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text or "Not enough data yet.")
        return
    
    # === TIMES ANALYSIS ===
    if content in ["times", "time", "hours"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        by_time = {}
        for bet in settled:
            bucket = bet.get("time_bucket", "unknown")
            if bucket not in by_time:
                by_time[bucket] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_time[bucket]["won"] += 1
            else:
                by_time[bucket]["lost"] += 1
            by_time[bucket]["pnl"] += bet.get("pnl", 0)
        
        text = "⏰ **Analysis by Time of Day**\n\n"
        for bucket, data in by_time.items():
            total = data["won"] + data["lost"]
            if total < 3:
                continue
            wr = data["won"] / total * 100 if total > 0 else 0
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} **{bucket}**: {wr:.0f}% win ({total} bets), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text or "Not enough data yet.")
        return
    
    # === HORSES ANALYSIS ===
    if content in ["horses", "horse"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        horse_bets = [b for b in all_bets if b.get("bet_type") == "horse" and b.get("status") in ["won", "lost"]]
        
        if len(horse_bets) < 5:
            await message.channel.send("🐎 Not enough horse racing data yet.")
            return
        
        won = [b for b in horse_bets if b["status"] == "won"]
        lost = [b for b in horse_bets if b["status"] == "lost"]
        total_pnl = sum(b.get("pnl", 0) for b in horse_bets)
        
        # By venue
        by_venue = {}
        for bet in horse_bets:
            v = bet.get("venue", "unknown")
            if v not in by_venue:
                by_venue[v] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_venue[v]["won"] += 1
            else:
                by_venue[v]["lost"] += 1
            by_venue[v]["pnl"] += bet.get("pnl", 0)
        
        # Sort by P&L
        sorted_venues = sorted(by_venue.items(), key=lambda x: x[1]["pnl"], reverse=True)
        
        text = f"""
🐎 **Horse Racing Analysis**

**Overall:** {len(won)}/{len(horse_bets)} won ({len(won)/len(horse_bets)*100:.0f}%), {total_pnl:+.1f}u

**Top Venues:**
"""
        for venue, data in sorted_venues[:10]:
            total = data["won"] + data["lost"]
            if total < 2:
                continue
            wr = data["won"] / total * 100
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} {venue}: {wr:.0f}% ({total}), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text)
        return
    
    # === DOGS ANALYSIS ===
    if content in ["dogs", "greyhound", "greyhounds"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        dog_bets = [b for b in all_bets if b.get("bet_type") == "greyhound" and b.get("status") in ["won", "lost"]]
        
        if len(dog_bets) < 5:
            await message.channel.send("🐕 Not enough greyhound data yet.")
            return
        
        won = [b for b in dog_bets if b["status"] == "won"]
        total_pnl = sum(b.get("pnl", 0) for b in dog_bets)
        
        text = f"""
🐕 **Greyhound Analysis**

**Overall:** {len(won)}/{len(dog_bets)} won ({len(won)/len(dog_bets)*100:.0f}%), {total_pnl:+.1f}u
"""
        await message.channel.send(text)
        return
    
    # === SPORTS ANALYSIS ===
    if content in ["sports", "sport"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        sports_bets = [b for b in all_bets if b.get("bet_type") == "sports" and b.get("status") in ["won", "lost"]]
        
        if len(sports_bets) < 5:
            await message.channel.send("⚽ Not enough sports data yet.")
            return
        
        # By sport
        by_sport = {}
        for bet in sports_bets:
            s = bet.get("sport_key", "unknown")
            if s not in by_sport:
                by_sport[s] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_sport[s]["won"] += 1
            else:
                by_sport[s]["lost"] += 1
            by_sport[s]["pnl"] += bet.get("pnl", 0)
        
        text = "⚽ **Sports Analysis**\n\n"
        for sport, data in sorted(by_sport.items(), key=lambda x: x[1]["pnl"], reverse=True):
            total = data["won"] + data["lost"]
            if total < 2:
                continue
            wr = data["won"] / total * 100
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} **{sport}**: {wr:.0f}% win ({total}), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text)
        return
    
    # === VENUES ANALYSIS ===
    if content in ["venues", "venue", "tracks"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        by_venue = {}
        for bet in settled:
            v = bet.get("venue", "unknown")
            if v == "unknown":
                continue
            if v not in by_venue:
                by_venue[v] = {"won": 0, "lost": 0, "pnl": 0}
            if bet["status"] == "won":
                by_venue[v]["won"] += 1
            else:
                by_venue[v]["lost"] += 1
            by_venue[v]["pnl"] += bet.get("pnl", 0)
        
        text = "📍 **Analysis by Venue**\n\n"
        for venue, data in sorted(by_venue.items(), key=lambda x: x[1]["pnl"], reverse=True)[:15]:
            total = data["won"] + data["lost"]
            if total < 2:
                continue
            wr = data["won"] / total * 100
            emoji = "✅" if data["pnl"] > 0 else "❌"
            text += f"{emoji} **{venue}**: {wr:.0f}% ({total}), {data['pnl']:+.1f}u\n"
        
        await message.channel.send(text or "Not enough venue data yet.")
        return
    
    # === LEARNING TOGGLE ===
    if content in ["learning on", "ml on"]:
        ML_CONFIG["learning_mode"] = True
        await message.channel.send("🟢 Learning mode is now **ON**. Observing all bets.")
        return
    
    if content in ["learning off", "ml off"]:
        ML_CONFIG["learning_mode"] = False
        await message.channel.send("🔴 Learning mode is now **OFF**. Only using existing patterns.")
        return
    
    # === RESET ML ===
    if content == "reset ml":
        await message.channel.send("⚠️ Are you sure? Reply `confirm reset` to clear all ML data.")
        return
    
    if content == "confirm reset":
        ml_learning_data["all_observed_bets"] = []
        ml_learning_data["profitable_patterns"] = {}
        ml_learning_data["discoveries"] = []
        ml_learning_data["last_analysis"] = None
        save_ml_data()
        await message.channel.send("✅ ML data has been reset.")
        return
    
    # === EXPORT ===
    if content in ["export", "data"]:
        all_bets = ml_learning_data.get("all_observed_bets", [])
        patterns = ml_learning_data.get("profitable_patterns", {})
        
        summary = f"""
📤 **Data Export Summary**

**ML Learning Data:**
• Total Observed: {len(all_bets)}
• Posted: {len([b for b in all_bets if b.get('was_posted')])}
• Won: {len([b for b in all_bets if b.get('status') == 'won'])}
• Lost: {len([b for b in all_bets if b.get('status') == 'lost'])}
• Pending: {len([b for b in all_bets if b.get('status') == 'pending'])}
• Profitable Patterns: {len(patterns)}

**Files Location:**
• ML Data: ml_learning_data.json
• Bet History: bet_history.json
• Settings: guild_settings.json
"""
        await message.channel.send(summary)
        return
    
    # === SETTINGS ===
    if content in ["settings", "config"]:
        text = "⚙️ **Guild Settings**\n\n"
        for guild_id, settings in list(guild_settings.items())[:5]:
            guild = bot.get_guild(guild_id)
            name = guild.name if guild else f"ID: {guild_id}"
            text += f"**{name}:**\n"
            text += f"• Horse Alerts: {'🟢' if settings.get('horse_alerts_running') else '🔴'}\n"
            text += f"• Sports Alerts: {'🟢' if settings.get('global_alerts_running') else '🔴'}\n"
            text += f"• Greyhound Alerts: {'🟢' if settings.get('greyhound_alerts_running') else '🔴'}\n"
            text += f"• AU Min: ${settings.get('au_horse_min_matched', 1000):,}\n"
            text += f"• UK Min: ${settings.get('uk_horse_min_matched', 1000):,}\n\n"
        
        await message.channel.send(text)
        return
    
    # Unknown command
    await message.channel.send("❓ Unknown command. Type `help` for available commands.")


class MainPanelView(discord.ui.View):
    """Main panel with organized sections"""
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        # Row 0: Racing Settings
        self.add_item(OpenHorseSettingsButton(guild))
        self.add_item(OpenGreyhoundSettingsButton(guild))
        self.add_item(OpenTrotsSettingsButton(guild))
        self.add_item(OpenSportsSettingsButton(guild))
        # Row 1: Configuration
        self.add_item(OpenReportSettingsButton(guild))
        self.add_item(SetMentionRoleButton(guild))
        self.add_item(QuickSetupButton(guild))
        self.add_item(ShowCurrentSettingsButton(guild))
        # Row 2: Tools & Sections
        self.add_item(TestFormatButton(guild))
        self.add_item(ResetUnitsButton(guild))
        self.add_item(OpenTrackingSectionButton(guild))
        self.add_item(OpenMLSectionButton(guild))


class OpenTrackingSectionButton(discord.ui.Button):
    """Opens the Bet Tracking section"""
    def __init__(self, guild):
        super().__init__(
            label="📊 Tracking",
            style=discord.ButtonStyle.primary,
            row=2
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Get tracking stats
            guild_bets = [b for b in tracked_bets if b.get("guild_id") == self.guild.id]
            pending = len([b for b in guild_bets if b.get("status") == "pending"])
            won = len([b for b in guild_bets if b.get("status") == "won"])
            lost = len([b for b in guild_bets if b.get("status") == "lost"])
            total_pnl = sum(b.get("pnl", 0) for b in guild_bets if b.get("status") in ["won", "lost"])
            
            embed = discord.Embed(
                title="📊 Bet Tracking",
                description=(
                    "Manage tracked bets and add missed bets.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                color=discord.Color.blue(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            embed.add_field(
                name="📈 Current Stats",
                value=(
                    f"**Total Tracked:** {len(guild_bets):,}\n"
                    f"**Pending:** {pending}\n"
                    f"**Won:** {won} | **Lost:** {lost}\n"
                    f"**Net P&L:** {total_pnl:+.1f} units"
                ),
                inline=False
            )
            
            embed.add_field(
                name="ℹ️ Add Missed Bets",
                value=(
                    "If bets were posted while the bot was down,\n"
                    "click **➕ Add Missed Bet** to manually add them."
                ),
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                view=TrackingSectionView(self.guild),
                ephemeral=True
            )
        except Exception as e:
            print(f"❌ Error opening tracking section: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error opening tracking section.", ephemeral=True)


class TrackingSectionView(discord.ui.View):
    """View containing bet tracking buttons"""
    def __init__(self, guild):
        super().__init__(timeout=300)
        self.guild = guild
        self.add_item(ShowTrackedBetsButton(guild))
        self.add_item(AddMissedBetButton(guild))
        self.add_item(DeleteTrackedBetButton(guild))
        self.add_item(AnalyzeResultsButton(guild))


class ShowTrackedBetsButton(discord.ui.Button):
    """Show currently tracked bets"""
    def __init__(self, guild):
        super().__init__(
            label="📋 Show Tracked Bets",
            style=discord.ButtonStyle.secondary,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            guild_bets = [b for b in tracked_bets if b.get("guild_id") == self.guild.id]
            
            if not guild_bets:
                await interaction.followup.send("📋 No tracked bets found.", ephemeral=True)
                return
            
            # Sort by timestamp, most recent first
            guild_bets.sort(key=lambda x: x.get("timestamp", datetime.datetime.min), reverse=True)
            
            # Show last 15 bets
            embed = discord.Embed(
                title="📋 Recently Tracked Bets",
                color=discord.Color.blue(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            bet_text = ""
            for bet in guild_bets[:15]:
                status = bet.get("status", "pending")
                if status == "won":
                    emoji = "✅"
                elif status == "lost":
                    emoji = "❌"
                else:
                    emoji = "⏳"
                
                runner = bet.get("runner_name", "Unknown")[:20]
                odds = bet.get("odds", 0)
                pnl = bet.get("pnl", 0)
                bet_type = bet.get("bet_type", "unknown")
                
                bet_text += f"{emoji} **{runner}** @ ${odds:.2f} ({bet_type})"
                if status != "pending":
                    bet_text += f" → {pnl:+.1f}u"
                bet_text += "\n"
            
            embed.description = bet_text or "No bets to show"
            embed.set_footer(text=f"Showing {min(15, len(guild_bets))} of {len(guild_bets)} bets")
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error showing tracked bets: {e}")
            await interaction.followup.send("❌ Error showing tracked bets.", ephemeral=True)


class DeleteTrackedBetButton(discord.ui.Button):
    """Delete a tracked bet"""
    def __init__(self, guild):
        super().__init__(
            label="🗑️ Delete Bet",
            style=discord.ButtonStyle.danger,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            guild_bets = [b for b in tracked_bets if b.get("guild_id") == self.guild.id]
            
            if not guild_bets:
                await interaction.followup.send("📋 No tracked bets to delete.", ephemeral=True)
                return
            
            # Sort by timestamp, most recent first
            guild_bets.sort(key=lambda x: x.get("timestamp", datetime.datetime.min), reverse=True)
            
            # Show last 20 bets in a select menu
            embed = discord.Embed(
                title="🗑️ Delete Tracked Bet",
                description="Select a bet to delete from tracking:",
                color=discord.Color.red()
            )
            
            await interaction.followup.send(
                embed=embed,
                view=SelectBetToDeleteView(self.guild, guild_bets[:20]),
                ephemeral=True
            )
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await interaction.followup.send("❌ Error loading bets.", ephemeral=True)


class SelectBetToDeleteView(discord.ui.View):
    """View to select a bet to delete"""
    def __init__(self, guild, bets: list):
        super().__init__(timeout=300)
        self.guild = guild
        self.bets = bets
        
        # Create select menu
        options = []
        for i, bet in enumerate(bets[:20]):
            runner = bet.get("runner_name", "Unknown")[:35]
            odds = bet.get("odds", 0)
            status = bet.get("status", "pending")
            
            status_emoji = "✅" if status == "won" else "❌" if status == "lost" else "⏳"
            
            timestamp = bet.get("timestamp")
            time_str = timestamp.strftime("%m/%d %H:%M") if isinstance(timestamp, datetime.datetime) else ""
            
            options.append(discord.SelectOption(
                label=f"{status_emoji} {runner}",
                description=f"${odds:.2f} | {status} | {time_str}",
                value=str(i)
            ))
        
        if options:
            select = discord.ui.Select(
                placeholder="Select bet to delete...",
                options=options
            )
            select.callback = self.select_callback
            self.add_item(select)
        
        # Add delete all button
        self.add_item(DeleteAllBetsButton(guild))
    
    async def select_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            selected_idx = int(interaction.data["values"][0])
            selected_bet = self.bets[selected_idx]
            
            # Show confirmation
            embed = discord.Embed(
                title="⚠️ Confirm Delete",
                description="Are you sure you want to delete this bet?",
                color=discord.Color.orange()
            )
            
            status = selected_bet.get("status", "pending")
            status_emoji = "✅" if status == "won" else "❌" if status == "lost" else "⏳"
            
            embed.add_field(
                name="Bet Details",
                value=(
                    f"**Selection:** {selected_bet.get('runner_name', 'Unknown')}\n"
                    f"**Odds:** ${selected_bet.get('odds', 0):.2f}\n"
                    f"**Status:** {status_emoji} {status.upper()}\n"
                    f"**P&L:** {selected_bet.get('pnl', 0):+.1f} units"
                ),
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                view=ConfirmDeleteBetView(self.guild, selected_bet),
                ephemeral=True
            )
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await interaction.followup.send("❌ Error selecting bet.", ephemeral=True)


class ConfirmDeleteBetView(discord.ui.View):
    """Confirm deletion of a bet"""
    def __init__(self, guild, bet: dict):
        super().__init__(timeout=60)
        self.guild = guild
        self.bet = bet
    
    @discord.ui.button(label="✅ Yes, Delete", style=discord.ButtonStyle.danger)
    async def confirm_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Find and remove the bet from tracked_bets
            bet_to_remove = None
            for i, b in enumerate(tracked_bets):
                # Match by multiple fields to find the exact bet
                if (b.get("guild_id") == self.guild.id and
                    b.get("runner_name") == self.bet.get("runner_name") and
                    b.get("odds") == self.bet.get("odds") and
                    b.get("timestamp") == self.bet.get("timestamp")):
                    bet_to_remove = i
                    break
            
            if bet_to_remove is not None:
                removed_bet = tracked_bets.pop(bet_to_remove)
                save_bet_history()
                
                embed = discord.Embed(
                    title="🗑️ Bet Deleted",
                    description=(
                        f"**Selection:** {removed_bet.get('runner_name', 'Unknown')}\n"
                        f"**Odds:** ${removed_bet.get('odds', 0):.2f}\n"
                        f"Bet has been removed from tracking."
                    ),
                    color=discord.Color.green()
                )
                await interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await interaction.followup.send("❌ Could not find bet to delete.", ephemeral=True)
                
        except Exception as e:
            print(f"❌ Error deleting bet: {e}")
            await interaction.followup.send("❌ Error deleting bet.", ephemeral=True)
    
    @discord.ui.button(label="❌ Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Deletion cancelled.", ephemeral=True)


class DeleteAllBetsButton(discord.ui.Button):
    """Delete all tracked bets for this guild"""
    def __init__(self, guild):
        super().__init__(
            label="🗑️ Delete ALL Bets",
            style=discord.ButtonStyle.danger,
            row=1
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        guild_bets = [b for b in tracked_bets if b.get("guild_id") == self.guild.id]
        
        if not guild_bets:
            await interaction.followup.send("📋 No tracked bets to delete.", ephemeral=True)
            return
        
        embed = discord.Embed(
            title="⚠️ DELETE ALL BETS?",
            description=(
                f"This will permanently delete **{len(guild_bets)}** tracked bets.\n\n"
                "**This action cannot be undone!**"
            ),
            color=discord.Color.red()
        )
        
        await interaction.followup.send(
            embed=embed,
            view=ConfirmDeleteAllBetsView(self.guild, len(guild_bets)),
            ephemeral=True
        )


class ConfirmDeleteAllBetsView(discord.ui.View):
    """Confirm deletion of all bets"""
    def __init__(self, guild, count: int):
        super().__init__(timeout=60)
        self.guild = guild
        self.count = count
    
    @discord.ui.button(label="⚠️ YES, DELETE ALL", style=discord.ButtonStyle.danger)
    async def confirm_delete_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        try:
            global tracked_bets
            
            # Keep bets from other guilds, remove this guild's bets
            original_count = len(tracked_bets)
            tracked_bets = [b for b in tracked_bets if b.get("guild_id") != self.guild.id]
            deleted_count = original_count - len(tracked_bets)
            
            save_bet_history()
            
            embed = discord.Embed(
                title="🗑️ All Bets Deleted",
                description=f"Successfully deleted **{deleted_count}** tracked bets.",
                color=discord.Color.green()
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error deleting all bets: {e}")
            await interaction.followup.send("❌ Error deleting bets.", ephemeral=True)
    
    @discord.ui.button(label="❌ Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_delete_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Deletion cancelled.", ephemeral=True)


class AddMissedBetButton(discord.ui.Button):
    """Add a bet that was posted but not tracked"""
    def __init__(self, guild):
        super().__init__(
            label="➕ Add Missed Bet",
            style=discord.ButtonStyle.success,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = discord.Embed(
                title="➕ Add Missed Bet",
                description=(
                    "Choose how to add the missed bet:\n\n"
                    "**Option 1:** Select from recent channel messages\n"
                    "**Option 2:** Enter bet details manually"
                ),
                color=discord.Color.green(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            await interaction.followup.send(
                embed=embed,
                view=AddMissedBetOptionsView(self.guild),
                ephemeral=True
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            await interaction.followup.send("❌ Error opening add bet menu.", ephemeral=True)


class AddMissedBetOptionsView(discord.ui.View):
    """Options for adding a missed bet"""
    def __init__(self, guild):
        super().__init__(timeout=300)
        self.guild = guild
        self.add_item(ScanChannelForBetsButton(guild))
        self.add_item(ManualAddBetButton(guild))


class ScanChannelForBetsButton(discord.ui.Button):
    """Scan channel messages for bet posts"""
    def __init__(self, guild):
        super().__init__(
            label="🔍 Find from Channel",
            style=discord.ButtonStyle.primary,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            settings = guild_settings.get(self.guild.id, {})
            
            # Get all configured alert channels
            channels_to_scan = []
            
            # Horse channels
            if settings.get("au_horse_alert_channel"):
                channels_to_scan.append(settings["au_horse_alert_channel"])
            if settings.get("uk_horse_alert_channel"):
                channels_to_scan.append(settings["uk_horse_alert_channel"])
            
            # Greyhound channel
            if settings.get("greyhound_alert_channel"):
                channels_to_scan.append(settings["greyhound_alert_channel"])
            
            # Trots channel
            if settings.get("trots_alert_channel"):
                channels_to_scan.append(settings["trots_alert_channel"])
            
            # Sports channels
            sports_channels = settings.get("sports_channels", {})
            for ch_id in sports_channels.values():
                if ch_id:
                    channels_to_scan.append(ch_id)
            
            # Remove duplicates
            channels_to_scan = list(set(channels_to_scan))
            
            if not channels_to_scan:
                await interaction.followup.send(
                    "❌ No alert channels configured. Set up channels first in the main panel.",
                    ephemeral=True
                )
                return
            
            # Scan channels for recent bot messages with bet embeds
            found_bets = []
            
            for channel_id in channels_to_scan[:5]:  # Limit to 5 channels
                try:
                    channel = self.guild.get_channel(channel_id)
                    if not channel:
                        continue
                    
                    # Get last 50 messages from the channel
                    async for message in channel.history(limit=50):
                        # Only look at bot's own messages with embeds
                        if message.author.id != interaction.client.user.id:
                            continue
                        if not message.embeds:
                            continue
                        
                        embed = message.embeds[0]
                        
                        # Try to extract bet info from embed
                        bet_info = extract_bet_from_embed(embed, message)
                        if bet_info:
                            # Check if already tracked
                            already_tracked = any(
                                b.get("message_id") == message.id 
                                for b in tracked_bets
                            )
                            if not already_tracked:
                                bet_info["message_id"] = message.id
                                bet_info["channel_id"] = channel_id
                                found_bets.append(bet_info)
                except Exception as e:
                    print(f"Error scanning channel {channel_id}: {e}")
                    continue
            
            if not found_bets:
                await interaction.followup.send(
                    "📭 No untracked bets found in your alert channels.\n"
                    "All recent bets may already be tracked, or use **Manual Add** to enter details.",
                    ephemeral=True
                )
                return
            
            # Show found bets (limit to 10)
            found_bets = found_bets[:10]
            
            embed = discord.Embed(
                title="🔍 Found Untracked Bets",
                description=f"Found **{len(found_bets)}** bet(s) not currently tracked.\nSelect one to add:",
                color=discord.Color.green()
            )
            
            await interaction.followup.send(
                embed=embed,
                view=SelectFoundBetView(self.guild, found_bets),
                ephemeral=True
            )
            
        except Exception as e:
            print(f"❌ Error scanning channels: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error scanning channels for bets.", ephemeral=True)


def extract_bet_from_embed(embed: discord.Embed, message: discord.Message) -> dict:
    """Extract bet information from a Discord embed"""
    try:
        if not embed.title and not embed.description:
            return None
        
        bet_info = {
            "timestamp": message.created_at,
            "embed_title": embed.title or "",
            "embed_description": embed.description or "",
        }
        
        # Try to find runner/selection name
        runner_name = None
        odds = None
        bet_type = "unknown"
        volume = 0
        event_name = ""
        
        # Check embed title for bet type
        title = (embed.title or "").lower()
        if "horse" in title or "🏇" in title or "🐎" in title:
            bet_type = "horse"
        elif "greyhound" in title or "🐕" in title:
            bet_type = "greyhound"
        elif "trots" in title or "harness" in title:
            bet_type = "trots"
        else:
            bet_type = "sports"
        
        # Parse fields for info
        for field in embed.fields:
            field_name = field.name.lower() if field.name else ""
            field_value = field.value or ""
            
            # Runner/Selection
            if any(x in field_name for x in ["runner", "selection", "horse", "dog", "team", "pick"]):
                runner_name = field_value.strip()
            
            # Odds
            if "odds" in field_name or "price" in field_name:
                # Extract number from value like "$2.50" or "2.50"
                import re
                odds_match = re.search(r'\$?([\d.]+)', field_value)
                if odds_match:
                    odds = float(odds_match.group(1))
            
            # Volume/Matched
            if any(x in field_name for x in ["matched", "volume", "stake", "amount"]):
                import re
                vol_match = re.search(r'\$?([\d,]+)', field_value.replace(',', ''))
                if vol_match:
                    volume = float(vol_match.group(1).replace(',', ''))
            
            # Event
            if any(x in field_name for x in ["event", "race", "match", "game"]):
                event_name = field_value.strip()
        
        # If no runner found in fields, try to get from title or description
        if not runner_name:
            # Try title
            if embed.title and not any(x in embed.title.lower() for x in ["alert", "big bet", "matched"]):
                runner_name = embed.title
            # Try first line of description
            elif embed.description:
                first_line = embed.description.split('\n')[0]
                if first_line and len(first_line) < 50:
                    runner_name = first_line
        
        if not runner_name:
            return None
        
        bet_info.update({
            "runner_name": runner_name[:50],  # Limit length
            "odds": odds or 2.0,
            "bet_type": bet_type,
            "volume": volume,
            "event_name": event_name[:100],
            "status": "pending",
        })
        
        return bet_info
        
    except Exception as e:
        print(f"Error extracting bet from embed: {e}")
        return None


class SelectFoundBetView(discord.ui.View):
    """View to select from found bets"""
    def __init__(self, guild, found_bets: list):
        super().__init__(timeout=300)
        self.guild = guild
        self.found_bets = found_bets
        
        # Create select menu with found bets
        options = []
        for i, bet in enumerate(found_bets[:10]):
            runner = bet.get("runner_name", "Unknown")[:40]
            odds = bet.get("odds", 0)
            bet_type = bet.get("bet_type", "unknown")
            timestamp = bet.get("timestamp")
            time_str = timestamp.strftime("%m/%d %H:%M") if timestamp else ""
            
            options.append(discord.SelectOption(
                label=f"{runner}",
                description=f"${odds:.2f} | {bet_type} | {time_str}",
                value=str(i)
            ))
        
        if options:
            select = discord.ui.Select(
                placeholder="Select a bet to add...",
                options=options
            )
            select.callback = self.select_callback
            self.add_item(select)
    
    async def select_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            selected_idx = int(interaction.data["values"][0])
            selected_bet = self.found_bets[selected_idx]
            
            # Show the bet and ask for win/loss status
            embed = discord.Embed(
                title="📝 Confirm Bet Details",
                color=discord.Color.blue()
            )
            
            embed.add_field(name="🏷️ Selection", value=selected_bet.get("runner_name", "Unknown"), inline=True)
            embed.add_field(name="💵 Odds", value=f"${selected_bet.get('odds', 0):.2f}", inline=True)
            embed.add_field(name="📊 Type", value=selected_bet.get("bet_type", "unknown").title(), inline=True)
            
            if selected_bet.get("event_name"):
                embed.add_field(name="🏟️ Event", value=selected_bet.get("event_name"), inline=False)
            
            embed.add_field(
                name="❓ What was the result?",
                value="Select the outcome below:",
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                view=ConfirmBetResultView(self.guild, selected_bet),
                ephemeral=True
            )
            
        except Exception as e:
            print(f"❌ Error selecting bet: {e}")
            await interaction.followup.send("❌ Error processing selection.", ephemeral=True)


class ConfirmBetResultView(discord.ui.View):
    """Confirm the result of a bet"""
    def __init__(self, guild, bet_info: dict):
        super().__init__(timeout=300)
        self.guild = guild
        self.bet_info = bet_info
        
        self.add_item(MarkBetWonButton(guild, bet_info))
        self.add_item(MarkBetLostButton(guild, bet_info))
        self.add_item(MarkBetPendingButton(guild, bet_info))


class MarkBetWonButton(discord.ui.Button):
    def __init__(self, guild, bet_info: dict):
        super().__init__(label="✅ WON", style=discord.ButtonStyle.success, row=0)
        self.guild = guild
        self.bet_info = bet_info

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await add_manual_bet(interaction, self.guild, self.bet_info, "won")


class MarkBetLostButton(discord.ui.Button):
    def __init__(self, guild, bet_info: dict):
        super().__init__(label="❌ LOST", style=discord.ButtonStyle.danger, row=0)
        self.guild = guild
        self.bet_info = bet_info

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await add_manual_bet(interaction, self.guild, self.bet_info, "lost")


class MarkBetPendingButton(discord.ui.Button):
    def __init__(self, guild, bet_info: dict):
        super().__init__(label="⏳ Still Pending", style=discord.ButtonStyle.secondary, row=0)
        self.guild = guild
        self.bet_info = bet_info

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await add_manual_bet(interaction, self.guild, self.bet_info, "pending")


async def add_manual_bet(interaction: discord.Interaction, guild, bet_info: dict, status: str):
    """Add a manually tracked bet"""
    try:
        odds = bet_info.get("odds", 2.0)
        
        # Calculate P&L based on 1 unit stake
        if status == "won":
            pnl = odds - 1.0  # Profit = odds - 1
        elif status == "lost":
            pnl = -1.0
        else:
            pnl = 0.0
        
        # Create the tracked bet entry
        new_bet = {
            "guild_id": guild.id,
            "runner_name": bet_info.get("runner_name", "Unknown"),
            "event_name": bet_info.get("event_name", ""),
            "odds": odds,
            "bet_type": bet_info.get("bet_type", "unknown"),
            "volume": bet_info.get("volume", 0),
            "status": status,
            "pnl": pnl,
            "timestamp": bet_info.get("timestamp", datetime.datetime.now(datetime.timezone.utc)),
            "settled_at": datetime.datetime.now(datetime.timezone.utc) if status != "pending" else None,
            "message_id": bet_info.get("message_id"),
            "channel_id": bet_info.get("channel_id"),
            "manually_added": True,
            "recommended_units": 1.0,
        }
        
        # Add to tracked_bets
        tracked_bets.append(new_bet)
        save_bet_history()
        
        # Also add to ML data if learning mode is on
        if ML_CONFIG.get("learning_mode"):
            ml_bet = {
                "timestamp": new_bet["timestamp"],
                "bet_type": new_bet["bet_type"],
                "odds": odds,
                "runner_name": new_bet["runner_name"],
                "event_name": new_bet["event_name"],
                "was_posted": True,
                "status": status,
                "pnl": pnl,
                "settled_at": new_bet["settled_at"],
                "manually_added": True,
            }
            ml_learning_data["all_observed_bets"].append(ml_bet)
            save_ml_data()
        
        # Build confirmation embed
        if status == "won":
            embed = discord.Embed(
                title="✅ Bet Added - WON",
                color=discord.Color.green()
            )
            embed.description = f"**{new_bet['runner_name']}** @ ${odds:.2f}\n**P&L:** +{pnl:.1f} units"
        elif status == "lost":
            embed = discord.Embed(
                title="❌ Bet Added - LOST",
                color=discord.Color.red()
            )
            embed.description = f"**{new_bet['runner_name']}** @ ${odds:.2f}\n**P&L:** {pnl:.1f} units"
        else:
            embed = discord.Embed(
                title="⏳ Bet Added - PENDING",
                color=discord.Color.orange()
            )
            embed.description = f"**{new_bet['runner_name']}** @ ${odds:.2f}\n*Will be tracked for result*"
        
        embed.set_footer(text="Bet has been added to tracking")
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        
        # Send DM confirmation
        try:
            dm_embed = discord.Embed(
                title="📊 Bet Manually Added",
                description=(
                    f"**Selection:** {new_bet['runner_name']}\n"
                    f"**Odds:** ${odds:.2f}\n"
                    f"**Status:** {status.upper()}\n"
                    f"**P&L:** {pnl:+.1f} units\n"
                    f"**Server:** {guild.name}"
                ),
                color=discord.Color.blue(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            await interaction.user.send(embed=dm_embed)
        except:
            pass
        
    except Exception as e:
        print(f"❌ Error adding manual bet: {e}")
        import traceback
        traceback.print_exc()
        await interaction.followup.send("❌ Error adding bet.", ephemeral=True)


class ManualAddBetButton(discord.ui.Button):
    """Manually enter bet details"""
    def __init__(self, guild):
        super().__init__(
            label="✏️ Manual Entry",
            style=discord.ButtonStyle.secondary,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        # Show a modal for manual entry
        modal = ManualBetEntryModal(self.guild)
        await interaction.response.send_modal(modal)


class ManualBetEntryModal(discord.ui.Modal, title="Add Missed Bet"):
    """Modal for manually entering bet details"""
    
    runner_name = discord.ui.TextInput(
        label="Selection/Runner Name",
        placeholder="e.g., Horse Name, Team Name",
        required=True,
        max_length=100
    )
    
    odds = discord.ui.TextInput(
        label="Odds (decimal)",
        placeholder="e.g., 2.50",
        required=True,
        max_length=10
    )
    
    bet_type = discord.ui.TextInput(
        label="Bet Type",
        placeholder="horse, greyhound, trots, or sports",
        required=True,
        max_length=20
    )
    
    result = discord.ui.TextInput(
        label="Result",
        placeholder="won, lost, or pending",
        required=True,
        max_length=10
    )
    
    def __init__(self, guild):
        super().__init__()
        self.guild = guild
    
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Parse inputs
            runner = self.runner_name.value.strip()
            
            try:
                odds_val = float(self.odds.value.replace("$", "").strip())
            except:
                await interaction.followup.send("❌ Invalid odds format. Use decimal like 2.50", ephemeral=True)
                return
            
            bet_type_val = self.bet_type.value.lower().strip()
            if bet_type_val not in ["horse", "greyhound", "trots", "sports"]:
                bet_type_val = "sports"
            
            result_val = self.result.value.lower().strip()
            if result_val not in ["won", "lost", "pending"]:
                await interaction.followup.send("❌ Result must be 'won', 'lost', or 'pending'", ephemeral=True)
                return
            
            # Create bet info
            bet_info = {
                "runner_name": runner,
                "odds": odds_val,
                "bet_type": bet_type_val,
                "event_name": "",
                "volume": 0,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
            }
            
            await add_manual_bet(interaction, self.guild, bet_info, result_val)
            
        except Exception as e:
            print(f"❌ Error in manual bet modal: {e}")
            await interaction.followup.send("❌ Error adding bet.", ephemeral=True)


class OpenMLSectionButton(discord.ui.Button):
    """Opens the Machine Learning settings section"""
    def __init__(self, guild):
        is_learning = ML_CONFIG.get("learning_mode", True)
        super().__init__(
            label=f"🧠 Machine Learning {'🟢' if is_learning else '🔴'}",
            style=discord.ButtonStyle.success if is_learning else discord.ButtonStyle.secondary,
            row=2
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Get ML stats for the embed
            all_bets = ml_learning_data.get("all_observed_bets", [])
            settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
            wins = len([b for b in settled if b["status"] == "won"])
            losses = len([b for b in settled if b["status"] == "lost"])
            win_rate = (wins / len(settled) * 100) if settled else 0
            patterns = len(ml_learning_data.get("profitable_patterns", {}))
            
            embed = discord.Embed(
                title="🧠 Machine Learning Settings",
                description=(
                    "Configure the ML algorithm that learns from betting patterns.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ),
                color=discord.Color.purple(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            # Current Status
            learning_status = "🟢 ON - Actively Learning" if ML_CONFIG.get("learning_mode") else "🔴 OFF - Using Learned Patterns"
            embed.add_field(
                name="📊 Current Status",
                value=(
                    f"**Learning Mode:** {learning_status}\n"
                    f"**Total Observed:** {len(all_bets):,} bets\n"
                    f"**Settled:** {len(settled):,} ({wins}W - {losses}L)\n"
                    f"**Win Rate:** {win_rate:.1f}%\n"
                    f"**Patterns Found:** {patterns}"
                ),
                inline=False
            )
            
            embed.add_field(
                name="ℹ️ What is Learning Mode?",
                value=(
                    "**ON:** Bot observes ALL bets to learn profitable patterns\n"
                    "**OFF:** Bot stops learning, only applies discovered strategies"
                ),
                inline=False
            )
            
            embed.set_footer(text="Use the buttons below to configure ML settings")
            
            await interaction.followup.send(
                embed=embed,
                view=MLSectionView(self.guild),
                ephemeral=True
            )
        except Exception as e:
            print(f"❌ Error opening ML section: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error opening ML section.", ephemeral=True)


class MLSectionView(discord.ui.View):
    """View containing all ML-related buttons"""
    def __init__(self, guild):
        super().__init__(timeout=300)
        self.guild = guild
        # Row 0: Main controls
        self.add_item(ToggleLearningModeButton(guild))
        self.add_item(RunMLAnalysisButton(guild))
        # Row 1: View data
        self.add_item(MLDiscoveriesButton(guild))
        self.add_item(MLStrategiesButton(guild))
        # Row 2: Filters
        self.add_item(MLOptimalSettingsButton(guild))


class ToggleLearningModeButton(discord.ui.Button):
    """Toggle ML learning mode on/off - controls whether the bot analyzes bets"""
    def __init__(self, guild):
        is_on = ML_CONFIG.get("learning_mode", True)
        super().__init__(
            label=f"{'🟢' if is_on else '🔴'} Learning Mode: {'ON' if is_on else 'OFF'}",
            style=discord.ButtonStyle.success if is_on else discord.ButtonStyle.danger,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Toggle learning mode
            current = ML_CONFIG.get("learning_mode", True)
            ML_CONFIG["learning_mode"] = not current
            new_state = ML_CONFIG["learning_mode"]
            
            # Save the config change to disk so it persists across restarts
            save_ml_data()
            
            # Build response embed
            embed = discord.Embed(
                title="🧠 Learning Mode Updated",
                color=discord.Color.green() if new_state else discord.Color.red(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            if new_state:
                embed.description = (
                    "✅ **Learning Mode is now ON**\n\n"
                    "The bot will:\n"
                    "• Observe and record ALL bets\n"
                    "• Learn from patterns and outcomes\n"
                    "• Build data for ML analysis\n\n"
                    "⚠️ **No alerts will be posted while learning!**\n"
                    "*The bot is silently collecting data.*"
                )
            else:
                embed.description = (
                    "🔴 **Learning Mode is now OFF**\n\n"
                    "The bot will:\n"
                    "• **POST alerts to Discord** ✅\n"
                    "• Apply learned profitable patterns\n"
                    "• Use ML-optimized filters\n\n"
                    "*Alerts are now LIVE! Bets will be posted.*"
                )
            
            # Get stats for embed
            all_bets = ml_learning_data.get("all_observed_bets", [])
            settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
            patterns = len(ml_learning_data.get("profitable_patterns", {}))
            
            embed.add_field(
                name="📊 Current ML Stats",
                value=(
                    f"**Total Observed:** {len(all_bets):,} bets\n"
                    f"**Settled:** {len(settled):,} bets\n"
                    f"**Patterns Found:** {patterns}"
                ),
                inline=False
            )
            
            # Send response in channel
            await interaction.followup.send(embed=embed, ephemeral=True)
            
            # Send DM confirmation to the user
            try:
                dm_embed = discord.Embed(
                    title="🧠 Learning Mode Changed",
                    description=(
                        f"**Status:** {'🟢 ON' if new_state else '🔴 OFF'}\n"
                        f"**Changed by:** You\n"
                        f"**Server:** {self.guild.name if self.guild else 'Unknown'}\n\n"
                        f"{'The bot is now actively learning from bets.' if new_state else 'The bot is now using learned patterns only.'}"
                    ),
                    color=discord.Color.green() if new_state else discord.Color.red(),
                    timestamp=datetime.datetime.now(datetime.timezone.utc)
                )
                dm_embed.set_footer(text="Betfair Bot - ML System")
                await interaction.user.send(embed=dm_embed)
            except discord.Forbidden:
                # User has DMs disabled
                pass
            except Exception as dm_error:
                print(f"⚠️ Could not send DM confirmation: {dm_error}")
            
        except Exception as e:
            print(f"❌ Error toggling learning mode: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error toggling learning mode.", ephemeral=True)


class QuickSetupButton(discord.ui.Button):
    """Quick setup - set all minimums at once"""
    def __init__(self, guild):
        super().__init__(label="⚡ Quick Setup", style=discord.ButtonStyle.success, row=1)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message(
            "**⚡ Quick Setup - Set All Minimums**\n"
            "Choose what you want to configure:",
            view=QuickSetupView(self.guild),
            ephemeral=True
        )


class QuickSetupView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=300)
        self.guild = guild
        self.add_item(SetAllMinimumsButton(guild))
        self.add_item(SetHorseMinButton(guild))
        self.add_item(SetDogTrotsMinButton(guild))


class SetAllMinimumsButton(discord.ui.Button):
    """Set the same minimum for ALL alert types"""
    def __init__(self, guild):
        super().__init__(label="💰 Set ALL Minimums", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        amount_options = [
            discord.SelectOption(label=f"${amt:,}", value=str(amt)) 
            for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
        ]
        amount_select = discord.ui.Select(placeholder="Set minimum for EVERYTHING", options=amount_options)

        async def amount_callback(amount_interaction: discord.Interaction):
            try:
                min_amt = int(amount_interaction.data['values'][0])
                settings = guild_settings.setdefault(self.guild.id, {})
                
                # Set for horses
                settings["au_horse_min_matched"] = min_amt
                settings["uk_horse_min_matched"] = min_amt
                
                # Set for greyhounds
                settings["greyhound_min_matched"] = min_amt
                
                # Set for trots
                settings["trots_min_matched"] = min_amt
                
                # Set for all sports
                sport_min = settings.setdefault("sports_min_matched", {})
                for category in SPORT_CATEGORIES:
                    sport_min[category["key"]] = min_amt
                
                settings["min_matched"] = min_amt  # Default fallback
                
                save_guild_settings()
                
                await amount_interaction.response.send_message(
                    f"✅ **All minimums set to ${min_amt:,}**\n\n"
                    f"🐎 AU/UK Horses: ${min_amt:,}\n"
                    f"🐕 Greyhounds: ${min_amt:,}\n"
                    f"🚗 Trots: ${min_amt:,}\n"
                    f"⚽ All Sports: ${min_amt:,}",
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error setting all minimums: {e}")
                await amount_interaction.response.send_message("❌ Error updating minimums.", ephemeral=True)

        amount_select.callback = amount_callback
        amount_view = discord.ui.View()
        amount_view.add_item(amount_select)
        await interaction.response.send_message(
            "**Set the same minimum $ matched for ALL alert types:**\n"
            "This will apply to: Horses (AU+UK), Greyhounds, Trots, and all Sports",
            view=amount_view, 
            ephemeral=True
        )


class SetHorseMinButton(discord.ui.Button):
    """Set minimum for both AU and UK horses"""
    def __init__(self, guild):
        super().__init__(label="🐎 Set Horse Min (AU+UK)", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        amount_options = [
            discord.SelectOption(label=f"${amt:,}", value=str(amt)) 
            for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
        ]
        amount_select = discord.ui.Select(placeholder="Set horse minimum", options=amount_options)

        async def amount_callback(amount_interaction: discord.Interaction):
            try:
                min_amt = int(amount_interaction.data['values'][0])
                settings = guild_settings.setdefault(self.guild.id, {})
                settings["au_horse_min_matched"] = min_amt
                settings["uk_horse_min_matched"] = min_amt
                save_guild_settings()
                
                await amount_interaction.response.send_message(
                    f"✅ **Horse minimums set to ${min_amt:,}**\n"
                    f"🇦🇺 AU Horses: ${min_amt:,}\n"
                    f"🇬🇧 UK Horses: ${min_amt:,}",
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error setting horse minimums: {e}")
                await amount_interaction.response.send_message("❌ Error updating.", ephemeral=True)

        amount_select.callback = amount_callback
        amount_view = discord.ui.View()
        amount_view.add_item(amount_select)
        await interaction.response.send_message(
            "**Set minimum for ALL horse racing (AU + UK):**",
            view=amount_view, 
            ephemeral=True
        )


class SetDogTrotsMinButton(discord.ui.Button):
    """Set minimum for greyhounds and trots together"""
    def __init__(self, guild):
        super().__init__(label="🐕🚗 Set Dogs + Trots Min", style=discord.ButtonStyle.secondary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        amount_options = [
            discord.SelectOption(label=f"${amt:,}", value=str(amt)) 
            for amt in [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
        ]
        amount_select = discord.ui.Select(placeholder="Set dogs + trots minimum", options=amount_options)

        async def amount_callback(amount_interaction: discord.Interaction):
            try:
                min_amt = int(amount_interaction.data['values'][0])
                settings = guild_settings.setdefault(self.guild.id, {})
                settings["greyhound_min_matched"] = min_amt
                settings["trots_min_matched"] = min_amt
                save_guild_settings()
                
                await amount_interaction.response.send_message(
                    f"✅ **Dogs + Trots minimums set to ${min_amt:,}**\n"
                    f"🐕 Greyhounds: ${min_amt:,}\n"
                    f"🚗 Trots: ${min_amt:,}",
                    ephemeral=True
                )
            except Exception as e:
                print(f"❌ Error setting dog/trots minimums: {e}")
                await amount_interaction.response.send_message("❌ Error updating.", ephemeral=True)

        amount_select.callback = amount_callback
        amount_view = discord.ui.View()
        amount_view.add_item(amount_select)
        await interaction.response.send_message(
            "**Set minimum for Greyhounds AND Trots:**",
            view=amount_view, 
            ephemeral=True
        )


# ============================================
# ML OPTIMAL SETTINGS - Configure ML Filters
# ============================================

class MLOptimalSettingsButton(discord.ui.Button):
    """Configure ML-discovered optimal betting filters"""
    def __init__(self, guild):
        super().__init__(label="⚙️ ML Filters", style=discord.ButtonStyle.primary, row=2)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        settings = guild_settings.get(self.guild.id, {})
        
        # Build status embed
        embed = discord.Embed(
            title="🧠 ML Optimal Filter Settings",
            description=(
                "These filters are based on patterns discovered by the ML algorithm.\n"
                "Based on 520+ settled bets with proven profitability.\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            color=discord.Color.gold(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        
        # Current settings
        max_odds = settings.get("max_odds")
        avoid_longshots = settings.get("avoid_longshots", False)
        weekday_only = settings.get("weekday_only", False)
        prefer_favorites = settings.get("prefer_favorites", False)
        sports_only = settings.get("sports_only_mode", False)
        priority_sports = settings.get("priority_sports", [])
        
        # Filter 1: Max Odds
        embed.add_field(
            name=f"💵 Max Odds: {'$' + str(max_odds) if max_odds else '❌ OFF'}",
            value=(
                "**89.3% win rate** on favorites (< $3.00)\n"
                "Filters out high-odds risky bets"
            ),
            inline=True
        )
        
        # Filter 2: Avoid Longshots
        embed.add_field(
            name=f"🎯 Avoid Longshots: {'✅ ON' if avoid_longshots else '❌ OFF'}",
            value=(
                "**88.1% win rate** avoiding odds > $10\n"
                "Skips high-risk longshot bets"
            ),
            inline=True
        )
        
        # Filter 3: Weekday Only
        embed.add_field(
            name=f"📅 Weekday Only: {'✅ ON' if weekday_only else '❌ OFF'}",
            value=(
                "**93.2% win rate** on weekdays\n"
                "Skips Saturday & Sunday bets"
            ),
            inline=True
        )
        
        # Filter 4: Prefer Favorites
        embed.add_field(
            name=f"⭐ Prefer Favorites: {'✅ ON' if prefer_favorites else '❌ OFF'}",
            value=(
                "**89.3% win rate** on odds < $3.00\n"
                "Only shows favorite selections"
            ),
            inline=True
        )
        
        # Filter 5: Sports Only
        embed.add_field(
            name=f"⚽ Sports Only Mode: {'✅ ON' if sports_only else '❌ OFF'}",
            value=(
                "**94.5% win rate** on sports\n"
                "Disables racing alerts entirely"
            ),
            inline=True
        )
        
        # Filter 6: Priority Sports
        embed.add_field(
            name=f"🏏 Priority Sports: {', '.join(priority_sports[:2]) if priority_sports else '❌ OFF'}",
            value=(
                "**98.3% win rate** on Cricket\n"
                "Focus on specific profitable sports"
            ),
            inline=True
        )
        
        embed.add_field(
            name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            value=(
                "🏆 **RECOMMENDED:** Apply All Optimal = 90%+ expected win rate\n"
                "Use the buttons below to configure individual filters or apply all at once."
            ),
            inline=False
        )
        
        embed.set_footer(text="Based on ML analysis of 520+ settled bets")
        
        view = MLOptimalSettingsView(self.guild)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)


class MLOptimalSettingsView(discord.ui.View):
    """View with buttons to configure ML optimal settings"""
    def __init__(self, guild):
        super().__init__(timeout=300)
        self.guild = guild
        
        # Add individual filter buttons
        self.add_item(ToggleMaxOddsButton(guild))
        self.add_item(ToggleAvoidLongshotsButton(guild))
        self.add_item(ToggleWeekdayOnlyButton(guild))
        self.add_item(TogglePreferFavoritesButton(guild))
        self.add_item(ToggleSportsOnlyButton(guild))
        self.add_item(SetPrioritySportsButton(guild))
        self.add_item(ApplyAllOptimalButton(guild))
        self.add_item(ClearAllFiltersButton(guild))


class ToggleMaxOddsButton(discord.ui.Button):
    """Set max odds filter"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        max_odds = settings.get("max_odds")
        label = f"💵 Max Odds: ${max_odds}" if max_odds else "💵 Set Max Odds"
        super().__init__(label=label, style=discord.ButtonStyle.primary, row=0)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        options = [
            discord.SelectOption(label="$2.00 (Strong favorites)", value="2.0"),
            discord.SelectOption(label="$2.50 (Solid favorites)", value="2.5"),
            discord.SelectOption(label="$3.00 (Favorites - 89% WR)", value="3.0", default=True),
            discord.SelectOption(label="$4.00 (Short odds)", value="4.0"),
            discord.SelectOption(label="$5.00 (Medium odds)", value="5.0"),
            discord.SelectOption(label="$10.00 (Avoid longshots)", value="10.0"),
            discord.SelectOption(label="❌ Disable (No limit)", value="0"),
        ]
        select = discord.ui.Select(placeholder="Select maximum odds...", options=options)
        
        async def select_callback(select_interaction: discord.Interaction):
            try:
                value = float(select_interaction.data['values'][0])
                settings = guild_settings.setdefault(self.guild.id, {})
                
                if value == 0:
                    settings.pop("max_odds", None)
                    msg = "✅ **Max Odds filter DISABLED**\nAll odds will be allowed."
                else:
                    settings["max_odds"] = value
                    msg = f"✅ **Max Odds set to ${value:.2f}**\nOnly bets with odds ≤ ${value:.2f} will be shown."
                
                save_guild_settings()
                await select_interaction.response.send_message(msg, ephemeral=True)
            except Exception as e:
                print(f"❌ Error setting max odds: {e}")
                await select_interaction.response.send_message("❌ Error updating.", ephemeral=True)
        
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message("**Set Maximum Odds Filter:**", view=view, ephemeral=True)


class ToggleAvoidLongshotsButton(discord.ui.Button):
    """Toggle avoid longshots filter"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        is_on = settings.get("avoid_longshots", False)
        super().__init__(
            label=f"🎯 Avoid Longshots: {'ON' if is_on else 'OFF'}",
            style=discord.ButtonStyle.success if is_on else discord.ButtonStyle.secondary,
            row=0
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        current = settings.get("avoid_longshots", False)
        settings["avoid_longshots"] = not current
        save_guild_settings()
        
        new_state = settings["avoid_longshots"]
        await interaction.response.send_message(
            f"✅ **Avoid Longshots {'ENABLED' if new_state else 'DISABLED'}**\n"
            f"{'Bets with odds > $10.00 will be skipped.' if new_state else 'All odds allowed (longshots included).'}",
            ephemeral=True
        )


class ToggleWeekdayOnlyButton(discord.ui.Button):
    """Toggle weekday only filter"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        is_on = settings.get("weekday_only", False)
        super().__init__(
            label=f"📅 Weekday Only: {'ON' if is_on else 'OFF'}",
            style=discord.ButtonStyle.success if is_on else discord.ButtonStyle.secondary,
            row=1
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        current = settings.get("weekday_only", False)
        settings["weekday_only"] = not current
        save_guild_settings()
        
        new_state = settings["weekday_only"]
        await interaction.response.send_message(
            f"✅ **Weekday Only {'ENABLED' if new_state else 'DISABLED'}**\n"
            f"{'Saturday & Sunday bets will be skipped.' if new_state else 'Bets allowed on all days.'}",
            ephemeral=True
        )


class TogglePreferFavoritesButton(discord.ui.Button):
    """Toggle prefer favorites filter"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        is_on = settings.get("prefer_favorites", False)
        super().__init__(
            label=f"⭐ Favorites Only: {'ON' if is_on else 'OFF'}",
            style=discord.ButtonStyle.success if is_on else discord.ButtonStyle.secondary,
            row=1
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        current = settings.get("prefer_favorites", False)
        settings["prefer_favorites"] = not current
        save_guild_settings()
        
        new_state = settings["prefer_favorites"]
        await interaction.response.send_message(
            f"✅ **Prefer Favorites {'ENABLED' if new_state else 'DISABLED'}**\n"
            f"{'Only bets with odds < $3.00 will be shown.' if new_state else 'All odds allowed.'}",
            ephemeral=True
        )


class ToggleSportsOnlyButton(discord.ui.Button):
    """Toggle sports only mode"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        is_on = settings.get("sports_only_mode", False)
        super().__init__(
            label=f"⚽ Sports Only: {'ON' if is_on else 'OFF'}",
            style=discord.ButtonStyle.success if is_on else discord.ButtonStyle.secondary,
            row=1
        )
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        current = settings.get("sports_only_mode", False)
        settings["sports_only_mode"] = not current
        save_guild_settings()
        
        new_state = settings["sports_only_mode"]
        await interaction.response.send_message(
            f"✅ **Sports Only Mode {'ENABLED' if new_state else 'DISABLED'}**\n"
            f"{'Horse racing, greyhounds, and trots will be skipped.' if new_state else 'All bet types allowed.'}",
            ephemeral=True
        )


class SetPrioritySportsButton(discord.ui.Button):
    """Set priority sports list"""
    def __init__(self, guild):
        settings = guild_settings.get(guild.id, {})
        priority = settings.get("priority_sports", [])
        label = f"🏏 Priority: {priority[0] if priority else 'None'}" if len(priority) <= 1 else f"🏏 Priority: {len(priority)} sports"
        super().__init__(label=label, style=discord.ButtonStyle.primary, row=2)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        options = [
            discord.SelectOption(label="🏏 Cricket (98.3% WR)", value="cricket", description="Best performing sport"),
            discord.SelectOption(label="⚽ Soccer", value="soccer"),
            discord.SelectOption(label="🏀 Basketball", value="basketball"),
            discord.SelectOption(label="🎾 Tennis", value="tennis"),
            discord.SelectOption(label="🏈 American Football", value="americanfootball"),
            discord.SelectOption(label="🏉 Rugby", value="rugby"),
            discord.SelectOption(label="⚾ Baseball", value="baseball"),
            discord.SelectOption(label="🏒 Ice Hockey", value="icehockey"),
            discord.SelectOption(label="❌ Clear All", value="clear"),
        ]
        select = discord.ui.Select(placeholder="Select priority sports...", options=options, max_values=5)
        
        async def select_callback(select_interaction: discord.Interaction):
            try:
                values = select_interaction.data['values']
                settings = guild_settings.setdefault(self.guild.id, {})
                
                if "clear" in values:
                    settings["priority_sports"] = []
                    settings["sports_only_priority"] = False
                    msg = "✅ **Priority Sports CLEARED**\nAll sports will be shown equally."
                else:
                    settings["priority_sports"] = values
                    settings["sports_only_priority"] = True
                    msg = f"✅ **Priority Sports Set:**\n{', '.join(values)}\n\nOther sports will be filtered out when in priority mode."
                
                save_guild_settings()
                await select_interaction.response.send_message(msg, ephemeral=True)
            except Exception as e:
                print(f"❌ Error setting priority sports: {e}")
                await select_interaction.response.send_message("❌ Error updating.", ephemeral=True)
        
        select.callback = select_callback
        view = discord.ui.View()
        view.add_item(select)
        await interaction.response.send_message(
            "**Select Priority Sports (up to 5):**\nThese sports will be prioritized. Cricket has 98.3% win rate!",
            view=view, 
            ephemeral=True
        )


class ApplyAllOptimalButton(discord.ui.Button):
    """Apply all ML optimal settings at once"""
    def __init__(self, guild):
        super().__init__(label="🏆 APPLY ALL OPTIMAL", style=discord.ButtonStyle.success, row=3)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        
        # Apply all optimal settings based on ML discoveries
        settings["max_odds"] = 3.0  # 89.3% win rate on favorites
        settings["avoid_longshots"] = True  # 88.1% win rate
        settings["weekday_only"] = True  # 93.2% win rate
        settings["prefer_favorites"] = True  # 89.3% win rate
        settings["sports_only_mode"] = False  # Keep racing but with filters
        settings["priority_sports"] = ["cricket"]  # 98.3% win rate
        settings["sports_only_priority"] = False  # Don't force only cricket
        
        save_guild_settings()
        
        embed = discord.Embed(
            title="🏆 ALL OPTIMAL FILTERS APPLIED!",
            description=(
                "Your server is now configured for maximum profitability.\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="✅ Filters Applied",
            value=(
                "💵 **Max Odds:** $3.00 (89% WR)\n"
                "🎯 **Avoid Longshots:** ON (88% WR)\n"
                "📅 **Weekday Only:** ON (93% WR)\n"
                "⭐ **Prefer Favorites:** ON (89% WR)\n"
                "🏏 **Priority:** Cricket (98% WR)"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📊 Expected Performance",
            value=(
                "Based on 520+ settled bets:\n"
                "• **Expected Win Rate:** ~90%+\n"
                "• **Lower volume** but higher quality\n"
                "• **Consistent profits** over time"
            ),
            inline=False
        )
        
        embed.set_footer(text="Settings saved • Takes effect immediately")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
        # Notify admin
        await notify_admin(
            f"🏆 **ML Optimal Filters Applied!**\n\n"
            f"Server: {self.guild.name}\n"
            f"Max Odds: $3.00\n"
            f"Avoid Longshots: ON\n"
            f"Weekday Only: ON\n"
            f"Prefer Favorites: ON\n"
            f"Priority Sport: Cricket"
        )


class ClearAllFiltersButton(discord.ui.Button):
    """Clear all ML filters"""
    def __init__(self, guild):
        super().__init__(label="🗑️ Clear All Filters", style=discord.ButtonStyle.danger, row=3)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        
        # Clear all ML filters
        settings.pop("max_odds", None)
        settings["avoid_longshots"] = False
        settings["weekday_only"] = False
        settings["prefer_favorites"] = False
        settings["sports_only_mode"] = False
        settings["priority_sports"] = []
        settings["sports_only_priority"] = False
        
        save_guild_settings()
        
        await interaction.response.send_message(
            "🗑️ **All ML Filters Cleared!**\n\n"
            "Your server is now back to default settings.\n"
            "All bets meeting minimum amounts will be shown.",
            ephemeral=True
        )


class ShowCurrentSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="📋 View Current Settings", style=discord.ButtonStyle.secondary, row=1)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.get(self.guild.id, {})
        
        # Build status embed
        embed = discord.Embed(
            title="⚙️ Current Bot Settings",
            color=discord.Color.blue(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        
        # Horse Racing Status
        horse_status = "🟢 Active" if settings.get("horse_alerts_running") else "🔴 Inactive"
        au_channel = settings.get("au_horse_alert_channel")
        uk_channel = settings.get("uk_horse_alert_channel")
        au_min = settings.get("au_horse_min_matched", 1000)
        uk_min = settings.get("uk_horse_min_matched", 1000)
        horse_text = f"**Status:** {horse_status}\n"
        horse_text += f"AU Channel: {f'<#{au_channel}>' if au_channel else '❌ Not set'} (Min: ${au_min:,})\n"
        horse_text += f"UK Channel: {f'<#{uk_channel}>' if uk_channel else '❌ Not set'} (Min: ${uk_min:,})"
        embed.add_field(name="🐎 Horse Racing", value=horse_text, inline=False)
        
        # Greyhound Status
        grey_status = "🟢 Active" if settings.get("greyhound_alerts_running") else "🔴 Inactive"
        grey_channel = settings.get("greyhound_alert_channel")
        grey_min = settings.get("greyhound_min_matched", 1000)
        grey_text = f"**Status:** {grey_status}\n"
        grey_text += f"Channel: {f'<#{grey_channel}>' if grey_channel else '❌ Not set'} (Min: ${grey_min:,})"
        embed.add_field(name="🐕 Greyhound Racing", value=grey_text, inline=False)
        
        # Sports Status
        sports_status = "🟢 Active" if settings.get("global_alerts_running") else "🔴 Inactive"
        sports_channels = settings.get("sports_channels", {})
        sports_text = f"**Status:** {sports_status}\n"
        if sports_channels:
            for sport_key, chan_id in list(sports_channels.items())[:5]:
                sport_meta = SPORT_CATEGORY_LOOKUP.get(sport_key, {})
                label = sport_meta.get("label", sport_key.title())
                sports_text += f"{label}: <#{chan_id}>\n"
            if len(sports_channels) > 5:
                sports_text += f"... and {len(sports_channels) - 5} more"
        else:
            fallback = settings.get("alert_channel")
            sports_text += f"Default: {f'<#{fallback}>' if fallback else '❌ Not set'}"
        embed.add_field(name="⚽ Sports", value=sports_text, inline=False)
        
        # Report Channel
        report_channel = settings.get("report_channel") or settings.get("alert_channel")
        report_text = f"Channel: {f'<#{report_channel}>' if report_channel else '❌ Not set'}"
        embed.add_field(name="📊 Daily Report", value=report_text, inline=True)
        
        # Mention Role
        mention_role = settings.get("mention_role")
        role_text = f"<@&{mention_role}>" if mention_role else "❌ Not set"
        embed.add_field(name="👤 Mention Role", value=role_text, inline=True)
        
        # ML Optimal Filters
        ml_filters_text = ""
        max_odds = settings.get("max_odds")
        avoid_longshots = settings.get("avoid_longshots", False)
        weekday_only = settings.get("weekday_only", False)
        prefer_favorites = settings.get("prefer_favorites", False)
        sports_only = settings.get("sports_only_mode", False)
        priority_sports = settings.get("priority_sports", [])
        
        ml_filters_text += f"**Max Odds:** {'$' + str(max_odds) if max_odds else '❌ Off'}\n"
        ml_filters_text += f"**Avoid Longshots:** {'✅ On' if avoid_longshots else '❌ Off'}\n"
        ml_filters_text += f"**Weekday Only:** {'✅ On' if weekday_only else '❌ Off'}\n"
        ml_filters_text += f"**Prefer Favorites:** {'✅ On' if prefer_favorites else '❌ Off'}\n"
        ml_filters_text += f"**Sports Only:** {'✅ On' if sports_only else '❌ Off'}\n"
        if priority_sports:
            ml_filters_text += f"**Priority Sports:** {', '.join(priority_sports[:3])}"
        
        embed.add_field(name="🧠 ML Optimal Filters", value=ml_filters_text, inline=False)
        
        embed.set_footer(text="Settings auto-save and persist across restarts")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)


class TestFormatButton(discord.ui.Button):
    """Preview how big bet alerts look"""
    def __init__(self, guild):
        super().__init__(label="🧪 Test Format", style=discord.ButtonStyle.secondary, row=2)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        # Create sample embeds for each bet type
        samples = [
            ("horse", "🇦🇺", "Winx", "Randwick R5", "Win", "$45,000", "$120,000", "3.50", "5m 30s", 2.5),
            ("sports", "⚽", "Liverpool", "Fulham v Liverpool", "Match Odds", "$145,000", "$145,000", "4.40", "STARTED", 1.0),
            ("greyhound", "🐕", "Blazing Speed", "Sandown R3", "Win", "$12,000", "$35,000", "2.80", "3m 15s", 1.5),
        ]
        
        embeds = []
        for bet_type, emoji, selection, event, market, bet_amt, total, odds, time_str, units in samples:
            # Determine color
            colors = {
                "horse": discord.Color.from_rgb(139, 90, 43),
                "sports": discord.Color.from_rgb(46, 204, 113),
                "greyhound": discord.Color.from_rgb(65, 105, 225),
            }
            
            embed = discord.Embed(color=colors.get(bet_type, discord.Color.gold()))
            
            # Header
            type_labels = {"horse": "Horse Racing", "sports": "Soccer", "greyhound": "Greyhound Racing"}
            embed.title = "🔔 LARGE BET SPOTTED"
            embed.description = f"{emoji} **{type_labels.get(bet_type, 'Sports')}** • {event} • {market}"
            
            # Selection info
            selection_label = "Team" if bet_type == "sports" else "Runner"
            embed.add_field(name=f"{selection_label}", value=f"**{selection}**", inline=True)
            embed.add_field(name="Bet Amount", value=bet_amt, inline=True)
            embed.add_field(name="Total Matched", value=total, inline=True)
            
            embed.add_field(name="Odds", value=odds, inline=True)
            embed.add_field(name="Time", value=f"⏰ {time_str}", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer
            
            # Betting guidelines
            embed.add_field(
                name="📊 Betting Guidelines",
                value=f"**BET {units:.1f} UNITS ON THIS SELECTION**",
                inline=False
            )
            
            embeds.append(embed)
        
        await interaction.response.send_message(
            "**🧪 Sample Alert Formats:**\nHere's how your bet alerts will look:",
            embeds=embeds,
            ephemeral=True
        )


class ResetUnitsButton(discord.ui.Button):
    """Reset unit tracking to start fresh"""
    def __init__(self, guild):
        super().__init__(label="🔄 Reset Units", style=discord.ButtonStyle.danger, row=2)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        # Confirmation view
        confirm_view = discord.ui.View(timeout=30)
        guild_id = self.guild.id
        
        async def confirm_callback(confirm_interaction: discord.Interaction):
            try:
                # Reset bet history for this guild by filtering out this guild's bets
                global tracked_bets
                old_count = len([b for b in tracked_bets if b.get("guild_id") == guild_id])
                tracked_bets = [b for b in tracked_bets if b.get("guild_id") != guild_id]
                save_bet_history()
                
                if old_count > 0:
                    await confirm_interaction.response.send_message(
                        f"✅ **Units Reset Complete!**\n"
                        f"Cleared {old_count} historical bets.\n"
                        f"Your P&L tracking now starts fresh from zero.",
                        ephemeral=True
                    )
                else:
                    await confirm_interaction.response.send_message(
                        "ℹ️ No bet history found to reset.",
                        ephemeral=True
                    )
            except Exception as e:
                print(f"❌ Error resetting units: {e}")
                await confirm_interaction.response.send_message("❌ Error resetting units.", ephemeral=True)
        
        async def cancel_callback(cancel_interaction: discord.Interaction):
            await cancel_interaction.response.send_message("❌ Reset cancelled.", ephemeral=True)
        
        confirm_btn = discord.ui.Button(label="✅ Yes, Reset Everything", style=discord.ButtonStyle.danger)
        cancel_btn = discord.ui.Button(label="❌ Cancel", style=discord.ButtonStyle.secondary)
        confirm_btn.callback = confirm_callback
        cancel_btn.callback = cancel_callback
        confirm_view.add_item(confirm_btn)
        confirm_view.add_item(cancel_btn)
        
        await interaction.response.send_message(
            "⚠️ **Are you sure you want to reset all unit tracking?**\n"
            "This will:\n"
            "• Clear all bet history\n"
            "• Reset daily/weekly/monthly/yearly P&L to zero\n"
            "• Start fresh with no historical data\n\n"
            "**This action cannot be undone!**",
            view=confirm_view,
            ephemeral=True
        )


class AnalyzeResultsButton(discord.ui.Button):
    """Analyze betting results to find most profitable combinations"""
    def __init__(self, guild):
        super().__init__(label="📈 Analyze Results", style=discord.ButtonStyle.primary, row=2)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Use tracked_bets which stores bet history, filter by guild_id
            history = [b for b in tracked_bets if b.get("guild_id") == self.guild.id]
            
            if len(history) < 10:
                await interaction.followup.send(
                    "📊 **Not enough data to analyze.**\n"
                    f"You have {len(history)} bets recorded. Need at least 10 for meaningful analysis.",
                    ephemeral=True
                )
                return
            
            # Analyze by bet type
            type_stats = {}
            for bet in history:
                bet_type = bet.get("bet_type", "unknown")
                if bet_type not in type_stats:
                    type_stats[bet_type] = {"bets": 0, "won": 0, "lost": 0, "pending": 0, "units_won": 0.0, "units_lost": 0.0}
                
                type_stats[bet_type]["bets"] += 1
                status = bet.get("status", "pending")
                units = bet.get("units", 1.0)
                odds = bet.get("odds", 2.0)
                
                if status == "won":
                    type_stats[bet_type]["won"] += 1
                    type_stats[bet_type]["units_won"] += units * (odds - 1)
                elif status == "lost":
                    type_stats[bet_type]["lost"] += 1
                    type_stats[bet_type]["units_lost"] += units
                else:
                    type_stats[bet_type]["pending"] += 1
            
            # Calculate P&L for each type
            results = []
            for bet_type, stats in type_stats.items():
                pnl = stats["units_won"] - stats["units_lost"]
                total_decided = stats["won"] + stats["lost"]
                win_rate = (stats["won"] / total_decided * 100) if total_decided > 0 else 0
                
                results.append({
                    "type": bet_type,
                    "bets": stats["bets"],
                    "won": stats["won"],
                    "lost": stats["lost"],
                    "pending": stats["pending"],
                    "pnl": pnl,
                    "win_rate": win_rate
                })
            
            # Sort by P&L (best first)
            results.sort(key=lambda x: x["pnl"], reverse=True)
            
            # Build analysis embed
            embed = discord.Embed(
                title="📊 Betting Performance Analysis",
                color=discord.Color.gold(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            # Summary
            total_bets = sum(r["bets"] for r in results)
            total_pnl = sum(r["pnl"] for r in results)
            embed.description = f"**Total Bets:** {total_bets} | **Net P&L:** {total_pnl:+.1f} units"
            
            # Best performers
            profitable = [r for r in results if r["pnl"] > 0]
            unprofitable = [r for r in results if r["pnl"] < 0]
            
            if profitable:
                best_text = ""
                for r in profitable[:3]:
                    emoji = {"horse": "🐎", "greyhound": "🐕", "trots": "🚗", "sports": "⚽"}.get(r["type"], "📌")
                    best_text += f"{emoji} **{r['type'].title()}**: +{r['pnl']:.1f}u ({r['win_rate']:.0f}% win rate, {r['bets']} bets)\n"
                embed.add_field(name="✅ KEEP - Most Profitable", value=best_text, inline=False)
            
            if unprofitable:
                worst_text = ""
                for r in unprofitable[:3]:
                    emoji = {"horse": "🐎", "greyhound": "🐕", "trots": "🚗", "sports": "⚽"}.get(r["type"], "📌")
                    worst_text += f"{emoji} **{r['type'].title()}**: {r['pnl']:.1f}u ({r['win_rate']:.0f}% win rate, {r['bets']} bets)\n"
                embed.add_field(name="❌ CONSIDER CUTTING - Losing Money", value=worst_text, inline=False)
            
            # Recommendations
            recommendations = []
            
            # Find best sport if applicable
            sport_results = [r for r in results if r["type"] == "sports"]
            if profitable and unprofitable:
                best = profitable[0]
                worst = unprofitable[-1]
                recommendations.append(f"💡 Focus on **{best['type'].title()}** (+{best['pnl']:.1f}u)")
                recommendations.append(f"🚫 Consider removing **{worst['type'].title()}** ({worst['pnl']:.1f}u)")
            
            if total_pnl > 0:
                recommendations.append(f"📈 Overall strategy is **profitable** at +{total_pnl:.1f} units")
            else:
                recommendations.append(f"📉 Overall strategy is **losing** at {total_pnl:.1f} units - review settings")
            
            if recommendations:
                embed.add_field(name="💡 Recommendations", value="\n".join(recommendations), inline=False)
            
            # Detailed breakdown
            breakdown = "```\n"
            breakdown += f"{'Type':<15} {'Bets':>6} {'Won':>5} {'Lost':>5} {'P&L':>8} {'Win%':>6}\n"
            breakdown += "-" * 50 + "\n"
            for r in results:
                breakdown += f"{r['type']:<15} {r['bets']:>6} {r['won']:>5} {r['lost']:>5} {r['pnl']:>+7.1f}u {r['win_rate']:>5.0f}%\n"
            breakdown += "```"
            embed.add_field(name="📋 Full Breakdown", value=breakdown, inline=False)
            
            embed.set_footer(text="Analysis based on historical bet data")
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error analyzing results.", ephemeral=True)


class MLDiscoveriesButton(discord.ui.Button):
    """View ML algorithm discoveries and profitable patterns"""
    def __init__(self, guild):
        super().__init__(label="📊 View Discoveries", style=discord.ButtonStyle.secondary, row=1)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            patterns = ml_learning_data.get("profitable_patterns", {})
            all_bets = ml_learning_data.get("all_observed_bets", [])
            strategies = get_best_strategies()
            
            embed = discord.Embed(
                title="🧠 ML Algorithm Discoveries",
                color=discord.Color.purple(),
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )
            
            # Learning status
            total_observed = len(all_bets)
            posted = len([b for b in all_bets if b.get("was_posted")])
            filtered = len([b for b in all_bets if not b.get("was_posted")])
            settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
            wins = len([b for b in settled if b["status"] == "won"])
            losses = len([b for b in settled if b["status"] == "lost"])
            overall_wr = (wins / len(settled) * 100) if settled else 0
            
            embed.description = (
                f"**Learning Mode:** {'🟢 ON' if ML_CONFIG.get('learning_mode') else '🔴 OFF'}\n"
                f"**Total Observed:** {total_observed:,} bets\n"
                f"**Settled:** {len(settled):,} ({wins}W - {losses}L = {overall_wr:.1f}% WR)\n"
                f"**Patterns Found:** {len(patterns)}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            # Show top strategies with details
            if strategies:
                top_text = ""
                for i, strat in enumerate(strategies[:5], 1):
                    win_rate = strat.get("win_rate", 0) * 100
                    avg_odds = strat.get("avg_odds", 2.0)
                    sample = strat.get("sample_size", 0)
                    pnl = strat.get("total_pnl", 0)
                    desc = strat.get("description", "")[:25]
                    
                    # Rating
                    if win_rate >= 90:
                        rating = "🔥"
                    elif win_rate >= 80:
                        rating = "💎"
                    elif win_rate >= 70:
                        rating = "⭐"
                    else:
                        rating = "✅"
                    
                    top_text += (
                        f"{rating} **#{i} {desc}**\n"
                        f"   {win_rate:.0f}% WR | ${avg_odds:.2f} avg | {sample} bets | +{pnl:.0f}u\n"
                    )
                
                embed.add_field(
                    name="🏆 Top Strategies (Click 📊 View Strategies to Apply)",
                    value=top_text,
                    inline=False
                )
                
                # Best odds insight
                best_odds = next((s for s in strategies if "odds" in s.get("category", "") or "favorite" in s.get("pattern", "").lower()), None)
                if best_odds:
                    embed.add_field(
                        name="💵 Best Odds Strategy",
                        value=(
                            f"**{best_odds.get('description', 'Unknown')}**\n"
                            f"Avg: ${best_odds.get('avg_odds', 2.0):.2f} | {best_odds.get('win_rate', 0)*100:.0f}% WR\n"
                            f"*{best_odds.get('recommended_action', '')}*"
                        ),
                        inline=True
                    )
                
                # Best sport
                best_sport = next((s for s in strategies if s.get("category") == "sport"), None)
                if best_sport:
                    embed.add_field(
                        name="⚽ Best Sport",
                        value=(
                            f"**{best_sport.get('description', 'Unknown')}**\n"
                            f"{best_sport.get('win_rate', 0)*100:.0f}% WR | +{best_sport.get('total_pnl', 0):.0f}u"
                        ),
                        inline=True
                    )
                
                # Best bet type
                best_type = next((s for s in strategies if s.get("category") == "bet_type"), None)
                if best_type:
                    embed.add_field(
                        name="🎯 Best Bet Type",
                        value=(
                            f"**{best_type.get('description', 'Unknown')}**\n"
                            f"{best_type.get('win_rate', 0)*100:.0f}% WR | +{best_type.get('total_pnl', 0):.0f}u"
                        ),
                        inline=True
                    )
                
                embed.add_field(
                    name="� How to Apply",
                    value=(
                        "Click **📊 View Strategies** button below to:\n"
                        "• See detailed strategy breakdowns\n"
                        "• Apply individual strategies\n"
                        "• Apply the best strategy to all servers"
                    ),
                    inline=False
                )
            else:
                embed.add_field(
                    name="💰 No Strategies Yet",
                    value=(
                        f"Need {ML_CONFIG['min_sample_size']} settled bets per pattern.\n"
                        f"Currently have {len(settled)} total settled bets.\n\n"
                        "*Keep betting - the algorithm is learning!*"
                    ),
                    inline=False
                )
            
            # Last analysis time
            last_analysis = ml_learning_data.get("last_analysis")
            if last_analysis:
                embed.set_footer(text=f"Last analysis: {last_analysis}")
            else:
                embed.set_footer(text="No analysis run yet")
            
            # Create view with strategy buttons if we have strategies
            if strategies:
                view = MLDiscoveriesQuickApplyView(self.guild, strategies)
                await interaction.followup.send(embed=embed, view=view, ephemeral=True)
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error showing ML discoveries: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error showing ML discoveries.", ephemeral=True)


class MLDiscoveriesQuickApplyView(discord.ui.View):
    """Quick apply view for ML discoveries"""
    def __init__(self, guild, strategies: List[dict]):
        super().__init__(timeout=300)
        self.guild = guild
        self.strategies = strategies
        
        # Add top strategy apply button
        if strategies:
            self.add_item(ApplyBestFromDiscoveriesButton(guild, strategies[0]))
        
        # Add view all strategies button
        self.add_item(ViewAllStrategiesFromDiscoveriesButton(guild, strategies))


class ApplyBestFromDiscoveriesButton(discord.ui.Button):
    """Apply best strategy from discoveries view"""
    def __init__(self, guild, strategy: dict):
        win_rate = strategy.get("win_rate", 0) * 100
        desc = strategy.get("description", "Best")[:15]
        super().__init__(
            label=f"🏆 Apply #{1}: {desc} ({win_rate:.0f}%)",
            style=discord.ButtonStyle.success,
            row=0
        )
        self.guild = guild
        self.strategy = strategy

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        strategy = self.strategy
        description = strategy.get("description", "Best Strategy")
        win_rate = strategy.get("win_rate", 0) * 100
        changes = get_strategy_settings_changes(strategy)
        
        if not changes:
            await interaction.followup.send(
                f"⚠️ **Cannot Apply This Strategy**\n\n"
                f"Strategy: {description}\n"
                f"This strategy is informational only.",
                ephemeral=True
            )
            return
        
        confirm_embed = discord.Embed(
            title="🏆 Apply BEST Strategy?",
            description=f"**{description.upper()}**",
            color=discord.Color.green()
        )
        
        confirm_embed.add_field(
            name="📊 Stats",
            value=(
                f"**Win Rate:** {win_rate:.1f}%\n"
                f"**Sample:** {strategy.get('sample_size', 0)} bets\n"
                f"**Avg Odds:** ${strategy.get('avg_odds', 2.0):.2f}\n"
                f"**P&L:** +{strategy.get('total_pnl', 0):.1f}u"
            ),
            inline=True
        )
        
        confirm_embed.add_field(
            name="⚙️ Will Apply",
            value=changes.get("description", "Optimal settings"),
            inline=True
        )
        
        confirm_embed.add_field(
            name="🌐 Servers",
            value=f"All {len(guild_settings)} configured servers",
            inline=False
        )
        
        view = ConfirmApplyStrategyView(self.guild, strategy)
        await interaction.followup.send(embed=confirm_embed, view=view, ephemeral=True)


class ViewAllStrategiesFromDiscoveriesButton(discord.ui.Button):
    """Navigate to full strategies view"""
    def __init__(self, guild, strategies: List[dict]):
        super().__init__(
            label="📊 View All Strategies & Apply",
            style=discord.ButtonStyle.primary,
            row=0
        )
        self.guild = guild
        self.strategies = strategies

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        embed = get_ml_strategies_embed()
        view = MLStrategiesView(self.guild, self.strategies)
        
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)


class RunMLAnalysisButton(discord.ui.Button):
    """Manually trigger ML analysis"""
    def __init__(self, guild):
        super().__init__(label="🔄 Run Analysis", style=discord.ButtonStyle.primary, row=0)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            all_bets = ml_learning_data.get("all_observed_bets", [])
            settled = len([b for b in all_bets if b.get("status") in ["won", "lost"]])
            
            if settled < ML_CONFIG["min_sample_size"]:
                await interaction.followup.send(
                    f"🧠 **Not enough data for analysis**\n\n"
                    f"Need at least {ML_CONFIG['min_sample_size']} settled bets.\n"
                    f"Currently have: {settled} settled bets.\n"
                    f"Total observed: {len(all_bets)} bets.\n\n"
                    f"*Keep betting! The algorithm is learning...*",
                    ephemeral=True
                )
                return
            
            # Run analysis
            await interaction.followup.send("🧠 Running ML analysis... This may take a moment.", ephemeral=True)
            
            await run_ml_analysis()
            
            # Show results
            patterns = ml_learning_data.get("profitable_patterns", {})
            
            if patterns:
                result_text = "**Profitable Patterns Found:**\n"
                for pattern_key, data in list(patterns.items())[:5]:
                    description = data.get("description", pattern_key)
                    win_rate = data.get("win_rate", 0) * 100
                    pnl = data.get("total_pnl", 0)
                    result_text += f"✅ {description}: {win_rate:.0f}% win rate, +{pnl:.1f}u\n"
                
                await interaction.followup.send(
                    f"🧠 **ML Analysis Complete!**\n\n{result_text}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    "🧠 **ML Analysis Complete**\n\n"
                    "No statistically significant profitable patterns found yet.\n"
                    "Need more data - keep betting!",
                    ephemeral=True
                )
                
        except Exception as e:
            print(f"❌ Error running ML analysis: {e}")
            await interaction.followup.send("❌ Error running analysis.", ephemeral=True)


class MLStrategiesButton(discord.ui.Button):
    """View and apply ML strategies"""
    def __init__(self, guild):
        super().__init__(label="� View Strategies", style=discord.ButtonStyle.secondary, row=1)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            strategies = get_best_strategies()
            
            if not strategies:
                await interaction.followup.send(
                    "🧠 **No Strategies Found Yet**\n\n"
                    f"The ML needs at least {ML_CONFIG['min_sample_size']} settled bets per pattern.\n"
                    "Keep the bot running to collect more data!",
                    ephemeral=True
                )
                return
            
            # Create embed
            embed = get_ml_strategies_embed()
            
            # Create view with strategy buttons
            view = MLStrategiesView(self.guild, strategies)
            
            await interaction.followup.send(embed=embed, view=view, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error showing ML strategies: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("❌ Error loading strategies.", ephemeral=True)


class MLStrategiesView(discord.ui.View):
    """View with buttons to apply different strategies"""
    def __init__(self, guild, strategies: List[dict]):
        super().__init__(timeout=300)
        self.guild = guild
        self.strategies = strategies
        
        # Add individual strategy buttons (top 4)
        for i, strat in enumerate(strategies[:4]):
            desc = strat.get("description", strat.get("pattern", ""))[:20]
            win_rate = strat.get("win_rate", 0) * 100
            self.add_item(ApplyStrategyButton(guild, strat, i+1, f"{desc} ({win_rate:.0f}%)"))
        
        # Add "Apply Best" button
        self.add_item(ApplyBestStrategyButton(guild, strategies[0] if strategies else None))
        
        # Add "View Details" button
        self.add_item(ViewStrategyDetailsButton(guild, strategies))


class ApplyStrategyButton(discord.ui.Button):
    """Apply a specific strategy to all servers"""
    def __init__(self, guild, strategy: dict, rank: int, label: str):
        # Color based on rank
        styles = [
            discord.ButtonStyle.success,  # 1st - green
            discord.ButtonStyle.primary,  # 2nd - blue
            discord.ButtonStyle.primary,  # 3rd - blue
            discord.ButtonStyle.secondary,  # 4th - gray
        ]
        super().__init__(
            label=f"#{rank}: {label}"[:40],
            style=styles[rank-1] if rank <= len(styles) else discord.ButtonStyle.secondary,
            row=1 if rank <= 2 else 2
        )
        self.guild = guild
        self.strategy = strategy
        self.rank = rank

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            strategy = self.strategy
            pattern = strategy.get("pattern", "")
            description = strategy.get("description", pattern)
            win_rate = strategy.get("win_rate", 0) * 100
            changes = get_strategy_settings_changes(strategy)
            
            if not changes:
                await interaction.followup.send(
                    f"⚠️ **Cannot Apply This Strategy**\n\n"
                    f"Strategy: {description}\n"
                    f"This strategy is informational only - no settings to change.\n"
                    f"The bot will naturally follow this pattern based on bet flow.",
                    ephemeral=True
                )
                return
            
            # Show confirmation
            confirm_embed = discord.Embed(
                title=f"🧠 Apply Strategy #{self.rank}?",
                description=f"**{description.upper()}**",
                color=discord.Color.gold()
            )
            
            confirm_embed.add_field(
                name="📊 Strategy Stats",
                value=(
                    f"**Win Rate:** {win_rate:.1f}%\n"
                    f"**Sample Size:** {strategy.get('sample_size', 0)} bets\n"
                    f"**P&L:** +{strategy.get('total_pnl', 0):.1f} units\n"
                    f"**Avg Odds:** ${strategy.get('avg_odds', 2.0):.2f}"
                ),
                inline=True
            )
            
            confirm_embed.add_field(
                name="⚙️ Changes To Apply",
                value=changes.get("description", "Various settings"),
                inline=True
            )
            
            confirm_embed.add_field(
                name="🌐 Servers Affected",
                value=f"All {len(guild_settings)} configured servers",
                inline=False
            )
            
            confirm_embed.set_footer(text="Click 'Confirm Apply' to proceed or 'Cancel' to abort")
            
            view = ConfirmApplyStrategyView(self.guild, strategy)
            await interaction.followup.send(embed=confirm_embed, view=view, ephemeral=True)
            
        except Exception as e:
            print(f"❌ Error applying strategy: {e}")
            await interaction.followup.send("❌ Error applying strategy.", ephemeral=True)


class ApplyBestStrategyButton(discord.ui.Button):
    """Apply the #1 best strategy"""
    def __init__(self, guild, strategy: dict):
        super().__init__(
            label="🏆 Apply Best Strategy",
            style=discord.ButtonStyle.success,
            row=3
        )
        self.guild = guild
        self.strategy = strategy

    async def callback(self, interaction: discord.Interaction):
        if not self.strategy:
            await interaction.response.send_message("No strategies available.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        strategy = self.strategy
        description = strategy.get("description", "Best Strategy")
        win_rate = strategy.get("win_rate", 0) * 100
        
        confirm_embed = discord.Embed(
            title="🏆 Apply BEST Strategy?",
            description=f"**{description.upper()}**\n\nThis is the highest-ranked strategy by the ML algorithm.",
            color=discord.Color.green()
        )
        
        confirm_embed.add_field(
            name="📊 Stats",
            value=(
                f"**Win Rate:** {win_rate:.1f}%\n"
                f"**Sample:** {strategy.get('sample_size', 0)} bets\n"
                f"**P&L:** +{strategy.get('total_pnl', 0):.1f}u"
            ),
            inline=True
        )
        
        changes = get_strategy_settings_changes(strategy)
        confirm_embed.add_field(
            name="⚙️ Will Apply",
            value=changes.get("description", "Optimal settings"),
            inline=True
        )
        
        view = ConfirmApplyStrategyView(self.guild, strategy)
        await interaction.followup.send(embed=confirm_embed, view=view, ephemeral=True)


class ViewStrategyDetailsButton(discord.ui.Button):
    """View detailed info about all strategies"""
    def __init__(self, guild, strategies: List[dict]):
        super().__init__(
            label="📋 View Full Details",
            style=discord.ButtonStyle.secondary,
            row=3
        )
        self.guild = guild
        self.strategies = strategies

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Send detailed breakdown
            messages = []
            
            header = (
                "🧠 **FULL STRATEGY BREAKDOWN**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            messages.append(header)
            
            for i, strat in enumerate(self.strategies[:10], 1):
                detail = format_strategy_for_display(strat, i)
                messages.append(detail)
            
            # Send in chunks to avoid message limit
            full_text = ""
            for msg in messages:
                if len(full_text) + len(msg) > 1900:
                    await interaction.followup.send(full_text, ephemeral=True)
                    full_text = msg
                else:
                    full_text += msg
            
            if full_text:
                await interaction.followup.send(full_text, ephemeral=True)
                
        except Exception as e:
            print(f"❌ Error showing strategy details: {e}")
            await interaction.followup.send("❌ Error loading details.", ephemeral=True)


class ConfirmApplyStrategyView(discord.ui.View):
    """Confirmation view for applying a strategy"""
    def __init__(self, guild, strategy: dict):
        super().__init__(timeout=60)
        self.guild = guild
        self.strategy = strategy
    
    @discord.ui.button(label="✅ Confirm Apply", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        try:
            count, messages = await apply_strategy_to_all_servers(self.strategy)
            
            description = self.strategy.get("description", "Strategy")
            win_rate = self.strategy.get("win_rate", 0) * 100
            
            result_embed = discord.Embed(
                title="✅ STRATEGY APPLIED!",
                description=(
                    f"**{description.upper()}** has been applied.\n\n"
                    f"📊 Strategy Win Rate: {win_rate:.1f}%\n"
                    f"🌐 Servers Updated: {count}\n"
                ),
                color=discord.Color.green()
            )
            
            changes = get_strategy_settings_changes(self.strategy)
            result_embed.add_field(
                name="⚙️ Changes Made",
                value=changes.get("description", "Settings updated"),
                inline=False
            )
            
            for msg in messages[:3]:
                result_embed.add_field(name="\u200b", value=msg, inline=False)
            
            result_embed.set_footer(text="Your servers are now using this ML strategy!")
            
            # Disable buttons
            for item in self.children:
                item.disabled = True
            
            await interaction.followup.send(embed=result_embed, ephemeral=True)
            
            # Notify admin
            await notify_admin(
                f"🧠 **Strategy Applied!**\n\n"
                f"Strategy: {description}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Servers: {count}\n"
                f"Changes: {changes.get('description', 'Various')}"
            )
            
        except Exception as e:
            print(f"❌ Error confirming strategy: {e}")
            await interaction.followup.send("❌ Error applying strategy.", ephemeral=True)
    
    @discord.ui.button(label="❌ Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("❌ Strategy application cancelled.", ephemeral=True)
        
        # Disable buttons
        for item in self.children:
            item.disabled = True


class OpenHorseSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🐎 Horse Settings", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.get(self.guild.id, {})
        status = "🟢 Active" if settings.get("horse_alerts_running") else "🔴 Inactive"
        au_ch = settings.get("au_horse_alert_channel")
        uk_ch = settings.get("uk_horse_alert_channel")
        
        status_msg = f"**Current Status:** {status}\n"
        status_msg += f"AU Channel: {f'<#{au_ch}>' if au_ch else 'Not set'}\n"
        status_msg += f"UK Channel: {f'<#{uk_ch}>' if uk_ch else 'Not set'}"
        
        await interaction.response.send_message(
            f"{status_msg}\n\n**Configure horse racing alerts:**",
            view=HorseSettingsView(self.guild),
            ephemeral=True
        )

class OpenSportsSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="⚽ Sports Settings", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.get(self.guild.id, {})
        status = "🟢 Active" if settings.get("global_alerts_running") else "🔴 Inactive"
        sports_channels = settings.get("sports_channels", {})
        
        status_msg = f"**Current Status:** {status}\n"
        if sports_channels:
            status_msg += f"Configured sports: {len(sports_channels)}"
        else:
            status_msg += "No sports configured yet"
        
        await interaction.response.send_message(
            f"{status_msg}\n\n**Configure sports alerts:**",
            view=SportsSettingsView(self.guild),
            ephemeral=True
        )

class OpenGreyhoundSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🐕 Greyhound Settings", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.get(self.guild.id, {})
        status = "🟢 Active" if settings.get("greyhound_alerts_running") else "🔴 Inactive"
        channel = settings.get("greyhound_alert_channel")
        min_bet = settings.get("greyhound_min_matched", 1000)
        
        status_msg = f"**Current Status:** {status}\n"
        status_msg += f"Channel: {f'<#{channel}>' if channel else 'Not set'}\n"
        status_msg += f"Min Bet: ${min_bet:,}"
        
        await interaction.response.send_message(
            f"{status_msg}\n\n**Configure greyhound alerts:**",
            view=GreyhoundSettingsView(self.guild),
            ephemeral=True
        )

class OpenTrotsSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🚗 Trots Settings", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.get(self.guild.id, {})
        status = "🟢 Active" if settings.get("trots_alerts_running") else "🔴 Inactive"
        channel = settings.get("trots_alert_channel")
        min_bet = settings.get("trots_min_matched", 1000)
        
        status_msg = f"**Current Status:** {status}\n"
        status_msg += f"Channel: {f'<#{channel}>' if channel else 'Not set'}\n"
        status_msg += f"Min Bet: ${min_bet:,}"
        
        await interaction.response.send_message(
            f"{status_msg}\n\n**Configure trots alerts:**",
            view=TrotsSettingsView(self.guild),
            ephemeral=True
        )

class OpenReportSettingsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="📊 Report Settings", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message(
            "Configure daily 6 PM Perth reports:",
            view=ReportSettingsView(self.guild),
            ephemeral=True
        )

class HorseSettingsView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        self.add_item(SetAUHorseAlertChannelButton(guild))
        self.add_item(SetAUHorseMinMatchedButton(guild))
        self.add_item(SetUKHorseAlertChannelButton(guild))
        self.add_item(SetUKHorseMinMatchedButton(guild))
        self.add_item(StartStopHorseAlertsButton(guild))

class SportsSettingsView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        self.add_item(SetSportsAlertChannelButton(guild))
        self.add_item(SetAllSportsMinButton(guild))  # Set ALL sports min at once
        self.add_item(SetSportsMinMatchedButton(guild))  # Set individual sport min
        self.add_item(StartStopSportsAlertsButton(guild))

class GreyhoundSettingsView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        self.add_item(SetGreyhoundAlertChannelButton(guild))
        self.add_item(SetGreyhoundMinMatchedButton(guild))
        self.add_item(StartStopGreyhoundAlertsButton(guild))

class TrotsSettingsView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        self.add_item(SetTrotsAlertChannelButton(guild))
        self.add_item(SetTrotsMinMatchedButton(guild))
        self.add_item(StartStopTrotsAlertsButton(guild))

class SetReportChannelButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="📊 Set Report Channel", style=discord.ButtonStyle.primary)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
            return
        channels = [c for c in guild.text_channels if c.permissions_for(guild.me).send_messages]
        if not channels:
            await interaction.followup.send("❌ No accessible text channels found.", ephemeral=True)
            return
        view = ChannelPaginationView(channels, "report", guild.id)
        await interaction.followup.send(
            f"Choose a channel for 6 PM Perth reports:\n**Found {len(channels)} channels**",
            view=view,
            ephemeral=True
        )

class ShowCurrentReportButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="🕒 Show Current Window", style=discord.ButtonStyle.success)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        guild = interaction.guild
        if not guild:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return
        settings = guild_settings.get(guild.id, {})
        channel_id = settings.get("report_channel") or settings.get("alert_channel")
        if not channel_id:
            await interaction.response.send_message("⚠️ Set a report channel first.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            sent = await send_report_for_guild(guild.id, end_awst=awst_now(), manual=True)
            if sent:
                await interaction.followup.send(f"✅ Report sent to <#{channel_id}>", ephemeral=True)
            else:
                await interaction.followup.send("⚠️ Could not send report (channel missing or no data).", ephemeral=True)
        except Exception as e:
            print(f"❌ Error sending manual report: {e}")
            await interaction.followup.send("❌ Error sending report.", ephemeral=True)

class ReportSettingsView(discord.ui.View):
    def __init__(self, guild):
        super().__init__(timeout=None)
        self.guild = guild
        self.add_item(SetReportChannelButton(guild))
        self.add_item(ShowCurrentReportButton(guild))


# Track sent bet alerts for win/loss editing and updates
sent_alerts = []  # List of dicts: {guild_id, channel_id, message_id, market_id, runner, emoji, status}
win_loss_stats = {"sent": 0, "won": 0, "lost": 0}

# Track active alerts per runner to enable updates instead of duplicates
active_runner_alerts = {}  # {(market_id, runner_id, guild_id): {message_id, channel_id, total_amount}}

# Rate limiting for duplicate alerts
alert_cache = {}  # {(market_id, runner_id, price, size): timestamp}
ALERT_COOLDOWN = 300  # 5 minutes cooldown for duplicate alerts

# Market data cache for tracking changes
market_data_cache = {}

# Cache size limits to prevent memory leaks
MAX_CACHE_SIZE = 10000
MAX_POSTED_EVENTS_AGE_HOURS = 24
MAX_MARKET_CACHE_AGE_HOURS = 12

def cleanup_caches():
    """Clean up old entries from caches to prevent memory leaks"""
    global posted_events, alert_cache, market_data_cache, active_runner_alerts, sent_alerts
    
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Clean posted_events - remove entries older than 24 hours
    cutoff_time = now - datetime.timedelta(hours=MAX_POSTED_EVENTS_AGE_HOURS)
    old_count = len(posted_events)
    posted_events = {k: v for k, v in posted_events.items() if v > cutoff_time}
    if old_count > len(posted_events):
        print(f"🧹 Cleaned {old_count - len(posted_events)} old posted_events entries")
    
    # Clean alert_cache - already has cooldown, but enforce max size
    if len(alert_cache) > MAX_CACHE_SIZE:
        # Sort by timestamp and keep newest half
        sorted_items = sorted(alert_cache.items(), key=lambda x: x[1], reverse=True)
        alert_cache.clear()
        for k, v in sorted_items[:MAX_CACHE_SIZE // 2]:
            alert_cache[k] = v
        print(f"🧹 Trimmed alert_cache to {len(alert_cache)} entries")
    
    # Clean market_data_cache - limit size
    if len(market_data_cache) > MAX_CACHE_SIZE:
        # Keep most recent entries (simple approach - clear oldest half)
        keys_to_remove = list(market_data_cache.keys())[:len(market_data_cache) // 2]
        for k in keys_to_remove:
            del market_data_cache[k]
        print(f"🧹 Trimmed market_data_cache to {len(market_data_cache)} entries")
    
    # Clean active_runner_alerts - limit size
    if len(active_runner_alerts) > MAX_CACHE_SIZE:
        keys_to_remove = list(active_runner_alerts.keys())[:len(active_runner_alerts) // 2]
        for k in keys_to_remove:
            del active_runner_alerts[k]
        print(f"🧹 Trimmed active_runner_alerts to {len(active_runner_alerts)} entries")
    
    # Clean sent_alerts - remove settled alerts older than 7 days
    week_ago = now - datetime.timedelta(days=7)
    old_alerts_count = len(sent_alerts)
    sent_alerts = [
        alert for alert in sent_alerts 
        if alert.get("status") == "pending" or 
        (isinstance(alert.get("timestamp"), datetime.datetime) and alert.get("timestamp") > week_ago)
    ]
    if old_alerts_count > len(sent_alerts):
        print(f"🧹 Cleaned {old_alerts_count - len(sent_alerts)} old sent_alerts entries")

@tasks.loop(hours=1)
async def cache_cleanup_task():
    """Periodic cache cleanup to prevent memory leaks"""
    try:
        cleanup_caches()
    except Exception as e:
        print(f"❌ Error in cache cleanup: {e}")

@cache_cleanup_task.before_loop
async def before_cache_cleanup():
    await bot.wait_until_ready()

# === ML ANALYSIS BACKGROUND TASK ===
@tasks.loop(hours=2)
async def ml_analysis_task():
    """Run ML pattern analysis every 2 hours to find profitable patterns"""
    try:
        print("🧠 Starting scheduled ML analysis...")
        
        # First, try to settle bets directly from Betfair API
        settled_from_betfair = await settle_ml_bets_from_betfair()
        
        # Also sync results from tracked_bets to ML data
        await update_ml_bet_results()
        
        # Clean up old data to prevent memory issues
        cleanup_ml_data()
        
        # Only run analysis if we have enough settled bets
        settled_count = len([b for b in ml_learning_data.get("all_observed_bets", []) if b.get("status") in ["won", "lost"]])
        if settled_count >= ML_CONFIG["min_sample_size"]:
            await run_ml_analysis()
        else:
            print(f"🧠 ML: Not enough settled bets for analysis ({settled_count}/{ML_CONFIG['min_sample_size']})")
        
    except Exception as e:
        print(f"❌ Error in ML analysis: {e}")
        traceback.print_exc()
        # Don't crash the task - it will retry in 6 hours

@ml_analysis_task.before_loop
async def before_ml_analysis():
    await bot.wait_until_ready()
    # Wait a bit after startup before first analysis
    await asyncio.sleep(60)  # 1 minute (reduced from 5 min for faster first analysis)
    
    # Import existing bet history on startup
    try:
        imported = import_existing_bet_history()
        if imported > 0:
            print(f"🧠 ML: Imported {imported} bets from existing history on startup")
    except Exception as e:
        print(f"⚠️ Could not import existing bet history: {e}")

async def update_ml_bet_results():
    """Update ML learning data with actual results from tracked_bets (real results only)"""
    try:
        # Get pending bets from ML data
        pending_bets = [b for b in ml_learning_data.get("all_observed_bets", []) if b.get("status") == "pending"]
        
        if not pending_bets:
            print("🧠 ML sync: No pending bets to update")
            return
        
        # Build multiple lookup keys for better matching
        settled_by_runner = {}  # runner_name -> list of settled bets
        settled_by_key = {}  # (runner_name, event_name) -> bet
        
        for b in tracked_bets:
            if b.get("status") in ["won", "lost"]:
                runner = b.get("runner_name", "")
                event = b.get("event_name", "")
                
                # Key by (runner, event) for exact match
                key = (runner, event)
                if key not in settled_by_key:
                    settled_by_key[key] = b
                
                # Also key by runner alone for fuzzy match
                if runner not in settled_by_runner:
                    settled_by_runner[runner] = []
                settled_by_runner[runner].append(b)
        
        if not settled_by_key and not settled_by_runner:
            print("🧠 ML sync: No settled bets in tracked_bets to sync from")
            return
        
        now = datetime.datetime.now(datetime.timezone.utc)
        updated_count = 0
        
        for bet in pending_bets:
            runner_name = bet.get("runner_name", "")
            event_name = bet.get("event_name", "")
            
            tracked_bet = None
            
            # Try exact match first
            key = (runner_name, event_name)
            if key in settled_by_key:
                tracked_bet = settled_by_key[key]
            # Fallback to runner-only match (get most recent)
            elif runner_name in settled_by_runner:
                # Get the most recent settled bet for this runner
                runner_bets = settled_by_runner[runner_name]
                if runner_bets:
                    tracked_bet = runner_bets[-1]  # Most recent
            
            if tracked_bet:
                # Use actual result from tracked_bets
                bet["status"] = tracked_bet.get("status")
                pnl = tracked_bet.get("pnl")
                if pnl is not None:
                    bet["pnl"] = pnl
                else:
                    # Calculate P&L if not provided
                    odds = bet.get("odds", 2.0) or 2.0
                    units = bet.get("recommended_units", 1.0) or 1.0
                    if bet["status"] == "won":
                        bet["pnl"] = (odds - 1) * units
                    else:
                        bet["pnl"] = -units
                bet["settled_at"] = now
                updated_count += 1
        
        if updated_count > 0:
            save_ml_data()
            print(f"🧠 ML sync: Updated {updated_count} bets with results from tracked_bets")
        else:
            print(f"🧠 ML sync: No matches found ({len(pending_bets)} pending, {len(settled_by_key)} settled)")
        
    except Exception as e:
        print(f"⚠️ Error updating ML bet results: {e}")
        traceback.print_exc()

async def settle_ml_bets_from_betfair():
    """Directly check Betfair for settled markets and update ML bets"""
    try:
        session_token = get_session_token()
        if not session_token:
            print("🧠 ML settle: No session token available")
            return 0
        
        # Get pending ML bets
        pending_bets = [b for b in ml_learning_data.get("all_observed_bets", []) if b.get("status") == "pending"]
        if not pending_bets:
            return 0
        
        # Group pending bets by market_id
        bets_by_market = defaultdict(list)
        for bet in pending_bets:
            market_id = bet.get("market_id")
            if market_id:
                bets_by_market[market_id].append(bet)
        
        if not bets_by_market:
            return 0
        
        print(f"🧠 ML settle: Checking {len(bets_by_market)} markets with {len(pending_bets)} pending bets...")
        
        # Check markets in batches (Betfair API allows up to 50 at a time)
        market_ids = list(bets_by_market.keys())
        updated_count = 0
        
        for i in range(0, len(market_ids), 40):
            batch = market_ids[i:i+40]
            
            try:
                # Get market status
                market_book_req = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/v1.0/listMarketBook",
                    "params": {
                        "marketIds": batch,
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}
                    },
                    "id": 1
                })
                
                response = call_betfair_api(market_book_req, session_token)
                if not response or not response.get("result"):
                    continue
                
                now = datetime.datetime.now(datetime.timezone.utc)
                
                for market_book in response.get("result", []):
                    market_id = market_book.get("marketId")
                    market_status = market_book.get("status")
                    
                    # Only process closed (settled) markets
                    if market_status != "CLOSED":
                        continue
                    
                    # Build runner results
                    runner_results = {}
                    for runner in market_book.get("runners", []):
                        sel_id = runner.get("selectionId")
                        runner_status = runner.get("status")
                        runner_results[sel_id] = runner_status
                    
                    # Update matching ML bets
                    for bet in bets_by_market.get(market_id, []):
                        sel_id = bet.get("selection_id")
                        runner_name = bet.get("runner_name", "")
                        
                        # Try to find by selection_id first, then by runner name matching
                        runner_status = None
                        if sel_id and sel_id in runner_results:
                            runner_status = runner_results[sel_id]
                        else:
                            # Look for WINNER status to determine if our runner won
                            for sid, status in runner_results.items():
                                if status == "WINNER":
                                    # Check if this is our runner - for now assume if there's a winner and we don't have selection_id, mark as lost
                                    runner_status = "LOSER" if status == "WINNER" else None
                                    break
                        
                        if runner_status:
                            won = runner_status == "WINNER"
                            bet["status"] = "won" if won else "lost"
                            bet["settled_at"] = now
                            
                            # Calculate P&L
                            odds = bet.get("odds", 2.0) or 2.0
                            units = bet.get("recommended_units", 1.0) or 1.0
                            if won:
                                bet["pnl"] = (odds - 1) * units
                            else:
                                bet["pnl"] = -units
                            
                            updated_count += 1
                
            except Exception as e:
                print(f"⚠️ Error checking market batch: {e}")
                continue
        
        if updated_count > 0:
            save_ml_data()
            print(f"🧠 ML settle: Updated {updated_count} bets directly from Betfair")
        
        return updated_count
        
    except Exception as e:
        print(f"⚠️ Error in settle_ml_bets_from_betfair: {e}")
        traceback.print_exc()
        return 0

@tasks.loop(minutes=5)
async def win_loss_check_task():
    """Check for win/loss results for sent bets"""
    try:
        session_token = get_session_token()
        if not session_token:
            return
            
        # Check alerts that are still pending
        pending_alerts = [alert for alert in sent_alerts if alert.get("status") == "pending"]
        
        for alert in pending_alerts:
            try:
                # Get market book to check if market is settled
                market_book_req = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/v1.0/listMarketBook",
                    "params": {
                        "marketIds": [alert["market_id"]],
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
                        "orderProjection": "ALL"
                    },
                    "id": 1
                })
                
                book_resp = call_betfair_api(market_book_req, session_token)
                if not book_resp or not book_resp.get("result"):
                    continue
                    
                market_book = book_resp["result"][0]
                market_status = market_book.get("status")
                
                # Only check settled markets
                if market_status != "CLOSED":
                    continue
                    
                # Find the runner
                for runner in market_book.get("runners", []):
                    if runner.get("selectionId") == alert.get("selection_id"):
                        runner_status = runner.get("status")
                        
                        # Determine win/loss
                        if runner_status == "WINNER":
                            alert["status"] = "won"
                            win_loss_stats["won"] += 1
                            update_bet_result(alert["market_id"], alert["runner"], alert["guild_id"], won=True)
                            
                            # Update ML learning data with real result
                            await update_ml_bet_with_result(alert["market_id"], alert["runner"], won=True)
                            
                            # Update Discord message with win emoji
                            try:
                                channel = bot.get_channel(alert["channel_id"])
                                if channel:
                                    msg = await channel.fetch_message(alert["message_id"])
                                    content = msg.content
                                    if not content.startswith("✅"):
                                        await msg.edit(content=f"✅ {content}")
                            except:
                                pass
                                
                        elif runner_status == "LOSER":
                            alert["status"] = "lost"
                            win_loss_stats["lost"] += 1
                            update_bet_result(alert["market_id"], alert["runner"], alert["guild_id"], won=False)
                            
                            # Update ML learning data with real result
                            await update_ml_bet_with_result(alert["market_id"], alert["runner"], won=False)
                            
                            # Update Discord message with loss emoji
                            try:
                                channel = bot.get_channel(alert["channel_id"])
                                if channel:
                                    msg = await channel.fetch_message(alert["message_id"])
                                    content = msg.content
                                    if not content.startswith("❌"):
                                        await msg.edit(content=f"❌ {content}")
                            except:
                                pass
                        break
                        
            except Exception as e:
                print(f"❌ Error checking alert result: {e}")
                continue
        
        # Also settle ML bets directly from Betfair (every 5 minutes)
        await settle_ml_bets_from_betfair()
                
    except Exception as e:
        print(f"❌ Error in win_loss_check_task: {e}")

@tasks.loop(seconds=1)
async def countdown_update_task():
    """Update countdown timers for active big bet embeds every second (debounced)."""
    try:
        await update_big_bet_countdowns()
        await check_race_closure()
    except Exception as e:
        print(f"❌ Error in countdown_update_task: {e}")

def should_send_alert(market_id, runner_id, price, size):
    """Check if we should send an alert based on rate limiting"""
    now = datetime.datetime.now(datetime.timezone.utc)
    key = (market_id, runner_id, price, size)
    
    # Clean old entries
    expired_keys = [k for k, timestamp in alert_cache.items() 
                   if (now - timestamp).total_seconds() > ALERT_COOLDOWN]
    for k in expired_keys:
        del alert_cache[k]
    
    # Check if this alert was recently sent
    if key in alert_cache:
        return False
    
    # Mark this alert as sent
    alert_cache[key] = now
    return True

async def check_market_for_big_bets(market_id, market_type_map, market_event_map, market_location_map, market_start_time_map, runner_name_map, market_sport_map, session_token):
    """Check a single market for big bets - with better error handling and AI analysis"""
    bets_found = 0
    try:
        # Get market book with more detailed data
        market_book_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "SportsAPING/v1.0/listMarketBook",
            "params": {
                "marketIds": [market_id],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"]},
                "matchProjection": "ROLLED_UP_BY_PRICE",
                "orderProjection": "ALL",
                "currencyCode": "AUD"
            },
            "id": 1
        })
        
        # Add timeout and error handling to prevent blocking
        try:
            book_resp = call_betfair_api(market_book_req, session_token)
            if not book_resp or not book_resp.get("result"):
                return 0
        except Exception as api_error:
            print(f"⚠️ API timeout for market {market_id[:8]}...")
            return 0
            
        market_book = book_resp["result"][0]
        # Fallback: if we don't yet have a start time from catalogue, try from market book definition
        try:
            if market_id not in market_start_time_map or not market_start_time_map[market_id]:
                market_def = market_book.get("runners", [])  # placeholder to access
                market_def = market_book.get("marketDefinition") if "marketDefinition" in market_book else None
                if market_def and market_def.get("marketTime"):
                    market_start_time_map[market_id] = market_def["marketTime"]
                    print(f"🧭 Fallback start time from book for {market_id[:8]}...: {market_start_time_map[market_id]}")
        except Exception as _e:
            pass

        # Normalize to aware UTC datetime where possible for consistent countdowns
        def parse_iso_utc(s: str):
            try:
                if not s:
                    return None
                return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
            except Exception:
                return None

        start_raw = market_start_time_map.get(market_id, "")
        parsed_start = parse_iso_utc(start_raw)
        if parsed_start:
            # Store normalized (ensure tz aware UTC)
            if parsed_start.tzinfo is None:
                parsed_start = parsed_start.replace(tzinfo=datetime.timezone.utc)
            market_start_time_map[market_id] = parsed_start.isoformat()
        event_name, market_name = market_event_map.get(market_id, ("", ""))
        country_code_raw, venue = market_location_map.get(market_id, ("", ""))
        combined_event_text = f"{event_name} {market_name} {venue}".lower()
        venue_lc = (venue or "").lower()
        bet_type = market_type_map.get(market_id)
        
        # Reclassify harness/trots and suppress entirely
        if bet_type == "horse" and any(keyword in combined_event_text for keyword in TROT_KEYWORDS):
            bet_type = "trots"
        
        # Suppress trots completely
        if bet_type == "trots":
            return 0
        
        # Precise AU vs UK routing using track sets
        country_code = (country_code_raw or "").upper()
        if venue_lc in UK_HORSE_TRACKS:
            is_au_race = False
        elif venue_lc in AU_HORSE_TRACKS:
            is_au_race = True
        elif country_code == "AU":
            is_au_race = True
        elif country_code == "GB":
            is_au_race = False
        else:
            # Fallback: check AU keywords in event text
            is_au_race = any(loc in combined_event_text for loc in AU_LOCATION_KEYWORDS)
        
        # Check if this market type has alerts enabled
        alerts_enabled = False
        for guild in bot.guilds:
            settings = guild_settings.get(guild.id, {})
            if bet_type == "horse" and settings.get("horse_alerts_running"):
                alerts_enabled = True
                break
            elif bet_type == "sports" and settings.get("global_alerts_running"):
                alerts_enabled = True
                break
            elif bet_type == "greyhound" and settings.get("greyhound_alerts_running"):
                alerts_enabled = True
                break
            elif bet_type == "trots" and settings.get("trots_alerts_running"):
                alerts_enabled = True
                break
        
        if not alerts_enabled:
            return 0
            
        # Process each runner
        for runner in market_book.get("runners", []):
            runner_id = runner.get("selectionId")
            if not runner_id:
                continue
                
            current_total_matched = runner.get("totalMatched", 0)
            # Ensure numeric type for comparisons
            if current_total_matched is None:
                current_total_matched = 0
            try:
                current_total_matched = float(current_total_matched)
            except Exception:
                current_total_matched = 0.0
            runner_name = runner_name_map.get((market_id, runner_id), f"Runner {runner_id}")
            
            # Get last known total for this runner
            cache_key = f"{market_id}_{runner_id}"
            last_total = market_data_cache.get(cache_key, 0)
            if last_total is None:
                last_total = 0
            try:
                last_total = float(last_total)
            except Exception:
                last_total = 0.0
            
            # Calculate new money matched since last check
            try:
                new_matched = float(current_total_matched) - float(last_total)
            except Exception:
                new_matched = 0.0
            
            # Update cache
            market_data_cache[cache_key] = current_total_matched
            
            # Only show important betting activity (reduce spam)
            significant_threshold = 1000  # Only log bets over $1000
            
            # Skip if no new money
            if new_matched is None or new_matched <= 0:
                continue
                
            # Get last traded price (removed debug logging to prevent spam)
            last_price = runner.get("lastPriceTraded")
            price_display = f"{last_price:.2f}" if last_price else "N/A"
            rounded_new_matched = round_bet_amount(new_matched)
            rounded_total_matched = round_bet_amount(current_total_matched)
            odds_value = None
            try:
                if last_price not in (None, "", 0):
                    odds_value = float(last_price)
            except Exception:
                odds_value = None
            recommended_units = None
            if odds_value is not None:
                try:
                    recommended_units = calculate_unit_size(odds_value, bet_type, rounded_total_matched)
                except Exception:
                    recommended_units = None
            
            # === ML LEARNING: Record ALL bets for learning (before any filtering) ===
            # This captures bets at ALL price points ($100, $1000, $10000+) for pattern analysis
            if ML_CONFIG.get("learning_mode") and new_matched >= 100:  # Record bets $100+
                # For sports, get the sport_key from the map; for racing, use bet_type as sport
                if bet_type == "sports":
                    sport_key_for_ml = market_sport_map.get(market_id)
                else:
                    sport_key_for_ml = bet_type  # horse, greyhound, or trots
                    
                ml_bet_data_global = {
                    "bet_type": bet_type,
                    "sport_key": sport_key_for_ml,
                    "odds": odds_value,
                    "bet_amount": new_matched,
                    "total_matched": current_total_matched,
                    "matched_percentage": (float(new_matched) / float(current_total_matched) * 100.0) if current_total_matched else 0,
                    "runner_name": runner_name,
                    "event_name": event_name,
                    "market_name": market_name,
                    "recommended_units": recommended_units,
                    "is_au_race": is_au_race if bet_type == "horse" else None,
                }
                # Only record if we haven't seen this exact bet recently
                ml_cache_key = f"ml_{market_id}_{runner_id}_{int(new_matched)}"
                if ml_cache_key not in market_data_cache:
                    market_data_cache[ml_cache_key] = True
                    record_observed_bet(ml_bet_data_global, was_posted=False, filter_reason="observed_for_learning")
                    if new_matched >= 5000:  # Log significant bets
                        print(f"🧠 ML: Observed ${new_matched:,.0f} bet on {runner_name} ({bet_type})")
            
            # Send alerts to guilds with proper settings
            for guild in bot.guilds:
                settings = guild_settings.get(guild.id, {})
                
                # Routing logic for each type with proper debug info
                channel_id = None
                min_bet = 1000  # Default fallback
                emoji = "🌐"
                alerts_running = False
                # For sports, get from map; for racing types, use bet_type as sport_key
                if bet_type == "sports":
                    sport_key = market_sport_map.get(market_id)
                else:
                    sport_key = bet_type  # horse, greyhound, or trots
                sport_meta = SPORT_CATEGORY_LOOKUP.get(sport_key, {}) if sport_key else {}
                sport_label = sport_meta.get("label", "Sports") if bet_type == "sports" else None
                
                if bet_type == "horse":
                    if is_au_race:
                        channel_id = settings.get("au_horse_alert_channel")
                        min_bet = settings.get("au_horse_min_matched", 1000)
                        emoji = "🇦🇺"
                    else:
                        channel_id = settings.get("uk_horse_alert_channel")
                        min_bet = settings.get("uk_horse_min_matched", 1000)
                        emoji = "🇬🇧"
                    alerts_running = settings.get("horse_alerts_running", False)
                elif bet_type == "sports":
                    sports_channels = settings.get("sports_channels", {})
                    sports_min = settings.get("sports_min_matched", {})
                    fallback_channel = settings.get("alert_channel")
                    
                    # Try sport-specific channel first, then fallback
                    if sport_key and sports_channels.get(sport_key):
                        channel_id = sports_channels[sport_key]
                    else:
                        channel_id = fallback_channel
                    
                    # Get min bet - sport-specific if configured, else global
                    if sport_key and sport_key in sports_min:
                        min_bet = sports_min[sport_key]
                    else:
                        min_bet = settings.get("min_matched", 1000)
                    
                    emoji = sport_meta.get("emoji", get_sport_emoji(event_name, market_name))
                    alerts_running = settings.get("global_alerts_running", False)
                elif bet_type == "greyhound":
                    channel_id = settings.get("greyhound_alert_channel")
                    min_bet = settings.get("greyhound_min_matched", 1000)
                    emoji = "🐕"
                    alerts_running = settings.get("greyhound_alerts_running", False)
                elif bet_type == "trots":
                    channel_id = settings.get("trots_alert_channel")
                    min_bet = settings.get("trots_min_matched", 1000)
                    emoji = "🚗"
                    alerts_running = settings.get("trots_alerts_running", False)
                
                # Safely coerce min_bet to a numeric value
                try:
                    min_bet = float(min_bet if min_bet is not None else 1000)
                except Exception:
                    min_bet = 1000.0
                
                # === ML LEARNING: Prepare bet data for recording ===
                ml_bet_data = {
                    "bet_type": bet_type,
                    "sport_key": sport_key,
                    "odds": odds_value,
                    "bet_amount": new_matched,
                    "total_matched": current_total_matched,
                    "matched_percentage": (float(new_matched) / float(current_total_matched) * 100.0) if current_total_matched else 0,
                    "runner_name": runner_name,
                    "event_name": event_name,
                    "market_name": market_name,
                    "recommended_units": recommended_units,
                }
                
                # Enforce 40% rule for horses: Bet Amount / Total Matched >= threshold
                threshold = BET_CONFIG.get('MIN_PERCENTAGE_THRESHOLD', 40.0)
                matched_warning = ""
                if bet_type == "horse" and new_matched and current_total_matched:
                    try:
                        matched_percentage = (float(new_matched) / float(current_total_matched)) * 100.0
                    except Exception:
                        matched_percentage = 0.0
                    
                    if matched_percentage < threshold:
                        # Record for ML learning (filtered bet)
                        if ML_CONFIG.get("learning_mode"):
                            record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"matched_percentage_{matched_percentage:.1f}_below_{threshold}")
                        # Skip low-volume horse bets
                        print(f"⏭️ Skipping horse bet: {matched_percentage:.2f}% < {threshold:.0f}% ({runner_name})")
                        continue
                    else:
                        matched_warning = (
                            f"\n✅ **Match Volume OK**\n"
                            f"💰 Bet amount is {matched_percentage:.1f}% of total matched"
                        )
                
                # Calculate matched warning for greyhounds (no skip, just warning)
                if bet_type == "greyhound" and new_matched and current_total_matched:
                    try:
                        matched_percentage = (float(new_matched) / float(current_total_matched)) * 100.0
                        matched_warning = (
                            f"\n✅ **Match Volume**\n"
                            f"💰 Bet amount is {matched_percentage:.1f}% of total matched"
                        )
                    except Exception:
                        pass
                
                # RESPECT USER'S MINIMUM BET SETTING - with ML learning
                if (not alerts_running) or (not channel_id) or (new_matched is None) or (new_matched < min_bet):
                    # Record for ML learning (filtered bet)
                    if ML_CONFIG.get("learning_mode") and new_matched is not None:
                        filter_reason = "alerts_off" if not alerts_running else "no_channel" if not channel_id else f"below_min_{min_bet}"
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason=filter_reason)
                    continue
                
                # === ML OPTIMAL FILTERS ===
                # These are based on profitable patterns discovered by the ML algorithm
                
                # 1. MAX ODDS FILTER (Favorites strategy - 89% win rate at odds < $3.00)
                max_odds = settings.get("max_odds")
                if max_odds and odds_value and odds_value > max_odds:
                    if ML_CONFIG.get("learning_mode"):
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"odds_{odds_value:.2f}_above_max_{max_odds}")
                    print(f"⏭️ ML Filter: Skipping bet - odds ${odds_value:.2f} > max ${max_odds:.2f} ({runner_name})")
                    continue
                
                # 2. AVOID LONGSHOTS FILTER (88% win rate when avoiding odds > $10)
                avoid_longshots = settings.get("avoid_longshots", False)
                if avoid_longshots and odds_value and odds_value > 10.0:
                    if ML_CONFIG.get("learning_mode"):
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"longshot_odds_{odds_value:.2f}")
                    print(f"⏭️ ML Filter: Skipping longshot - odds ${odds_value:.2f} > $10.00 ({runner_name})")
                    continue
                
                # 3. WEEKDAY ONLY FILTER (93% win rate on weekdays vs lower on weekends)
                weekday_only = settings.get("weekday_only", False)
                if weekday_only:
                    current_day = datetime.datetime.now(datetime.timezone.utc).weekday()
                    if current_day >= 5:  # Saturday (5) or Sunday (6)
                        if ML_CONFIG.get("learning_mode"):
                            record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"weekend_day_{current_day}")
                        print(f"⏭️ ML Filter: Skipping weekend bet - day {current_day} ({runner_name})")
                        continue
                
                # 4. PRIORITY SPORTS FILTER (e.g., Cricket has 98% win rate)
                priority_sports = settings.get("priority_sports", [])
                sports_only_mode = settings.get("sports_only_priority", False)
                if sports_only_mode and priority_sports and bet_type == "sports":
                    if sport_key and sport_key not in priority_sports:
                        if ML_CONFIG.get("learning_mode"):
                            record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"sport_{sport_key}_not_priority")
                        print(f"⏭️ ML Filter: Skipping non-priority sport {sport_key} ({runner_name})")
                        continue
                
                # 5. PREFER FAVORITES FILTER (focus on odds < $3.00)
                prefer_favorites = settings.get("prefer_favorites", False)
                if prefer_favorites and odds_value and odds_value >= 3.0:
                    if ML_CONFIG.get("learning_mode"):
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"not_favorite_odds_{odds_value:.2f}")
                    print(f"⏭️ ML Filter: Skipping non-favorite - odds ${odds_value:.2f} >= $3.00 ({runner_name})")
                    continue
                
                # 6. SPORTS ONLY MODE (94.5% win rate on sports vs lower on racing)
                sports_only = settings.get("sports_only_mode", False)
                if sports_only and bet_type != "sports":
                    if ML_CONFIG.get("learning_mode"):
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason=f"sports_only_mode_{bet_type}")
                    print(f"⏭️ ML Filter: Skipping {bet_type} - sports only mode ({runner_name})")
                    continue
                    
                channel = bot.get_channel(channel_id)
                if not channel:
                    print(f"❌ Channel {channel_id} not found for guild {guild.id}")
                    continue
                
                # PREVENT MULTIPLE BETS PER EVENT - Only post one bet per game/event
                event_key = f"{event_name}_{guild.id}"  # Include guild_id to track per server
                current_time = datetime.datetime.now(datetime.timezone.utc)
                
                # Check if we've already posted a bet for this event in the last 4 hours
                if event_key in posted_events:
                    last_posted = posted_events[event_key]
                    time_diff = current_time - last_posted
                    if time_diff.total_seconds() < 14400:  # 4 hours in seconds
                        print(f"🚫 Skipping bet for {event_name} - already posted bet for this event {time_diff.total_seconds()/3600:.1f}h ago")
                        continue
                
                # Mark this event as posted
                posted_events[event_key] = current_time
                
                # Check if we already have an active alert for this runner in this guild
                alert_key = (market_id, runner_id, guild.id)
                existing_alert = active_runner_alerts.get(alert_key)

                # Only surface actionable bets (skip "no bet" situations)
                if not existing_alert and odds_value is not None and recommended_units is None:
                    print(f"🚫 Skipping {bet_type} alert for {runner_name} — odds {odds_value:.2f} flagged as NO BET")
                    continue
                
                # Get market start time for countdown display
                current_time = datetime.datetime.now(datetime.timezone.utc)
                market_start_time_raw = market_start_time_map.get(market_id, "")
                print(f"🔍 DEBUG: Market {market_id[:8]}... start time raw: '{market_start_time_raw}'")
                start_dt = None
                if market_start_time_raw:
                    try:
                        # Accept already-normalized ISO or raw Z-form
                        start_dt = datetime.datetime.fromisoformat(str(market_start_time_raw).replace('Z', '+00:00'))
                    except Exception as e:
                        print(f"⚠️ Error parsing market start time {market_start_time_raw}: {e}")
                
                if start_dt is not None:
                    time_str = start_dt.strftime("%H:%M UTC")
                    print(f"🔍 DEBUG: Parsed start time: {start_dt} -> {time_str}")
                else:
                    print(f"🔍 DEBUG: No valid start time, using current time")
                    time_str = current_time.strftime("%H:%M UTC")
                
                # Create market info based on type
                market_info = ""
                if bet_type == "horse":
                    # Use the emoji determined above (AU or UK/ROW)
                    market_info = f"{emoji} **{event_name}** • {market_name}"
                elif bet_type == "greyhound":
                    market_info = f"🐕 **{event_name}** • {market_name}"
                elif bet_type == "trots":
                    market_info = f"🚗 **{event_name}** • {market_name}"
                else:
                    market_info = f"{emoji} **{sport_label or 'Sports'}** • {event_name} • {market_name}"
                
                # Check for mention role
                mention_text = ""
                mention_role_id = settings.get("mention_role")
                if mention_role_id:
                    mention_text = f"<@&{mention_role_id}> "
                
                if existing_alert:
                    # UPDATE existing alert instead of posting new one
                    
                    # === LEARNING MODE: Record stake update but DON'T update message ===
                    if ML_CONFIG.get("learning_mode"):
                        additional_stake = rounded_total_matched - existing_alert["total_amount"]
                        if additional_stake > 0:
                            stake_update_data = {
                                "bet_type": bet_type,
                                "sport_key": sport_key,
                                "odds": existing_alert.get("original_odds", odds_value),
                                "bet_amount": additional_stake,
                                "total_matched": rounded_total_matched,
                                "matched_percentage": (float(additional_stake) / float(rounded_total_matched) * 100.0) if rounded_total_matched else 0,
                                "runner_name": runner_name,
                                "event_name": event_name,
                                "market_name": market_name,
                                "recommended_units": existing_alert.get("original_units"),
                                "is_au_race": emoji == "🇦🇺" if bet_type in ["horse", "greyhound", "trots"] else None,
                                "is_stake_update": True,
                                "previous_total": existing_alert["total_amount"],
                            }
                            record_observed_bet(stake_update_data, was_posted=False, filter_reason="learning_mode_active")
                            print(f"🧠 ML LEARNING MODE: Recorded stake update but NOT posting - +${additional_stake:,} on {runner_name}")
                            # Update tracking so we don't re-record
                            active_runner_alerts[alert_key]["total_amount"] = rounded_total_matched
                        continue  # Skip posting, just record
                    
                    try:
                        msg = await channel.fetch_message(existing_alert["message_id"])
                        
                        # Calculate the additional stake
                        additional_stake = rounded_total_matched - existing_alert["total_amount"]
                        
                        # Get countdown using precise datetime if available
                        race_start_time = None
                        countdown_display = "⏰ **Time Unavailable**"
                        try:
                            if start_dt is not None:
                                print(f"🔎 COUNTDOWN DEBUG(update): market_id={market_id} parsed={start_dt.isoformat()}")
                                countdown_display, race_start_time = get_race_countdown_from_datetime(start_dt)
                            else:
                                countdown_display, race_start_time = get_race_countdown(time_str)
                        except Exception as _:
                            countdown_display, race_start_time = get_race_countdown(time_str)
                        
                        # Use ORIGINAL odds from when bet was first placed, not current odds
                        original_odds = existing_alert.get("original_odds", odds_value if odds_value else float(str(price_display).replace('$', '')))
                        original_units = existing_alert.get("original_units", 1.0)
                        
                        # Display original odds
                        original_odds_display = f"{original_odds:.2f}"
                        
                        # Get appropriate selection label (Team/Player for sports, Horse/Dog for racing)
                        selection_label = get_selection_label(bet_type, sport_key if bet_type == "sports" else None)
                        
                        # Get type label for header
                        type_label = {
                            "horse": "Horse Racing",
                            "greyhound": "Greyhound Racing",
                            "trots": "Harness Racing"
                        }.get(bet_type, sport_label or "Sports")
                        
                        # Stake update embed format
                        embed = discord.Embed(
                            color=discord.Color.from_rgb(255, 165, 0),  # Orange for updates
                            timestamp=datetime.datetime.now(datetime.timezone.utc)
                        )
                        
                        # Title
                        embed.title = "🔄 STAKE INCREASED"
                        
                        # Description with sport type, event, and market
                        embed.description = f"{emoji} **{type_label}** • {event_name} • {market_name}"
                        
                        # Selection info with update
                        embed.add_field(name=selection_label, value=f"**{runner_name}**", inline=True)
                        embed.add_field(name="New Total", value=f"**${rounded_total_matched:,}**", inline=True)
                        embed.add_field(name="Added", value=f"+${additional_stake:,}", inline=True)
                        
                        # Use ORIGINAL odds, not current
                        embed.add_field(name="Odds", value=f"{original_odds_display}", inline=True)
                        embed.add_field(name="Time", value=f"⏰ {countdown_display.replace('**', '')}", inline=True)
                        embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer
                        
                        # Use ORIGINAL units recommendation
                        if original_units and original_units > 0:
                            embed.add_field(
                                name="📊 Betting Guidelines",
                                value=f"**BET {original_units:.1f} UNITS ON THIS SELECTION**",
                                inline=False
                            )
                        
                        # Footer
                        footer_label = {
                            "horse": "🐎 Horse Racing",
                            "greyhound": "🐕 Greyhound Racing",
                            "trots": "🚗 Harness Racing"
                        }.get(bet_type, f"{emoji} {sport_label or 'Sports'}")
                        
                        embed.set_footer(text=f"STAKE INCREASED • {footer_label}")
                        
                        embed_description = f"{emoji} {runner_name} • {event_name}"
                        
                        await msg.edit(content=mention_text, embed=embed)
                        
                        # Update the tracking
                        active_runner_alerts[alert_key]["total_amount"] = rounded_total_matched
                        
                        # Update countdown tracking for the updated embed
                        if race_start_time:
                            fields_data = []
                            for field in embed.fields:
                                fields_data.append({
                                    "name": field.name,
                                    "value": field.value,
                                    "inline": field.inline
                                })
                                
                            active_big_bet_embeds[market_id] = {
                                "message_id": msg.id,
                                "channel_id": channel_id,
                                "start_time": race_start_time,
                                "embed_data": {
                                    "title": embed.title,
                                    "original_description": embed_description,
                                    "color": embed.color.value,
                                    "timestamp": embed.timestamp,
                                    "fields": fields_data
                                },
                                "last_display": countdown_display
                            }
                        
                        print(f"🔄 Updated {bet_type} alert: ${additional_stake:,} additional on {runner_name} (Total: ${rounded_total_matched:,})")
                        bets_found += 1
                        
                        # === ML LEARNING: Record stake update as separate observation ===
                        if ML_CONFIG.get("learning_mode") and additional_stake > 0:
                            try:
                                stake_update_data = {
                                    "bet_type": bet_type,
                                    "sport_key": sport_key,
                                    "odds": original_odds,
                                    "bet_amount": additional_stake,  # The added amount
                                    "total_matched": rounded_total_matched,  # New total
                                    "matched_percentage": matched_percentage if 'matched_percentage' in dir() else None,
                                    "runner_name": runner_name,
                                    "event_name": event_name,
                                    "market_name": market_name,
                                    "recommended_units": original_units,
                                    "is_au_race": emoji == "🇦🇺" if bet_type in ["horse", "greyhound", "trots"] else None,
                                    "is_stake_update": True,  # Mark as stake update
                                    "previous_total": existing_alert["total_amount"],
                                }
                                record_observed_bet(stake_update_data, was_posted=True, filter_reason=None)
                                print(f"🧠 ML: Recorded stake update: +${additional_stake:,} on {runner_name} (Total: ${rounded_total_matched:,})")
                            except Exception as ml_err:
                                print(f"⚠️ ML stake update recording failed: {ml_err}")
                        
                    except Exception as e:
                        print(f"❌ Error updating message: {e}")
                        # Remove broken alert from tracking
                        if alert_key in active_runner_alerts:
                            del active_runner_alerts[alert_key]
                        
                else:
                    # POST new alert
                    
                    # === LEARNING MODE: Record bet but DON'T post ===
                    if ML_CONFIG.get("learning_mode"):
                        # Record for ML learning (will be posted = False in learning mode)
                        ml_bet_data["is_au_race"] = emoji == "🇦🇺" if bet_type in ["horse", "greyhound", "trots"] else None
                        record_observed_bet(ml_bet_data, was_posted=False, filter_reason="learning_mode_active")
                        print(f"🧠 ML LEARNING MODE: Recorded but NOT posting - {runner_name} @ {odds_value:.2f} (${rounded_total_matched:,})")
                        bets_found += 1
                        continue  # Skip posting, just record
                    
                    # Get countdown display and race start time (prefer precise datetime)
                    race_start_time = None
                    try:
                        if start_dt is not None:
                            print(f"🔎 COUNTDOWN DEBUG(new): market_id={market_id} parsed={start_dt.isoformat()}")
                            countdown_display, race_start_time = get_race_countdown_from_datetime(start_dt)
                        else:
                            countdown_display, race_start_time = get_race_countdown(time_str)
                    except Exception as _:
                        countdown_display, race_start_time = get_race_countdown(time_str)
                    
                    # AI filter removed - now using ML optimal filters instead
                    # ML filters (max_odds, avoid_longshots, weekday_only, etc.) are applied earlier
                    # and are based on 520+ settled bets with proven 94.5% win rate for sports
                    
                    # Calculate units for display
                    try:
                        odds_for_units = odds_value if odds_value else float(str(price_display).replace('$', ''))
                        units_display = recommended_units if recommended_units else calculate_unit_size(odds_for_units, bet_type, rounded_total_matched)
                    except:
                        units_display = 1.0
                    
                    # Determine embed color based on bet type
                    if bet_type == "horse":
                        embed_color = discord.Color.from_rgb(139, 90, 43)  # Rich brown for horses
                    elif bet_type == "greyhound":
                        embed_color = discord.Color.from_rgb(65, 105, 225)  # Royal blue for dogs
                    elif bet_type == "trots":
                        embed_color = discord.Color.from_rgb(148, 103, 189)  # Purple for trots
                    elif bet_type == "sports":
                        embed_color = discord.Color.from_rgb(46, 204, 113)  # Emerald green for sports
                    else:
                        embed_color = discord.Color.from_rgb(241, 196, 15)  # Gold default
                    
                    # Get appropriate selection label (Team/Player for sports, Horse/Dog for racing)
                    selection_label = get_selection_label(bet_type, sport_key if bet_type == "sports" else None)
                    
                    # Get type label for header
                    type_label = {
                        "horse": "Horse Racing",
                        "greyhound": "Greyhound Racing",
                        "trots": "Harness Racing"
                    }.get(bet_type, sport_label or "Sports")
                    
                    # New embed format - LARGE BET SPOTTED style
                    embed = discord.Embed(
                        color=embed_color,
                        timestamp=datetime.datetime.now(datetime.timezone.utc)
                    )
                    
                    # Title
                    embed.title = "🔔 LARGE BET SPOTTED"
                    
                    # Description with sport type, event, and market
                    embed.description = f"{emoji} **{type_label}** • {event_name} • {market_name}"
                    
                    # Selection/Runner/Team info
                    embed.add_field(name=selection_label, value=f"**{runner_name}**", inline=True)
                    embed.add_field(name="Bet Amount", value=f"${rounded_new_matched:,}", inline=True)
                    embed.add_field(name="Total Matched", value=f"${rounded_total_matched:,}", inline=True)
                    
                    embed.add_field(name="Odds", value=f"{price_display}", inline=True)
                    embed.add_field(name="Time", value=f"⏰ {countdown_display.replace('**', '')}", inline=True)
                    embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer for alignment
                    
                    # Betting Guidelines
                    if units_display and units_display > 0:
                        embed.add_field(
                            name="📊 Betting Guidelines",
                            value=f"**BET {units_display:.1f} UNITS ON THIS SELECTION**",
                            inline=False
                        )
                    
                    # Footer with bet type
                    footer_label = {
                        "horse": "🐎 Horse Racing",
                        "greyhound": "🐕 Greyhound Racing",
                        "trots": "🚗 Harness Racing"
                    }.get(bet_type, f"{emoji} {sport_label or 'Sports'}")
                    
                    embed.set_footer(text=footer_label)
                    
                    # Store description for countdown updates
                    embed_description = f"{emoji} {runner_name} • {event_name}"
                    
                    try:
                        msg = await channel.send(content=mention_text, embed=embed)
                        
                        # Track this new alert - store original odds for updates
                        active_runner_alerts[alert_key] = {
                            "message_id": msg.id,
                            "channel_id": channel_id,
                            "total_amount": rounded_total_matched,
                            "original_odds": odds_value if odds_value else float(str(price_display).replace('$', '')),
                            "original_units": units_display
                        }
                        
                        # Track this embed for countdown updates
                        if race_start_time:
                            fields_data = []
                            for field in embed.fields:
                                fields_data.append({
                                    "name": field.name,
                                    "value": field.value,
                                    "inline": field.inline
                                })
                                
                            active_big_bet_embeds[market_id] = {
                                "message_id": msg.id,
                                "channel_id": channel_id,
                                "start_time": race_start_time,
                                "embed_data": {
                                    "title": embed.title,
                                    "original_description": embed_description,
                                    "color": embed.color.value,
                                    "timestamp": embed.timestamp,
                                    "fields": fields_data
                                },
                                "last_display": countdown_display
                            }
                        
                        # Track for win/loss stats (use original amounts for accuracy)
                        sent_alerts.append({
                            "guild_id": guild.id,
                            "channel_id": channel_id,
                            "message_id": msg.id,
                            "market_id": market_id,
                            "runner": runner_name,
                            "selection_id": runner_id,
                            "emoji": emoji,
                            "status": "pending"
                        })
                        win_loss_stats["sent"] += 1
                        
                        # Track bet in profit tracking system
                        try:
                            odds_value = float(str(price_display).replace('$', ''))
                            bet_data = track_new_bet(
                                market_id=market_id,
                                runner_name=runner_name,
                                odds=odds_value,
                                bet_type=bet_type,
                                amount=rounded_total_matched,
                                sport_key=sport_key,
                                guild_id=guild.id,
                                channel_id=channel_id,
                                message_id=msg.id
                            )
                            print(f"📊 Bet tracked: {runner_name} @ ${odds_value} ({bet_data['recommended_units']} units)")
                            
                            # === ML LEARNING: Mark this bet as POSTED (was observed before) ===
                            if ML_CONFIG.get("learning_mode"):
                                # Update the previously observed bet to mark it as posted
                                for ml_bet in reversed(ml_learning_data.get("all_observed_bets", [])):
                                    if (ml_bet.get("runner_name") == runner_name and 
                                        ml_bet.get("event_name") == event_name and
                                        not ml_bet.get("was_posted")):
                                        ml_bet["was_posted"] = True
                                        ml_bet["filter_reason"] = None
                                        save_ml_data()
                                        break
                                print(f"🧠 ML: Marked bet as POSTED: {runner_name}")
                            
                        except Exception as track_error:
                            print(f"⚠️ Error tracking bet: {track_error}")
                        
                        print(f"✅ Sent NEW {bet_type} alert: ${rounded_new_matched:,} bet on {runner_name}")
                        bets_found += 1
                        
                    except Exception as e:
                        print(f"❌ Error sending message to channel {channel_id}: {e}")
                    
    except Exception as e:
        print(f"⚠️ Market {market_id[:8]} check failed")
    
    return bets_found

# --- Track total staked ---
  # key: (market_id, runner_name), value: float

class StartStopHorseAlertsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="Start/Stop Horse Alerts", style=discord.ButtonStyle.success)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        running = settings.get("horse_alerts_running", False)
        
        # Check if channels are configured before allowing start
        au_channel = settings.get("au_horse_alert_channel")
        uk_channel = settings.get("uk_horse_alert_channel")
        
        if not running and not (au_channel or uk_channel):
            await interaction.response.send_message(
                "⚠️ Cannot start horse alerts - no channels configured!\n"
                "Please set up AU and/or UK horse alert channels first.",
                ephemeral=True
            )
            return
            
        settings["horse_alerts_running"] = not running
        status = "started" if not running else "stopped"
        
        # Start or stop live stream based on alert status
        if not running:
            await start_live_stream_if_needed()
        else:
            await stop_live_stream_if_not_needed()
            
        config_status = []
        if au_channel:
            config_status.append(f"AU Channel: <#{au_channel}>")
        if uk_channel:
            config_status.append(f"UK Channel: <#{uk_channel}>")
            
        await interaction.response.send_message(
            f"🐎 Horse alerts {status}.\n"
            f"Current configuration:\n" + "\n".join(config_status),
            ephemeral=True
        )

class StartStopSportsAlertsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="Start/Stop Sports Alerts", style=discord.ButtonStyle.success)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        settings = guild_settings.setdefault(self.guild.id, {})
        running = settings.get("global_alerts_running", False)
        settings["global_alerts_running"] = not running
        status = "started" if not running else "stopped"
        
        # Start or stop live stream based on alert status
        if not running:
            await start_live_stream_if_needed()
        else:
            await stop_live_stream_if_not_needed()
        
        await save_guild_settings_async()

        sports_channels = settings.get("sports_channels", {})
        fallback_channel = settings.get("alert_channel")
        if sports_channels:
            configured = ", ".join(
                f"{SPORT_CATEGORY_LOOKUP.get(key, {}).get('label', key.title())}: <#{chan_id}>"
                for key, chan_id in sports_channels.items()
                if chan_id
            )
        elif fallback_channel:
            configured = f"Default channel: <#{fallback_channel}>"
        else:
            configured = "No sport channels configured yet."

        await interaction.response.send_message(
            f"⚽ Sports alerts {status}.\n{configured}",
            ephemeral=True
        )

class StartStopGreyhoundAlertsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="Start/Stop Greyhound Alerts", style=discord.ButtonStyle.success)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        running = guild_settings.setdefault(self.guild.id, {}).get("greyhound_alerts_running", False)
        guild_settings[self.guild.id]["greyhound_alerts_running"] = not running
        status = "started" if not running else "stopped"
        
        # Start or stop live stream based on alert status
        if not running:
            await start_live_stream_if_needed()
        else:
            await stop_live_stream_if_not_needed()
            
        await interaction.response.send_message(f"🐕 Greyhound alerts {status}.", ephemeral=True)

class StartStopTrotsAlertsButton(discord.ui.Button):
    def __init__(self, guild):
        super().__init__(label="Start/Stop Trots Alerts", style=discord.ButtonStyle.success)
        self.guild = guild

    async def callback(self, interaction: discord.Interaction):
        running = guild_settings.setdefault(self.guild.id, {}).get("trots_alerts_running", False)
        guild_settings[self.guild.id]["trots_alerts_running"] = not running
        status = "started" if not running else "stopped"
        
        # Start or stop live stream based on alert status
        if not running:
            await start_live_stream_if_needed()
        else:
            await stop_live_stream_if_not_needed()
            
        await interaction.response.send_message(f"🚗 Trots alerts {status}.", ephemeral=True)

@bot.event
async def on_error(event, *args, **kwargs):
    print(f"❌ Bot error in {event}: {args}")
    traceback.print_exc()

@tree.command(name="start", description="Show Betfair bot panel")
async def start(interaction: discord.Interaction):
    try:
        # Defer immediately to prevent "application didn't respond" timeout
        await interaction.response.defer(ephemeral=True)
        
        # Check user authorization
        if interaction.user.id not in AUTHORIZED_USER_IDS:
            await interaction.followup.send(
                "❌ **Access Denied**\nYou are not authorized to use this bot.", 
                ephemeral=True
            )
            return
        
        embed = discord.Embed(
            title="🎛️ Betfair Bot Panel",
            description="Choose which settings to configure.",
            color=discord.Color.blurple()
        )
        await interaction.followup.send(
            embed=embed,
            view=MainPanelView(interaction.guild),
            ephemeral=True
        )
    except Exception as e:
        print(f"❌ Error in start command: {e}")
        try:
            await interaction.followup.send("❌ Error loading panel", ephemeral=True)
        except:
            pass

@tree.command(name="stats", description="Show win/loss rate for posted bets")
async def stats(interaction: discord.Interaction):
    # Defer immediately to prevent timeout
    await interaction.response.defer(ephemeral=True)
    
    # Check user authorization
    if interaction.user.id not in AUTHORIZED_USER_IDS:
        await interaction.followup.send(
            "❌ **Access Denied**\nYou are not authorized to use this bot.", 
            ephemeral=True
        )
        return
    
    # Show both old stats and new profit tracking stats
    sent = win_loss_stats["sent"]
    won = win_loss_stats["won"]
    lost = win_loss_stats["lost"]
    win_rate = (won / sent * 100) if sent else 0
    
    # Get current daily summary
    daily_summary = get_daily_summary()
    
    embed = discord.Embed(
        title="📊 Betting Statistics",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    
    # Legacy stats
    embed.add_field(
        name="📈 Legacy Stats",
        value=f"**Sent:** {sent}\n**Won:** {won}\n**Lost:** {lost}\n**Win Rate:** {win_rate:.1f}%",
        inline=True
    )
    
    # New profit tracking stats
    embed.add_field(
        name="💰 Today's P&L",
        value=(
            f"**Total Bets:** {daily_summary['total_bets']}\n"
            f"**Pending:** {daily_summary['pending_bets']}\n"
            f"**Won:** {daily_summary['won_bets']}\n"
            f"**Lost:** {daily_summary['lost_bets']}\n"
            f"**Win Rate:** {daily_summary['win_rate']:.1f}%\n"
            f"**Net P&L:** {daily_summary['net_pnl']:+.1f} units"
        ),
        inline=True
    )
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="record_loss", description="Record a betting loss for cooldown tracking")
async def record_loss_command(interaction: discord.Interaction):
    # Defer immediately to prevent timeout
    await interaction.response.defer(ephemeral=True)
    
    # Check user authorization
    if interaction.user.id not in AUTHORIZED_USER_IDS:
        await interaction.followup.send(
            "❌ **Access Denied**\nYou are not authorized to use this bot.", 
            ephemeral=True
        )
        return
    
    # Record the loss
    record_loss()
    
    # Check if cooldown is now active
    cooldown_message, is_in_cooldown = check_cooldown_status()
    
    losses_count = len(loss_tracking['losses'])
    
    if is_in_cooldown:
        embed = discord.Embed(
            title="🛑 Loss Recorded - Cooldown Activated",
            description=f"**Losses in last hour:** {losses_count}/3\n\n{cooldown_message}",
            color=discord.Color.red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
    else:
        embed = discord.Embed(
            title="📝 Loss Recorded",
            description=f"**Losses in last hour:** {losses_count}/3\n⚠️ **Warning:** {3-losses_count} more losses will trigger 1-hour cooldown",
            color=discord.Color.orange(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="force_start", description="Force start all alerts (emergency restart)")
async def force_start_command(interaction: discord.Interaction):
    """Emergency command to quickly restart all alerts after a bot restart"""
    await interaction.response.defer(ephemeral=True)
    
    if interaction.user.id not in AUTHORIZED_USER_IDS:
        await interaction.followup.send("❌ **Access Denied**", ephemeral=True)
        return
    
    guild_id = interaction.guild_id
    channel = interaction.channel
    
    # Initialize settings for this guild
    if guild_id not in guild_settings:
        guild_settings[guild_id] = {}
    
    settings = guild_settings[guild_id]
    
    # Enable all alert types
    settings["horse_alerts_running"] = True
    settings["global_alerts_running"] = True
    settings["greyhound_alerts_running"] = True
    settings["trots_alerts_running"] = False  # Usually disabled
    
    # Set this channel as default for all alerts
    settings["au_horse_alert_channel"] = channel.id
    settings["uk_horse_alert_channel"] = channel.id
    settings["alert_channel"] = channel.id
    settings["greyhound_alert_channel"] = channel.id
    
    # Set reasonable default minimums
    settings["au_horse_min_matched"] = 2000
    settings["uk_horse_min_matched"] = 2000
    settings["min_matched"] = 5000
    settings["greyhound_min_matched"] = 2000
    
    # Save settings
    save_guild_settings()
    
    # Start the live stream
    await start_live_stream_if_needed()
    
    embed = discord.Embed(
        title="🚀 Force Started All Alerts",
        description=f"All alerts have been enabled and will post to {channel.mention}",
        color=discord.Color.green(),
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    embed.add_field(
        name="✅ Enabled",
        value="🐎 Horse Racing (AU & UK)\n🌐 Sports\n🐕 Greyhounds",
        inline=True
    )
    embed.add_field(
        name="💰 Minimums",
        value="Horse: $2,000\nSports: $5,000\nGreyhound: $2,000",
        inline=True
    )
    embed.add_field(
        name="📡 Status",
        value=f"Stream: {'🟢 Running' if live_stream_running else '🔴 Starting...'}",
        inline=False
    )
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="diagnose", description="Diagnose why alerts aren't working")
async def diagnose_command(interaction: discord.Interaction):
    """Comprehensive diagnostic command to find out why alerts aren't posting"""
    await interaction.response.defer(ephemeral=True)
    
    # Check user authorization
    if interaction.user.id not in AUTHORIZED_USER_IDS:
        await interaction.followup.send("❌ **Access Denied**", ephemeral=True)
        return
    
    guild_id = interaction.guild_id
    settings = guild_settings.get(guild_id, {})
    
    issues = []
    checks = []
    
    # 1. Check certificate files
    if CERT_FILE and os.path.exists(CERT_FILE):
        checks.append("✅ Certificate file found")
    else:
        issues.append("❌ Certificate file missing - Betfair API won't work!")
    
    if KEY_FILE and os.path.exists(KEY_FILE):
        checks.append("✅ Key file found")
    else:
        issues.append("❌ Key file missing - Betfair API won't work!")
    
    # 2. Check session token
    try:
        token = get_session_token()
        if token:
            checks.append("✅ Betfair session token valid")
        else:
            issues.append("❌ Betfair session token FAILED - check credentials/certs!")
    except Exception as e:
        issues.append(f"❌ Session token error: {str(e)[:50]}")
    
    # 3. Check if live stream is running
    if live_stream_running:
        checks.append("✅ Live stream monitor running")
    else:
        issues.append("⚠️ Live stream NOT running - no alerts will be sent!")
    
    # 4. Check if any alerts are enabled
    alerts_enabled = []
    if settings.get("horse_alerts_running"):
        alerts_enabled.append("🐎 Horse")
    if settings.get("global_alerts_running"):
        alerts_enabled.append("🌐 Sports")
    if settings.get("greyhound_alerts_running"):
        alerts_enabled.append("🐕 Greyhound")
    if settings.get("trots_alerts_running"):
        alerts_enabled.append("🚗 Trots")
    
    if alerts_enabled:
        checks.append(f"✅ Alerts enabled: {', '.join(alerts_enabled)}")
    else:
        issues.append("❌ NO ALERTS ENABLED - run /start to enable alerts!")
    
    # 5. Check if channels are configured
    channels_configured = []
    if settings.get("au_horse_alert_channel"):
        channels_configured.append("AU Horse")
    if settings.get("uk_horse_alert_channel"):
        channels_configured.append("UK Horse")
    if settings.get("alert_channel"):
        channels_configured.append("Sports")
    if settings.get("greyhound_alert_channel"):
        channels_configured.append("Greyhound")
    if settings.get("trots_alert_channel"):
        channels_configured.append("Trots")
    
    if channels_configured:
        checks.append(f"✅ Channels configured: {', '.join(channels_configured)}")
    else:
        issues.append("❌ NO CHANNELS CONFIGURED - alerts have nowhere to post!")
    
    # 6. Check min bet settings
    min_bets = []
    if settings.get("au_horse_min_matched"):
        min_bets.append(f"AU Horse: ${settings['au_horse_min_matched']:,}")
    if settings.get("uk_horse_min_matched"):
        min_bets.append(f"UK Horse: ${settings['uk_horse_min_matched']:,}")
    if settings.get("min_matched"):
        min_bets.append(f"Sports: ${settings['min_matched']:,}")
    if settings.get("greyhound_min_matched"):
        min_bets.append(f"Greyhound: ${settings['greyhound_min_matched']:,}")
    
    if min_bets:
        checks.append(f"📊 Min bet thresholds: {', '.join(min_bets)}")
    
    # 7. Check ML learning mode
    if ML_CONFIG.get("learning_mode"):
        checks.append("🧠 ML Learning mode: ON (observing all bets)")
    
    # 8. Check disconnect count
    if disconnect_count > 10:
        issues.append(f"⚠️ High disconnect count: {disconnect_count} - check network!")
    else:
        checks.append(f"📡 Disconnect count: {disconnect_count}")
    
    # Build embed
    if issues:
        color = discord.Color.red()
        title = "🔴 Issues Found"
    else:
        color = discord.Color.green()
        title = "🟢 All Systems OK"
    
    embed = discord.Embed(
        title=title,
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    
    if checks:
        embed.add_field(name="✅ Working", value="\n".join(checks), inline=False)
    if issues:
        embed.add_field(name="❌ Issues", value="\n".join(issues), inline=False)
    
    # Add quick stats
    stats = (
        f"• Tracked Bets: {len(tracked_bets)}\n"
        f"• ML Observed: {len(ml_learning_data.get('all_observed_bets', []))}\n"
        f"• Alerts Sent: {len(sent_alerts)}\n"
        f"• Guilds: {len(bot.guilds)}"
    )
    embed.add_field(name="📊 Stats", value=stats, inline=False)
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="cooldown_status", description="Check current cooldown status")
async def cooldown_status_command(interaction: discord.Interaction):
    # Defer immediately to prevent timeout
    await interaction.response.defer(ephemeral=True)
    
    # Check user authorization
    if interaction.user.id not in AUTHORIZED_USER_IDS:
        await interaction.followup.send(
            "❌ **Access Denied**\nYou are not authorized to use this bot.", 
            ephemeral=True
        )
        return
    
    cooldown_message, is_in_cooldown = check_cooldown_status()
    losses_count = len(loss_tracking['losses'])
    
    if is_in_cooldown:
        embed = discord.Embed(
            title="🛑 Cooldown Active",
            description=cooldown_message,
            color=discord.Color.red(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
    else:
        embed = discord.Embed(
            title="✅ No Cooldown",
            description=f"**Losses in last hour:** {losses_count}/3\n🟢 **Status:** Betting allowed",
            color=discord.Color.green(),
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
    
    await interaction.followup.send(embed=embed, ephemeral=True)

# --- Live Stream Monitoring Functions ---
live_stream_task = None
live_stream_running = False
scan_lock = asyncio.Lock()
SCAN_INTERVAL_SECONDS = 25  # Base scan cadence (seconds)
SCAN_INTERVAL_SLOW = 45     # Slower scan when no activity
live_stream_error_count = 0  # Track consecutive errors
MAX_STREAM_ERRORS = 10  # Max errors before forcing restart

async def start_live_stream_if_needed():
    global live_stream_task, live_stream_running
    alerts_needed = False
    configured_guilds = []
    
    for guild_id, settings in guild_settings.items():
        # Check if guild has necessary configuration
        has_config = False
        if settings.get("horse_alerts_running"):
            has_config = settings.get("au_horse_alert_channel") or settings.get("uk_horse_alert_channel")
        if settings.get("global_alerts_running"):
            sports_channels = settings.get("sports_channels", {})
            has_config = has_config or settings.get("alert_channel") or any(sports_channels.values())
        if settings.get("greyhound_alerts_running"):
            has_config = has_config or settings.get("greyhound_alert_channel")
        if settings.get("trots_alerts_running"):
            has_config = has_config or settings.get("trots_alert_channel")
            
        if has_config:
            alerts_needed = True
            configured_guilds.append(str(guild_id))
            
    if alerts_needed and not live_stream_running:
        # Start a single global monitor task if not already running
        if not live_stream_task or live_stream_task.done():
            live_stream_task = asyncio.create_task(live_stream_monitor(), name="global_live_stream_monitor")
            print(f"🚀 Started monitoring for configured guilds: {', '.join(configured_guilds)}")
        live_stream_running = True
    elif not alerts_needed:
        print("ℹ️ Not starting monitor - no guilds fully configured yet")

async def stop_live_stream_if_not_needed():
    global live_stream_task, live_stream_running
    alerts_needed = any(
        settings.get("horse_alerts_running") or settings.get("global_alerts_running") or settings.get("greyhound_alerts_running") or settings.get("trots_alerts_running")
        for settings in guild_settings.values()
    )
    if not alerts_needed and live_stream_running:
        if live_stream_task and not live_stream_task.done():
            live_stream_task.cancel()
            try:
                await live_stream_task
            except asyncio.CancelledError:
                pass
        live_stream_running = False
        live_stream_task = None
        print("🛑 Stopped global live stream monitoring")

async def live_stream_monitor():
    global live_stream_running, live_stream_error_count
    print("🔄 Live stream monitoring started (global)")
    live_stream_error_count = 0
    scan_count = 0  # Track number of scans for logging
    
    while live_stream_running:
        try:
            # Check if bot is still connected before processing
            if bot.is_closed() or not bot.is_ready():
                print("⚠️ Bot not ready, waiting...")
                await asyncio.sleep(30)
                continue
            
            session_token = get_session_token()
            if not session_token:
                print("❌ No session token, retrying in 120 seconds...")
                live_stream_error_count += 1
                await asyncio.sleep(120)
                continue
            
            # Ensure only one scan runs at a time globally
            async with scan_lock:
                bets_found = await monitor_markets(session_token)
                # Be robust if monitor_markets returns None
                if not isinstance(bets_found, int):
                    bets_found = 0
            
            # Reset error count on successful scan
            live_stream_error_count = 0
            scan_count += 1
            
            # Log status every 30 scans (about every hour with 120s interval)
            if scan_count % 30 == 0:
                print(f"📊 Live stream status: {scan_count} scans completed, still running")
            
            # Adaptive scanning - slower when no activity, with jitter to prevent thundering herd
            import random
            base_interval = SCAN_INTERVAL_SECONDS if bets_found > 0 else SCAN_INTERVAL_SLOW
            jitter = random.uniform(-10, 20)  # Add randomness to prevent synchronized API calls
            interval = max(30, base_interval + jitter)  # Minimum 30 seconds
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            print("🛑 Live stream monitoring cancelled")
            break
        except Exception as e:
            live_stream_error_count += 1
            print(f"❌ Error in live stream monitor ({live_stream_error_count}/{MAX_STREAM_ERRORS}): {e}")
            
            # If too many errors, restart the task
            if live_stream_error_count >= MAX_STREAM_ERRORS:
                print("🔄 Too many consecutive errors, pausing for 5 minutes before resuming...")
                live_stream_error_count = 0
                # Longer pause on persistent errors
                await asyncio.sleep(300)
            else:
                await asyncio.sleep(90)  # Longer wait on errors
    
    print("🛑 Live stream monitor loop ended")

async def monitor_markets(session_token):
    """Monitor markets for big bets - enhanced implementation with proper minimum bet respect"""
    try:
        # Get current time range (next 12 hours)
        now = datetime.datetime.now(datetime.timezone.utc)
        later = now + datetime.timedelta(hours=12)
        
        print(f"🔍 Monitoring markets from {now.strftime('%H:%M')} to {later.strftime('%H:%M')} UTC")
        
        # Market types to monitor based on active alerts AND configuration
        market_type_ids = set()
        active_guilds = 0
        sport_event_types = ensure_sport_event_type_cache(session_token)

        for guild_id, settings in guild_settings.items():
            guild_has_active_alerts = False

            if settings.get("horse_alerts_running") and (settings.get("au_horse_alert_channel") or settings.get("uk_horse_alert_channel")):
                market_type_ids.add("7")  # Horse racing
                guild_has_active_alerts = True

            if settings.get("greyhound_alerts_running") and settings.get("greyhound_alert_channel"):
                market_type_ids.add("4339")  # Greyhound racing  
                guild_has_active_alerts = True

            if settings.get("trots_alerts_running") and settings.get("trots_alert_channel"):
                market_type_ids.add("7")  # Harness racing (within horse racing)
                guild_has_active_alerts = True

            if settings.get("global_alerts_running"):
                sports_channels = settings.get("sports_channels", {})
                fallback_channel = settings.get("alert_channel")
                enabled_sports = []
                missing_sports = []
                for category in SPORT_CATEGORIES:
                    sport_key = category["key"]
                    # Check if this sport has a channel configured
                    if sports_channels:
                        channel_configured = sports_channels.get(sport_key)
                    else:
                        channel_configured = fallback_channel
                    if channel_configured:
                        event_type_id = sport_event_types.get(sport_key)
                        if event_type_id:
                            market_type_ids.add(str(event_type_id))
                            guild_has_active_alerts = True
                            enabled_sports.append(category["label"])
                        else:
                            missing_sports.append(category["label"])
                
                # Log once which sports are active vs missing (every 10 minutes)
                if enabled_sports and now.minute % 10 == 0:
                    print(f"✅ Sports enabled: {', '.join(enabled_sports[:5])}{'...' if len(enabled_sports) > 5 else ''}")
                if missing_sports and now.minute % 10 == 0:
                    print(f"⚠️ Sports not in Betfair cache: {', '.join(missing_sports)}")

            if guild_has_active_alerts:
                active_guilds += 1
        
        if not market_type_ids:
            print("⚠️ No alerts enabled, skipping market monitoring")
            return 0
            
        print(f"📊 Active alerts in {active_guilds} guilds, monitoring {len(market_type_ids)} market types")
        
        # Reduce frequency of minimum bet logging 
        current_minute = now.minute
        if current_minute % 5 == 0:  # Only log every 5 minutes
            for guild_id, settings in guild_settings.items():
                guild = bot.get_guild(guild_id)
                guild_name = guild.name if guild else f"Guild {guild_id}"
                sports_min = settings.get("sports_min_matched", {})
                if sports_min:
                    sport_parts = []
                    for key, amount in list(sports_min.items())[:4]:  # Limit log length
                        label = SPORT_CATEGORY_LOOKUP.get(key, {}).get("label", key.title())
                        sport_parts.append(f"{label}=${amount:,}")
                    if len(sports_min) > 4:
                        sport_parts.append("…")
                    sports_summary = ", ".join(sport_parts)
                else:
                    sports_summary = f"Default=${settings.get('min_matched', 1000):,}"

                print(
                    "💰 "
                    f"{guild_name} minimums: Horse=${settings.get('au_horse_min_matched', settings.get('uk_horse_min_matched', 1000)):,}, "
                    f"Sports={sports_summary}, "
                    f"Greyhound=${settings.get('greyhound_min_matched', 1000):,}, "
                    f"Trots=${settings.get('trots_min_matched', 1000):,}"
                )
        
        markets_checked = 0
        bets_found = 0
        
        # Get markets for each type
        for event_type_id in market_type_ids:
            try:
                market_catalogue_req = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/v1.0/listMarketCatalogue",
                    "params": {
                        "filter": {
                            "eventTypeIds": [event_type_id],
                            "marketTypeCodes": ["WIN", "MATCH_ODDS"],
                            "marketStartTime": {
                                "from": now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                "to": later.strftime('%Y-%m-%dT%H:%M:%SZ')
                            }
                        },
                        "sort": "FIRST_TO_START",
                        "maxResults": "25",  # Reduced from 50 to 25 to reduce load
                        "marketProjection": ["RUNNER_METADATA", "EVENT", "MARKET_START_TIME"]
                    },
                    "id": 1
                })
                
                catalogue_resp = call_betfair_api(market_catalogue_req, session_token)
                if not catalogue_resp or not catalogue_resp.get("result"):
                    # Enhanced debug info for market type lookup
                    sport_names = {
                        "1": "Soccer", "2": "Tennis", "3": "Cricket",
                        "7": "Horse Racing", "4339": "Greyhound Racing"
                    }
                    sport = sport_names.get(event_type_id, f"Sport {event_type_id}")
                    print(f"⚠️ No markets found for {sport} (event type {event_type_id})")
                    if not catalogue_resp:
                        print(f"❌ API call failed for {sport}")
                    elif not catalogue_resp.get("result"):
                        print(f"📝 Empty result list for {sport}")
                    continue
                    
                markets = catalogue_resp["result"]
                # Enhanced scanning message with sport name
                sport_names = {
                    "7": "Horse Racing", "4339": "Greyhound Racing",
                    **{sport_event_types.get(cat["key"]): cat["label"] for cat in SPORT_CATEGORIES if sport_event_types.get(cat["key"]) }
                }
                sport_betfair_names = {
                    sport_event_types.get(cat["key"]): cat["betfair_name"]
                    for cat in SPORT_CATEGORIES
                    if sport_event_types.get(cat["key"])
                }
                sport = sport_names.get(event_type_id, f"Sport {event_type_id}")
                if len(markets) >= 10:
                    print(f"🏁 Scanning {len(markets)} {sport} markets")
                else:
                    print(f"🔍 Found {len(markets)} {sport} markets")
                
                # Process markets
                market_type_map = {}
                market_event_map = {}
                market_location_map = {}
                market_start_time_map = {}
                runner_name_map = {}
                market_sport_map = {}
                
                for market in markets:
                    market_id = market["marketId"]
                    event = market.get("event", {})
                    event_name = event.get("name", "")
                    market_name = market.get("marketName", "")
                    market_start_time = market.get("marketStartTime", "")
                    if not market_start_time:
                        # Fallback to event openDate if present
                        market_start_time = event.get("openDate", "")
                    
                    # Enhanced bet type determination
                    if event_type_id == "7":
                        if any(keyword in event_name.lower() for keyword in ["harness", "trots", "standardbred"]):
                            bet_type = "trots"
                        else:
                            bet_type = "horse"
                    elif event_type_id == "4339":
                        bet_type = "greyhound"
                    else:
                        bet_type = "sports"
                        
                    market_type_map[market_id] = bet_type
                    market_event_map[market_id] = (event_name, market_name)
                    market_location_map[market_id] = (event.get("countryCode", ""), event.get("venue", ""))
                    market_start_time_map[market_id] = market_start_time

                    if bet_type == "sports":
                        sport_key = SPORT_EVENT_TYPE_ID_TO_KEY.get(str(event_type_id))
                        if not sport_key:
                            betfair_name = sport_betfair_names.get(event_type_id, "")
                            if not betfair_name:
                                # Also try string lookup
                                betfair_name = sport_betfair_names.get(str(event_type_id), "")
                            sport_key = SPORT_NAME_TO_KEY.get(betfair_name.lower()) if betfair_name else None
                        if sport_key:
                            market_sport_map[market_id] = sport_key
                        else:
                            # Log when we can't identify a sport for debugging
                            if markets_checked % 50 == 0:  # Limit log spam
                                print(f"⚠️ Unknown sport type {event_type_id} for market {market_id[:8]}...")
                    
                    # Store runner names
                    for runner in market.get("runners", []):
                        runner_name_map[(market_id, runner["selectionId"])] = runner.get("runnerName", "Unknown")
                    
                    # Check this market for big bets
                    markets_checked += 1
                    try:
                        found_bets = await check_market_for_big_bets(
                            market_id,
                            market_type_map,
                            market_event_map,
                            market_location_map,
                            market_start_time_map,
                            runner_name_map,
                            market_sport_map,
                            session_token
                        )
                        if found_bets:
                            bets_found += found_bets
                        
                        # Add small delay every 10 markets to prevent blocking
                        if markets_checked % 10 == 0:
                            await asyncio.sleep(0.1)
                            
                    except Exception as market_error:
                        print(f"⚠️ Error checking market {market_id[:8]}...: {str(market_error)[:50]}")
                        continue
                
            except Exception as e:
                print(f"❌ Error processing event type {event_type_id}: {e}")
                continue
                
        if bets_found > 0:
            print(f"🎯 Scan complete: {markets_checked} markets checked, {bets_found} big bets found and sent")
        else:
            print(f"✅ Scan complete: {markets_checked} markets checked, no qualifying bets found")
        return bets_found
                
    except Exception as e:
        print(f"❌ Error in monitor_markets: {e}")
        traceback.print_exc()
        return 0

# --- Daily Summary Report Task (6 PM Perth) ---
last_awst_report_date = None  # Tracks last AWST date a report was sent

def awst_now():
    return datetime.datetime.now(datetime.timezone.utc).astimezone(AWST_TZ)

def get_awst_window(end_awst=None):
    end_awst = (end_awst or awst_now()).astimezone(AWST_TZ)
    start_awst = end_awst - datetime.timedelta(hours=24)
    return start_awst, end_awst

def _gather_settled_bets_for_window(guild_id, start_awst, end_awst):
    start_awst = start_awst.astimezone(AWST_TZ)
    end_awst = end_awst.astimezone(AWST_TZ)
    settled = []
    for bet in tracked_bets:
        if bet.get("guild_id") != guild_id:
            continue
        if bet.get("status") not in ("won", "lost"):
            continue
        settled_at = bet.get("settled_at")
        settled_dt = _coerce_datetime(settled_at)
        if not settled_dt:
            continue
        if settled_dt.tzinfo is None:
            settled_dt = settled_dt.replace(tzinfo=datetime.timezone.utc)
        settled_awst = settled_dt.astimezone(AWST_TZ)
        if settled_awst < start_awst or settled_awst > end_awst:
            continue
        settled.append({**bet, "settled_awst": settled_awst})
    return settled

def _gather_all_settled_bets(guild_id):
    """Get all settled bets for a guild regardless of time"""
    settled = []
    for bet in tracked_bets:
        if bet.get("guild_id") != guild_id:
            continue
        if bet.get("status") not in ("won", "lost"):
            continue
        settled.append(bet)
    return settled

def _calculate_pnl_for_period(guild_id, days=None):
    """Calculate P&L for a specific period. None = all time"""
    now = datetime.datetime.now(datetime.timezone.utc)
    total_pnl = 0.0
    total_staked = 0.0
    won = 0
    lost = 0
    
    for bet in tracked_bets:
        if bet.get("guild_id") != guild_id:
            continue
        if bet.get("status") not in ("won", "lost"):
            continue
        
        settled_at = bet.get("settled_at")
        settled_dt = _coerce_datetime(settled_at)
        if not settled_dt:
            continue
        if settled_dt.tzinfo is None:
            settled_dt = settled_dt.replace(tzinfo=datetime.timezone.utc)
        
        # Check if within period
        if days is not None:
            cutoff = now - datetime.timedelta(days=days)
            if settled_dt < cutoff:
                continue
        
        pnl = bet.get("pnl") or 0
        stake = bet.get("recommended_units") or 0
        total_pnl += pnl
        total_staked += stake
        if bet.get("status") == "won":
            won += 1
        else:
            lost += 1
    
    return {
        "pnl": total_pnl,
        "staked": total_staked,
        "won": won,
        "lost": lost,
        "total": won + lost
    }

def _sport_label_and_emoji(sport_key, bet_type):
    if sport_key and sport_key in SPORT_CATEGORY_LOOKUP:
        meta = SPORT_CATEGORY_LOOKUP[sport_key]
        return meta.get("label", sport_key.title()), meta.get("emoji", "🏟️")
    mapping = {
        "horse": ("Horse Racing", "🐎"),
        "greyhound": ("Greyhound Racing", "🐕"),
        "trots": ("Harness Racing", "🚗"),
    }
    return mapping.get(bet_type, ("Sports", "🏟️"))

def build_report_embed(guild, start_awst, end_awst):
    bets = _gather_settled_bets_for_window(guild.id, start_awst, end_awst)
    window_text = f"{start_awst.strftime('%d %b %H:%M')}–{end_awst.strftime('%d %b %H:%M')} AWST"
    
    # Calculate stats for all periods
    daily_stats = _calculate_pnl_for_period(guild.id, days=1)
    weekly_stats = _calculate_pnl_for_period(guild.id, days=7)
    monthly_stats = _calculate_pnl_for_period(guild.id, days=30)
    yearly_stats = _calculate_pnl_for_period(guild.id, days=365)
    alltime_stats = _calculate_pnl_for_period(guild.id, days=None)
    
    # Determine color based on daily P&L
    if daily_stats["pnl"] > 0:
        embed_color = discord.Color.green()
    elif daily_stats["pnl"] < 0:
        embed_color = discord.Color.red()
    else:
        embed_color = discord.Color.blue()
    
    embed = discord.Embed(
        title="📊 Daily P&L Report",
        description=f"**{window_text}**",
        color=embed_color,
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    
    # Performance Summary with all periods
    def format_pnl(val):
        return f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
    
    def format_record(stats):
        if stats["total"] == 0:
            return "0-0"
        return f"{stats['won']}-{stats['lost']}"
    
    performance_text = (
        f"```\n"
        f"{'Period':<10} {'P&L':>8} {'Record':>8}\n"
        f"{'─'*28}\n"
        f"{'Today':<10} {format_pnl(daily_stats['pnl']):>7}u {format_record(daily_stats):>8}\n"
        f"{'Week':<10} {format_pnl(weekly_stats['pnl']):>7}u {format_record(weekly_stats):>8}\n"
        f"{'Month':<10} {format_pnl(monthly_stats['pnl']):>7}u {format_record(monthly_stats):>8}\n"
        f"{'Year':<10} {format_pnl(yearly_stats['pnl']):>7}u {format_record(yearly_stats):>8}\n"
        f"{'All Time':<10} {format_pnl(alltime_stats['pnl']):>7}u {format_record(alltime_stats):>8}\n"
        f"```"
    )
    
    embed.add_field(
        name="📈 Performance Summary",
        value=performance_text,
        inline=False
    )

    if not bets:
        embed.add_field(name="Today's Bets", value="No settled bets in this window.", inline=False)
    else:
        grouped = {}
        for bet in bets:
            key = bet.get("sport_key") or bet.get("bet_type") or "sports"
            grouped.setdefault(key, []).append(bet)

        for key, items in grouped.items():
            label, emoji_icon = _sport_label_and_emoji(key if key != "sports" else items[0].get("sport_key"), items[0].get("bet_type"))
            won = [b for b in items if b.get("status") == "won"]
            lost = [b for b in items if b.get("status") == "lost"]
            units_staked = sum((b.get("recommended_units") or 0) for b in items)
            pnl = sum((b.get("pnl") or 0) for b in items)

            lines = []
            for bet in sorted(items, key=lambda b: b.get("settled_awst") or awst_now())[:8]:
                status_icon = "✅" if bet.get("status") == "won" else "❌"
                odds_val = bet.get("odds") or 0
                stake = bet.get("recommended_units") or 0
                pnl_val = bet.get("pnl") or 0
                runner = bet.get("runner_name", "Selection")[:20]
                lines.append(f"{status_icon} {runner} @ {odds_val:.2f} • {pnl_val:+.1f}u")

            summary = (
                f"**{len(won)}W-{len(lost)}L** | Staked: {units_staked:.1f}u | P/L: {pnl:+.1f}u\n"
                + "\n".join(lines)
            )
            embed.add_field(name=f"{emoji_icon} {label}", value=summary[:1024], inline=False)

    embed.set_footer(text="🕕 Report generated at 6 PM AWST • Settled bets only")
    return embed

async def send_report_for_guild(guild_id, end_awst=None, manual=False):
    guild = bot.get_guild(guild_id)
    if not guild:
        return False
    settings = guild_settings.get(guild_id, {})
    channel_id = settings.get("report_channel") or settings.get("alert_channel")
    if not channel_id:
        return False
    channel = bot.get_channel(channel_id)
    if not channel:
        return False

    if end_awst is None:
        now_awst = awst_now()
        end_awst = now_awst.replace(hour=18, minute=0, second=0, microsecond=0)
        if now_awst < end_awst:
            end_awst = now_awst

    start_awst, end_awst = get_awst_window(end_awst)
    embed = build_report_embed(guild, start_awst, end_awst)
    await channel.send(embed=embed)
    print(f"📊 Sent P&L report to {guild.name} #{channel.name} (manual={manual})")
    return True

@tasks.loop(minutes=5)
async def daily_summary_report():
    """Send daily P&L summary at 6 PM Perth time (AWST)."""
    global last_awst_report_date
    try:
        now_awst = awst_now()
        target_end = now_awst.replace(hour=18, minute=0, second=0, microsecond=0)
        if now_awst < target_end:
            return

        if last_awst_report_date == target_end.date():
            return

        sent_any = False
        for guild_id in list(guild_settings.keys()):
            try:
                sent = await send_report_for_guild(guild_id, end_awst=target_end)
                sent_any = sent_any or sent
            except Exception as send_error:
                print(f"❌ Error sending 6 PM report for guild {guild_id}: {send_error}")
                continue

        if sent_any:
            last_awst_report_date = target_end.date()
            print(f"📊 6 PM AWST reports sent for {last_awst_report_date}")
    except Exception as e:
        print(f"❌ Error in daily summary report: {e}")
        traceback.print_exc()

@daily_summary_report.before_loop
async def before_daily_summary():
    await bot.wait_until_ready()
    print("📊 Daily summary report task (6 PM AWST) started")

# Run the bot with rate limiting protection and auto-reconnect
if __name__ == "__main__":
    async def main_with_reconnect():
        """Main loop with automatic reconnection on disconnect"""
        global bot_start_time, last_gateway_refresh, live_stream_running, live_stream_task
        
        reconnect_delay = 5  # Start with 5 seconds
        max_reconnect_delay = 300  # Max 5 minutes between attempts
        consecutive_failures = 0
        total_restarts = 0
        
        while True:
            try:
                total_restarts += 1
                print(f"🚀 Starting Discord bot... (restart #{total_restarts})")
                
                # Reset live stream state before reconnecting
                live_stream_running = False
                live_stream_task = None
                
                # Try safe login first
                login_success = await safe_bot_login(bot, BOT_TOKEN)
                if not login_success:
                    print("❌ Could not login to Discord - check token or try again later")
                    consecutive_failures += 1
                    wait_time = min(reconnect_delay * (2 ** consecutive_failures), max_reconnect_delay)
                    print(f"⏱️ Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Reset failure counter on successful login
                consecutive_failures = 0
                reconnect_delay = 5
                
                # Update tracking
                bot_start_time = datetime.datetime.now(datetime.timezone.utc)
                last_gateway_refresh = bot_start_time
                
                print(f"✅ Bot logged in, connecting to gateway...")
                
                # Connect with reconnect=True for automatic reconnection
                await bot.connect(reconnect=True)
                
            except discord.ConnectionClosed as e:
                print(f"⚠️ Discord connection closed (code {e.code}): {e.reason}")
                consecutive_failures += 1
                wait_time = min(reconnect_delay * (2 ** consecutive_failures), max_reconnect_delay)
                print(f"🔄 Reconnecting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
            except discord.GatewayNotFound:
                print("❌ Discord Gateway not found - Discord may be having issues")
                await asyncio.sleep(60)
                
            except discord.HTTPException as e:
                if e.status == 429:  # Rate limited
                    retry_after = getattr(e, 'retry_after', 60)
                    print(f"⏱️ Rate limited! Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                else:
                    print(f"❌ HTTP error: {e}")
                    await asyncio.sleep(30)
                    
            except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                traceback.print_exc()
                consecutive_failures += 1
                wait_time = min(reconnect_delay * (2 ** consecutive_failures), max_reconnect_delay)
                print(f"🔄 Restarting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            
            finally:
                # Stop live stream before reconnecting
                live_stream_running = False
                if live_stream_task and not live_stream_task.done():
                    live_stream_task.cancel()
                    try:
                        await asyncio.wait_for(live_stream_task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                live_stream_task = None
                
                # Clean up before reconnecting
                if not bot.is_closed():
                    try:
                        await bot.close()
                    except:
                        pass
                
                print(f"🔄 Cleanup complete, preparing for reconnection...")

# =============================================================================
# WEB DASHBOARD API (FastAPI)
# =============================================================================

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    api = FastAPI(
        title="BetTracker Dashboard API",
        description="API for controlling and monitoring the BetTracker Discord bot",
        version="1.0.0"
    )

    # CORS - Allow frontend to connect from anywhere
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Models
    class SettingsUpdate(BaseModel):
        min_matched: Optional[int] = None
        horse_min_matched: Optional[int] = None
        greyhound_min_matched: Optional[int] = None
        sports_min_matched: Optional[Dict[str, int]] = None

    class ScanControl(BaseModel):
        action: str  # "start" or "stop"

    class WebhookTest(BaseModel):
        webhook_url: str

    # API Endpoints
    @api.get("/")
    async def api_root():
        return {"status": "ok", "message": "BetTracker Dashboard API", "bot_ready": bot.is_ready() if bot else False}

    @api.get("/api/health")
    async def api_health():
        return {"status": "healthy", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    @api.get("/api/bot/status")
    async def api_bot_status():
        """Get current bot connection status"""
        global disconnect_count, last_disconnect_time, live_stream_running, bot_start_time, ml_learning_data
        
        is_connected = bot.is_ready() if bot else False
        latency = bot.latency * 1000 if bot and is_connected else 0
        guilds = []
        
        if is_connected:
            guilds = [{"id": g.id, "name": g.name, "member_count": g.member_count} for g in bot.guilds]
        
        uptime_seconds = 0
        if bot_start_time:
            uptime_seconds = (datetime.datetime.now(datetime.timezone.utc) - bot_start_time).total_seconds()
        
        # Get ML stats (use all_observed_bets - kept in sync with observed_bets for compatibility)
        observed_bets = len(ml_learning_data.get("all_observed_bets", []))
        
        return {
            "status": "ok",
            "bot": {
                "connected": is_connected,
                "ready": is_connected,
                "latency_ms": round(latency, 2),
                "guilds": guilds,
                "guild_count": len(guilds),
                "start_time": bot_start_time.isoformat() if bot_start_time else None,
                "uptime_seconds": uptime_seconds,
                "disconnect_count": disconnect_count,
                "last_disconnect": last_disconnect_time.isoformat() if last_disconnect_time else None,
                "scan_running": live_stream_running,
                "ml_observed_bets": observed_bets,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    @api.get("/api/bot/guilds")
    async def api_bot_guilds():
        """Get list of guilds the bot is in"""
        if not bot or not bot.is_ready():
            return {"guilds": [], "count": 0}
        guilds = [{"id": g.id, "name": g.name, "member_count": g.member_count} for g in bot.guilds]
        return {"guilds": guilds, "count": len(guilds)}

    @api.get("/api/guild/{guild_id}")
    async def api_get_guild_details(guild_id: int):
        """Get detailed information about a specific guild"""
        if not bot or not bot.is_ready():
            raise HTTPException(status_code=503, detail="Bot not connected")
        
        guild = bot.get_guild(guild_id)
        if not guild:
            raise HTTPException(status_code=404, detail="Guild not found")
        
        settings = guild_settings.get(guild_id, {})
        
        # Get channels for the guild
        text_channels = [{"id": c.id, "name": c.name, "category": c.category.name if c.category else None} 
                        for c in guild.text_channels]
        
        return {
            "guild": {
                "id": guild.id,
                "name": guild.name,
                "member_count": guild.member_count,
                "icon_url": str(guild.icon.url) if guild.icon else None,
            },
            "settings": {
                # Alert toggles
                "horse_alerts_running": settings.get("horse_alerts_running", False),
                "greyhound_alerts_running": settings.get("greyhound_alerts_running", False),
                "trots_alerts_running": settings.get("trots_alerts_running", False),
                "global_alerts_running": settings.get("global_alerts_running", False),
                
                # Minimum matched thresholds
                "min_matched": settings.get("min_matched", 1000),
                "au_horse_min_matched": settings.get("au_horse_min_matched", 1000),
                "uk_horse_min_matched": settings.get("uk_horse_min_matched", 1000),
                "greyhound_min_matched": settings.get("greyhound_min_matched", 1000),
                "trots_min_matched": settings.get("trots_min_matched", 1000),
                "sports_min_matched": settings.get("sports_min_matched", 1000),
                
                # Alert channels
                "horse_alert_channel": settings.get("horse_alert_channel"),
                "au_horse_alert_channel": settings.get("au_horse_alert_channel"),
                "uk_horse_alert_channel": settings.get("uk_horse_alert_channel"),
                "greyhound_alert_channel": settings.get("greyhound_alert_channel"),
                "trots_alert_channel": settings.get("trots_alert_channel"),
                "alert_channel": settings.get("alert_channel"),
                "report_channel": settings.get("report_channel"),
                
                # Sports-specific channels
                "sports_channels": settings.get("sports_channels", {}),
                
                # Big bets settings
                "bigbets_enabled": settings.get("bigbets_enabled", False),
                "bigbets_threshold": settings.get("bigbets_threshold", 10000),
                "bigbets_channel": settings.get("bigbets_channel"),
                
                # ML Learning
                "ml_learning_enabled": settings.get("ml_learning_enabled", True),
            },
            "channels": text_channels
        }

    @api.post("/api/guild/{guild_id}/settings")
    async def api_update_guild_full_settings(guild_id: int, settings: dict):
        """Update all settings for a guild"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        # Update all provided settings
        allowed_keys = [
            # Alert toggles
            "horse_alerts_running", "greyhound_alerts_running", 
            "trots_alerts_running", "global_alerts_running",
            # Thresholds
            "min_matched", "au_horse_min_matched", "uk_horse_min_matched",
            "greyhound_min_matched", "trots_min_matched", "sports_min_matched",
            # Channels
            "horse_alert_channel", "au_horse_alert_channel", "uk_horse_alert_channel",
            "greyhound_alert_channel", "trots_alert_channel", "alert_channel", "report_channel",
            # Big bets
            "bigbets_enabled", "bigbets_threshold", "bigbets_channel",
            # ML
            "ml_learning_enabled",
        ]
        
        for key in allowed_keys:
            if key in settings:
                guild_settings[guild_id][key] = settings[key]
        
        # Handle sports_channels separately (it's a dict)
        if "sports_channels" in settings:
            guild_settings[guild_id]["sports_channels"] = settings["sports_channels"]
        
        # Save settings
        save_guild_settings()
        
        return {"status": "ok", "guild_id": guild_id, "message": "Settings updated"}

    @api.post("/api/guild/{guild_id}/alerts/{alert_type}")
    async def api_toggle_guild_alert(guild_id: int, alert_type: str, enabled: bool = True):
        """Toggle a specific alert type for a guild"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        alert_key_map = {
            "horse": "horse_alerts_running",
            "greyhound": "greyhound_alerts_running",
            "trots": "trots_alerts_running",
            "sports": "global_alerts_running",
            "global": "global_alerts_running",
        }
        
        alert_key = alert_key_map.get(alert_type)
        if not alert_key:
            raise HTTPException(status_code=400, detail=f"Invalid alert type: {alert_type}")
        
        guild_settings[guild_id][alert_key] = enabled
        save_guild_settings()
        
        # Start/stop scanning if needed
        if enabled:
            await start_live_stream_if_needed()
        
        return {"status": "ok", "alert_type": alert_type, "enabled": enabled}

    @api.post("/api/guild/{guild_id}/channel/{channel_type}")
    async def api_set_guild_channel(guild_id: int, channel_type: str, channel_id: int):
        """Set a channel for a specific alert type"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        channel_key_map = {
            "horse": "horse_alert_channel",
            "au_horse": "au_horse_alert_channel",
            "uk_horse": "uk_horse_alert_channel",
            "greyhound": "greyhound_alert_channel",
            "trots": "trots_alert_channel",
            "sports": "alert_channel",
            "report": "report_channel",
            "bigbets": "bigbets_channel",
        }
        
        channel_key = channel_key_map.get(channel_type)
        if not channel_key:
            raise HTTPException(status_code=400, detail=f"Invalid channel type: {channel_type}")
        
        guild_settings[guild_id][channel_key] = channel_id
        save_guild_settings()
        
        return {"status": "ok", "channel_type": channel_type, "channel_id": channel_id}

    @api.post("/api/guild/{guild_id}/threshold/{threshold_type}")
    async def api_set_guild_threshold(guild_id: int, threshold_type: str, value: int):
        """Set minimum matched threshold for a bet type"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        threshold_key_map = {
            "default": "min_matched",
            "au_horse": "au_horse_min_matched",
            "uk_horse": "uk_horse_min_matched",
            "greyhound": "greyhound_min_matched",
            "trots": "trots_min_matched",
            "sports": "sports_min_matched",
        }
        
        threshold_key = threshold_key_map.get(threshold_type)
        if not threshold_key:
            raise HTTPException(status_code=400, detail=f"Invalid threshold type: {threshold_type}")
        
        guild_settings[guild_id][threshold_key] = value
        save_guild_settings()
        
        return {"status": "ok", "threshold_type": threshold_type, "value": value}

    @api.post("/api/guild/{guild_id}/bigbets")
    async def api_configure_bigbets(guild_id: int, enabled: bool = True, threshold: int = 10000, channel_id: int = None):
        """Configure big bets alerts for a guild"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        guild_settings[guild_id]["bigbets_enabled"] = enabled
        guild_settings[guild_id]["bigbets_threshold"] = threshold
        if channel_id:
            guild_settings[guild_id]["bigbets_channel"] = channel_id
        
        save_guild_settings()
        
        return {"status": "ok", "bigbets_enabled": enabled, "threshold": threshold}

    @api.get("/api/settings")
    async def api_get_settings():
        """Get current bot settings"""
        return {
            "settings": {
                "guild_settings": {str(k): v for k, v in guild_settings.items()},
                "ml_config": ML_CONFIG,
                "scan_interval": SCAN_INTERVAL_SECONDS,
                "scan_interval_slow": SCAN_INTERVAL_SLOW,
            }
        }

    @api.post("/api/settings/guild/{guild_id}")
    async def api_update_guild_settings(guild_id: int, settings: dict):
        """Update guild-specific settings"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        # Update allowed settings
        allowed_keys = [
            "min_matched", "au_horse_min_matched", "uk_horse_min_matched",
            "greyhound_min_matched", "trots_min_matched", "sports_min_matched",
            "horse_alerts_running", "global_alerts_running", 
            "greyhound_alerts_running", "trots_alerts_running"
        ]
        
        for key in allowed_keys:
            if key in settings:
                guild_settings[guild_id][key] = settings[key]
        
        # Save settings
        save_guild_settings()
        
        return {"status": "ok", "guild_id": guild_id, "settings": guild_settings[guild_id]}

    @api.get("/api/ml/patterns")
    async def api_ml_patterns():
        """Get ML learning patterns and statistics"""
        global ml_learning_data, ML_CONFIG
        
        observed = ml_learning_data.get("observed_bets", [])
        total_bets = len(observed)
        settled_bets = len([b for b in observed if b.get("result") in ["won", "lost"]])
        won_bets = len([b for b in observed if b.get("result") == "won"])
        lost_bets = len([b for b in observed if b.get("result") == "lost"])
        
        # Get profitable patterns if any exist (can be dict or list)
        patterns_data = ml_learning_data.get("profitable_patterns", {})
        # Convert to list if it's a dict
        if isinstance(patterns_data, dict):
            patterns = list(patterns_data.values())
        else:
            patterns = patterns_data if patterns_data else []
        
        # Calculate win rate if we have settled bets
        win_rate = (won_bets / settled_bets * 100) if settled_bets > 0 else 0
        
        return {
            "total_bets": total_bets,
            "settled_bets": settled_bets,
            "pending_bets": total_bets - settled_bets,
            "won_bets": won_bets,
            "lost_bets": lost_bets,
            "win_rate": round(win_rate, 2),
            "patterns_found": len(patterns),
            "patterns": patterns[:50],  # Return top 50 patterns
            "last_analysis": ml_learning_data.get("last_analysis"),
            "config": ML_CONFIG  # Include ML config for settings page
        }

    @api.get("/api/ml/pattern/{pattern_id}")
    async def api_ml_pattern_detail(pattern_id: str):
        """Get detailed information about a specific pattern including matching bets"""
        global ml_learning_data
        
        # URL decode the pattern_id
        import urllib.parse
        pattern_key = urllib.parse.unquote(pattern_id)
        
        # Get pattern from profitable_patterns
        patterns_data = ml_learning_data.get("profitable_patterns", {})
        pattern = patterns_data.get(pattern_key)
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Get all bets matching this pattern
        all_bets = ml_learning_data.get("all_observed_bets", [])
        
        # Parse pattern to get filter criteria
        # Pattern format: "key=value" or "combo:key1=value1+key2=value2"
        matching_bets = []
        
        if pattern_key.startswith("combo:"):
            # Handle combo patterns like "combo:bet_type=sports+odds_bucket=short"
            parts = pattern_key[6:].split("+")
            filters = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    filters[k] = v
            
            for bet in all_bets:
                matches = True
                for k, v in filters.items():
                    bet_value = str(bet.get(k, ""))
                    if bet_value != v:
                        matches = False
                        break
                if matches:
                    matching_bets.append(bet)
        else:
            # Simple pattern like "bet_type=sports"
            if "=" in pattern_key:
                key, value = pattern_key.split("=", 1)
                matching_bets = [b for b in all_bets if str(b.get(key, "")) == value]
        
        # Separate settled and pending
        settled_bets = [b for b in matching_bets if b.get("status") in ["won", "lost"]]
        pending_bets = [b for b in matching_bets if b.get("status") == "pending"]
        
        # Calculate P&L over time (by day)
        pnl_by_day = defaultdict(lambda: {"pnl": 0, "bets": 0, "wins": 0, "losses": 0})
        for bet in settled_bets:
            ts = bet.get("timestamp")
            if isinstance(ts, datetime.datetime):
                day = ts.strftime("%Y-%m-%d")
            elif isinstance(ts, str):
                day = ts[:10]
            else:
                continue
            
            pnl_by_day[day]["pnl"] += bet.get("pnl", 0)
            pnl_by_day[day]["bets"] += 1
            if bet.get("status") == "won":
                pnl_by_day[day]["wins"] += 1
            else:
                pnl_by_day[day]["losses"] += 1
        
        # Convert to sorted list with cumulative P&L
        pnl_timeline = []
        cumulative_pnl = 0
        for day in sorted(pnl_by_day.keys()):
            cumulative_pnl += pnl_by_day[day]["pnl"]
            pnl_timeline.append({
                "date": day,
                "daily_pnl": round(pnl_by_day[day]["pnl"], 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "bets": pnl_by_day[day]["bets"],
                "wins": pnl_by_day[day]["wins"],
                "losses": pnl_by_day[day]["losses"]
            })
        
        # Breakdown by sport (for sports bets)
        sports_breakdown = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pnl": 0})
        for bet in settled_bets:
            sport = bet.get("sport_key") or bet.get("bet_type") or "unknown"
            sports_breakdown[sport]["bets"] += 1
            sports_breakdown[sport]["pnl"] += bet.get("pnl", 0)
            if bet.get("status") == "won":
                sports_breakdown[sport]["wins"] += 1
            else:
                sports_breakdown[sport]["losses"] += 1
        
        # Convert to list
        sports_data = []
        for sport, data in sports_breakdown.items():
            win_rate = (data["wins"] / data["bets"] * 100) if data["bets"] > 0 else 0
            sports_data.append({
                "sport": sport,
                "bets": data["bets"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(win_rate, 1),
                "pnl": round(data["pnl"], 2)
            })
        sports_data.sort(key=lambda x: x["bets"], reverse=True)
        
        # Breakdown by odds bucket
        odds_breakdown = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pnl": 0})
        for bet in settled_bets:
            odds_bucket = bet.get("odds_bucket", "unknown")
            odds_breakdown[odds_bucket]["bets"] += 1
            odds_breakdown[odds_bucket]["pnl"] += bet.get("pnl", 0)
            if bet.get("status") == "won":
                odds_breakdown[odds_bucket]["wins"] += 1
            else:
                odds_breakdown[odds_bucket]["losses"] += 1
        
        odds_data = []
        for bucket, data in odds_breakdown.items():
            win_rate = (data["wins"] / data["bets"] * 100) if data["bets"] > 0 else 0
            odds_data.append({
                "bucket": bucket,
                "bets": data["bets"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(win_rate, 1),
                "pnl": round(data["pnl"], 2)
            })
        
        # Breakdown by hour
        hour_breakdown = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pnl": 0})
        for bet in settled_bets:
            hour = bet.get("hour_of_day", 12)
            hour_breakdown[hour]["bets"] += 1
            hour_breakdown[hour]["pnl"] += bet.get("pnl", 0)
            if bet.get("status") == "won":
                hour_breakdown[hour]["wins"] += 1
            else:
                hour_breakdown[hour]["losses"] += 1
        
        hour_data = []
        for h in range(24):
            data = hour_breakdown[h]
            win_rate = (data["wins"] / data["bets"] * 100) if data["bets"] > 0 else 0
            hour_data.append({
                "hour": h,
                "bets": data["bets"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(win_rate, 1),
                "pnl": round(data["pnl"], 2)
            })
        
        # Recent bets (last 50)
        recent_bets = []
        for bet in sorted(matching_bets, key=lambda x: str(x.get("timestamp", "")), reverse=True)[:50]:
            ts = bet.get("timestamp")
            if isinstance(ts, datetime.datetime):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            
            recent_bets.append({
                "runner": bet.get("runner_name", ""),
                "event": bet.get("event_name", ""),
                "odds": bet.get("odds", 0),
                "amount": bet.get("bet_amount", 0),
                "status": bet.get("status", "pending"),
                "pnl": bet.get("pnl", 0),
                "sport": bet.get("sport_key") or bet.get("bet_type", ""),
                "timestamp": ts_str
            })
        
        return {
            "pattern": pattern,
            "pattern_key": pattern_key,
            "total_bets": len(matching_bets),
            "settled_bets": len(settled_bets),
            "pending_bets": len(pending_bets),
            "pnl_timeline": pnl_timeline,
            "sports_breakdown": sports_data,
            "odds_breakdown": odds_data,
            "hour_breakdown": hour_data,
            "recent_bets": recent_bets
        }

    @api.get("/api/ml/summary")
    async def api_ml_summary():
        """Get ML learning summary with breakdowns for dashboard"""
        global ml_learning_data
        
        all_bets = ml_learning_data.get("all_observed_bets", [])
        settled = [b for b in all_bets if b.get("status") in ["won", "lost"]]
        
        # Overall stats
        wins = len([b for b in settled if b["status"] == "won"])
        losses = len([b for b in settled if b["status"] == "lost"])
        total_pnl = sum(b.get("pnl", 0) for b in settled)
        win_rate = (wins / len(settled) * 100) if settled else 0
        
        # By bet type
        by_type = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pnl": 0, "pending": 0})
        for bet in all_bets:
            bt = bet.get("bet_type", "unknown")
            if bet.get("status") == "pending":
                by_type[bt]["pending"] += 1
            elif bet.get("status") == "won":
                by_type[bt]["bets"] += 1
                by_type[bt]["wins"] += 1
                by_type[bt]["pnl"] += bet.get("pnl", 0)
            elif bet.get("status") == "lost":
                by_type[bt]["bets"] += 1
                by_type[bt]["losses"] += 1
                by_type[bt]["pnl"] += bet.get("pnl", 0)
        
        type_data = []
        for bt, data in by_type.items():
            wr = (data["wins"] / data["bets"] * 100) if data["bets"] > 0 else 0
            type_data.append({
                "type": bt,
                "settled": data["bets"],
                "pending": data["pending"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(wr, 1),
                "pnl": round(data["pnl"], 2)
            })
        
        # By sport/bet_type (include all types with sport_key OR use bet_type as fallback)
        by_sport = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pnl": 0})
        for bet in settled:
            # Use sport_key if available, otherwise use bet_type as the category
            sport = bet.get("sport_key") or bet.get("bet_type", "unknown")
            by_sport[sport]["bets"] += 1
            by_sport[sport]["pnl"] += bet.get("pnl", 0)
            if bet.get("status") == "won":
                by_sport[sport]["wins"] += 1
            else:
                by_sport[sport]["losses"] += 1
        
        sport_data = []
        for sport, data in by_sport.items():
            wr = (data["wins"] / data["bets"] * 100) if data["bets"] > 0 else 0
            sport_data.append({
                "sport": sport,
                "bets": data["bets"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(wr, 1),
                "pnl": round(data["pnl"], 2)
            })
        sport_data.sort(key=lambda x: x["bets"], reverse=True)
        
        # P&L by day
        pnl_by_day = defaultdict(lambda: {"pnl": 0, "bets": 0})
        for bet in settled:
            ts = bet.get("timestamp")
            if isinstance(ts, datetime.datetime):
                day = ts.strftime("%Y-%m-%d")
            elif isinstance(ts, str):
                day = ts[:10]
            else:
                continue
            pnl_by_day[day]["pnl"] += bet.get("pnl", 0)
            pnl_by_day[day]["bets"] += 1
        
        pnl_timeline = []
        cumulative = 0
        for day in sorted(pnl_by_day.keys()):
            cumulative += pnl_by_day[day]["pnl"]
            pnl_timeline.append({
                "date": day,
                "daily_pnl": round(pnl_by_day[day]["pnl"], 2),
                "cumulative_pnl": round(cumulative, 2),
                "bets": pnl_by_day[day]["bets"]
            })
        
        return {
            "total_observed": len(all_bets),
            "total_settled": len(settled),
            "total_pending": len(all_bets) - len(settled),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "by_type": type_data,
            "by_sport": sport_data,
            "pnl_timeline": pnl_timeline
        }

    @api.get("/api/ml/bets")
    async def api_ml_bets(limit: int = 100, result: str = None):
        """Get ML observed bets with optional filtering"""
        global ml_learning_data
        
        observed = ml_learning_data.get("all_observed_bets", [])
        
        # Filter by status if specified (use 'result' param for backwards compat)
        if result:
            observed = [b for b in observed if b.get("status") == result]
        
        # Sort by timestamp descending
        sorted_bets = sorted(observed, key=lambda x: str(x.get("timestamp", "")), reverse=True)
        
        # Format for API
        formatted = []
        for bet in sorted_bets[:limit]:
            formatted.append({
                "id": bet.get("market_id", ""),
                "runner": bet.get("runner_name", ""),
                "event": bet.get("event_name", ""),
                "odds": bet.get("odds", 0),
                "stake": bet.get("bet_amount", 0),
                "bet_type": bet.get("bet_type", ""),
                "result": bet.get("status", "pending"),
                "timestamp": str(bet.get("timestamp", ""))
            })
        
        return {
            "bets": formatted,
            "total": len(ml_learning_data.get("all_observed_bets", [])),
            "showing": len(formatted)
        }

    @api.get("/api/alerts")
    async def api_get_alerts(limit: int = 50):
        """Get recent alerts from sent_alerts"""
        global sent_alerts
        
        # Get recent alerts
        recent = sent_alerts[-limit:] if sent_alerts else []
        
        # Format for API
        formatted = []
        for alert in reversed(recent):
            formatted.append({
                "id": alert.get("market_id", ""),
                "type": alert.get("bet_type", "unknown"),
                "runner": alert.get("runner_name", ""),
                "event": alert.get("event_name", ""),
                "odds": alert.get("odds", 0),
                "amount": alert.get("amount", 0),
                "timestamp": alert.get("timestamp").isoformat() if isinstance(alert.get("timestamp"), datetime.datetime) else str(alert.get("timestamp", "")),
                "result": alert.get("result", "pending")
            })
        
        return {"alerts": formatted, "total": len(sent_alerts)}

    @api.get("/api/bets")
    async def api_get_bets(limit: int = 100):
        """Get tracked bets"""
        global tracked_bets
        
        # Sort by timestamp descending
        sorted_bets = sorted(tracked_bets, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Format for API
        formatted = []
        for bet in sorted_bets[:limit]:
            formatted.append({
                "id": bet.get("market_id", ""),
                "runner": bet.get("runner_name", ""),
                "event": bet.get("event_name", ""),
                "odds": bet.get("odds", 0),
                "stake": bet.get("amount", 0),
                "result": bet.get("result", "pending"),
                "timestamp": bet.get("timestamp").isoformat() if isinstance(bet.get("timestamp"), datetime.datetime) else str(bet.get("timestamp", ""))
            })
        
        won = len([b for b in tracked_bets if b.get("result") == "won"])
        lost = len([b for b in tracked_bets if b.get("result") == "lost"])
        pending = len([b for b in tracked_bets if b.get("result") == "pending"])
        
        return {
            "bets": formatted,
            "total": len(tracked_bets),
            "won": won,
            "lost": lost,
            "pending": pending
        }

    @api.post("/api/scan/control")
    async def api_scan_control(control: ScanControl):
        """Start or stop market scanning"""
        global live_stream_running, live_stream_task
        
        if control.action == "start":
            if not live_stream_running:
                live_stream_running = True
                live_stream_task = asyncio.create_task(live_stream_monitor(), name="global_live_stream_monitor")
            return {"status": "ok", "message": "Scanning started", "scan_running": True}
        elif control.action == "stop":
            live_stream_running = False
            if live_stream_task and not live_stream_task.done():
                live_stream_task.cancel()
            return {"status": "ok", "message": "Scanning stopped", "scan_running": False}
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'start' or 'stop'")

    @api.post("/api/alerts/toggle/{alert_type}")
    async def api_toggle_alerts(alert_type: str, guild_id: int, enabled: bool):
        """Toggle alerts on/off for a guild"""
        global guild_settings
        
        if guild_id not in guild_settings:
            guild_settings[guild_id] = {}
        
        alert_key = f"{alert_type}_alerts_running"
        if alert_key in ["horse_alerts_running", "global_alerts_running", "greyhound_alerts_running", "trots_alerts_running"]:
            guild_settings[guild_id][alert_key] = enabled
            save_guild_settings()
            
            # Start/stop live stream if needed
            if enabled:
                await start_live_stream_if_needed()
            
            return {"status": "ok", "alert_type": alert_type, "enabled": enabled}
        else:
            raise HTTPException(status_code=400, detail=f"Invalid alert type: {alert_type}")

    @api.post("/api/webhook/test")
    async def api_webhook_test(webhook: WebhookTest):
        """Test a Discord webhook"""
        try:
            response = requests.post(
                webhook.webhook_url,
                json={
                    "content": "🧪 **Test Alert from BetTracker Dashboard**\nWebhook is working correctly!",
                    "username": "BetTracker"
                },
                timeout=10
            )
            if response.status_code == 204:
                return {"status": "ok", "message": "Webhook test successful"}
            else:
                return {"status": "error", "message": f"Webhook returned status {response.status_code}"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Webhook test failed: {str(e)}")

    @api.post("/api/ml/analyze")
    async def api_trigger_ml_analysis():
        """Manually trigger ML pattern analysis"""
        try:
            # First, settle bets from Betfair directly
            settled_count = await settle_ml_bets_from_betfair()
            
            # Also sync from tracked_bets
            await update_ml_bet_results()
            
            # Run ML analysis
            await run_ml_analysis()
            save_ml_data()
            
            patterns = ml_learning_data.get("profitable_patterns", [])
            
            return {
                "status": "ok",
                "message": f"ML analysis completed, settled {settled_count} bets from Betfair",
                "patterns_found": len(patterns) if patterns else 0,
                "bets_settled": settled_count
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

    @api.get("/api/stats/summary")
    async def api_stats_summary():
        """Get overall statistics summary"""
        global tracked_bets, sent_alerts, ml_learning_data, disconnect_count, bot_start_time, live_stream_running
        
        # Get ML observed bets (prefer all_observed_bets which stores ALL observations)
        observed = ml_learning_data.get("all_observed_bets", [])
        # Note: ML data uses "status" field, not "result"
        ml_settled = len([b for b in observed if b.get("status") in ["won", "lost"]])
        ml_won = len([b for b in observed if b.get("status") == "won"])
        
        # Calculate profit/loss from ML data (use pnl field if available, otherwise calculate)
        total_profit = 0
        for bet in observed:
            if bet.get("status") in ["won", "lost"]:
                pnl = bet.get("pnl")
                if pnl is not None:
                    total_profit += pnl
                elif bet.get("status") == "won":
                    odds = bet.get("odds", 2.0) or 2.0
                    units = bet.get("recommended_units", 1.0) or 1.0
                    total_profit += (odds - 1) * units
                else:
                    units = bet.get("recommended_units", 1.0) or 1.0
                    total_profit -= units
        
        uptime_hours = 0
        if bot_start_time:
            uptime_hours = (datetime.datetime.now(datetime.timezone.utc) - bot_start_time).total_seconds() / 3600
        
        patterns = ml_learning_data.get("profitable_patterns", [])
        win_rate = (ml_won / ml_settled * 100) if ml_settled > 0 else 0
        
        return {
            "bot_status": "online" if (bot and bot.is_ready()) else "offline",
            "uptime_hours": round(uptime_hours, 2),
            "total_bets": len(tracked_bets),
            "ml_observed_bets": len(observed),
            "ml_settled_bets": ml_settled,
            "ml_won_bets": ml_won,
            "win_rate": round(win_rate, 2),
            "total_alerts": len(sent_alerts),
            "ml_patterns": len(patterns),
            "total_profit_units": round(total_profit, 2),
            "disconnect_count": disconnect_count,
            "guilds_connected": len(bot.guilds) if (bot and bot.is_ready()) else 0,
            "scan_running": live_stream_running
        }

    @api.get("/api/ml/observed")
    async def api_ml_observed(page: int = 0, limit: int = 50):
        """Get paginated ML observed bets"""
        global ml_learning_data
        
        observed = ml_learning_data.get("all_observed_bets", [])
        
        # Sort by timestamp descending (most recent first)
        sorted_bets = sorted(observed, key=lambda x: str(x.get("timestamp", "")), reverse=True)
        
        # Paginate
        start = page * limit
        end = start + limit
        page_bets = sorted_bets[start:end]
        
        # Format for API
        formatted = []
        for bet in page_bets:
            formatted.append({
                "event_id": bet.get("event_id", ""),
                "market_id": bet.get("market_id", ""),
                "selection_id": bet.get("selection_id", ""),
                "runner_name": bet.get("runner_name", ""),
                "event_name": bet.get("event_name", ""),
                "market_type": bet.get("bet_type", ""),
                "sport": bet.get("sport", "unknown"),
                "odds_at_alert": bet.get("odds", 0),
                "alert_time": str(bet.get("timestamp", "")),
                "settled": bet.get("result") in ["won", "lost"],
                "won": bet.get("result") == "won" if bet.get("result") in ["won", "lost"] else None,
                "profit_loss": (bet.get("odds", 1) - 1) if bet.get("result") == "won" else (-1 if bet.get("result") == "lost" else 0)
            })
        
        return {
            "bets": formatted,
            "total": len(observed),
            "page": page,
            "limit": limit,
            "total_pages": (len(observed) + limit - 1) // limit
        }

    @api.post("/api/settings/ml")
    async def api_update_ml_config(config: dict):
        """Update ML configuration"""
        global ML_CONFIG
        
        if "min_sample_size" in config:
            ML_CONFIG["min_sample_size"] = int(config["min_sample_size"])
        if "min_profit_per_100" in config:
            ML_CONFIG["min_profit_per_100"] = float(config["min_profit_per_100"])
        if "max_patterns" in config:
            ML_CONFIG["max_patterns"] = int(config["max_patterns"])
        if "sync_interval_seconds" in config:
            ML_CONFIG["sync_interval_seconds"] = int(config["sync_interval_seconds"])
        
        return {"status": "ok", "config": ML_CONFIG}

    def run_api_server():
        """Run the FastAPI server in a separate thread"""
        uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Start API server in background thread if FastAPI is available
    if FASTAPI_AVAILABLE:
        print("🌐 Starting Dashboard API server on http://0.0.0.0:8000")
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        time.sleep(1)  # Give API server time to start
        print("✅ Dashboard API server started")
    else:
        print("⚠️ Dashboard API not available - install fastapi uvicorn pydantic")
    
    print("🤖 Starting Discord bot...")
    
    # For Railway deployment, use the robust reconnect approach
    try:
        asyncio.run(main_with_reconnect())
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        # Fallback to simple bot.run for compatibility
        print("🔄 Falling back to simple bot.run()")
        bot.run(BOT_TOKEN)

