import discord
from discord.ext import commands
import aiohttp
import json
from datetime import datetime, timedelta
import asyncio
import os
import sys
from typing import List, Dict, Optional

# Force unbuffered output for Railway logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '401d62b208c11ee6fdfab511972c67b7')
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CHANNEL_ID = int(os.getenv('CHANNEL_ID', '0'))  # Set your channel ID

# Queue for pending searches
search_queue = []
queue_lock = asyncio.Lock()

# Market priority (higher number = higher priority)
MARKET_PRIORITY = {
    'spreads': 4,
    'totals': 4,
    'h2h': 3,
    'player_props': 2,
}

# Supported Australian bookmakers
# Note: bet365_au requires paid API subscription and only covers AFL/NRL
SUPPORTED_BOOKMAKERS = [
    'sportsbet', 'tab', 'pointsbetau', 'ladbrokes_au', 'neds', 'unibet',
    'betright', 'betr_au', 'bet365_au', 'betfair_ex_au', 'playup', 'boombet', 'tabtouch'
]

# Soccer-related keywords to filter out
SOCCER_KEYWORDS = [
    'soccer', 'football', 'epl', 'uefa', 'champions', 'premier', 'serie', 'la liga',
    'bundesliga', 'ligue', 'mls', 'fifa', 'world cup', 'euro', 'copa', 'arsenal',
    'chelsea', 'liverpool', 'manchester', 'barcelona', 'real madrid', 'juventus',
    'psg', 'bayern', 'dortmund', 'atletico', 'tottenham', 'milan', 'inter'
]

def create_interface_embed():
    """Create the main interface embed"""
    embed = discord.Embed(
        title="🎯 Want to use your bonus bet smart?",
        description="Click below to begin:",
        color=0x5865F2
    )
    return embed

class ArbitrageBot:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.search_task = None
        self.session = None
        # Cache durations (in seconds)
        self.SPORTS_CACHE_DURATION = 3600  # 1 hour for sports list
        self.ODDS_CACHE_DURATION = 300     # 5 minutes for odds data
        self.last_full_fetch = None        # Track last complete data fetch
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def add_to_queue(self, user_id: int, user_mention: str, amount: float, bookmaker: str, search_mode: str, interaction: discord.Interaction):
        """Add a search request to the queue"""
        async with queue_lock:
            search_queue.append({
                'user_id': user_id,
                'user_mention': user_mention,
                'amount': amount,
                'bookmaker': bookmaker,
                'search_mode': search_mode,
                'interaction': interaction,
                'attempts': 0,
                'added_at': datetime.now()
            })
            print(f"Added search to queue for user {user_id}: ${amount} on {bookmaker} ({search_mode} mode)")
    
    async def process_queue(self):
        """Background task that processes the search queue every 15 minutes"""
        await bot.wait_until_ready()
        print("Queue processor started - checking every 15 minutes")
        
        while not bot.is_closed():
            try:
                async with queue_lock:
                    if search_queue:
                        print(f"Processing {len(search_queue)} queued searches...")
                        
                        # Fetch all odds data ONCE for all queued searches
                        all_opportunities = await self.fetch_all_opportunities_cached()
                        
                        for search in search_queue[:]:  # Copy to avoid modification during iteration
                            search['attempts'] += 1
                            print(f"Checking cached data for user {search['user_id']} (attempt {search['attempts']})")
                            
                            try:
                                # Find best opportunity from cached data
                                opportunity = self.find_opportunity_from_cache(
                                    all_opportunities,
                                    search['bookmaker'], 
                                    search['amount'], 
                                    search['search_mode']
                                )
                                
                                if opportunity:
                                    # Found an opportunity! Notify the user via DM
                                    embed = self.create_opportunity_embed(opportunity, search['search_mode'])
                                    embed.set_footer(text=f"✅ Found after {search['attempts']} search(es) | Searched every 15 minutes")
                                    
                                    try:
                                        # Send as DM to keep it private
                                        user = await bot.fetch_user(search['user_id'])
                                        if user:
                                            await user.send(
                                                content=f"🎉 **Your bonus bet opportunity is ready!**",
                                                embed=embed
                                            )
                                            print(f"✅ Sent DM to user {search['user_id']} - opportunity found!")
                                        else:
                                            print(f"⚠ Could not fetch user {search['user_id']}")
                                    except discord.Forbidden:
                                        print(f"⚠ Cannot DM user {search['user_id']} - DMs disabled")
                                        # Fallback: try to send in channel as last resort
                                        try:
                                            channel = bot.get_channel(CHANNEL_ID)
                                            if channel:
                                                await channel.send(
                                                    content=f"{search['user_mention']} 🎉 Your bonus bet opportunity is ready! (Enable DMs for private results)",
                                                    embed=embed,
                                                    delete_after=60  # Delete after 1 minute for privacy
                                                )
                                        except:
                                            pass
                                    except Exception as e:
                                        print(f"Error notifying user: {e}")
                                    
                                    # Remove from queue
                                    search_queue.remove(search)
                                else:
                                    print(f"No opportunity found yet for user {search['user_id']}")
                                    
                                    # Remove after 24 hours (96 attempts at 15 min intervals)
                                    if search['attempts'] >= 96:
                                        try:
                                            user = await bot.fetch_user(search['user_id'])
                                            if user:
                                                await user.send(
                                                    content=f"⏰ Your bonus bet search has expired after 24 hours. Please try again with different parameters."
                                                )
                                        except:
                                            pass
                                        search_queue.remove(search)
                                        print(f"Removed expired search for user {search['user_id']}")
                                
                            except Exception as e:
                                print(f"Error processing search for user {search['user_id']}: {e}")
                    
                    else:
                        print("Queue is empty, waiting...")
                
            except Exception as e:
                print(f"Error in queue processor: {e}")
            
            # Wait 15 minutes before next check (saves API credits)
            await asyncio.sleep(900)  # 900 seconds = 15 minutes
    
    def is_soccer_related(self, text: str) -> bool:
        """Check if text contains soccer-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in SOCCER_KEYWORDS)
    
    async def get_sports(self) -> List[Dict]:
        """Fetch available sports from The Odds API (with caching)"""
        cache_key = 'sports_list'
        now = datetime.now()
        
        # Check cache first
        if cache_key in self.cache and cache_key in self.cache_expiry:
            if now < self.cache_expiry[cache_key]:
                print("Using cached sports list")
                return self.cache[cache_key]
        
        try:
            print("Fetching sports list from API...")
            session = await self.get_session()
            url = f"{ODDS_API_BASE}/sports"
            params = {'apiKey': ODDS_API_KEY}
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                print(f"Sports API response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error fetching sports: HTTP {response.status} - {error_text}")
                    # Return cached data if available, even if expired
                    return self.cache.get(cache_key, [])
                
                sports = await response.json()
                print(f"Fetched {len(sports)} total sports from API")
                
                # Filter out soccer and inactive sports
                # Limit to popular Australian sports to reduce API calls
                priority_sports = ['aussierules_afl', 'rugbyleague_nrl', 'basketball_nba', 'cricket_big_bash']
                filtered_sports = []
                
                # First add priority sports if they're active
                for sport in sports:
                    if not sport.get('active', False):
                        continue
                    if self.is_soccer_related(sport.get('title', '')):
                        continue
                    if sport.get('key') in priority_sports:
                        filtered_sports.append(sport)
                        print(f"  ✓ Added priority sport: {sport.get('title')}")
                
                # Then add other sports up to limit of 10
                for sport in sports:
                    if len(filtered_sports) >= 10:
                        break
                    if not sport.get('active', False):
                        continue
                    if self.is_soccer_related(sport.get('title', '')):
                        continue
                    if sport not in filtered_sports:
                        filtered_sports.append(sport)
                        print(f"  ✓ Added sport: {sport.get('title')}")
                
                print(f"Filtered to {len(filtered_sports)} sports for scanning")
                
                # Cache the results
                result = filtered_sports[:10]
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = now + timedelta(seconds=self.SPORTS_CACHE_DURATION)
                
                return result  # Hard limit to prevent too many API calls
        except Exception as e:
            print(f"Error fetching sports: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def get_odds(self, sport_key: str, markets: str) -> List[Dict]:
        """Fetch odds for a specific sport and market (with caching)"""
        cache_key = f'odds_{sport_key}_{markets}'
        now = datetime.now()
        
        # Check cache first
        if cache_key in self.cache and cache_key in self.cache_expiry:
            if now < self.cache_expiry[cache_key]:
                cached_data = self.cache[cache_key]
                print(f"  → Using cached {len(cached_data)} events for {sport_key}/{markets}")
                return cached_data
        
        try:
            session = await self.get_session()
            url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'au',
                'markets': markets,
                'oddsFormat': 'decimal',
                'bookmakers': ','.join(SUPPORTED_BOOKMAKERS)
            }
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 401:
                    print(f"  ⚠ API key unauthorized for {sport_key}/{markets}")
                    return self.cache.get(cache_key, [])
                elif response.status == 422:
                    # Market not available for this sport, cache empty result
                    self.cache[cache_key] = []
                    self.cache_expiry[cache_key] = now + timedelta(seconds=self.ODDS_CACHE_DURATION)
                    return []
                elif response.status != 200:
                    error_text = await response.text()
                    print(f"  ⚠ Error fetching odds for {sport_key}/{markets}: HTTP {response.status}")
                    return self.cache.get(cache_key, [])
                
                events = await response.json()
                print(f"  → Fetched {len(events)} events for {sport_key}/{markets}")
                
                # Cache the results
                self.cache[cache_key] = events
                self.cache_expiry[cache_key] = now + timedelta(seconds=self.ODDS_CACHE_DURATION)
                
                return events
        except asyncio.TimeoutError:
            print(f"  ⚠ Timeout fetching odds for {sport_key}/{markets}")
            return self.cache.get(cache_key, [])
        except Exception as e:
            print(f"  ⚠ Error fetching odds for {sport_key}/{markets}: {e}")
            return self.cache.get(cache_key, [])
    
    async def fetch_all_opportunities_cached(self) -> List[Dict]:
        """Fetch all odds data once and extract all possible opportunities.
        This dramatically reduces API calls by fetching once and reusing for all queue items.
        """
        print("\n" + "="*60)
        print("Fetching all opportunities (single API batch)")
        print("="*60)
        
        all_opportunities = []
        sports = await self.get_sports()
        if not sports:
            print("❌ No sports available")
            return []
        
        # Fetch h2h,spreads,totals in ONE call per sport (saves 2 API calls per sport)
        markets_combined = 'h2h,spreads,totals'
        
        for sport in sports:
            sport_key = sport['key']
            sport_title = sport['title']
            print(f"\n📊 Fetching {sport_title}...")
            
            events = await self.get_odds(sport_key, markets_combined)
            
            for event in events:
                # Filter events happening within 7 days
                try:
                    commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                    if commence_time > datetime.now().astimezone() + timedelta(days=7):
                        continue
                except:
                    continue
                
                home_team = event.get('home_team', '')
                away_team = event.get('away_team', '')
                
                if self.is_soccer_related(home_team) or self.is_soccer_related(away_team):
                    continue
                
                bookmakers = event.get('bookmakers', [])
                
                # Extract all 2-way opportunities from this event
                for bookmaker in bookmakers:
                    bookmaker_key = bookmaker['key']
                    
                    for market in bookmaker.get('markets', []):
                        market_type = market['key']
                        outcomes = market.get('outcomes', [])
                        
                        if len(outcomes) != 2:  # Only 2-way markets
                            continue
                        
                        for i, bonus_outcome in enumerate(outcomes):
                            hedge_outcome = outcomes[1 - i]
                            bonus_odds = bonus_outcome['price']
                            
                            # Find best hedge odds from other bookmakers
                            best_hedge_odds = 0
                            best_hedge_bookmaker = None
                            
                            for other_bookmaker in bookmakers:
                                if other_bookmaker['key'] == bookmaker_key:
                                    continue
                                
                                for other_market in other_bookmaker.get('markets', []):
                                    if other_market['key'] != market_type:
                                        continue
                                    
                                    for outcome in other_market.get('outcomes', []):
                                        if outcome['name'] == hedge_outcome['name']:
                                            if outcome['price'] > best_hedge_odds:
                                                best_hedge_odds = outcome['price']
                                                best_hedge_bookmaker = other_bookmaker['key']
                            
                            if not best_hedge_bookmaker or best_hedge_odds == 0:
                                continue
                            
                            market_display = {
                                'h2h': 'Head to Head',
                                'spreads': 'Spread',
                                'totals': 'Totals'
                            }.get(market_type, market_type)
                            
                            all_opportunities.append({
                                'sport_title': sport_title,
                                'home_team': home_team,
                                'away_team': away_team,
                                'market_type': market_type,
                                'market_display': market_display,
                                'bonus_bookmaker': bookmaker_key,
                                'bonus_outcome': bonus_outcome['name'],
                                'bonus_odds_decimal': bonus_odds,
                                'hedge_bookmaker': best_hedge_bookmaker,
                                'hedge_outcome': hedge_outcome['name'],
                                'hedge_odds_decimal': best_hedge_odds,
                            })
            
            # Small delay between sports to be nice to the API
            await asyncio.sleep(0.3)
        
        print(f"\n✅ Extracted {len(all_opportunities)} potential opportunities")
        return all_opportunities
    
    def find_opportunity_from_cache(self, all_opportunities: List[Dict], selected_bookmaker: str, amount: float, search_mode: str = 'best') -> Optional[Dict]:
        """Find the best opportunity for a specific bookmaker from pre-fetched data.
        This uses NO API calls - just filters the cached opportunities.
        """
        best_opportunity = None
        best_return = float('-inf')
        quick_threshold = amount * 0.60  # 60% minimum for quick mode
        
        for opp in all_opportunities:
            if opp['bonus_bookmaker'] != selected_bookmaker:
                continue
            
            # Calculate returns for this amount
            calc = self.calculate_bonus_bet_opportunity(
                opp['bonus_odds_decimal'],
                opp['hedge_odds_decimal'],
                amount
            )
            
            if calc['guaranteed_return'] > best_return:
                best_return = calc['guaranteed_return']
                best_opportunity = {
                    **opp,
                    'bonus_amount': amount,
                    **calc
                }
                
                # In quick mode, return first opportunity that meets threshold
                if search_mode == 'quick' and calc['guaranteed_return'] >= quick_threshold:
                    return best_opportunity
        
        return best_opportunity
    
    def calculate_bonus_bet_opportunity(self, bonus_odds: float, hedge_odds: float, amount: float) -> Dict:
        """Calculate the returns for a bonus bet opportunity"""
        # Bonus bet: you don't get the stake back, only the winnings
        bonus_payout_if_wins = (amount * bonus_odds) - amount
        
        # Hedge bet: calculate how much to bet to cover the bonus payout
        hedge_amount = bonus_payout_if_wins / hedge_odds
        hedge_payout_if_wins = hedge_amount * hedge_odds
        
        # Calculate guaranteed return (minimum profit)
        # If bonus wins: profit = bonus_payout - hedge_amount (what you spent)
        # If hedge wins: profit = hedge_payout - amount (bonus was free) - hedge_amount
        profit_if_bonus_wins = bonus_payout_if_wins - hedge_amount
        profit_if_hedge_wins = hedge_payout_if_wins - hedge_amount
        
        guaranteed_return = min(profit_if_bonus_wins, profit_if_hedge_wins)
        return_percentage = (guaranteed_return / amount) * 100
        
        return {
            'bonus_payout_if_wins': round(bonus_payout_if_wins, 2),
            'hedge_amount': round(hedge_amount, 2),
            'hedge_payout_if_wins': round(hedge_payout_if_wins, 2),
            'guaranteed_return': round(guaranteed_return, 2),
            'return_percentage': round(return_percentage, 1)
        }
    
    async def find_best_opportunity(self, selected_bookmaker: str, amount: float, search_mode: str = 'best') -> Optional[Dict]:
        """Find the single best 2-way opportunity for the selected bookmaker
        
        Uses combined market fetch to minimize API calls (1 call per sport instead of 3).
        
        Args:
            selected_bookmaker: The bookmaker where the bonus bet will be placed
            amount: The bonus bet amount
            search_mode: 'quick' for fast 60-70% return, 'best' for maximum return
        """
        print(f"\n{'='*60}")
        print(f"Starting search for {selected_bookmaker} - ${amount} ({search_mode} mode)")
        print(f"{'='*60}")
        
        # Use the batch fetch and filter approach to minimize API calls
        all_opportunities = await self.fetch_all_opportunities_cached()
        
        if not all_opportunities:
            print("❌ No opportunities available")
            return None
        
        # Find best opportunity from the fetched data
        best_opportunity = self.find_opportunity_from_cache(
            all_opportunities, 
            selected_bookmaker, 
            amount, 
            search_mode
        )
        
        if best_opportunity:
            print(f"✅ Found opportunity: {best_opportunity['return_percentage']:.1f}% return")
        else:
            print(f"❌ No opportunities found for {selected_bookmaker}")
        
        return best_opportunity

    def create_opportunity_embed(self, opportunity: Dict, search_mode: str = 'best') -> discord.Embed:
        mode_emoji = "⚡" if search_mode == "quick" else "🏆"
        mode_text = "Quick Return" if search_mode == "quick" else "Best Return"
        
        embed = discord.Embed(
            title=f"{mode_emoji} Your Bonus Bet Opportunity ({mode_text})",
            color=0x00ff88
        )
        embed.add_field(
            name="🏟️ Event",
            value=f"**{opportunity['sport_title']}** - {opportunity['home_team']} vs {opportunity['away_team']}",
            inline=False
        )
        embed.add_field(
            name="📊 Market",
            value=f"**{opportunity['market_display']}** → {opportunity['bonus_outcome']} / {opportunity['hedge_outcome']}",
            inline=False
        )
        embed.add_field(
            name="💰 Return Estimate",
            value=f"**{opportunity['return_percentage']:.1f}%** (${opportunity['guaranteed_return']:.2f} from ${opportunity['bonus_amount']:,.0f} bonus)",
            inline=False
        )
        embed.add_field(
            name="🎲 Bonus Bet",
            value=(
                f"📱 **{opportunity['bonus_bookmaker'].title()}**\n"
                f"🟢 **{opportunity['bonus_outcome']}** @ {opportunity['bonus_odds_decimal']}\n"
                f"**Stake:** ${opportunity['bonus_amount']:,.0f} (bonus)\n"
                f"**Payout if wins:** ${opportunity['bonus_payout_if_wins']:,.2f}"
            ),
            inline=True
        )
        embed.add_field(
            name="🛡️ Hedge Bet",
            value=(
                f"📱 **{opportunity['hedge_bookmaker'].title()}**\n"
                f"🔴 **{opportunity['hedge_outcome']}** @ {opportunity['hedge_odds_decimal']}\n"
                f"**Stake:** ${opportunity['hedge_amount']:,.2f} (cash)\n"
                f"**Payout if wins:** ${opportunity['hedge_payout_if_wins']:,.2f}"
            ),
            inline=True
        )
        embed.add_field(name="", value="", inline=True)
        embed.add_field(
            name="✅ Outcome",
            value=f"You'll receive at least ${opportunity['guaranteed_return']:.2f}, regardless of who wins."
        )
        return embed

arb_bot = ArbitrageBot()

class SearchModeView(discord.ui.View):
    def __init__(self, amount: float, bookmaker: str, user_id: int, user_mention: str):
        super().__init__(timeout=180)
        self.amount = amount
        self.bookmaker = bookmaker
        self.user_id = user_id
        self.user_mention = user_mention
        
        # Create select menu for search mode
        select = discord.ui.Select(
            placeholder="Choose your search mode...",
            options=[
                discord.SelectOption(
                    label="🚀 Quick Return (60-70%)",
                    value="quick",
                    description="Fast search - finds first good opportunity",
                    emoji="⚡"
                ),
                discord.SelectOption(
                    label="💎 Best Return Possible",
                    value="best",
                    description="Thorough search - finds highest return (slower)",
                    emoji="🏆"
                ),
            ],
            custom_id="search_mode_select"
        )
        select.callback = self.select_callback
        self.add_item(select)
    
    async def select_callback(self, interaction: discord.Interaction):
        search_mode = interaction.data['values'][0]
        
        mode_text = "**Quick Return**" if search_mode == "quick" else "**Best Return Possible**"
        loading_embed = discord.Embed(
            title="🔍 Searching for Opportunity...",
            description=f"Mode: {mode_text}\nBookmaker: **{self.bookmaker.title()}**\nAmount: **${self.amount:,.0f}**\n\n⏳ Searching now...",
            color=0xffaa00
        )
        await interaction.response.edit_message(embed=loading_embed, view=None)
        
        try:
            # Try to find immediately first
            opportunity = await arb_bot.find_best_opportunity(self.bookmaker, self.amount, search_mode)
            
            if opportunity:
                # Found immediately!
                embed = arb_bot.create_opportunity_embed(opportunity, search_mode)
                embed.set_footer(text="✅ Found immediately!")
                await interaction.edit_original_response(embed=embed)
            else:
                # Not found - add to queue for continuous searching
                await arb_bot.add_to_queue(
                    self.user_id,
                    self.user_mention,
                    self.amount,
                    self.bookmaker,
                    search_mode,
                    interaction
                )
                
                queued_embed = discord.Embed(
                    title="⏰ Added to Search Queue",
                    description=(
                        f"No immediate opportunity found for **{self.bookmaker.title()}**.\n\n"
                        f"**Don't worry!** I'll keep searching for you:\n"
                        f"• Checking every **15 minutes**\n"
                        f"• You'll be **@mentioned** when found\n"
                        f"• Search expires after **24 hours**\n\n"
                        f"**Your search:**\n"
                        f"💰 Amount: ${self.amount:,.0f}\n"
                        f"📱 Bookmaker: {self.bookmaker.title()}\n"
                        f"⚙️ Mode: {mode_text}"
                    ),
                    color=0xffa500
                )
                queued_embed.set_footer(text="You can close this message - I'll ping you when ready!")
                await interaction.edit_original_response(embed=queued_embed)
                
        except Exception as e:
            print(f"Error in bonus bet generation: {e}")
            error_embed = discord.Embed(
                title="❌ Error",
                description="Something went wrong while finding opportunities. Please try again.",
                color=0xff0000
            )
            await interaction.edit_original_response(embed=error_embed)

class BookmakerSelectView(discord.ui.View):
    def __init__(self, amount: float):
        super().__init__(timeout=180)
        self.amount = amount
        self.selected_bookmaker = None
        
        # Create select menu with all bookmakers
        select = discord.ui.Select(
            placeholder="Choose your bookmaker...",
            options=[
                discord.SelectOption(label="Sportsbet", value="sportsbet", emoji="🎰"),
                discord.SelectOption(label="TAB", value="tab", emoji="🏇"),
                discord.SelectOption(label="PointsBet", value="pointsbetau", emoji="🎯"),
                discord.SelectOption(label="Ladbrokes", value="ladbrokes_au", emoji="🎲"),
                discord.SelectOption(label="Neds", value="neds", emoji="🏈"),
                discord.SelectOption(label="Unibet", value="unibet", emoji="⚽"),
                discord.SelectOption(label="BetRight", value="betright", emoji="✅"),
                discord.SelectOption(label="Betr", value="betr_au", emoji="💵"),
                discord.SelectOption(label="Bet365", value="bet365_au", emoji="🟢", description="AFL & NRL only"),
                discord.SelectOption(label="Betfair Exchange", value="betfair_ex_au", emoji="🔄"),
                discord.SelectOption(label="PlayUp", value="playup", emoji="🎮"),
                discord.SelectOption(label="BoomBet", value="boombet", emoji="💥"),
            ],
            custom_id="bookmaker_select"
        )
        select.callback = self.select_callback
        self.add_item(select)
    
    async def select_callback(self, interaction: discord.Interaction):
        self.selected_bookmaker = interaction.data['values'][0]
        
        # Show search mode selection
        mode_embed = discord.Embed(
            title="⚙️ Choose Search Mode",
            description=(
                f"**Bookmaker:** {self.selected_bookmaker.title()}\n"
                f"**Bonus Amount:** ${self.amount:,.0f}\n\n"
                "**Select your preferred search mode:**"
            ),
            color=0x00aaff
        )
        mode_embed.add_field(
            name="⚡ Quick Return (60-70%)",
            value="Fast search that finds the first good opportunity. Typically returns 60-70% of your bonus bet.",
            inline=False
        )
        mode_embed.add_field(
            name="🏆 Best Return Possible",
            value="Comprehensive search across all sports and markets. Takes longer but finds the absolute best return available.",
            inline=False
        )
        
        view = SearchModeView(self.amount, self.selected_bookmaker, interaction.user.id, interaction.user.mention)
        await interaction.response.edit_message(embed=mode_embed, view=view)

class BonusBetModal(discord.ui.Modal, title='Enter Your Bonus Bet Amount'):
    def __init__(self):
        super().__init__(timeout=300)

    bonus_amount = discord.ui.TextInput(
        label='Bonus Bet Amount ($)',
        placeholder='Enter amount (e.g., 50, 100, 250)',
        required=True,
        max_length=10
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        try:
            amount = float(self.bonus_amount.value.replace('$', '').replace(',', ''))
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError:
            await interaction.followup.send(
                "❌ Please enter a valid bonus bet amount (e.g., 100, 250, 500)", ephemeral=True
            )
            return

        # Show bookmaker selection
        select_embed = discord.Embed(
            title="📱 Select Your Bookmaker",
            description=f"Choose the bookmaker where you have your **${amount:,.0f}** bonus bet:",
            color=0x00aaff
        )
        view = BookmakerSelectView(amount)
        await interaction.followup.send(embed=select_embed, view=view, ephemeral=True)

class PersistentView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(
        label='Generate a Bonus Bet for Me',
        style=discord.ButtonStyle.primary,
        custom_id='generate_bonus_bet_button'
    )
    async def generate_bonus_bet(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = BonusBetModal()
        await interaction.response.send_modal(modal)

@bot.event
async def on_ready():
    print(f'{bot.user} has logged in!')
    bot.add_view(PersistentView())
    
    # Start the queue processor
    if not arb_bot.search_task or arb_bot.search_task.done():
        arb_bot.search_task = bot.loop.create_task(arb_bot.process_queue())
        print("Started background queue processor")

    if CHANNEL_ID:
        try:
            channel = bot.get_channel(CHANNEL_ID)
            if channel:
                async for message in channel.history(limit=50):
                    if message.author == bot.user and message.embeds:
                        embed = message.embeds[0]
                        if "Want to use your bonus bet smart" in embed.title:
                            await message.edit(embed=create_interface_embed(), view=PersistentView())
                            return
                embed = create_interface_embed()
                view = PersistentView()
                await channel.send(embed=embed, view=view)
        except Exception as e:
            print(f"Error setting up channel interface: {e}")

if __name__ == "__main__":
    if not DISCORD_TOKEN or not ODDS_API_KEY:
        print("Error: Missing DISCORD_TOKEN or ODDS_API_KEY environment variables")
    elif not CHANNEL_ID:
        print("Error: Missing CHANNEL_ID environment variable")
    else:
        print("Starting Discord 2-Way Bonus Bet Turnover Bot...")
        try:
            bot.run(DISCORD_TOKEN)
        finally:
            # Cleanup
            if arb_bot.session and not arb_bot.session.closed:
                asyncio.run(arb_bot.close_session())
