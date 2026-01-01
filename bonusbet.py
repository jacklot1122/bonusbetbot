import discord
from discord.ext import commands
import requests
import json
from datetime import datetime, timedelta
import asyncio
import os
from typing import List, Dict, Optional

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CHANNEL_ID = int(os.getenv('CHANNEL_ID', '0'))  # Set your channel ID

# Market priority (higher number = higher priority)
MARKET_PRIORITY = {
    'spreads': 4,
    'totals': 4,
    'h2h': 3,
    'player_props': 2,
}

# Supported Australian bookmakers
SUPPORTED_BOOKMAKERS = [
    'sportsbet', 'tab', 'pointsbet', 'ladbrokes', 'neds', 'unibet',
    'betright', 'bluebet', 'topbetta', 'betr', 'picklebet'
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
        title="🎯 2-Way Bonus Bet Turnover Bot",
        description=(
            "**Convert your bonus bet into guaranteed profit!**\n\n"
            "This bot finds **THE single best 2-way opportunity** using your selected bookmaker and bonus amount.\n\n"
            "✅ **2-Way Markets Only:** Spreads, Totals, 2-way H2H, Player Props\n"
            "🚫 **Soccer Excluded:** No 3-way market complications\n"
            "💰 **Return Shown Regardless of %**\n"
            "⏰ **Time Filter:** Next 7 days only\n"
            "🌏 **Region:** Australian bookmakers"
        ),
        color=0x00ff88
    )

    embed.add_field(
        name="🚀 How It Works",
        value=(
            "1. Click the button below\n"
            "2. Enter your bonus bet amount\n"
            "3. Select your bookmaker\n"
            "4. Get THE best opportunity"
        ),
        inline=False
    )

    embed.add_field(
        name="🎯 What You Get",
        value=(
            "• **Single best result** tailored to your bookmaker\n"
            "• **Exact hedge amounts** for your bonus bet size\n"
            "• **Return % even if low**\n"
            "• **Private results** (ephemeral responses)"
        ),
        inline=False
    )

    embed.set_footer(text="🔒 All interactions are private • 2-way markets only • Soccer excluded")

    return embed

class ArbitrageBot:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
    
    def is_soccer_related(self, text: str) -> bool:
        """Check if text contains soccer-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in SOCCER_KEYWORDS)
    
    async def get_sports(self) -> List[Dict]:
        """Fetch available sports from The Odds API"""
        try:
            url = f"{ODDS_API_BASE}/sports"
            params = {'apiKey': ODDS_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            sports = response.json()
            
            # Filter out soccer and inactive sports
            filtered_sports = [
                sport for sport in sports 
                if sport.get('active', False) and not self.is_soccer_related(sport.get('title', ''))
            ]
            return filtered_sports
        except Exception as e:
            print(f"Error fetching sports: {e}")
            return []
    
    async def get_odds(self, sport_key: str, markets: str) -> List[Dict]:
        """Fetch odds for a specific sport and market"""
        try:
            url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'au',
                'markets': markets,
                'oddsFormat': 'decimal',
                'bookmakers': ','.join(SUPPORTED_BOOKMAKERS)
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching odds for {sport_key}/{markets}: {e}")
            return []
    
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
    
    async def find_best_opportunity(self, selected_bookmaker: str, amount: float) -> Optional[Dict]:
        """Find the single best 2-way opportunity for the selected bookmaker"""
        best_opportunity = None
        best_return = float('-inf')
        
        # Get available sports
        sports = await self.get_sports()
        if not sports:
            return None
        
        # Markets to check (all 2-way)
        markets_to_check = ['h2h', 'spreads', 'totals']
        
        for sport in sports:
            sport_key = sport['key']
            sport_title = sport['title']
            
            # Check each market type
            for market_type in markets_to_check:
                events = await self.get_odds(sport_key, market_type)
                
                for event in events:
                    # Filter events happening within 7 days
                    commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                    if commence_time > datetime.now().astimezone() + timedelta(days=7):
                        continue
                    
                    home_team = event.get('home_team', '')
                    away_team = event.get('away_team', '')
                    
                    # Skip if teams contain soccer keywords
                    if self.is_soccer_related(home_team) or self.is_soccer_related(away_team):
                        continue
                    
                    bookmakers = event.get('bookmakers', [])
                    
                    # Find odds from selected bookmaker
                    bonus_bookmaker_data = None
                    for bookmaker in bookmakers:
                        if bookmaker['key'] == selected_bookmaker:
                            bonus_bookmaker_data = bookmaker
                            break
                    
                    if not bonus_bookmaker_data:
                        continue
                    
                    # Get markets for this bookmaker
                    for market in bonus_bookmaker_data.get('markets', []):
                        if market['key'] != market_type:
                            continue
                        
                        outcomes = market.get('outcomes', [])
                        if len(outcomes) != 2:  # Only 2-way markets
                            continue
                        
                        # Try both outcomes as the bonus bet
                        for i, bonus_outcome in enumerate(outcomes):
                            hedge_outcome = outcomes[1 - i]
                            
                            bonus_odds = bonus_outcome['price']
                            
                            # Find best hedge odds from other bookmakers
                            best_hedge_odds = 0
                            best_hedge_bookmaker = None
                            
                            for other_bookmaker in bookmakers:
                                if other_bookmaker['key'] == selected_bookmaker:
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
                            
                            # Calculate opportunity
                            calc = self.calculate_bonus_bet_opportunity(bonus_odds, best_hedge_odds, amount)
                            
                            # Track best opportunity
                            if calc['guaranteed_return'] > best_return:
                                best_return = calc['guaranteed_return']
                                
                                market_display = {
                                    'h2h': 'Head to Head',
                                    'spreads': 'Spread',
                                    'totals': 'Totals'
                                }.get(market_type, market_type)
                                
                                best_opportunity = {
                                    'sport_title': sport_title,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'market_display': market_display,
                                    'bonus_bookmaker': selected_bookmaker,
                                    'bonus_outcome': bonus_outcome['name'],
                                    'bonus_odds_decimal': bonus_odds,
                                    'bonus_amount': amount,
                                    'hedge_bookmaker': best_hedge_bookmaker,
                                    'hedge_outcome': hedge_outcome['name'],
                                    'hedge_odds_decimal': best_hedge_odds,
                                    **calc
                                }
        
        return best_opportunity

    def create_opportunity_embed(self, opportunity: Dict) -> discord.Embed:
        embed = discord.Embed(
            title="🧠 Your Bonus Bet Opportunity",
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
                discord.SelectOption(label="PointsBet", value="pointsbet", emoji="🎯"),
                discord.SelectOption(label="Ladbrokes", value="ladbrokes", emoji="🎲"),
                discord.SelectOption(label="Neds", value="neds", emoji="🏈"),
                discord.SelectOption(label="Unibet", value="unibet", emoji="⚽"),
                discord.SelectOption(label="BetRight", value="betright", emoji="✅"),
                discord.SelectOption(label="BlueBet", value="bluebet", emoji="🔵"),
                discord.SelectOption(label="TopBetta", value="topbetta", emoji="🔝"),
                discord.SelectOption(label="Betr", value="betr", emoji="💰"),
                discord.SelectOption(label="PickleBet", value="picklebet", emoji="🥒"),
            ],
            custom_id="bookmaker_select"
        )
        select.callback = self.select_callback
        self.add_item(select)
    
    async def select_callback(self, interaction: discord.Interaction):
        self.selected_bookmaker = interaction.data['values'][0]
        
        loading_embed = discord.Embed(
            title="🔍 Finding Your Best 2-Way Opportunity...",
            description=f"Scanning all sports (excluding soccer) for opportunities where your ${self.amount:,.0f} bonus bet is on **{self.selected_bookmaker.title()}**...",
            color=0xffaa00
        )
        await interaction.response.edit_message(embed=loading_embed, view=None)
        
        try:
            opportunity = await arb_bot.find_best_opportunity(self.selected_bookmaker, self.amount)
            
            if not opportunity:
                error_embed = discord.Embed(
                    title="❌ No Opportunities Found",
                    description=f"Could not find any 2-way opportunities for **{self.selected_bookmaker.title()}** at this time. Please try again later.",
                    color=0xff0000
                )
                await interaction.edit_original_response(embed=error_embed)
                return
            
            embed = arb_bot.create_opportunity_embed(opportunity)
            await interaction.edit_original_response(embed=embed)
        except Exception as e:
            print(f"Error in bonus bet generation: {e}")
            error_embed = discord.Embed(
                title="❌ Error",
                description="Something went wrong while finding opportunities. Please try again.",
                color=0xff0000
            )
            await interaction.edit_original_response(embed=error_embed)

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
        label='🧠 Generate a Bonus Bet for Me',
        style=discord.ButtonStyle.primary,
        emoji='🧠',
        custom_id='generate_bonus_bet_button'
    )
    async def generate_bonus_bet(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = BonusBetModal()
        await interaction.response.send_modal(modal)

@bot.event
async def on_ready():
    print(f'{bot.user} has logged in!')
    bot.add_view(PersistentView())

    if CHANNEL_ID:
        try:
            channel = bot.get_channel(CHANNEL_ID)
            if channel:
                async for message in channel.history(limit=50):
                    if message.author == bot.user and message.embeds:
                        embed = message.embeds[0]
                        if "2-Way Bonus Bet Turnover" in embed.title:
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
        bot.run(DISCORD_TOKEN)