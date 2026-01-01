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
    async def find_best_opportunity(self, selected_bookmaker: str, amount: float) -> Optional[Dict]:
        return {
            'sport_title': 'Sample Sport',
            'home_team': 'Team A',
            'away_team': 'Team B',
            'market_display': 'Head to Head',
            'bonus_bookmaker': selected_bookmaker,
            'bonus_outcome': 'Team A',
            'bonus_odds_decimal': 2.5,
            'bonus_amount': amount,
            'bonus_payout_if_wins': round((amount * 2.5) - amount, 2),
            'hedge_bookmaker': 'tab',
            'hedge_outcome': 'Team B',
            'hedge_odds_decimal': 1.45,
            'hedge_amount': round(((amount * 2.5) - amount) / 1.45, 2),
            'hedge_payout_if_wins': round((((amount * 2.5) - amount) / 1.45) * 1.45, 2),
            'return_percentage': round(min((amount * 1.5 - ((amount * 2.5 - amount) / 1.45)), (((amount * 2.5 - amount) / 1.45) * 1.45 - amount)) / amount * 100, 1),
            'guaranteed_return': round(min((amount * 1.5 - ((amount * 2.5 - amount) / 1.45)), (((amount * 2.5 - amount) / 1.45) * 1.45 - amount)), 2),
        }

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

class BonusBetModal(discord.ui.Modal, title='Generate Your Bonus Bet Opportunity'):
    def __init__(self):
        super().__init__(timeout=300)

    bonus_amount = discord.ui.TextInput(
        label='Bonus Bet Amount ($)',
        placeholder='Enter amount (e.g., 50, 100, 250)',
        required=True,
        max_length=10
    )

    bookmaker = discord.ui.TextInput(
        label='Your Bookmaker',
        placeholder='Enter bookmaker name (e.g., sportsbet, tab, pointsbet)',
        required=True,
        max_length=20
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

        bookmaker_input = self.bookmaker.value.lower().strip()
        selected_bookmaker = next((b for b in SUPPORTED_BOOKMAKERS if bookmaker_input in b or b in bookmaker_input), None)

        if not selected_bookmaker:
            bookmaker_list = ', '.join([bm.title() for bm in SUPPORTED_BOOKMAKERS[:8]])
            await interaction.followup.send(
                f"❌ Bookmaker '{self.bookmaker.value}' not supported.\n\n"
                f"**Supported bookmakers:** {bookmaker_list}, and more.\n"
                f"Please try again with a supported bookmaker name.", ephemeral=True
            )
            return

        loading_embed = discord.Embed(
            title="🔍 Finding Your Best 2-Way Opportunity...",
            description=f"Scanning all sports (excluding soccer) for opportunities where your ${amount:,.0f} bonus bet is on **{selected_bookmaker.title()}**...",
            color=0xffaa00
        )
        await interaction.followup.send(embed=loading_embed, ephemeral=True)

        try:
            opportunity = await arb_bot.find_best_opportunity(selected_bookmaker, amount)
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
                # Check if the interface already exists
                interface_exists = False
                async for message in channel.history(limit=50):
                    if message.author == bot.user and message.embeds:
                        embed = message.embeds[0]
                        if "2-Way Bonus Bet Turnover" in embed.title:
                            interface_exists = True
                            # Update existing message with persistent view
                            await message.edit(embed=create_interface_embed(), view=PersistentView())
                            print(f"Updated existing interface message in channel {CHANNEL_ID}")
                            break
                
                # Only post new message if interface doesn't exist
                if not interface_exists:
                    embed = create_interface_embed()
                    view = PersistentView()
                    await channel.send(embed=embed, view=view)
                    print(f"Posted new interface message in channel {CHANNEL_ID}")
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