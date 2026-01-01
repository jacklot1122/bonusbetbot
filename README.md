# 2-Way Bonus Bet Turnover Discord Bot

A Discord bot that helps users find the best 2-way arbitrage opportunities for bonus bet turnover on Australian bookmakers.

## Features

- 🎯 Finds single best 2-way opportunities
- 💰 Calculates exact hedge amounts
- 🚫 Excludes soccer (no 3-way markets)
- 🌏 Supports Australian bookmakers
- 🔒 Private ephemeral responses
- ✅ No spam posting - intelligently updates existing interface

## Deployment on Railway

### Prerequisites

1. Discord Bot Token (from [Discord Developer Portal](https://discord.com/developers/applications))
2. The Odds API Key (from [The Odds API](https://the-odds-api.com/))
3. Discord Channel ID where the bot interface will be posted

### Steps to Deploy

1. **Fork or upload this repository to GitHub**

2. **Go to [Railway.app](https://railway.app/) and sign in**

3. **Create a new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Set environment variables:**
   - Go to your project settings
   - Add the following variables:
     - `DISCORD_TOKEN` - Your Discord bot token
     - `ODDS_API_KEY` - Your Odds API key
     - `CHANNEL_ID` - Your Discord channel ID (numbers only)

5. **Deploy:**
   - Railway will automatically detect the `Procfile` and deploy your bot
   - Check the logs to ensure it started successfully

### Environment Variables

```bash
DISCORD_TOKEN=your_discord_bot_token_here
ODDS_API_KEY=your_odds_api_key_here
CHANNEL_ID=1234567890123456789
```

### Getting Your Discord Channel ID

1. Enable Developer Mode in Discord (User Settings → Advanced → Developer Mode)
2. Right-click on the channel where you want the bot interface
3. Click "Copy Channel ID"

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

3. Fill in your credentials in `.env`

4. Run the bot:
```bash
python bonusbet.py
```

## Supported Bookmakers

- Sportsbet
- TAB
- PointsBet
- Ladbrokes
- Neds
- Unibet
- BetRight
- BlueBet
- TopBetta
- Betr
- PickleBet

## How It Works

1. User clicks the "Generate a Bonus Bet for Me" button
2. Bot prompts for bonus bet amount and bookmaker
3. Bot searches for the best 2-way opportunity across all non-soccer sports
4. Returns a private message with:
   - Best market and odds
   - Exact bonus bet stake
   - Exact hedge bet amount
   - Guaranteed return percentage

## Technical Details

- Built with discord.py
- Uses The Odds API for live betting odds
- Persistent views prevent button spam
- Smart message detection prevents duplicate interface posts
- Ephemeral responses keep user data private

## License

MIT License
