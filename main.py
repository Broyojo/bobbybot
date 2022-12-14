import json
import logging
from pathlib import Path

import discord
from alive_progress import Show, alive_bar

discord.utils.setup_logging()


class DiscordScraperClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super(DiscordScraperClient, self).__init__(intents=intents)
        logging.info(f"Discord scraper client has been loaded")

    async def on_ready(self):
        logging.info(f"Logged in as {self.user}")
        await self.scrape_all_messages()
        await self.close()

    async def scrape_all_messages(self):
        logging.info("Starting to scrape messages from all channels")
        self.data = {}
        for guild in self.guilds:
            self.data[guild.name] = {}
            for channel in guild.channels:
                self.data[guild.name][channel.name] = []
                logging.info(f"Scraping messages from {guild.name}: #{channel.name}")
                if isinstance(channel, discord.CategoryChannel):
                    continue
                with alive_bar() as message_bar:
                    async for message in channel.history(limit=None, oldest_first=True):
                        if message.author == self.user:
                            continue
                        self.data[guild.name][channel.name].append(
                            {
                                "jump_url": message.jump_url,
                                "created_at": str(message.created_at),
                                "author": str(message.author),
                                "clean_content": message.clean_content,
                            }
                        )
                        message_bar()
        logging.info("Done with scraping")
        await self.close()


class DiscordMessageScraper:
    def __init__(self, token, output_file="data2.json"):
        self.token = token
        self.output_file = output_file
        self.client = DiscordScraperClient()
        logging.info(f"Discord message scraper has been loaded")

    def scrape_all_messages(self):
        self.client.run(self.token)
        with open(self.output_file, "w") as f:
            f.write(json.dumps(self.client.data))


scraper = DiscordMessageScraper(token=Path("./token").read_text())
scraper.scrape_all_messages()
