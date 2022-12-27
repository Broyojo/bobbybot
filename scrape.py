import json
import logging

import discord
from alive_progress import alive_bar

discord.utils.setup_logging()


class Scraper(discord.Client):
    def __init__(self, output_file):
        self.output_file = output_file
        intents = discord.Intents.default()
        intents.message_content = True
        super(Scraper, self).__init__(intents=intents)

    async def on_ready(self):
        logging.info(f"Logged in as {self.user}")
        data = {}
        with alive_bar() as bar:
            for guild in self.guilds:
                data[guild.name] = {}
                for channel in guild.channels:
                    data[guild.name][channel.name] = []
                    logging.info(
                        f"Scraping messages from {guild.name}: #{channel.name}"
                    )
                    if isinstance(channel, discord.CategoryChannel):
                        continue
                    async for message in channel.history(limit=None, oldest_first=True):
                        if message.author == self.user:
                            continue
                        data[guild.name][channel.name].append(
                            {
                                "jump_url": message.jump_url,
                                "created_at": str(message.created_at),
                                "author": str(message.author.display_name),
                                "clean_content": message.clean_content,
                            }
                        )
                        bar()
        with open(self.output_file, "w") as f:
            f.write(json.dumps(data))
        await self.close()


def main():
    with open("./homike_token") as f:
        Scraper("./data/scraped_data_logic_world.json").run(f.read())


if __name__ == "__main__":
    main()
