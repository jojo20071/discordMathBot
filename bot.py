import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user}')

@bot.command(name='add')
async def add(ctx, *numbers: float):
    result = sum(numbers)
    await ctx.send(f"The sum is: {result}")

@bot.command(name='subtract')
async def subtract(ctx, a: float, b: float):
    result = a - b
    await ctx.send(f"The result of subtraction is: {result}")

@bot.command(name='multiply')
async def multiply(ctx, *numbers: float):
    result = 1
    for num in numbers:
        result *= num
    await ctx.send(f"The product is: {result}")

@bot.command(name='divide')
async def divide(ctx, a: float, b: float):
    if b == 0:
        await ctx.send("Cannot divide by zero.")
    else:
        result = a / b
        await ctx.send(f"The result of division is: {result}")

bot.run('YOUR_BOT_TOKEN')