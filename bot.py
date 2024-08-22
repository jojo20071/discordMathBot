import discord
from discord.ext import commands
import sympy as sp

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')

@bot.command(name='solve')
async def solve(ctx, *, equation: str):
    try:
        expr = sp.sympify(equation)
        result = sp.solve(expr)
        await ctx.send(f'The solution is: {result}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

@bot.command(name='differentiate')
async def differentiate(ctx, *, function: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(function)
        derivative = sp.diff(expr, x)
        await ctx.send(f'The derivative is: {derivative}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

@bot.command(name='integrate')
async def integrate(ctx, *, function: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(function)
        integral = sp.integrate(expr, x)
        await ctx.send(f'The integral is: {integral}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

@bot.command(name='simplify')
async def simplify(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        simplified = sp.simplify(expr)
        await ctx.send(f'The simplified expression is: {simplified}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

@bot.command(name='limit')
async def limit(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        limit_value = sp.limit(expr, x, 0)
        await ctx.send(f'The limit is: {limit_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

@bot.command(name='expand')
async def expand(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        expanded = sp.expand(expr)
        await ctx.send(f'The expanded expression is: {expanded}')
    except Exception as e:
        await ctx.send(f'Error: {e}')

bot.run('YOUR_BOT_TOKEN')
