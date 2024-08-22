import discord
from discord.ext import commands
import sympy as sp
import logging

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user.name} ({bot.user.id})')
    logging.info('------')

@bot.command(name='solve')
async def solve(ctx, *, equation: str):
    try:
        expr = sp.sympify(equation)
        result = sp.solve(expr)
        await ctx.send(f'The solution is: {result}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error solving equation: {e}')

@bot.command(name='differentiate')
async def differentiate(ctx, *, function: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(function)
        derivative = sp.diff(expr, x)
        await ctx.send(f'The derivative is: {derivative}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error differentiating function: {e}')

@bot.command(name='integrate')
async def integrate(ctx, *, function: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(function)
        integral = sp.integrate(expr, x)
        await ctx.send(f'The integral is: {integral}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error integrating function: {e}')

@bot.command(name='simplify')
async def simplify(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        simplified = sp.simplify(expr)
        await ctx.send(f'The simplified expression is: {simplified}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error simplifying expression: {e}')

@bot.command(name='limit')
async def limit(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        limit_value = sp.limit(expr, x, 0)
        await ctx.send(f'The limit is: {limit_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error finding limit: {e}')

@bot.command(name='expand')
async def expand(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        expanded = sp.expand(expr)
        await ctx.send(f'The expanded expression is: {expanded}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error expanding expression: {e}')

@bot.command(name='series')
async def series(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        series_exp = sp.series(expr, x)
        await ctx.send(f'The series expansion is: {series_exp}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating series: {e}')

@bot.command(name='substitute')
async def substitute(ctx, *, expression: str, value: float):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        substituted = expr.subs(x, value)
        await ctx.send(f'The result after substitution is: {substituted}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error substituting value: {e}')

@bot.command(name='factor')
async def factor(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        factors = sp.factor(expr)
        await ctx.send(f'The factors are: {factors}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error factoring expression: {e}')

@bot.command(name='expand_trig')
async def expand_trig(ctx, *, expression: str):
    try:
        expr = sp.sympify(expression)
        expanded_trig = sp.expand_trig(expr)
        await ctx.send(f'The expanded trigonometric expression is: {expanded_trig}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error expanding trigonometric expression: {e}')

bot.run('YOUR_BOT_TOKEN')
