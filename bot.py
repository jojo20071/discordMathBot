import discord
from discord.ext import commands
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy import stats

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

@bot.command(name='plot')
async def plot(ctx, *, function: str):
    try:
        x = np.linspace(-10, 10, 400)
        expr = sp.sympify(function)
        f = sp.lambdify(sp.Symbol('x'), expr, 'numpy')
        y = f(x)
        plt.figure()
        plt.plot(x, y)
        plt.title(f'Plot of {function}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.savefig('plot.png')
        await ctx.send(file=discord.File('plot.png'))
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error plotting function: {e}')

@bot.command(name='mean')
async def mean(ctx, *values: float):
    try:
        mean_value = np.mean(values)
        await ctx.send(f'The mean is: {mean_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating mean: {e}')

@bot.command(name='median')
async def median(ctx, *values: float):
    try:
        median_value = np.median(values)
        await ctx.send(f'The median is: {median_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating median: {e}')

@bot.command(name='mode')
async def mode(ctx, *values: float):
    try:
        mode_value = stats.mode(values)
        await ctx.send(f'The mode is: {mode_value.mode[0]}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating mode: {e}')

@bot.command(name='std_dev')
async def std_dev(ctx, *values: float):
    try:
        std_dev_value = np.std(values)
        await ctx.send(f'The standard deviation is: {std_dev_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating standard deviation: {e}')

bot.run('YOUR_BOT_TOKEN')
