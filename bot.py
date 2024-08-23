import discord
from discord.ext import commands
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

logging.basicConfig(level=logging.ERROR)

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.command(name='solve')
async def solve(ctx, *, equation: str):
    try:
        x = sp.Symbol('x')
        solutions = sp.solve(equation, x)
        await ctx.send(f'Solutions: {solutions}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error solving equation: {e}')

@bot.command(name='integrate')
async def integrate(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        integral = sp.integrate(expression, x)
        await ctx.send(f'Integral: {integral}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error integrating expression: {e}')

@bot.command(name='differentiate')
async def differentiate(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        derivative = sp.diff(expression, x)
        await ctx.send(f'Derivative: {derivative}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error differentiating expression: {e}')

@bot.command(name='limit')
async def limit(ctx, *, expression: str):
    try:
        x = sp.Symbol('x')
        lim = sp.limit(expression, x, 0)
        await ctx.send(f'Limit: {lim}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating limit: {e}')

@bot.command(name='expand')
async def expand(ctx, *, expression: str):
    try:
        expanded = sp.expand(expression)
        await ctx.send(f'Expanded expression: {expanded}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error expanding expression: {e}')

@bot.command(name='factor')
async def factor(ctx, *, expression: str):
    try:
        factored = sp.factor(expression)
        await ctx.send(f'Factored expression: {factored}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error factoring expression: {e}')

@bot.command(name='simplify')
async def simplify(ctx, *, expression: str):
    try:
        simplified = sp.simplify(expression)
        await ctx.send(f'Simplified expression: {simplified}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error simplifying expression: {e}')

@bot.command(name='evaluate')
async def evaluate(ctx, *, expression: str):
    try:
        evaluated = sp.sympify(expression)
        await ctx.send(f'Evaluated result: {evaluated}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error evaluating expression: {e}')

@bot.command(name='polynomial')
async def polynomial(ctx, *coefficients: float):
    try:
        x = sp.Symbol('x')
        poly = sum(c * x**i for i, c in enumerate(coefficients[::-1]))
        evaluated = sp.expand(poly)
        await ctx.send(f'Polynomial: {evaluated}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error evaluating polynomial: {e}')

@bot.command(name='variance')
async def variance(ctx, *values: float):
    try:
        variance_value = np.var(values)
        await ctx.send(f'The variance is: {variance_value}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating variance: {e}')

@bot.command(name='covariance')
async def covariance(ctx, *values: float):
    try:
        values = np.array(values)
        mean = np.mean(values)
        covariance_matrix = np.cov(values - mean, rowvar=False)
        await ctx.send(f'The covariance is: {covariance_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating covariance: {e}')

@bot.command(name='polyfit')
async def polyfit(ctx, degree: int, *points: str):
    try:
        x = []
        y = []
        for point in points:
            xi, yi = map(float, point.split(','))
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        coefficients = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coefficients)
        await ctx.send(f'Polynomial fitting result: {poly_eq}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error in polynomial fitting: {e}')

@bot.command(name='residuals')
async def residuals(ctx, degree: int, *points: str):
    try:
        x = []
        y = []
        for point in points:
            xi, yi = map(float, point.split(','))
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        coefficients = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coefficients)
        residuals = y - poly_eq(x)
        await ctx.send(f'Residuals: {residuals}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating residuals: {e}')

@bot.command(name='chi_square')
async def chi_square(ctx, observed: str, expected: str):
    try:
        observed_values = np.array(list(map(float, observed.split(','))))
        expected_values = np.array(list(map(float, expected.split(','))))
        chi2, p = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
        await ctx.send(f'Chi-square statistic: {chi2}, p-value: {p}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error in chi-square test: {e}')

@bot.command(name='anova')
async def anova(ctx, *groups: str):
    try:
        data_groups = [list(map(float, group.split(','))) for group in groups]
        f_val, p_val = stats.f_oneway(*data_groups)
        await ctx.send(f'ANOVA result: F-value: {f_val}, p-value: {p_val}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error in ANOVA: {e}')

@bot.command(name='ttest')
async def ttest(ctx, group1: str, group2: str):
    try:
        data_group1 = list(map(float, group1.split(',')))
        data_group2 = list(map(float, group2.split(',')))
        t_stat, p_val = stats.ttest_ind(data_group1, data_group2)
        await ctx.send(f'T-test result: t-statistic: {t_stat}, p-value: {p_val}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error in T-test: {e}')

@bot.command(name='plot_fit')
async def plot_fit(ctx, degree: int, *points: str):
    try:
        x = []
        y = []
        for point in points:
            xi, yi = map(float, point.split(','))
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        coefficients = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coefficients)
        xp = np.linspace(min(x), max(x), 100)
        plt.figure()
        plt.plot(x, y, 'o', label='Data points')
        plt.plot(xp, poly_eq(xp), '-', label=f'Fit: degree {degree}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Polynomial Fit')
        plt.savefig('fit_plot.png')
        await ctx.send(file=discord.File('fit_plot.png'))
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error plotting polynomial fit: {e}')

@bot.command(name='predict')
async def predict(ctx, degree: int, point: float, *points: str):
    try:
        x = []
        y = []
        for point_str in points:
            xi, yi = map(float, point_str.split(','))
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        coefficients = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coefficients)
        prediction = poly_eq(point)
        await ctx.send(f'Prediction at x={point}: {prediction}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error in prediction: {e}')

@bot.command(name='fourier_transform')
async def fourier_transform(ctx, *, expression: str):
    try:
        t = sp.Symbol('t')
        expr = sp.sympify(expression)
        fourier_expr = sp.fourier_transform(expr, t, sp.Symbol('w'))
        await ctx.send(f'Fourier Transform: {fourier_expr}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating Fourier transform: {e}')

@bot.command(name='inverse_fourier')
async def inverse_fourier(ctx, *, expression: str):
    try:
        w = sp.Symbol('w')
        expr = sp.sympify(expression)
        inv_fourier_expr = sp.inverse_fourier_transform(expr, w, sp.Symbol('t'))
        await ctx.send(f'Inverse Fourier Transform: {inv_fourier_expr}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating inverse Fourier transform: {e}')

@bot.command(name='matrix_addition')
async def matrix_addition(ctx, *, matrices: str):
    try:
        mat1_str, mat2_str = matrices.split(';')
        mat1 = sp.Matrix(sp.sympify(mat1_str))
        mat2 = sp.Matrix(sp.sympify(mat2_str))
        sum_matrix = mat1 + mat2
        await ctx.send(f'Matrix addition result:\n{sum_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing matrix addition: {e}')

@bot.command(name='matrix_subtraction')
async def matrix_subtraction(ctx, *, matrices: str):
    try:
        mat1_str, mat2_str = matrices.split(';')
        mat1 = sp.Matrix(sp.sympify(mat1_str))
        mat2 = sp.Matrix(sp.sympify(mat2_str))
        diff_matrix = mat1 - mat2
        await ctx.send(f'Matrix subtraction result:\n{diff_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing matrix subtraction: {e}')

@bot.command(name='laplace_transform')
async def laplace_transform(ctx, *, expression: str):
    try:
        t = sp.Symbol('t')
        expr = sp.sympify(expression)
        laplace_expr = sp.laplace_transform(expr, t, sp.Symbol('s'))[0]
        await ctx.send(f'Laplace Transform: {laplace_expr}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating Laplace transform: {e}')

@bot.command(name='inverse_laplace')
async def inverse_laplace(ctx, *, expression: str):
    try:
        s = sp.Symbol('s')
        expr = sp.sympify(expression)
        inv_laplace_expr = sp.inverse_laplace_transform(expr, s, sp.Symbol('t'))
        await ctx.send(f'Inverse Laplace Transform: {inv_laplace_expr}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating inverse Laplace transform: {e}')

@bot.command(name='matrix_determinant')
async def matrix_determinant(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        determinant = mat.det()
        await ctx.send(f'Determinant of the matrix: {determinant}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating matrix determinant: {e}')

@bot.command(name='matrix_inverse')
async def matrix_inverse(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        inverse_matrix = mat.inv()
        await ctx.send(f'Inverse of the matrix:\n{inverse_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating matrix inverse: {e}')

@bot.command(name='matrix_rank')
async def matrix_rank(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        rank = mat.rank()
        await ctx.send(f'Rank of the matrix: {rank}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating matrix rank: {e}')

@bot.command(name='matrix_eigen')
async def matrix_eigen(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        eigenvals = mat.eigenvals()
        eigenvects = mat.eigenvects()
        await ctx.send(f'Eigenvalues: {eigenvals}\nEigenvectors: {eigenvects}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating eigenvalues and eigenvectors: {e}')

@bot.command(name='matrix_svd')
async def matrix_svd(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        U, S, V = mat.singular_value_decomposition()
        await ctx.send(f'Singular Value Decomposition (SVD):\nU:\n{U}\nS:\n{S}\nV:\n{V}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing SVD: {e}')

@bot.command(name='matrix_power')
async def matrix_power(ctx, n: int, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        power_matrix = mat**n
        await ctx.send(f'Matrix to the power of {n}:\n{power_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating matrix power: {e}')

@bot.command(name='matrix_decompose')
async def matrix_decompose(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        L, U, _ = mat.LUdecomposition()
        await ctx.send(f'LU Decomposition:\nL:\n{L}\nU:\n{U}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing LU decomposition: {e}')

@bot.command(name='qr_decomposition')
async def qr_decomposition(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        Q, R = mat.QRdecomposition()
        await ctx.send(f'QR Decomposition:\nQ:\n{Q}\nR:\n{R}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing QR decomposition: {e}')

@bot.command(name='cholesky')
async def cholesky(ctx, *, matrix: str):
    try:
        mat = sp.Matrix(sp.sympify(matrix))
        chol = mat.cholesky()
        await ctx.send(f'Cholesky Decomposition:\n{chol}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error performing Cholesky decomposition: {e}')

@bot.command(name='correlation')
async def correlation(ctx, *, data: str):
    try:
        data = np.array([list(map(float, row.split(','))) for row in data.split(';')])
        corr_matrix = np.corrcoef(data, rowvar=False)
        await ctx.send(f'Correlation matrix:\n{corr_matrix}')
    except Exception as e:
        await ctx.send(f'Error: {e}')
        logging.error(f'Error calculating correlation matrix: {e}')



bot.run('YOUR_BOT_TOKEN')
