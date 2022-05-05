<div align="center"><h1>Riemann Sum</h1> </div>

<div align="center"><h3>Integral approximation using Riemann sum</h3> </div>

<br>

<img align='center' src="./media/one.gif">

<br>

<h2>Installation</h2>

```bash
git clone https://github.com/CodePleaseRun/riemann-sum.git
cd riemann-sum
```

**Dependencies:**

- `matplotlib`
- `numpy`
- `sympy`

<br>

<h2>Usage</h2>

```python
python main.py

```

<br>

<h2>Points to be noted</h2>

- To be specific, the program is calculating Left Riemann sum. If the function `f(x)` is assumed to be a smooth function, all three Riemann sums (Left, Right, Middle) will converge to the same value as the number of subdivisions goes to infinity.
- Left Limit & Right Limit are evaluated first. That means input like `gamma(3)`, `exp(2)` will be evaluated and converted to float before being used.
- The `Points` input determines the smoothness of the the function `f(x)`. It is independent from the number of subdivisions used for calculating Riemann sum.

<br>

<h2>Licence</h2>
 <h3>MIT</h3>
