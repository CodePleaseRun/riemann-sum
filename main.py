import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Rectangle

RLIM = 10
LLIM = -10
SLEVEL = 500
SUBDIVISIONS = 1


class Function:
    def __init__(self) -> None:
        self.function = None
        self.rlim = 10
        self.llim = -10
        self.slevel = 500  # how smooth will be the plotted curve
        self.subd = 1

    def create(self, expr):
        try:
            self.function = sp.sympify(expr, convert_xor=True)
        except:
            self.function = None
        finally:
            if type(self.function) == sp.core.function.FunctionClass:
                self.function = None
        if self.function:
            self.refresh()

    def refresh(self):
        if self.function != None:
            self.range = np.linspace(self.llim, self.rlim, self.slevel + 1)
            self.value = get_func_values(self.function, self.range)

    def set_slevel(self, slevel):
        if self.function != None:
            self.slevel = slevel
            self.refresh()

    def set_llim(self, llim):
        if self.function != None:
            self.llim = llim
            self.refresh()

    def set_rlim(self, rlim):
        if self.function != None:
            self.rlim = rlim
            self.refresh()


def func_eval(x, function, func_range):  # when lambdify from Sympy fails
    func_values = np.zeros_like(func_range)
    for index, i in enumerate(func_range):
        try:
            func_values[index] = function.subs(x, i)
        except TypeError:
            func_values[index] = np.NaN
    return func_values


def create_subd_tbox():
    txt_axex = plt.axes([0.14, 0.02, 0.04, 0.04])
    subd_tbox = TextBox(ax=txt_axex,
                        label=f'subdivisions (min {SUBDIVISIONS}): ',
                        initial=f'{SUBDIVISIONS}',
                        textalignment='center',
                        color='lightcyan',
                        hovercolor='linen')
    return subd_tbox


def create_func_tbox():
    txt_axex = plt.axes([0.2205, 0.02, 0.3, 0.04])
    func_tbox = TextBox(ax=txt_axex,
                        label='f(x): ',
                        initial='',
                        textalignment='center',
                        color='lightcyan',
                        hovercolor='linen')
    return func_tbox


def create_llim_tbox():
    txt_axex = plt.axes([0.60, 0.02, 0.075, 0.04])
    llim_tbox = TextBox(ax=txt_axex,
                        label='Left Limit: ',
                        initial=str(LLIM),
                        textalignment='center',
                        color='lightcyan',
                        hovercolor='linen')
    return llim_tbox


def create_rlim_tbox():
    txt_axex = plt.axes([0.7575, 0.02, 0.075, 0.04])
    rlim_tbox = TextBox(ax=txt_axex,
                        label='Right Limit: ',
                        initial=str(RLIM),
                        textalignment='center',
                        color='lightcyan',
                        hovercolor='linen')
    return rlim_tbox


def create_slevel_tbox():
    txt_axex = plt.axes([0.951, 0.02, 0.04, 0.04])
    slvl_tbox = TextBox(ax=txt_axex,
                        label=f'Points (min {SLEVEL}): ',
                        initial=str(SLEVEL),
                        textalignment='center',
                        color='lightcyan',
                        hovercolor='linen')
    return slvl_tbox


def get_func_values(function: Function, func_range):
    x = sp.Symbol('x')
    try:
        func_lambda = sp.lambdify(x, function, modules='numpy')
        func_values = func_lambda(func_range)
    except:
        func_values = func_eval(x, function, func_range)
    return func_values


def create_tboxes(func, graph, ax):
    func_tbox = create_func_tbox()
    llim_tbox = create_llim_tbox()
    rlim_tbox = create_rlim_tbox()
    slevel_tbox = create_slevel_tbox()
    subd_tbox = create_subd_tbox()
    func_tbox.on_submit(lambda x: plot_func(x, func, graph, ax))
    llim_tbox.on_submit(lambda x: update_llim(x, func, graph, ax))
    rlim_tbox.on_submit(lambda x: update_rlim(x, func, graph, ax))
    slevel_tbox.on_submit(lambda x: update_slevel(x, func,  graph, ax))
    subd_tbox.on_submit(lambda x: get_subd(x, func,  graph, ax))
    plt.show()


def update_graph(func, graph, ax, caller=''):
    if func.function == None:
        return
    if caller != 'get_subd':
        graph.set_data(func.range, func.value)
        try:
            ax.set_xlim(np.nanmin(func.range), np.nanmax(func.range))
            ax.set_ylim(np.nanmin(func.value), np.nanmax(func.value))
        except ValueError:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
    plot_subdivision(func, ax)
    plt.draw()


def update_rlim(rlim, func, graph, ax):
    try:
        rlim = sp.sympify(rlim, convert_xor=True)
        rlim = rlim.evalf()
    except:
        return
    else:
        if type(rlim) == sp.core.numbers.Float and float(rlim) > func.llim:
            func.set_rlim(float(rlim))
            update_graph(func, graph, ax)
        else:
            return


def update_llim(llim, func, graph, ax):
    try:
        llim = sp.sympify(llim, convert_xor=True)
        llim = llim.evalf()
    except:
        return
    else:
        if type(llim) == sp.core.numbers.Float and float(llim) < func.rlim:
            func.set_llim(float(llim))
            update_graph(func, graph, ax)
        else:
            return


def update_slevel(slevel, func,  graph, ax):
    try:
        slevel = sp.sympify(slevel, convert_xor=True)
        slevel = slevel.evalf()
    except:
        return
    else:
        if type(slevel) == sp.core.numbers.Float and int(slevel) > SLEVEL:
            func.set_slevel(int(slevel))
            update_graph(func, graph, ax)
        else:
            return


def get_subd(subd, func, graph,  ax):
    try:
        subd = int(subd)
    except:
        return
    else:
        name = 'get_subd'
        subd = max(int(subd), SUBDIVISIONS)
        func.subd = subd
        update_graph(func, graph, ax, caller=name)


def plot_func(expr, func, graph, ax):
    func.create(expr)
    update_graph(func, graph, ax)


def clear_prev_patches(ax):
    patches = ax.patches
    while len(patches) != 0:
        for each_patch in patches:
            each_patch.remove()


def approximate_area(func_val, width):
    height_sum = 0
    for each in func_val:
        height_sum += 0 if np.isnan(each) else each
    approx_area = height_sum*width
    return approx_area


def set_title(func, approx):
    title = f'Riemann Sum (with {func.subd} subdivisions) = {approx}'
    plt.suptitle(title, fontsize=20)


def plot_subdivision(func, ax):
    clear_prev_patches(ax)
    xCords, width = np.linspace(
        func.llim, func.rlim, func.subd+1, retstep=True)
    xCords = xCords[:-1]
    func_val = get_func_values(func.function, xCords)
    lw = 1.0 if func.subd <= 400 else 0.1
    for i, height in zip(xCords, func_val):
        ax_rect = Rectangle((i, 0), width, height,
                            ec='green',
                            fc='lightgreen',
                            alpha=.6,
                            linewidth=lw)
        ax.add_patch(ax_rect)
    approx_area = approximate_area(func_val, width)
    set_title(func, approx_area)


def init():
    func = Function()
    fig, ax = plt.subplots()
    ax.set_facecolor('linen')
    plt.rcParams.update({'font.size': 17})
    graph, = ax.plot([], [], color='red', linewidth=2)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.grid()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.subplots_adjust(top=.93, bottom=.1, right=.99, left=.04)
    plt.suptitle('Integral approximation using Riemann sum', fontsize=20)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    create_tboxes(func, graph, ax)


if __name__ == '__main__':
    init()
