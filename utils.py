import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




def _set_style():
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })
    except Exception:
        plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })


def f(w, c):
    return c * np.abs(w[0] + w[1]) + np.abs(w[0] - w[1])


def sign_s(x, s):
    x = np.asarray(x)
    return np.where(x==0, s, np.sign(x))


def subgrad_f(w, s, c, p, q): 
    return c * sign_s(w[0] + w[1], s) * p + sign_s(w[0] - w[1], s) * q


def sign_subgradient_descent(w0, s, c, p, q, alpha_0=1, max_iters=1000):
    w = np.copy(w0)
    logging = {"loss": [], "w": []}
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        D = sign_s(G, s)  
        alpha_t = alpha_0 / (t + 1)**0.51
        w = w - alpha_t * D
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy())
    return w, logging


def sign_subgradient_descent_ef(w0, s, c, p, q, max_iters=1000, fixed_gamma=True, gamma0=None):
    w = np.copy(w0)
    E = np.zeros_like(w0, dtype=float)
    gamma = 1 / (max_iters**0.5) if gamma0 is None else gamma0
    logging = {"loss": [], "w": []}
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        gamma_t = 1 / ((t+1)**0.5) if not fixed_gamma else gamma
        M = gamma_t * G + E
        D = sign_s(M, s)
        rank = (M != 0).sum()
        delta = (np.abs(M).sum() / rank) * D
        w = w - delta
        E = M - delta
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy())
    return w, logging


def sign_subgradient_descent_ef_momentum(w0, s, c, p, q, beta, max_iters=1000, fixed_gamma=True, gamma0=None):
    w = np.copy(w0)
    E = np.zeros_like(w0, dtype=float)
    M = np.zeros_like(w0, dtype=float)
    gamma = 1 / (max_iters**0.5) if gamma0 is None else gamma0
    logging = {"loss": [], "w": []}
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        M = beta * M + (1 - beta) * G 
        gamma_t = 1 / ((t+1)**0.5) if not fixed_gamma else gamma
        P = gamma_t * M + E
        D = sign_s(P, s)
        rank = (P != 0).sum()
        delta = (np.abs(P).sum() / rank) * D
        w = w - delta
        E = P - delta
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy()) 
    return w, logging


def sign_subgradient_descent_momentum(w0, s, c, p, q, beta, max_iters=1000):
    w = np.copy(w0)
    M = np.zeros_like(w0, dtype=float) 
    logging = {"loss": [], "w": []}
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        alpha_t = 1 / (t + 1)
        M = beta * M + (1-beta) * G 
        D = sign_s(M, s) 
        w = w - alpha_t * D 
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy())
    return w, logging


def sign_subgradient_descent_momentum_beta_to_1(w0, s, c, p, q, max_iters=1000):
    w = np.copy(w0)
    M = np.zeros_like(w0, dtype=float) 
    logging = {"loss": [], "w": []}
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        alpha_t = 1 / (t + 1)
        beta = 1 - 1 / (t + 1)
        M = beta * M + (1-beta) * G 
        D = sign_s(M, s) 
        w = w - alpha_t * D 
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy())
    return w, logging


def sign_subgradient_descent_polyak(w0, s, c, p, q, f_star=0, max_iters=1000):
    w = np.copy(w0)
    logging = {"loss": [], "w": []}
    logging["loss"].append(f(w, c).item())
    logging["w"].append(w.copy())
    for t in range(max_iters):
        G = subgrad_f(w, s, c, p, q)
        D = sign_s(G, s)  
        alpha_t = (logging["loss"][-1] - f_star) / (np.abs(G).sum())
        w = w - alpha_t * D
        logging["loss"].append(f(w, c).item())
        logging["w"].append(w.copy())
    return w, logging


def plot_loss_and_w_sum(ws, logging, filename=None, xlog=False, max_iter=None):
    _set_style()

    if max_iter is None: max_iter = len(logging["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(logging["loss"][:max_iter], label=r"$f(W_t)$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Iteration $t$")
    axes[0].set_ylabel(r"$f(W_t)$")
    # axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    w_sum = ws[:max_iter].sum(axis=1)
    axes[1].plot(w_sum, label=r"$(W_t)_{1,1} + (W_t)_{2,2}$")
    # axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Iteration $t$")
    axes[1].set_ylabel(r"$(W_t)_{1,1} + (W_t)_{2, 2}$")
    if w_sum.max() - w_sum.min() < 1e-10:
        mid = w_sum.mean()
        axes[1].set_ylim(mid * 0.95, mid * 1.05)
    # axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if xlog:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    plt.tight_layout()
    if filename:
        plt.savefig(f"plots/{filename}.pdf", bbox_inches="tight")
    plt.show()


def plot_trajectory(ws, w0, c, n_show=100, filename=None, figsize=(7, 4.5)):
    _set_style()
    ws_show = ws[:n_show]

    shift = 0.1
    x1_min = min(ws_show[:, 0].min(), 0) - shift
    x1_max = ws_show[:, 0].max() + shift
    x2_min = min(ws_show[:, 1].min(), 0) - shift
    x2_max = ws_show[:, 1].max() + shift

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.grid(False)

    x1 = np.linspace(x1_min, x1_max, 300)
    x2 = np.linspace(x2_min, x2_max, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = c * np.abs(X1 + X2) + np.abs(X1 - X2)
    ax.contourf(X1, X2, Z, levels=30, cmap="coolwarm", alpha=0.6, zorder=0)
    ax.contour(X1, X2, Z, levels=30, colors="white", linewidths=0.4, alpha=0.4, zorder=0)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    x_line = np.linspace(x1_min, x1_max, 200)
    ax.plot(x_line, w0.sum() - x_line, color="forestgreen", linewidth=1., linestyle="-.",
            label=rf"$W_{{1, 1}} + W_{{2, 2}} = {int(w0.sum())}$")
    ax.plot(x_line, x_line, color="maroon",
            linewidth=1., linestyle="--", label=r"$W_{1, 1} = W_{2, 2}$")

    ax.plot(ws_show[:, 0], ws_show[:, 1], color="gray", alpha=0.3, linewidth=0.8, zorder=1)
    ax.scatter(0, 0, color="magenta", s=100, zorder=3, marker="*", label=r"$(W_{1,1}^\star, W_{2,2}^\star)$")
    sc = ax.scatter(ws_show[:, 0], ws_show[:, 1], c=np.arange(n_show), cmap="viridis", s=10, zorder=2)
    cbar = plt.colorbar(sc, ax=ax, label=r"Iteration $t$")
    cbar.set_ticks(np.linspace(1, n_show, 5, dtype=int))

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel(r"$W_{1, 1}$")
    ax.set_ylabel(r"$W_{2, 2}$")
    plt.tight_layout()
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    if filename is not None:
        plt.savefig(f"plots/{filename}_trajectory.pdf", bbox_inches="tight")
    plt.show()


def countex_level_sets(c, xlim, ylim, filename=None, figsize=(6, 4), show_ticks=True):
    _set_style()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.grid(False)

    x1 = np.linspace(xlim[0], xlim[1], 300)
    x2 = np.linspace(ylim[0], ylim[1], 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = c * np.abs(X1 + X2) + np.abs(X1 - X2)
    cf = ax.contourf(X1, X2, Z, levels=30, cmap="coolwarm", alpha=0.6, zorder=0)
    ax.contour(X1, X2, Z, levels=30, colors="white", linewidths=0.4, alpha=0.4, zorder=0)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x_line = np.linspace(xlim[0], xlim[1], 200)
    ax.plot(x_line, x_line, color="maroon",
            linewidth=1., linestyle="--", label=r"$W_{1, 1} = W_{2, 2}$")

    ax.scatter(0, 0, color="magenta", s=100, zorder=3, marker="*", label=r"$(W_{1,1}^\star, W_{2,2}^\star)$")

    cbar = plt.colorbar(cf, ax=ax, label=r"$f(W)$")
    cbar.locator = plt.MaxNLocator(4)
    cbar.update_ticks()
    ax.set_xlabel(r"$W_{1, 1}$")
    ax.set_ylabel(r"$W_{2, 2}$")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    if filename is not None:
        plt.savefig(f"plots/{filename}_levelsets.pdf", bbox_inches="tight")
    plt.show()
