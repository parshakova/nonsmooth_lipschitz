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


def plot_trajectory_side_by_side(ws_left, ws_right, w0, c, n_show=100,
                                 titles=(None, None), filename=None):
    _set_style()
    ws_left_show = ws_left[:n_show]
    ws_right_show = ws_right[:n_show]

    all_ws = np.concatenate([ws_left_show, ws_right_show], axis=0)
    shift = 0.1
    x1_min = min(all_ws[:, 0].min(), 0) - shift
    x1_max = all_ws[:, 0].max() + shift
    x2_min = min(all_ws[:, 1].min(), 0) - shift
    x2_max = all_ws[:, 1].max() + shift

    # Each plot panel is 7 wide (matching plot_trajectory), height 4.5,
    # colorbar gets a thin column (0.3), with small gaps.
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[14.3, 0.3], wspace=0.05)
    gs_plots = gs[0, 0].subgridspec(1, 2, wspace=0.175)
    ax_left = fig.add_subplot(gs_plots[0, 0])
    ax_right = fig.add_subplot(gs_plots[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 1])
    axes = [ax_left, ax_right]

    x1 = np.linspace(x1_min, x1_max, 300)
    x2 = np.linspace(x2_min, x2_max, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = c * np.abs(X1 + X2) + np.abs(X1 - X2)
    x_line = np.linspace(x1_min, x1_max, 200)

    sc = None
    for i, (ax, ws_show) in enumerate(zip(axes, [ws_left_show, ws_right_show])):
        ax.set_facecolor("white")
        ax.grid(False)
        ax.contourf(X1, X2, Z, levels=30, cmap="coolwarm", alpha=0.6, zorder=0)
        ax.contour(X1, X2, Z, levels=30, colors="white", linewidths=0.4, alpha=0.4, zorder=0)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

        ax.plot(x_line, w0.sum() - x_line, color="forestgreen", linewidth=1., linestyle="-.",
                label=rf"$W_{{1, 1}} + W_{{2, 2}} = {int(w0.sum())}$")
        ax.plot(x_line, x_line, color="maroon",
                linewidth=1., linestyle="--", label=r"$W_{1, 1} = W_{2, 2}$")

        ax.plot(ws_show[:, 0], ws_show[:, 1], color="gray", alpha=0.3, linewidth=0.8, zorder=1)
        ax.scatter(0, 0, color="magenta", s=100, zorder=3, marker="*",
                   label=r"$(W_{1,1}^\star, W_{2,2}^\star)$")
        sc = ax.scatter(ws_show[:, 0], ws_show[:, 1], c=np.arange(n_show),
                        cmap="viridis", s=10, zorder=2, vmin=0, vmax=n_show - 1)

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel(r"$W_{1, 1}$", fontsize=22)
        ax.set_ylabel(r"$W_{2, 2}$", fontsize=22)
        ax.tick_params(labelsize=18)
        if titles[i] is not None:
            ax.set_title(titles[i], fontsize=22)

    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(r"Iteration $t$", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks(np.linspace(1, n_show, 5, dtype=int))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.9),
               ncol=3, frameon=False, fontsize=18)
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




def get_best_trial(df, label):
    """Return rows for the config with the lowest final validation loss."""
    final = df.dropna(subset=["valid/ce_loss"]).groupby("config")["valid/ce_loss"].last()
    best_config = final.idxmin()
    best_df = df[df["config"] == best_config].copy().sort_values("global_step")
    print(f"{label}: best config = {best_config}, final valid loss = {final.min():.4f}")
    return best_df


def plot_loss_vs_step(trials, metric, ylabel, filename=None, sparse=False):
    """Plot a metric vs. step for multiple optimizer trials."""
    COLORS = {
        "MuonMax": "#00518F",
        "EF-MuonMax": "#FF6B35",
        "EF-Muon": "#8A2BE2",
        "Vanilla Muon": "#2CA02C",
        "Muon": "#FF7F0E",
    }
    fig, ax = plt.subplots(figsize=(4, 3))
    for label, df in trials:
        sub = df.dropna(subset=[metric]) if sparse else df
        steps = sub["global_step"].values
        vals = sub[metric].values
        kwargs = dict(marker="o", markersize=3) if sparse else {}
        ax.plot(steps, vals, label=label, color=COLORS[label],
                linewidth=2, alpha=0.85, **kwargs)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    ax.legend(loc="upper right", fontsize=10)
    fig.subplots_adjust(top=0.95, bottom=0.17, left=0.17, right=0.95)
    if filename:
        plt.savefig(f"plots/{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_loss_vs_step_side_by_side(cifar_trials, trials, metric, ylabel,
                                   titles=("Image classification", "Language modeling"),
                                   filename=None, sparse=False):
    """Plot a metric vs. step for CIFAR and nanoGPT trials side by side with unified legend."""
    COLORS = {
        "MuonMax": "#00518F",
        "EF-MuonMax": "#FF6B35",
        "EF-Muon": "#8A2BE2",
        "Vanilla Muon": "#2CA02C",
        "Muon": "#FF7F0E",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, trial_list, title in zip(axes, [cifar_trials, trials], titles):
        for label, df in trial_list:
            sub = df.dropna(subset=[metric]) if sparse else df
            steps = sub["global_step"].values
            vals = sub[metric].values
            kwargs = dict(marker="o", markersize=3) if sparse else {}
            ax.plot(steps, vals, label=label, color=COLORS[label],
                    linewidth=2, alpha=0.85, **kwargs)
        ax.set_xlabel("Iteration", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(title, fontsize=22)
        ax.tick_params(labelsize=18)
        ax.grid(axis="both", lw=0.2, ls="--", zorder=0)

    # Unified legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False, fontsize=18)
    fig.subplots_adjust(top=0.85, bottom=0.17, left=0.10, right=0.95, wspace=0.20)
    if filename:
        plt.savefig(f"plots/{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_final_metric_vs_lr(dfs, metric, ylabel, filename=None):
    """Plot final value of a (sparse) metric vs. effective LR for each config."""
    COLORS = {
        "MuonMax": "#00518F",
        "EF-MuonMax": "#FF6B35",
        "EF-Muon": "#8A2BE2",
        "Vanilla Muon": "#2CA02C",
        "Muon": "#FF7F0E",
    }
    fig, ax = plt.subplots(figsize=(4, 3))
    for label, df in dfs:
        final = (df.dropna(subset=[metric])
                   .groupby("config")
                   .agg(lr=("base_lr", "first"),
                        mult=("lr_multiplier", "first"),
                        val=(metric, "last")))
        # final["eff_lr"] = final["lr"] * final["mult"]
        final["eff_lr"] = final["lr"]
        final = final.sort_values("eff_lr")
        ax.plot(final["eff_lr"], final["val"], label=label, color=COLORS[label],
                linewidth=2, alpha=0.85, marker="o", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    ax.legend(loc="upper left", fontsize=10)
    fig.subplots_adjust(top=0.95, bottom=0.17, left=0.17, right=0.95)
    if filename:
        plt.savefig(f"plots/{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
