from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

COLORS = {"neg": "#2b6cb0", "pos": "#c05621", "line": "#1a202c", "muted": "#a0aec0", "accent": "#2f855a", "accent2": "#c14665"}
def _save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _node_label(model, node, feature_names, class_names, category_names):
    dist = node.value / max(node.value.sum(), 1e-12)
    pred = class_names[int(np.argmax(node.value))]
    head = f"n={node.n_samples}  {model.criterion}={node.impurity:.3f}\nP(canceled)={dist[-1]:.2f} → {pred}"
    if node.is_leaf:
        return head
    name = feature_names[node.feature]
    if node.threshold is not None:
        test = f"{name} ≤ {node.threshold:,.4g}"
    else:
        cats = category_names.get(node.feature) if category_names else None
        values = [cats[int(c)] if cats else str(int(c)) for c in node.left_categories]
        text = ", ".join(values)
        test = f"{name} ∈ {{{text if len(text) < 40 else text[:37] + '…'}}}"
    nan = "missing→left" if node.nan_left else "missing→right"
    return f"{test}\n{head}\n({nan})"


def plot_tree(model, feature_names, top_n=15, class_names=("not canceled", "canceled"),
              category_names=None, order="bfs", path=None, figsize=None):
    internal = [n for n in model.nodes() if not n.is_leaf]
    if order == "samples":
        chosen = sorted(internal, key=lambda n: (-n.weight, n.node_id))[:top_n]
    else:
        chosen = sorted(internal, key=lambda n: (n.depth, n.node_id))[:top_n]
    chosen_ids = {n.node_id for n in chosen}
    parent_of = {}
    for n in model.nodes():
        if not n.is_leaf:
            parent_of[n.left.node_id] = n
            parent_of[n.right.node_id] = n
    for n in list(chosen):
        p = parent_of.get(n.node_id)
        while p is not None and p.node_id not in chosen_ids:
            chosen_ids.add(p.node_id)
            p = parent_of.get(p.node_id)

    positions, edges, boxes = {}, [], []
    counter = [0]

    def layout(node, depth):
        expanded = (not node.is_leaf) and node.node_id in chosen_ids
        if not expanded:
            positions[node.node_id] = (counter[0], -depth)
            counter[0] += 1
            boxes.append((node, depth, node.is_leaf))
            return positions[node.node_id][0]
        xl = layout(node.left, depth + 1)
        xr = layout(node.right, depth + 1)
        positions[node.node_id] = ((xl + xr) / 2, -depth)
        boxes.append((node, depth, True))
        edges.append((node.node_id, node.left.node_id, "yes"))
        edges.append((node.node_id, node.right.node_id, "no"))
        return positions[node.node_id][0]

    layout(model.root_, 0)
    width = max(counter[0], 1)
    depth = max(-y for _, y in positions.values()) + 1
    fig, ax = plt.subplots(figsize=figsize or (max(10, width * 2.3), max(4, depth * 2.0)))
    ax.set_axis_off()
    for parent, child, label in edges:
        (x0, y0), (x1, y1) = positions[parent], positions[child]
        ax.plot([x0, x1], [y0 - 0.25, y1 + 0.25], color=COLORS["muted"], lw=1, zorder=1)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=7, ha="center",
                color=COLORS["line"], bbox=dict(fc="white", ec="none", pad=0.5))
    for node, _, shown in boxes:
        x, y = positions[node.node_id]
        if shown:
            text = _node_label(model, node, feature_names, class_names, category_names)
            p1 = node.value[-1] / max(node.value.sum(), 1e-12)
            face = plt.cm.RdBu_r(0.15 + 0.7 * p1)
            alpha = 0.35
        else:
            text = f"… subtree\n({node.n_samples} samples)"
            face, alpha = "#edf2f7", 1.0
        ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=2,
                bbox=dict(boxstyle="round,pad=0.4", fc=face, ec=COLORS["line"], lw=0.6, alpha=alpha))
    ax.set_xlim(-0.8, width - 0.2)
    ax.set_ylim(-depth + 0.4, 0.6)
    ax.set_title(f"CART — top {min(top_n, len(internal))} of {len(internal)} splits "
                 f"(depth {model.get_depth()}, {model.get_n_leaves()} leaves)")
    return _save(fig, path)


def log_loss_surface(x, y, theta0_grid, theta1_grid):
    T0, T1 = np.meshgrid(theta0_grid, theta1_grid)
    z = T0[..., None] + T1[..., None] * x[None, None, :]
    loss = np.logaddexp(0.0, z) - y[None, None, :] * z
    return T0, T1, loss.mean(axis=-1)


def animate_logreg_contour(x, y, trajectories, path, fps=20, max_frames=200,
                           padding=0.6, title="Log-loss contour and trajectory (θ₀, θ₁)"):
    allp = np.vstack(list(trajectories.values()))
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 0.5)
    t0 = np.linspace(lo[0] - padding * span[0], hi[0] + padding * span[0], 160)
    t1 = np.linspace(lo[1] - padding * span[1], hi[1] + padding * span[1], 160)
    T0, T1, J = log_loss_surface(np.asarray(x, float), np.asarray(y, float), t0, t1)
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    levels = np.quantile(J, np.linspace(0, 0.9, 25))
    cs = ax.contour(T0, T1, J, levels=np.unique(levels), cmap="viridis", linewidths=0.8)
    ax.clabel(cs, cs.levels[::4], fontsize=6, fmt="%.3f")
    i_min = np.unravel_index(np.argmin(J), J.shape)
    ax.plot(T0[i_min], T1[i_min], marker="*", color="k", ms=12, label="grid minimum")
    ax.set_xlabel("θ₀ (intercept)")
    ax.set_ylabel("θ₁ (feature coefficient)")
    ax.set_title(title, fontsize=10)
    palette = [COLORS["pos"], COLORS["accent2"], COLORS["accent"], COLORS["neg"]]
    names = list(trajectories)
    lines, dots, loss_lines, losses = [], [], [], []
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    for c, name in zip(palette, names):
        traj = trajectories[name]
        line, = ax.plot([], [], "-", color=c, lw=1.6, label=name)
        dot, = ax.plot([], [], "o", color=c, ms=6)
        z = traj[:, :1] + traj[:, 1:2] * xs[None, :]
        loss = (np.logaddexp(0.0, z) - ys[None, :] * z).mean(axis=1)
        ll, = axl.plot([], [], color=c, lw=1.6, label=name)
        lines.append(line)
        dots.append(dot)
        loss_lines.append(ll)
        losses.append(loss)
    ax.legend(fontsize=8, loc="upper right")
    longest = max(len(t) for t in trajectories.values())
    axl.set_xlim(0, longest)
    axl.set_ylim(min(l.min() for l in losses) * 0.98, max(l.max() for l in losses) * 1.02)
    axl.set_xlabel("step")
    axl.set_ylabel("log-loss")
    axl.set_title("Log-loss during training", fontsize=10)
    axl.legend(fontsize=8)
    frames = np.unique(np.linspace(0, longest - 1, min(max_frames, longest)).astype(int))

    def update(frame):
        artists = []
        for traj, line, dot, ll, loss in zip(trajectories.values(), lines, dots, loss_lines, losses):
            k = min(frame, len(traj) - 1)
            line.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
            dot.set_data([traj[k, 0]], [traj[k, 1]])
            ll.set_data(np.arange(k + 1), loss[:k + 1])
            artists += [line, dot, ll]
        return artists

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif" or not animation.writers.is_available("ffmpeg"):
        path = path.with_suffix(".gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    update(frames[-1])
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_svm_boundary(model, X2, y, ax=None, title=None, feature_names=("component 1", "component 2"),
                      resolution=250, path=None):
    X2 = np.asarray(X2, float)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    pad = 0.5
    xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, resolution),
                         np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    f = model.decision_function(grid).reshape(xx.shape)
    lim = np.quantile(np.abs(f), 0.98)
    ax.contourf(xx, yy, np.clip(f, -lim, lim), levels=30, cmap="RdBu_r", alpha=0.25)
    ax.contour(xx, yy, f, levels=[-1, 0, 1], colors=[COLORS["neg"], COLORS["line"], COLORS["pos"]],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2, 1.2])
    y = np.asarray(y)
    for label, color, name in [(model.classes_[0], COLORS["neg"], "not canceled"),
                               (model.classes_[1], COLORS["pos"], "canceled")]:
        m = y == label
        ax.scatter(X2[m, 0], X2[m, 1], s=9, color=color, alpha=0.6, label=name, edgecolors="none")
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=40, facecolors="none", edgecolors="k", linewidths=0.7,
               label=f"support vector ({len(sv)})")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title or f"SVM kernel {model.kernel}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    if own:
        _save(fig, path)
    return fig


def plot_bayes_network(feature_names, class_name="is_canceled", parents=None, edge_weights=None,
                       class_weights=None, title=None, path=None):
    d = len(feature_names)
    angle = np.linspace(0, 2 * np.pi, d, endpoint=False) + np.pi / 2
    pos = np.c_[np.cos(angle), np.sin(angle)] * 4.0
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    cw = np.ones(d) if class_weights is None else np.asarray(class_weights, float)
    cw_norm = cw / cw.max() if cw.max() > 0 else cw
    for i in range(d):
        ax.annotate("", xy=pos[i] * 0.86, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=0.6 + 3 * cw_norm[i],
                                    shrinkA=28, shrinkB=4))
    if parents is not None:
        ew = edge_weights if edge_weights is not None else np.ones((d, d))
        vals = [ew[i, p] for i, p in enumerate(parents) if p >= 0]
        top = max(vals) if vals else 1.0
        for i, p in enumerate(parents):
            if p < 0:
                continue
            ax.annotate("", xy=pos[i], xytext=pos[p],
                        arrowprops=dict(arrowstyle="-|>", color=COLORS["accent2"],
                                        lw=0.8 + 3 * ew[i, p] / top, shrinkA=22, shrinkB=22,
                                        connectionstyle="arc3,rad=0.15"))
    ax.add_patch(FancyBboxPatch((-1.1, -0.35), 2.2, 0.7, boxstyle="round,pad=0.1",
                                fc="#fefcbf", ec=COLORS["line"]))
    ax.text(0, 0, class_name, ha="center", va="center", fontsize=11, weight="bold")
    for i, name in enumerate(feature_names):
        is_root = parents is not None and parents[i] < 0
        ax.text(*pos[i], name.replace("_", "_\n", 1) if len(name) > 16 else name, ha="center",
                va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#c6f6d5" if is_root else "#ebf8ff",
                          ec=COLORS["line"], lw=0.6))
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    kind = "Tree-Augmented Naive Bayes" if parents is not None else "Naive Bayes"
    ax.set_title(title or f"Bayesian Network structure — {kind}")
    return _save(fig, path)


def plot_confusion(cm, ax, title, labels=("not canceled", "canceled")):
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=9)


def plot_metric_comparison(table, metric_cols, path=None, title="Model comparison"):
    fig, ax = plt.subplots(figsize=(max(8, len(table) * 1.2), 4.5))
    width = 0.8 / len(metric_cols)
    x = np.arange(len(table))
    palette = [COLORS["neg"], COLORS["pos"], COLORS["accent"], COLORS["accent2"], COLORS["muted"]]
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width - 0.4 + width / 2, table[col], width, label=col,
               color=palette[i % len(palette)])
    ax.set_xticks(x, table.index, rotation=30, ha="right", fontsize=8)
    lo = max(0.0, float(table[metric_cols].min().min()) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    return _save(fig, path)


def plot_threshold_curve(thresholds, curves: dict, path=None, ylabel="macro F1", title=None,
                         mark=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        ax.plot(thresholds, values, label=name)
    if mark is not None:
        for x, label in mark:
            ax.axvline(x, color=COLORS["muted"], ls="--", lw=1)
            ax.text(x, ax.get_ylim()[1], label, fontsize=7, rotation=90, va="top", ha="right")
    ax.set_xlabel("threshold P(canceled)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return _save(fig, path)