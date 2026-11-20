import re
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons

plt.style.use("dark_background")
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
G = 1.0


# ----------------------------------------------------------------------------
# Fysik
# ----------------------------------------------------------------------------
class Sim:
    def __init__(self, n, m, x, v):
        self.n = n
        self.m0, self.x0, self.v0 = m.copy(), x.copy(), v.copy()
        self.reset()

    def reset(self):
        self.m, self.x, self.v = self.m0.copy(), self.x0.copy(), self.v0.copy()
        self.t = 0.0
        self.E0 = self.energy()

    def acc(self, x):
        diff = x[None, :, :] - x[:, None, :]          # diff[i, j] = x_j - x_i
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        w = self.m[None, :] / dist ** self.n
        np.fill_diagonal(w, 0.0)
        return G * (w[:, :, None] * diff).sum(axis=1)

    def step(self, dt, method="RK4"):
        x, v = self.x, self.v
        if method == "Euler":
            a = self.acc(x)
            self.x = x + v * dt
            self.v = v + a * dt
        else:  # RK4
            k1x, k1v = v, self.acc(x)
            k2x, k2v = v + dt / 2 * k1v, self.acc(x + dt / 2 * k1x)
            k3x, k3v = v + dt / 2 * k2v, self.acc(x + dt / 2 * k2x)
            k4x, k4v = v + dt * k3v, self.acc(x + dt * k3x)
            self.x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            self.v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.t += dt

    def pair_dist(self):
        return np.linalg.norm(self.x[:, None, :] - self.x[None, :, :], axis=2)

    def radius(self):
        return 0.12 * self.m ** (1.0 / 3.0)

    def energy(self):
        ke = 0.5 * np.sum(self.m * np.sum(self.v ** 2, axis=1))
        d = self.pair_dist()
        i, j = np.triu_indices(len(self.m), 1)
        pe = -G * np.sum(self.m[i] * self.m[j] / ((self.n - 2) * d[i, j] ** (self.n - 2)))
        return ke + pe

    def collision(self):
        d = self.pair_dist()
        r = self.radius()
        i, j = np.triu_indices(len(self.m), 1)
        hit = np.where(d[i, j] < r[i] + r[j])[0]
        if hit.size:
            return int(i[hit[0]]), int(j[hit[0]])
        return None


def random_unit(n, rng):
    u = rng.normal(size=n)
    return u / np.linalg.norm(u)


def make_solar(n, N, rng):
    M = 100.0
    m = np.empty(N)
    x = np.zeros((N, n))
    v = np.zeros((N, n))
    m[0] = M
    for i in range(1, N):
        m[i] = rng.uniform(0.2, 1.0)
        r = 2.5 + 2.0 * (i - 1)
        u = random_unit(n, rng)
        t = random_unit(n, rng)
        t -= t.dot(u) * u
        t /= np.linalg.norm(t)
        x[i] = r * u
        v[i] = np.sqrt(G * M / r ** (n - 2)) * t
    v[0] = -(m[1:, None] * v[1:]).sum(axis=0) / M      # total rörelsemängd = 0
    return m, x, v


def make_random(n, N, rng):
    m = rng.uniform(1.0, 10.0, size=N)
    x = rng.normal(size=(N, n)) * 3.0
    v = rng.normal(size=(N, n)) * 0.5
    x -= (m[:, None] * x).sum(axis=0) / m.sum()        # masscentrum i origo
    v -= (m[:, None] * v).sum(axis=0) / m.sum()        # och i vila
    return m, x, v


# ----------------------------------------------------------------------------
# Projektion n-D -> 3D
# ----------------------------------------------------------------------------
def project(P, theta, dist):
    P = np.array(P, dtype=float)
    n = P.shape[-1]
    c, s = np.cos(theta), np.sin(theta)
    a, b = P[..., 0].copy(), P[..., n - 1].copy()
    P[..., 0] = c * a - s * b
    P[..., n - 1] = s * a + c * b
    for k in range(n, 3, -1):
        den = np.maximum(dist - P[..., k - 1], 0.1 * dist)
        P = P[..., : k - 1] * (dist / den)[..., None]
    return P


def parse_vec(text, n):
    parts = [p for p in re.split(r"[,\s;]+", text.strip()) if p]
    vals = [float(p) for p in parts]
    if len(vals) != n:
        raise ValueError(f"Ange exakt {n} tal")
    return np.array(vals)


def fmt_vec(a):
    return ", ".join(f"{c:.4g}" for c in a)


# ----------------------------------------------------------------------------
# Användargränssnitt
# ----------------------------------------------------------------------------
class App:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.n, self.N = 4, 4
        self.running = False
        self.method = "RK4"
        self.trail_on = True
        self.autorot = False
        self.theta = 0.0
        self.sel = 0
        self.trail = deque(maxlen=800)
        self.lines_body, self.lines_trail = [], []

        self.fig = plt.figure(figsize=(14, 8))
        try:
            self.fig.canvas.manager.set_window_title("n-dimensionell gravitation")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.0, 0.02, 0.70, 0.96], projection="3d")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.build_widgets()
        self.new_preset("solar")
        self.anim = FuncAnimation(self.fig, self.update, interval=30, cache_frame_data=False)

    # Widgets
    def build_widgets(self):
        f = self.fig
        f.text(0.84, 0.955, "n-dimensionell gravitation", ha="center", fontsize=14, weight="bold")

        def slider(y, label, lo, hi, init, step=None):
            ax = f.add_axes([0.80, y, 0.17, 0.025])
            s = Slider(ax, label, lo, hi, valinit=init, valstep=step, color="#4c9be8")
            s.label.set_fontsize(9)
            s.valtext.set_fontsize(9)
            return s

        self.sl_n = slider(0.90, "Dimensioner n  ", 3, 8, self.n, 1)
        self.sl_N = slider(0.86, "Antal kroppar  ", 2, 10, self.N, 1)
        self.sl_dt = slider(0.82, "Tidssteg dt  ", 0.001, 0.02, 0.005)
        self.sl_speed = slider(0.78, "Hastighet  ", 1, 30, 5, 1)
        self.sl_rot = slider(0.74, "Rotation (x-w)  ", 0, 2 * np.pi, 0.0)
        self.sl_n.on_changed(lambda v: self.on_dims())
        self.sl_N.on_changed(lambda v: self.on_dims())
        self.sl_rot.on_changed(self.on_rot)

        f.text(0.73, 0.700, "Numerisk metod", fontsize=9)
        ax = f.add_axes([0.73, 0.625, 0.10, 0.07])
        self.radio = RadioButtons(ax, ("RK4", "Euler"), active=0, activecolor="#4c9be8")
        self.radio.on_clicked(self.on_method)

        def button(x, y, label, cb, w=0.11):
            ax = f.add_axes([x, y, w, 0.045])
            b = Button(ax, label, color="#2b2f36", hovercolor="#3d4450")
            b.on_clicked(cb)
            return b

        self.b_run = button(0.73, 0.555, "Start", self.on_run)
        self.b_reset = button(0.85, 0.555, "Återställ", lambda e: self.apply_ic())
        self.b_solar = button(0.73, 0.500, "Solsystem", lambda e: self.new_preset("solar"))
        self.b_rand = button(0.85, 0.500, "Slumpa", lambda e: self.new_preset("random"))
        self.b_trail = button(0.73, 0.445, "Spår: på", self.on_trail)
        self.b_rot = button(0.85, 0.445, "Auto-rotera", self.on_autorot)

        f.text(0.73, 0.395, "Redigera kropp (startvärden)", fontsize=10, weight="bold")
        self.sl_body = slider(0.355, "Kropp nr  ", 1, self.N, 1, 1)
        self.sl_body.on_changed(self.on_body)

        def tbox(y, label):
            ax = f.add_axes([0.80, y, 0.17, 0.04])
            return TextBox(ax, label, textalignment="left", color="#2b2f36", hovercolor="#3d4450")

        self.tb_m = tbox(0.30, "Massa  ")
        self.tb_x = tbox(0.25, "Position  ")
        self.tb_v = tbox(0.20, "Hastighet  ")
        self.tb_m.on_submit(self.on_edit)
        self.tb_x.on_submit(self.on_edit)
        self.tb_v.on_submit(self.on_edit)
        f.text(0.73, 0.15, "Skriv n tal separerade med kommatecken\n(t.ex. 1, 0, 0, 0) och tryck Enter.",
               fontsize=8, color="#aaaaaa")

        self.status = f.text(0.02, 0.95, "", color="#ff6b6b", fontsize=12, weight="bold")
        self.info = f.text(0.02, 0.03, "", fontsize=10, family="monospace")
# Tillstånd
    def new_preset(self, kind):
        self.n, self.N = int(self.sl_n.val), int(self.sl_N.val)
        maker = make_solar if kind == "solar" else make_random
        m, x, v = maker(self.n, self.N, self.rng)
        self.sim = Sim(self.n, m, x, v)
        self.sel = min(self.sel, self.N - 1)
        self.sl_body.valmax = self.N
        self.sl_body.ax.set_xlim(1, self.N)
        self.sync_boxes()
        self.apply_ic()

    def apply_ic(self):
        self.sim.reset()
        self.running = False
        self.b_run.label.set_text("Start")
        self.status.set_text("")
        self.trail.clear()
        self.trail.append(self.sim.x.copy())
        ext = np.abs(self.sim.x0).max()
        self.L = max(1.3 * ext, 3.0)
        self.dist = max(5 * ext, 20.0)
        self.rebuild_artists()
        self.draw()

    def rebuild_artists(self):
        for l in self.lines_body + self.lines_trail:
            l.remove()
        self.lines_body, self.lines_trail = [], []
        mmax = self.sim.m.max()
        for i in range(self.N):
            c = COLORS[i % 10]
            ms = 4 + 8 * (self.sim.m[i] / mmax) ** (1 / 3)
            self.lines_trail.append(self.ax.plot([], [], [], "-", color=c, lw=1.0, alpha=0.6)[0])
            self.lines_body.append(self.ax.plot([], [], [], "o", color=c, ms=ms)[0])
        self.set_limits()

    def set_limits(self):
        L = self.L
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_zlim(-L, L)
        self.ax.set_box_aspect((1, 1, 1))

    def sync_boxes(self):
        s = self.sim
        i = self.sel
        self.tb_m.set_val(f"{s.m0[i]:.4g}")
        self.tb_x.set_val(fmt_vec(s.x0[i]))
        self.tb_v.set_val(fmt_vec(s.v0[i]))

    # callbacks
    def on_dims(self):
        self.new_preset("solar")

    def on_rot(self, val):
        self.theta = val
        self.draw()

    def on_method(self, label):
        self.method = label
        self.apply_ic()

    def on_run(self, _):
        if self.status.get_text():
            self.apply_ic()
        self.running = not self.running
        self.b_run.label.set_text("Paus" if self.running else "Start")

    def on_trail(self, _):
        self.trail_on = not self.trail_on
        self.b_trail.label.set_text("Spår: på" if self.trail_on else "Spår: av")
        if self.trail_on:
            self.trail.clear()
            self.trail.append(self.sim.x.copy())

    def on_autorot(self, _):
        self.autorot = not self.autorot

    def on_body(self, val):
        self.sel = int(val) - 1
        self.sync_boxes()

    def on_edit(self, _):
        i, n, s = self.sel, self.n, self.sim
        try:
            mass = float(self.tb_m.text)
            if mass <= 0:
                raise ValueError("Massan måste vara > 0")
            s.m0[i] = mass
            s.x0[i] = parse_vec(self.tb_x.text, n)
            s.v0[i] = parse_vec(self.tb_v.text, n)
        except ValueError as e:
            self.status.set_text(f"Ogiltigt värde: {e}")
            return
        self.apply_ic()

    # animation
    def update(self, _):
        s = self.sim
        if self.running:
            for _ in range(int(self.sl_speed.val)):
                s.step(self.sl_dt.val, self.method)
                hit = s.collision()
                if hit is not None:
                    self.status.set_text(f"Kropp {hit[0] + 1} och {hit[1] + 1} har kolliderat – pausad")
                    self.running = False
                    break
                if not np.isfinite(s.x).all():
                    self.status.set_text("Numeriskt fel – pausad")
                    self.running = False
                    break
            if not self.running:
                self.b_run.label.set_text("Start")
            if self.trail_on:
                self.trail.append(s.x.copy())
        if self.autorot:
            self.theta = (self.theta + 0.01) % (2 * np.pi)
            self.sl_rot.eventson = False
            self.sl_rot.set_val(self.theta)
            self.sl_rot.eventson = True
        self.draw()

    def draw(self):
        s = self.sim
        P = project(s.x, self.theta, self.dist)
        T = project(np.array(self.trail), self.theta, self.dist)     # (T, N, 3)
        big = max(np.abs(P).max(), 0.0)
        if big > self.L:
            self.L = 1.2 * big
            self.set_limits()
        for i in range(self.N):
            self.lines_body[i].set_data_3d([P[i, 0]], [P[i, 1]], [P[i, 2]])
            if self.trail_on and len(T) > 1:
                self.lines_trail[i].set_data_3d(T[:, i, 0], T[:, i, 1], T[:, i, 2])
            else:
                self.lines_trail[i].set_data_3d([], [], [])
        E = s.energy()
        dE = (E - s.E0) / abs(s.E0) if s.E0 != 0 else 0.0
        self.ax.set_title(f"{self.n}D -> 3D (perspektivprojektion)", fontsize=11)
        self.info.set_text(f"t = {s.t:8.3f}   metod: {self.method:5s}   relativt energifel ΔE/|E0| = {dE: .2e}")


if __name__ == "__main__":
    app = App()
    plt.show()
