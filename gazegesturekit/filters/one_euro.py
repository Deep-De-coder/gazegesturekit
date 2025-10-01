import math, time

class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
        self.x_prev=None; self.dx_prev=0.0; self.t_prev=None
    def alpha(self, cutoff): return 1.0/(1.0 + (self.freq/(2*math.pi*cutoff)))
    def __call__(self, x, t=None):
        t = t or time.time()
        if self.t_prev is None:
            self.t_prev=t; self.x_prev=x; return x
        dt = max(t - self.t_prev, 1e-6); self.freq = 1.0/dt; self.t_prev=t
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff); dx_hat = a_d*dx + (1-a_d)*self.dx_prev; self.dx_prev = dx_hat
        cutoff = self.min_cutoff + self.beta*abs(dx_hat); a = self.alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev; self.x_prev = x_hat
        return x_hat
