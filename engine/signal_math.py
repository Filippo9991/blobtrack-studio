import math

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0):
        self.min_cutoff = min_cutoff # Frequenza di taglio minima (stabilizza quando lento)
        self.beta = beta             # Risposta alla velocità (riduce latenza quando veloce)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev # Evita divisioni per zero

        # Calcola la derivata (velocità del cambiamento)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(self.smoothing_factor(t_e, 1.0), dx, self.dx_prev)

        # Calcola cutoff dinamico
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class HysteresisTrigger:
    """Gestisce gli stati (es. IDLE -> PEAK) con memoria per evitare sfarfallio"""
    def __init__(self, threshold_high, threshold_low):
        self.high = threshold_high
        self.low = threshold_low
        self.state = False # False = Idle, True = Active

    def update(self, value):
        if not self.state and value > self.high:
            self.state = True
        elif self.state and value < self.low:
            self.state = False
        return self.state